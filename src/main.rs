mod codegen;
mod editor;
mod graph;

use eframe::{egui, egui_wgpu, wgpu};
use graph::{Graph, MathOp, NodeKind, Tiling};

/// shortcut: fixed-size param buffer (16 KiB); grow on demand if graphs ever need more.
const MAX_PARAMS: usize = 4096;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    cam_pos: [f32; 3],
    aspect: f32,
    cam_fwd: [f32; 3],
    amount: f32,
    cam_right: [f32; 3],
    step_scale: f32,
    cam_up: [f32; 3],
    seed: u32,
    tile_size: f32,
    _pad: [f32; 3],
}

struct Gpu {
    pipeline: wgpu::RenderPipeline,
    uniforms: wgpu::Buffer,
    params: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

struct Viewport {
    uniforms: Uniforms,
    params: Vec<f32>,
}

impl egui_wgpu::CallbackTrait for Viewport {
    fn prepare(
        &self,
        _device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen: &egui_wgpu::ScreenDescriptor,
        _encoder: &mut wgpu::CommandEncoder,
        res: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        let gpu: &Gpu = res.get().unwrap();
        queue.write_buffer(&gpu.uniforms, 0, bytemuck::bytes_of(&self.uniforms));
        queue.write_buffer(&gpu.params, 0, bytemuck::cast_slice(&self.params));
        Vec::new()
    }

    fn paint(&self, _info: egui::PaintCallbackInfo, pass: &mut wgpu::RenderPass<'static>, res: &egui_wgpu::CallbackResources) {
        let gpu: &Gpu = res.get().unwrap();
        pass.set_pipeline(&gpu.pipeline);
        pass.set_bind_group(0, &gpu.bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
}

fn build_gpu(rs: &egui_wgpu::RenderState, graph_wgsl: &str) -> Gpu {
    let device = &rs.device;
    let src = format!("{}\n{graph_wgsl}\n{}", codegen::NOISE_LIB, include_str!("raymarch.wgsl"));
    // shortcut: no error scope here; codegen output is naga-validated in tests. Add one if users hit shader panics.
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("raymarch"),
        source: wgpu::ShaderSource::Wgsl(src.into()),
    });
    let entry = |binding, ty| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Buffer { ty, has_dynamic_offset: false, min_binding_size: None },
        count: None,
    };
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            entry(0, wgpu::BufferBindingType::Uniform),
            entry(1, wgpu::BufferBindingType::Storage { read_only: true }),
        ],
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[Some(&bgl)],
        immediate_size: 0,
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("raymarch"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs"),
            compilation_options: Default::default(),
            targets: &[Some(rs.target_format.into())],
        }),
        primitive: Default::default(),
        depth_stencil: None,
        multisample: Default::default(),
        multiview_mask: None,
        cache: None,
    });
    let buffer = |label, size, usage| {
        device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size, usage: usage | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false })
    };
    let uniforms = buffer("uniforms", std::mem::size_of::<Uniforms>() as u64, wgpu::BufferUsages::UNIFORM);
    let params = buffer("params", (MAX_PARAMS * 4) as u64, wgpu::BufferUsages::STORAGE);
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: uniforms.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: params.as_entire_binding() },
        ],
    });
    Gpu { pipeline, uniforms, params, bind_group }
}

struct App {
    rs: egui_wgpu::RenderState,
    graph: Graph,
    editor: editor::Editor,
    /// File the graph was last opened from / saved to.
    path: Option<std::path::PathBuf>,
    status: String,
    /// WGSL the current pipeline was built from; a different compile result triggers a rebuild.
    built_wgsl: Option<String>,
    yaw: f32,
    pitch: f32,
    dist: f32,
    amount: f32,
    tile_size: f32,
    seed: u32,
    step_scale: f32,
}

impl App {
    /// Preview defaults: displacement, tile size, global seed, step safety.
    const DEFAULTS: (f32, f32, u32, f32) = (0.25, 1.0, 0, 0.6);

    fn new(cc: &eframe::CreationContext) -> Self {
        egui_extras::install_image_loaders(&cc.egui_ctx);
        let (amount, tile_size, seed, step_scale) = Self::DEFAULTS;
        Self {
            rs: cc.wgpu_render_state.clone().expect("Noiser needs the wgpu renderer"),
            graph: Graph::default_graph(),
            editor: editor::Editor::default(),
            path: None,
            status: String::new(),
            built_wgsl: None,
            yaw: 0.6,
            pitch: 0.35,
            dist: 3.5,
            amount,
            tile_size,
            seed,
            step_scale,
        }
    }

    fn uniforms(&self, aspect: f32) -> Uniforms {
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        let pos = [self.dist * cp * sy, self.dist * sp, self.dist * cp * cy];
        let fwd = norm([-pos[0], -pos[1], -pos[2]]);
        let right = norm(cross(fwd, [0.0, 1.0, 0.0]));
        let up = cross(right, fwd);
        Uniforms {
            cam_pos: pos,
            aspect,
            cam_fwd: fwd,
            amount: self.amount,
            cam_right: right,
            step_scale: self.step_scale,
            cam_up: up,
            seed: self.seed,
            tile_size: self.tile_size,
            _pad: [0.0; 3],
        }
    }

    fn file_dialog() -> rfd::FileDialog {
        rfd::FileDialog::new().add_filter("Noiser graph", &["noisegraph"])
    }

    fn open(&mut self) {
        let Some(path) = Self::file_dialog().pick_file() else { return };
        match std::fs::read_to_string(&path).map_err(|e| e.to_string()).and_then(|s| Graph::from_json(&s)) {
            Ok(g) => {
                self.graph = g;
                self.editor.selected.clear();
                self.status = format!("Opened {}", path.display());
                self.path = Some(path);
            }
            Err(e) => self.status = format!("Could not open {}: {e}", path.display()),
        }
    }

    fn save(&mut self, ask: bool) {
        let path = match (&self.path, ask) {
            (Some(p), false) => p.clone(),
            _ => match Self::file_dialog().set_file_name("untitled.noisegraph").save_file() {
                Some(p) => p,
                None => return,
            },
        };
        match std::fs::write(&path, self.graph.to_json()) {
            Ok(()) => {
                self.status = format!("Saved {}", path.display());
                self.path = Some(path);
            }
            Err(e) => self.status = format!("Could not save {}: {e}", path.display()),
        }
    }
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

fn norm(v: [f32; 3]) -> [f32; 3] {
    let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    [v[0] / l, v[1] / l, v[2] / l]
}

/// Small icon button that puts `value` back to `default`; disabled while it already is.
fn reset<T: PartialEq>(ui: &mut egui::Ui, value: &mut T, default: T) {
    let icon = egui::Image::new(egui::include_image!("../assets/icons/restart-alt.svg"))
        .fit_to_exact_size(egui::vec2(14.0, 14.0))
        .tint(ui.visuals().text_color());
    let btn = ui.add_enabled(*value != default, egui::Button::image(icon).frame(false));
    if btn.on_hover_text("Reset to default").clicked() {
        *value = default;
    }
}

/// A row with a reset button in front of the widget.
fn row<T: PartialEq>(ui: &mut egui::Ui, value: &mut T, default: T, widget: impl FnOnce(&mut egui::Ui, &mut T)) {
    ui.horizontal(|ui| {
        reset(ui, value, default);
        widget(ui, value);
    });
}

/// Properties of one node.
fn node_props(ui: &mut egui::Ui, kind: &mut NodeKind) {
    match kind {
        NodeKind::Float(_) | NodeKind::Vec3(_) => {
            let is_vec = matches!(kind, NodeKind::Vec3(_));
            ui.horizontal(|ui| {
                if ui.selectable_label(!is_vec, "Float").clicked() && is_vec {
                    if let NodeKind::Vec3(v) = kind {
                        *kind = NodeKind::Float(v[0]);
                    }
                }
                if ui.selectable_label(is_vec, "Vector").clicked() && !is_vec {
                    if let NodeKind::Float(v) = kind {
                        *kind = NodeKind::Vec3([*v; 3]);
                    }
                }
            });
            let drag = |ui: &mut egui::Ui, x: &mut f32| _ = ui.add(egui::DragValue::new(x).speed(0.01));
            match kind {
                NodeKind::Float(v) => row(ui, v, 1.0, drag),
                NodeKind::Vec3(v) => row(ui, v, [1.0; 3], |ui, v| v.iter_mut().for_each(|x| drag(ui, x))),
                _ => {}
            }
        }
        NodeKind::Math { op, a, b } => {
            row(ui, op, MathOp::Add, |ui, op| {
                egui::ComboBox::from_label("operation").selected_text(format!("{op:?}")).show_ui(ui, |ui| {
                    for o in [MathOp::Add, MathOp::Subtract, MathOp::Multiply, MathOp::Divide, MathOp::Min, MathOp::Max] {
                        ui.selectable_value(op, o, format!("{o:?}"));
                    }
                });
            });
            ui.label("Used when the input is not connected:");
            row(ui, a, 0.0, |ui, a| _ = ui.add(egui::DragValue::new(a).speed(0.01).prefix("A ")));
            row(ui, b, 0.0, |ui, b| _ = ui.add(egui::DragValue::new(b).speed(0.01).prefix("B ")));
        }
        NodeKind::Normal { flip } => row(ui, flip, false, |ui, f| _ = ui.checkbox(f, "flip")),
        NodeKind::Noise(n) => {
            let d = graph::NoiseParams::default();
            row(ui, &mut n.scale, d.scale, |ui, v| _ = ui.add(egui::Slider::new(v, 0.5..=32.0).text("scale (cells/tile)")));
            row(ui, &mut n.normalize, d.normalize, |ui, v| _ = ui.checkbox(v, "normalize 0-1"));
            row(ui, &mut n.octaves, d.octaves, |ui, v| _ = ui.add(egui::Slider::new(v, 1..=8).text("octaves")));
            row(ui, &mut n.lacunarity, d.lacunarity, |ui, v| _ = ui.add(egui::Slider::new(v, 1.0..=4.0).text("lacunarity")));
            row(ui, &mut n.roughness, d.roughness, |ui, v| _ = ui.add(egui::Slider::new(v, 0.0..=1.0).text("roughness")));
            row(ui, &mut n.distortion, d.distortion, |ui, v| _ = ui.add(egui::Slider::new(v, 0.0..=2.0).text("distortion")));
            row(ui, &mut n.seed, d.seed, |ui, seed| {
                let mut locked = seed.is_some();
                ui.checkbox(&mut locked, "seed override");
                if locked {
                    ui.add(egui::DragValue::new(seed.get_or_insert(0)));
                } else {
                    *seed = None;
                }
            });
        }
        NodeKind::Position => _ = ui.label("World position. Feeds noise inputs by default."),
        NodeKind::OutDensity | NodeKind::OutNormal => {}
    }
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let tiling = self.graph.tiling();

        egui::Panel::top("menu").show(ui, |ui| {
            egui::MenuBar::new().ui(ui, |ui| {
                ui.menu_button("File", |ui| {
                    if ui.button("New").clicked() {
                        self.graph = Graph::default_graph();
                        self.editor.selected.clear();
                        self.path = None;
                    }
                    if ui.button("Open…").clicked() {
                        self.open();
                    }
                    if ui.button("Save").clicked() {
                        self.save(false);
                    }
                    if ui.button("Save As…").clicked() {
                        self.save(true);
                    }
                });
                ui.weak(&self.status);
            });
        });

        egui::Panel::right("props").default_size(320.0).show(ui, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.heading("Preview");
                let d = Self::DEFAULTS;
                row(ui, &mut self.amount, d.0, |ui, v| _ = ui.add(egui::Slider::new(v, 0.0..=1.0).text("displacement")));
                row(ui, &mut self.tile_size, d.1, |ui, v| _ = ui.add(egui::Slider::new(v, 0.25..=4.0).logarithmic(true).text("tile size")));
                row(ui, &mut self.seed, d.2, |ui, v| _ = ui.add(egui::Slider::new(v, 0..=100).text("global seed")));
                row(ui, &mut self.step_scale, d.3, |ui, v| _ = ui.add(egui::Slider::new(v, 0.1..=1.0).text("step safety")));
                ui.separator();
                let sel: Vec<_> = self.editor.selected.iter().copied().collect();
                match sel.as_slice() {
                    [id] if self.graph.nodes.contains_key(id) => {
                        let node = self.graph.nodes.get_mut(id).unwrap();
                        ui.heading(editor::title(&node.kind));
                        if let Some(Tiling::Broken(why)) = tiling.get(id) {
                            ui.colored_label(editor::ORANGE, format!("Does not tile: {why}"));
                        }
                        node_props(ui, &mut node.kind);
                    }
                    [] => _ = ui.weak("Select a node to edit it."),
                    _ => _ = ui.weak(format!("{} nodes selected", sel.len())),
                }
            });
        });

        egui::Panel::bottom("graph").resizable(true).default_size(330.0).show(ui, |ui| {
            self.editor.show(ui, &mut self.graph, &tiling);
        });

        let compiled = match codegen::compile(&self.graph) {
            Ok(c) if c.params.len() > MAX_PARAMS => Err(format!("graph uses {} parameters (max {MAX_PARAMS})", c.params.len())),
            r => r,
        };
        if let Ok(c) = &compiled {
            if self.built_wgsl.as_deref() != Some(&c.wgsl) {
                let gpu = build_gpu(&self.rs, &c.wgsl);
                self.rs.renderer.write().callback_resources.insert(gpu);
                self.built_wgsl = Some(c.wgsl.clone());
            }
        }

        egui::CentralPanel::default().frame(egui::Frame::NONE).show(ui, |ui| {
            let (rect, resp) = ui.allocate_exact_size(ui.available_size(), egui::Sense::drag());
            let d = resp.drag_delta();
            self.yaw -= d.x * 0.01;
            self.pitch = (self.pitch + d.y * 0.01).clamp(-1.5, 1.5);
            if resp.hovered() {
                let scroll = ui.input(|i| i.smooth_scroll_delta.y);
                self.dist = (self.dist * (1.0 - scroll * 0.002)).clamp(1.2, 15.0);
            }
            match compiled {
                Ok(c) => {
                    let uniforms = self.uniforms(rect.width() / rect.height().max(1.0));
                    let cb = Viewport { uniforms, params: c.params };
                    ui.painter().add(egui_wgpu::Callback::new_paint_callback(rect, cb));
                }
                Err(e) => {
                    ui.painter().rect_filled(rect, 0.0, egui::Color32::from_gray(30));
                    ui.painter().text(rect.center(), egui::Align2::CENTER_CENTER, e, egui::FontId::proportional(16.0), egui::Color32::RED);
                }
            }
        });
    }
}


fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        renderer: eframe::Renderer::Wgpu,
        viewport: egui::ViewportBuilder::default().with_inner_size([1280.0, 800.0]).with_title("Noiser"),
        ..Default::default()
    };
    eframe::run_native("Noiser", options, Box::new(|cc| Ok(Box::new(App::new(cc)))))
}
