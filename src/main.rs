mod codegen;
mod graph;

use eframe::{egui, egui_wgpu, wgpu};
use graph::{Graph, NodeKind, Tiling};

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
    fn new(cc: &eframe::CreationContext) -> Self {
        Self {
            rs: cc.wgpu_render_state.clone().expect("Noiser needs the wgpu renderer"),
            graph: Graph::default_graph(),
            built_wgsl: None,
            yaw: 0.6,
            pitch: 0.35,
            dist: 3.5,
            amount: 0.25,
            tile_size: 1.0,
            seed: 0,
            step_scale: 0.6,
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
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

fn norm(v: [f32; 3]) -> [f32; 3] {
    let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    [v[0] / l, v[1] / l, v[2] / l]
}

const ORANGE: egui::Color32 = egui::Color32::from_rgb(255, 150, 40);

/// Temporary node list (milestone 1) until the node editor lands in milestone 2.
fn node_list(ui: &mut egui::Ui, graph: &mut Graph) {
    let tiling = graph.tiling();
    for (id, node) in graph.nodes.iter_mut() {
        let title = match &node.kind {
            NodeKind::Noise(n) => format!("#{id} {:?} noise", n.kind),
            NodeKind::Math { op, .. } => format!("#{id} Math {op:?}"),
            k => format!("#{id} {}", format!("{k:?}").split(['(', ' ', '{']).next().unwrap_or("")),
        };
        let header = egui::CollapsingHeader::new(title).id_salt(id).default_open(true);
        let resp = header.show(ui, |ui| match &mut node.kind {
            NodeKind::Float(v) => {
                ui.add(egui::DragValue::new(v).speed(0.01));
            }
            NodeKind::Vec3(v) => {
                ui.horizontal(|ui| v.iter_mut().for_each(|x| _ = ui.add(egui::DragValue::new(x).speed(0.01))));
            }
            NodeKind::Math { a, b, .. } => {
                ui.add(egui::DragValue::new(a).speed(0.01).prefix("a "));
                ui.add(egui::DragValue::new(b).speed(0.01).prefix("b "));
            }
            NodeKind::Normal { flip } => {
                ui.checkbox(flip, "flip");
            }
            NodeKind::Noise(n) => {
                ui.add(egui::Slider::new(&mut n.scale, 0.5..=32.0).text("scale (cells/tile)"));
                ui.checkbox(&mut n.normalize, "normalize 0-1");
                ui.add(egui::Slider::new(&mut n.octaves, 1..=8).text("octaves"));
                ui.add(egui::Slider::new(&mut n.lacunarity, 1.0..=4.0).text("lacunarity"));
                ui.add(egui::Slider::new(&mut n.roughness, 0.0..=1.0).text("roughness"));
                ui.add(egui::Slider::new(&mut n.distortion, 0.0..=2.0).text("distortion"));
                let mut locked = n.seed.is_some();
                ui.horizontal(|ui| {
                    ui.checkbox(&mut locked, "seed override");
                    if locked {
                        let s = n.seed.get_or_insert(0);
                        ui.add(egui::DragValue::new(s));
                    } else {
                        n.seed = None;
                    }
                });
            }
            NodeKind::Position | NodeKind::OutDensity | NodeKind::OutNormal => {}
        });
        if let Some(Tiling::Broken(why)) = tiling.get(id) {
            let r = resp.header_response.rect;
            ui.painter().rect_stroke(r, 3.0, egui::Stroke::new(1.5, ORANGE), egui::StrokeKind::Outside);
            resp.header_response.on_hover_text(format!("Does not tile: {why}"));
        }
    }
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let compiled = codegen::compile(&self.graph);

        egui::Panel::right("params").default_size(320.0).show(ui, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.heading("Preview");
                ui.add(egui::Slider::new(&mut self.amount, 0.0..=1.0).text("displacement"));
                ui.add(egui::Slider::new(&mut self.tile_size, 0.25..=4.0).logarithmic(true).text("tile size"));
                ui.add(egui::Slider::new(&mut self.seed, 0..=100).text("global seed"));
                ui.add(egui::Slider::new(&mut self.step_scale, 0.1..=1.0).text("step safety"));
                ui.separator();
                ui.heading("Graph");
                if let Err(e) = &compiled {
                    ui.colored_label(egui::Color32::RED, e);
                }
                node_list(ui, &mut self.graph);
            });
        });

        let compiled = match compiled {
            Ok(c) if c.params.len() <= MAX_PARAMS => c,
            _ => return, // error shown in the panel; keep the last good frame
        };
        if self.built_wgsl.as_deref() != Some(&compiled.wgsl) {
            let gpu = build_gpu(&self.rs, &compiled.wgsl);
            self.rs.renderer.write().callback_resources.insert(gpu);
            self.built_wgsl = Some(compiled.wgsl);
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
            let uniforms = self.uniforms(rect.width() / rect.height().max(1.0));
            let cb = Viewport { uniforms, params: compiled.params };
            ui.painter().add(egui_wgpu::Callback::new_paint_callback(rect, cb));
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
