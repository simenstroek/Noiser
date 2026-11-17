use eframe::{egui, egui_wgpu, wgpu};

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
    gain: f32,
    period: u32,
    octaves: u32,
    lacunarity: u32,
    seed: u32,
}

struct Gpu {
    pipeline: wgpu::RenderPipeline,
    buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

struct Viewport(Uniforms);

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
        queue.write_buffer(&gpu.buffer, 0, bytemuck::bytes_of(&self.0));
        Vec::new()
    }

    fn paint(
        &self,
        _info: egui::PaintCallbackInfo,
        pass: &mut wgpu::RenderPass<'static>,
        res: &egui_wgpu::CallbackResources,
    ) {
        let gpu: &Gpu = res.get().unwrap();
        pass.set_pipeline(&gpu.pipeline);
        pass.set_bind_group(0, &gpu.bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
}

struct App {
    yaw: f32,
    pitch: f32,
    dist: f32,
    amount: f32,
    period: u32,
    octaves: u32,
    lacunarity: u32,
    gain: f32,
    seed: u32,
    step_scale: f32,
}

impl App {
    fn new(cc: &eframe::CreationContext) -> Self {
        let rs = cc.wgpu_render_state.as_ref().expect("Noiser needs the wgpu renderer");
        let device = &rs.device;
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("raymarch"),
            source: wgpu::ShaderSource::Wgsl(include_str!("raymarch.wgsl").into()),
        });
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
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
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("uniforms"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: buffer.as_entire_binding() }],
        });
        rs.renderer.write().callback_resources.insert(Gpu { pipeline, buffer, bind_group });

        Self {
            yaw: 0.6,
            pitch: 0.35,
            dist: 3.5,
            amount: 0.25,
            period: 4,
            octaves: 4,
            lacunarity: 2,
            gain: 0.5,
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
            gain: self.gain,
            period: self.period,
            octaves: self.octaves,
            lacunarity: self.lacunarity,
            seed: self.seed,
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

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        egui::Panel::right("params").show(ui, |ui| {
            ui.heading("Perlin fBm");
            ui.add(egui::Slider::new(&mut self.amount, 0.0..=1.0).text("displacement"));
            ui.add(egui::Slider::new(&mut self.period, 1..=32).text("scale (cells/tile)"));
            ui.add(egui::Slider::new(&mut self.octaves, 1..=8).text("octaves"));
            ui.add(egui::Slider::new(&mut self.lacunarity, 1..=4).text("lacunarity"));
            ui.add(egui::Slider::new(&mut self.gain, 0.0..=1.0).text("roughness"));
            ui.add(egui::Slider::new(&mut self.seed, 0..=100).text("seed"));
            ui.separator();
            ui.add(egui::Slider::new(&mut self.step_scale, 0.1..=1.0).text("step safety"));
        });
        egui::CentralPanel::default().frame(egui::Frame::NONE).show(ui, |ui| {
            let (rect, resp) = ui.allocate_exact_size(ui.available_size(), egui::Sense::drag());
            let d = resp.drag_delta();
            self.yaw -= d.x * 0.01;
            self.pitch = (self.pitch + d.y * 0.01).clamp(-1.5, 1.5);
            if resp.hovered() {
                let scroll = ui.input(|i| i.smooth_scroll_delta.y);
                self.dist = (self.dist * (1.0 - scroll * 0.002)).clamp(1.2, 15.0);
            }
            let u = self.uniforms(rect.width() / rect.height().max(1.0));
            ui.painter().add(egui_wgpu::Callback::new_paint_callback(rect, Viewport(u)));
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
