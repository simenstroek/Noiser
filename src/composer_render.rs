//! The SDF Composer's viewport: the camera and a CPU ray marcher that runs on a worker thread. A picture is drawn
//! coarse first and refined, and a newer request makes the worker drop the one it is on.

use crate::sdf_model::{LayerField, V3};
use eframe::egui;
use rayon::prelude::*;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, mpsc};

/// Vertical field of view of the viewport camera, in radians.
pub const FIELD_OF_VIEW: f64 = 0.8;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Camera {
    pub yaw: f64,
    pub pitch: f64,
    pub distance: f64,
    pub target: V3,
}

impl Default for Camera {
    fn default() -> Self {
        Camera { yaw: 0.7, pitch: 0.4, distance: 8.0, target: [0.0; 3] }
    }
}

fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: V3, b: V3) -> V3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

fn normalized(v: V3) -> V3 {
    let length = dot(v, v).sqrt().max(1e-12);
    [v[0] / length, v[1] / length, v[2] / length]
}

fn add_scaled(a: V3, b: V3, k: f64) -> V3 {
    [a[0] + b[0] * k, a[1] + b[1] * k, a[2] + b[2] * k]
}

impl Camera {
    pub fn eye(&self) -> V3 {
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        add_scaled(self.target, [cp * sy, sp, cp * cy], self.distance)
    }

    /// Forward, right and up, with +y as the world's up.
    pub fn basis(&self) -> (V3, V3, V3) {
        let eye = self.eye();
        let forward = normalized([self.target[0] - eye[0], self.target[1] - eye[1], self.target[2] - eye[2]]);
        let right = normalized(cross(forward, [0.0, 1.0, 0.0]));
        let up = cross(right, forward);
        (forward, right, up)
    }

    /// The ray through a point of the view, `ndc` running from −1 to 1 with y up.
    pub fn ray(&self, ndc: [f64; 2], aspect: f64) -> (V3, V3) {
        let (forward, right, up) = self.basis();
        let half = (FIELD_OF_VIEW * 0.5).tan();
        let mut direction = forward;
        direction = add_scaled(direction, right, ndc[0] * half * aspect);
        direction = add_scaled(direction, up, ndc[1] * half);
        (self.eye(), normalized(direction))
    }

    /// Where a world point lands in a view `size` pixels across, or `None` behind the camera.
    pub fn project(&self, p: V3, size: [f32; 2]) -> Option<[f32; 2]> {
        let (forward, right, up) = self.basis();
        let eye = self.eye();
        let v = [p[0] - eye[0], p[1] - eye[1], p[2] - eye[2]];
        let depth = dot(v, forward);
        if depth < 0.01 {
            return None;
        }
        let half = (FIELD_OF_VIEW * 0.5).tan();
        let aspect = size[0] as f64 / size[1] as f64;
        let x = dot(v, right) / (depth * half * aspect);
        let y = dot(v, up) / (depth * half);
        Some([((x * 0.5 + 0.5) * size[0] as f64) as f32, ((0.5 - y * 0.5) * size[1] as f64) as f32])
    }

    /// World units per screen pixel at the depth of `p`, for a view `height` pixels tall.
    pub fn units_per_pixel(&self, p: V3, height: f32) -> f64 {
        let (forward, ..) = self.basis();
        let eye = self.eye();
        let depth = dot([p[0] - eye[0], p[1] - eye[1], p[2] - eye[2]], forward).max(0.01);
        2.0 * depth * (FIELD_OF_VIEW * 0.5).tan() / height as f64
    }

    /// A camera that frames the box with corners `lo` and `hi`.
    pub fn framing(&self, lo: V3, hi: V3) -> Camera {
        let center = [(lo[0] + hi[0]) * 0.5, (lo[1] + hi[1]) * 0.5, (lo[2] + hi[2]) * 0.5];
        let radius = ((hi[0] - lo[0]).powi(2) + (hi[1] - lo[1]).powi(2) + (hi[2] - lo[2]).powi(2)).sqrt() * 0.5;
        Camera { target: center, distance: (radius / (FIELD_OF_VIEW * 0.5).sin() * 1.1).max(0.5), ..*self }
    }
}

// ---- ray marching -------------------------------------------------------------------------------------------------

fn march(field: &LayerField, origin: V3, direction: V3, far: f64) -> Option<(f64, V3)> {
    let mut t = 0.0;
    for _ in 0..220 {
        let p = add_scaled(origin, direction, t);
        let d = field.assembly_distance(p)?;
        if d < 0.0004 * (1.0 + t) {
            return Some((t, p));
        }
        t += (d * 0.85).max(0.0002);
        if t > far {
            return None;
        }
    }
    None
}

fn surface_normal(field: &LayerField, p: V3, t: f64) -> V3 {
    let h = 0.0008 * (1.0 + t);
    let directions: [V3; 4] = [[1.0, -1.0, -1.0], [-1.0, -1.0, 1.0], [-1.0, 1.0, -1.0], [1.0, 1.0, 1.0]];
    let mut normal = [0.0; 3];
    for k in directions {
        let d = field.assembly_distance(add_scaled(p, k, h)).unwrap_or(0.0);
        normal = add_scaled(normal, k, d);
    }
    normalized(normal)
}

/// Finds the shape under a point of the view, for picking.
pub fn pick(field: &LayerField, camera: &Camera, ndc: [f64; 2], aspect: f64) -> Option<Vec<usize>> {
    let (origin, direction) = camera.ray(ndc, aspect);
    let far = camera.distance * 3.0 + 60.0;
    let (_, p) = march(field, origin, direction, far)?;
    field
        .leaf_paths()
        .into_iter()
        .filter_map(|path| field.node_distance(&path, p).map(|d| (d.abs(), path)))
        .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
        .map(|(_, path)| path)
}

struct Shaded {
    pixels: Vec<egui::Color32>,
    lo: V3,
    hi: V3,
    any: bool,
}

fn render_image(
    field: &LayerField,
    camera: &Camera,
    size: [usize; 2],
    selected: Option<&[usize]>,
    current: &AtomicU64,
    generation: u64,
) -> Option<Shaded> {
    let (w, h) = (size[0], size[1]);
    let aspect = w as f64 / h as f64;
    let (forward, right, up) = camera.basis();
    let light = normalized([-forward[0] * 0.6 - right[0] * 0.5 + up[0] * 0.7, -forward[1] * 0.6 - right[1] * 0.5 + up[1] * 0.7, -forward[2] * 0.6 - right[2] * 0.5 + up[2] * 0.7]);
    let far = camera.distance * 3.0 + 60.0;
    let rows: Vec<Option<(Vec<egui::Color32>, V3, V3, bool)>> = (0..h)
        .into_par_iter()
        .map(|row| {
            if current.load(Ordering::Relaxed) != generation {
                return None;
            }
            let mut line = Vec::with_capacity(w);
            let (mut lo, mut hi, mut any) = ([f64::MAX; 3], [f64::MIN; 3], false);
            for column in 0..w {
                let ndc = [2.0 * (column as f64 + 0.5) / w as f64 - 1.0, 1.0 - 2.0 * (row as f64 + 0.5) / h as f64];
                let (origin, direction) = camera.ray(ndc, aspect);
                let shade = match march(field, origin, direction, far) {
                    Some((t, p)) => {
                        any = true;
                        for axis in 0..3 {
                            lo[axis] = lo[axis].min(p[axis]);
                            hi[axis] = hi[axis].max(p[axis]);
                        }
                        let n = surface_normal(field, p, t);
                        let diffuse = dot(n, light).max(0.0);
                        let rim = (1.0 - dot(n, [-direction[0], -direction[1], -direction[2]]).max(0.0)).powi(3) * 0.15;
                        let mut color = [0.74_f64, 0.75, 0.78];
                        if let Some(path) = selected
                            && field.node_distance(path, p).is_some_and(|d| d.abs() < 0.0025 * (1.0 + t))
                        {
                            color = [0.98, 0.58, 0.18];
                        }
                        let light_amount = 0.28 + 0.72 * diffuse + rim;
                        egui::Color32::from_rgb(
                            (color[0] * light_amount * 255.0).min(255.0) as u8,
                            (color[1] * light_amount * 255.0).min(255.0) as u8,
                            (color[2] * light_amount * 255.0).min(255.0) as u8,
                        )
                    }
                    None => {
                        let shade = (0.20 + 0.10 * (1.0 - (row as f64 / h as f64))) * 255.0;
                        egui::Color32::from_gray(shade as u8)
                    }
                };
                line.push(shade);
            }
            Some((line, lo, hi, any))
        })
        .collect();
    let mut pixels = Vec::with_capacity(w * h);
    let (mut lo, mut hi, mut any) = ([f64::MAX; 3], [f64::MIN; 3], false);
    for row in rows {
        let (line, row_lo, row_hi, row_any) = row?;
        pixels.extend(line);
        if row_any {
            any = true;
            for axis in 0..3 {
                lo[axis] = lo[axis].min(row_lo[axis]);
                hi[axis] = hi[axis].max(row_hi[axis]);
            }
        }
    }
    Some(Shaded { pixels, lo, hi, any })
}

// ---- the worker -----------------------------------------------------------------------------------------------------

pub struct Request {
    pub generation: u64,
    pub field: Arc<LayerField>,
    pub camera: Camera,
    /// The size of the finest picture, in pixels.
    pub size: [usize; 2],
    pub selected: Option<Vec<usize>>,
}

pub struct Frame {
    pub generation: u64,
    /// Whether this is the finest picture of its request.
    pub finest: bool,
    pub image: egui::ColorImage,
    /// The box around what is visible, when anything is.
    pub bounds: Option<(V3, V3)>,
}

pub struct Renderer {
    sender: mpsc::Sender<Request>,
    receiver: mpsc::Receiver<Frame>,
    current: Arc<AtomicU64>,
    next: u64,
}

const TIER_DIVISORS: [f64; 3] = [8.0, 3.0, 1.0];

impl Renderer {
    pub fn new(context: egui::Context) -> Renderer {
        let (sender, requests) = mpsc::channel::<Request>();
        let (frames, receiver) = mpsc::channel::<Frame>();
        let current = Arc::new(AtomicU64::new(0));
        let worker_current = current.clone();
        std::thread::spawn(move || {
            while let Ok(mut request) = requests.recv() {
                while let Ok(newer) = requests.try_recv() {
                    request = newer;
                }
                for (tier, divisor) in TIER_DIVISORS.iter().enumerate() {
                    if worker_current.load(Ordering::Relaxed) != request.generation {
                        break;
                    }
                    let size = [((request.size[0] as f64 / divisor) as usize).max(8), ((request.size[1] as f64 / divisor) as usize).max(8)];
                    let Some(shaded) = render_image(&request.field, &request.camera, size, request.selected.as_deref(), &worker_current, request.generation) else {
                        break;
                    };
                    let image = egui::ColorImage { size, source_size: egui::vec2(size[0] as f32, size[1] as f32), pixels: shaded.pixels };
                    let bounds = shaded.any.then_some((shaded.lo, shaded.hi));
                    let sent = frames.send(Frame { generation: request.generation, finest: tier + 1 == TIER_DIVISORS.len(), image, bounds });
                    context.request_repaint();
                    if sent.is_err() {
                        return;
                    }
                }
            }
        });
        Renderer { sender, receiver, current, next: 0 }
    }

    /// Asks for a new picture; whatever the worker is on is dropped. Returns the request's generation.
    pub fn submit(&mut self, field: Arc<LayerField>, camera: Camera, size: [usize; 2], selected: Option<Vec<usize>>) -> u64 {
        self.next += 1;
        self.current.store(self.next, Ordering::Relaxed);
        let _ = self.sender.send(Request { generation: self.next, field, camera, size, selected });
        self.next
    }

    pub fn poll(&self) -> Vec<Frame> {
        self.receiver.try_iter().collect()
    }
}
