//! The SDF Composer's camera, and the CPU ray march used to pick the shape under the cursor. The picture itself is drawn
//! by the GPU (`composer_gpu`).

use crate::sdf_model::{LayerField, V3};

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
