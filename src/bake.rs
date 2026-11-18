//! Baking maps for a low-poly mesh straight from the field it approximates: no high-poly mesh in between.
//!
//! Every texel of the UV atlas that a triangle covers has a point on the low-poly surface. That point is moved onto
//! the true surface with Newton steps along the field's gradient, and the maps are read from the field there:
//! the normal (in the tangent space of the low-poly mesh, MikkTSpace like the game engines), height above the
//! low-poly surface, ambient occlusion, curvature, thickness and position. Islands are padded outwards so
//! filtering does not pull in the empty background.
//!
//! The UVs are read with the origin at the top left, as xatlas writes them (V grows downwards, like Direct3D
//! and Unreal); a row of an image is a row of the atlas. The normal map has its green channel pointing down
//! (the Unreal convention) unless `Settings::green_up` is set (Blender, Unity, glTF).

use crate::mesh::V3;
use crate::uvgen::UvMesh;

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Channel {
    /// Tangent-space normal map.
    Normal,
    /// Ambient occlusion, 1 on open surface, towards 0 in creases.
    Occlusion,
    /// Mean curvature: 0.5 on flat surface, towards 1 on convex and 0 on concave parts (scaled by its 95th percentile).
    Curvature,
    /// Height of the true surface above the low-poly one along its normal: 0.5 where they meet.
    Depth,
    /// How deep the solid is behind the surface (0 at the surface, 1 at one mesh radius deep).
    Thickness,
    /// The point on the surface, as a fraction of the mesh's bounding box, in RGB.
    Position,
}

impl Channel {
    pub const ALL: [Channel; 6] = [Channel::Normal, Channel::Occlusion, Channel::Curvature, Channel::Depth, Channel::Thickness, Channel::Position];

    pub fn name(self) -> &'static str {
        match self {
            Channel::Normal => "Normal",
            Channel::Occlusion => "Occlusion",
            Channel::Curvature => "Curvature",
            Channel::Depth => "Depth",
            Channel::Thickness => "Thickness",
            Channel::Position => "Position",
        }
    }

    /// Colour channels in the image: 3 for the normal and position maps, 1 for the rest.
    pub fn components(self) -> usize {
        if matches!(self, Channel::Normal | Channel::Position) { 3 } else { 1 }
    }
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct Settings {
    /// Texture size in texels (square).
    pub size: u32,
    /// Texels islands are extended outwards by.
    pub padding: u32,
    pub channels: Vec<Channel>,
    /// Names of extra maps taken from fields wired into the Bake node (values 0 to 1), after the standard ones.
    pub custom: Vec<String>,
    /// A normal map with green up (OpenGL: Blender, Unity, glTF) instead of down (DirectX: Unreal).
    pub green_up: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Settings { size: 2048, padding: 8, channels: vec![Channel::Normal, Channel::Occlusion, Channel::Curvature, Channel::Depth], custom: Vec::new(), green_up: false }
    }
}

/// How an image output node writes its image.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ImageFormat {
    #[default]
    Png16,
    Png8,
    Exr,
}

impl ImageFormat {
    pub const ALL: [ImageFormat; 3] = [ImageFormat::Png16, ImageFormat::Png8, ImageFormat::Exr];

    pub fn label(self) -> &'static str {
        match self {
            ImageFormat::Png16 => "PNG 16-bit",
            ImageFormat::Png8 => "PNG 8-bit",
            ImageFormat::Exr => "OpenEXR 32-bit float",
        }
    }

    pub fn extension(self) -> &'static str {
        if self == ImageFormat::Exr { "exr" } else { "png" }
    }
}

/// One baked map: `size` by `size` texels, `components` values (0 to 1) per texel, the top row first.
#[derive(Clone, Debug)]
pub struct Image {
    /// The standard map this is, if it is one (custom maps and processed images are not).
    #[cfg_attr(not(test), allow(dead_code))]
    pub channel: Option<Channel>,
    /// The name its file gets (`Normal`, `Normal_R`...).
    pub label: String,
    pub size: u32,
    pub components: usize,
    pub data: Vec<f32>,
}

impl Image {
    /// Writes the image as `format` (PNG 8 or 16 bit, or a 32-bit float OpenEXR) by its extension.
    pub fn write(&self, path: &std::path::Path, format: ImageFormat) -> Result<(), String> {
        match format {
            ImageFormat::Png16 => self.write_png(path),
            ImageFormat::Png8 => {
                let data: Vec<u8> = self.data.iter().map(|&v| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8).collect();
                let n = self.size;
                let r = match self.components {
                    4 => image::ImageBuffer::<image::Rgba<u8>, _>::from_raw(n, n, data).ok_or("bad image size")?.save(path),
                    3 => image::ImageBuffer::<image::Rgb<u8>, _>::from_raw(n, n, data).ok_or("bad image size")?.save(path),
                    _ => image::ImageBuffer::<image::Luma<u8>, _>::from_raw(n, n, data).ok_or("bad image size")?.save(path),
                };
                r.map_err(|e| format!("{}: {e}", path.display()))
            }
            ImageFormat::Exr => {
                use exr::prelude::*;
                let c = self.components;
                let names: &[&'static str] = match c {
                    1 => &["Y"],
                    3 => &["R", "G", "B"],
                    _ => &["R", "G", "B", "A"],
                };
                let channels: SmallVec<[AnyChannel<FlatSamples>; 4]> = names.iter().enumerate().map(|(k, name)| AnyChannel::new(*name, FlatSamples::F32(self.data.iter().skip(k).step_by(c).copied().collect()))).collect();
                let layer = Layer::new((self.size as usize, self.size as usize), LayerAttributes::named("noiser"), Encoding::FAST_LOSSLESS, AnyChannels::sort(channels));
                Image::from_layer(layer).write().to_file(path).map_err(|e| format!("{}: {e}", path.display()))
            }
        }
    }

    /// Writes a 16-bit PNG (grey, RGB or RGBA).
    pub fn write_png(&self, path: &std::path::Path) -> Result<(), String> {
        let q = |v: f32| (v.clamp(0.0, 1.0) * 65535.0 + 0.5) as u16;
        let data: Vec<u16> = self.data.iter().map(|&v| q(v)).collect();
        let n = self.size;
        let r = if self.components == 4 {
            image::ImageBuffer::<image::Rgba<u16>, _>::from_raw(n, n, data).ok_or("bad image size")?.save(path)
        } else if self.components == 3 {
            image::ImageBuffer::<image::Rgb<u16>, _>::from_raw(n, n, data).ok_or("bad image size")?.save(path)
        } else {
            image::ImageBuffer::<image::Luma<u16>, _>::from_raw(n, n, data).ok_or("bad image size")?.save(path)
        };
        r.map_err(|e| format!("{}: {e}", path.display()))
    }
}

/// The field being baked: distance (negative inside) and gradient at points, in the mesh's space.
pub trait Field {
    /// (distance, gradient x, y, z) at each point.
    fn gradient(&mut self, points: &[V3]) -> Vec<[f32; 4]>;
    /// The distance alone, which is cheaper.
    fn distance(&mut self, points: &[V3]) -> Vec<f32>;
}

fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn add(a: V3, b: V3) -> V3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
fn scale(a: V3, s: f32) -> V3 {
    [a[0] * s, a[1] * s, a[2] * s]
}
fn dot(a: V3, b: V3) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn cross(a: V3, b: V3) -> V3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
fn unit(a: V3) -> V3 {
    let l = dot(a, a).sqrt();
    if l > 1e-20 { scale(a, 1.0 / l) } else { [0.0, 0.0, 1.0] }
}

/// MikkTSpace tangents for every corner of every triangle: (tangent xyz, bitangent sign).
fn tangents(mesh: &UvMesh) -> Vec<[[f32; 4]; 3]> {
    struct Geo<'a> {
        mesh: &'a UvMesh,
        out: Vec<[[f32; 4]; 3]>,
    }
    impl mikktspace::Geometry for Geo<'_> {
        fn num_faces(&self) -> usize {
            self.mesh.triangles.len()
        }
        fn num_vertices_of_face(&self, _: usize) -> usize {
            3
        }
        fn position(&self, f: usize, v: usize) -> [f32; 3] {
            self.mesh.positions[self.mesh.triangles[f][v] as usize]
        }
        fn normal(&self, f: usize, v: usize) -> [f32; 3] {
            self.mesh.normals[self.mesh.triangles[f][v] as usize]
        }
        fn tex_coord(&self, f: usize, v: usize) -> [f32; 2] {
            self.mesh.uvs[self.mesh.triangles[f][v] as usize]
        }
        fn set_tangent_encoded(&mut self, t: [f32; 4], f: usize, v: usize) {
            self.out[f][v] = t;
        }
    }
    let mut geo = Geo { mesh, out: vec![[[1.0, 0.0, 0.0, 1.0]; 3]; mesh.triangles.len()] };
    mikktspace::generate_tangents(&mut geo);
    geo.out
}

/// A texel centre inside a triangle.
struct Texel {
    index: u32,
    tri: u32,
    b: [f32; 2],
}

/// The texels whose centres lie in some triangle of the UV layout (the first triangle wins a shared edge).
fn rasterize(mesh: &UvMesh, size: usize) -> Vec<Texel> {
    let mut taken = vec![false; size * size];
    let mut out = Vec::new();
    let n = size as f32;
    for (f, tri) in mesh.triangles.iter().enumerate() {
        let q = tri.map(|i| {
            let uv = mesh.uvs[i as usize];
            [uv[0] * n, uv[1] * n]
        });
        let area = (q[1][0] - q[0][0]) * (q[2][1] - q[0][1]) - (q[2][0] - q[0][0]) * (q[1][1] - q[0][1]);
        if area.abs() < 1e-12 {
            continue;
        }
        let span = |c: usize| (q.iter().map(|p| p[c]).fold(f32::MAX, f32::min).floor().max(0.0) as usize, (q.iter().map(|p| p[c]).fold(f32::MIN, f32::max).ceil().max(0.0) as usize).min(size));
        let ((x0, x1), (y0, y1)) = (span(0), span(1));
        let before = out.len();
        for y in y0..y1 {
            for x in x0..x1 {
                let (cx, cy) = (x as f32 + 0.5, y as f32 + 0.5);
                // Barycentric coordinates of the centre; a small tolerance keeps shared edges from leaving holes.
                let b1 = ((cx - q[0][0]) * (q[2][1] - q[0][1]) - (q[2][0] - q[0][0]) * (cy - q[0][1])) / area;
                let b2 = ((q[1][0] - q[0][0]) * (cy - q[0][1]) - (cx - q[0][0]) * (q[1][1] - q[0][1])) / area;
                let b0 = 1.0 - b1 - b2;
                const EPS: f32 = -1e-4;
                if b0 >= EPS && b1 >= EPS && b2 >= EPS && !taken[y * size + x] {
                    taken[y * size + x] = true;
                    out.push(Texel { index: (y * size + x) as u32, tri: f as u32, b: [b1, b2] });
                }
            }
        }
        // A triangle too small to cover a texel centre still gets the texel under its centroid, sampled at the centroid,
        // so tiny islands are baked rather than filled in from their neighbours.
        if out.len() == before {
            let (cx, cy) = ((q[0][0] + q[1][0] + q[2][0]) / 3.0, (q[0][1] + q[1][1] + q[2][1]) / 3.0);
            let (x, y) = (cx.floor() as i64, cy.floor() as i64);
            if x >= 0 && y >= 0 && (x as usize) < size && (y as usize) < size && !taken[y as usize * size + x as usize] {
                taken[y as usize * size + x as usize] = true;
                out.push(Texel { index: (y as usize * size + x as usize) as u32, tri: f as u32, b: [1.0 / 3.0, 1.0 / 3.0] });
            }
        }
    }
    out
}

/// Bakes `settings.channels` for `mesh` against `field`. `progress` is called with (done, total) texel counts.
pub fn bake(mesh: &UvMesh, field: &mut dyn Field, customs: &mut [&mut dyn Field], settings: &Settings, progress: &dyn Fn(usize, usize)) -> Result<Vec<Image>, String> {
    let size = settings.size.max(16) as usize;
    if mesh.triangles.is_empty() {
        return Err("nothing to bake: the mesh has no triangles".into());
    }
    let tan = tangents(mesh);
    let texels = rasterize(mesh, size);
    if texels.is_empty() {
        return Err("the UV layout covers no texels".into());
    }
    let (mut lo, mut hi) = ([f32::MAX; 3], [f32::MIN; 3]);
    for p in &mesh.positions {
        for a in 0..3 {
            lo[a] = lo[a].min(p[a]);
            hi[a] = hi[a].max(p[a]);
        }
    }
    // One mesh radius: the unit the probe distances and ranges are given in.
    let radius = (0..3).map(|a| hi[a] - lo[a]).fold(0.0, f32::max) * 0.5;
    let channels: Vec<Channel> = Channel::ALL.into_iter().filter(|c| settings.channels.contains(c)).collect();
    let mut images: Vec<Vec<f32>> = channels.iter().map(|c| vec![0.0; size * size * c.components()]).collect();
    let mut valid = vec![false; size * size];
    let mut custom_images: Vec<Vec<f32>> = customs.iter().map(|_| vec![0.0; size * size]).collect();
    // The height map is scaled by its largest value once all texels are known.
    let mut heights: Vec<f32> = Vec::new();
    let mut curvatures: Vec<f32> = Vec::new();
    let has = |c: Channel| channels.contains(&c);
    const CHUNK: usize = 1 << 17;
    for (ci, chunk) in texels.chunks(CHUNK).enumerate() {
        progress(ci * CHUNK, texels.len());
        // The point, normal and tangent frame of the low-poly surface at each texel.
        let mut base: Vec<V3> = Vec::with_capacity(chunk.len());
        let mut frame: Vec<(V3, V3, V3)> = Vec::with_capacity(chunk.len());
        for t in chunk {
            let tri = mesh.triangles[t.tri as usize];
            let w = [1.0 - t.b[0] - t.b[1], t.b[0], t.b[1]];
            let mix = |f: &dyn Fn(usize) -> V3| (0..3).fold([0.0; 3], |s, k| add(s, scale(f(k), w[k])));
            let p = mix(&|k| mesh.positions[tri[k] as usize]);
            let n = unit(mix(&|k| mesh.normals[tri[k] as usize]));
            let tt = mix(&|k| [tan[t.tri as usize][k][0], tan[t.tri as usize][k][1], tan[t.tri as usize][k][2]]);
            // Gram-Schmidt against the interpolated normal; the bitangent sign is the corner's.
            let tt = unit(sub(tt, scale(n, dot(tt, n))));
            let sign = tan[t.tri as usize][0][3];
            base.push(p);
            frame.push((n, tt, scale(cross(n, tt), if sign < 0.0 { -1.0 } else { 1.0 })));
        }
        // Onto the true surface: Newton steps along the gradient.
        let mut p = base.clone();
        let max_step = 0.25 * radius;
        let mut grad = field.gradient(&p);
        for _ in 0..3 {
            for (q, s) in p.iter_mut().zip(&grad) {
                let g = [s[1], s[2], s[3]];
                let g2 = dot(g, g);
                if g2 > 1e-12 && s[0].is_finite() {
                    let step = scale(g, s[0] / g2);
                    let l = dot(step, step).sqrt();
                    *q = sub(*q, if l > max_step { scale(step, max_step / l) } else { step });
                }
            }
            grad = field.gradient(&p);
        }
        let normal: Vec<V3> = grad.iter().zip(&frame).map(|(s, f)| if dot([s[1], s[2], s[3]], [s[1], s[2], s[3]]) > 1e-12 { unit([s[1], s[2], s[3]]) } else { f.0 }).collect();
        // Probes along the normal and around the point, for the channels that need the field beyond the surface.
        let occlusion = if has(Channel::Occlusion) {
            const STEPS: usize = 5;
            let pts: Vec<V3> = p.iter().zip(&normal).flat_map(|(q, n)| (1..=STEPS).map(move |k| add(*q, scale(*n, (0.01 + 0.07 * k as f32) * radius)))).collect();
            let d = field.distance(&pts);
            d.chunks(STEPS)
                .map(|row| {
                    let (mut occ, mut sca) = (0.0, 1.0);
                    for (k, dk) in row.iter().enumerate() {
                        occ += ((0.01 + 0.07 * (k + 1) as f32) * radius - dk) / radius * sca;
                        sca *= 0.85;
                    }
                    (1.0 - 2.0 * occ).clamp(0.0, 1.0)
                })
                .collect()
        } else {
            Vec::new()
        };
        let curvature = if has(Channel::Curvature) {
            let e = 0.02 * radius;
            let mut pts: Vec<V3> = Vec::with_capacity(chunk.len() * 6);
            for q in &p {
                for a in 0..3 {
                    for s in [-1.0, 1.0] {
                        let mut r = *q;
                        r[a] += s * e;
                        pts.push(r);
                    }
                }
            }
            let around = field.distance(&pts);
            around
                .chunks(6)
                .zip(&grad)
                .map(|(row, s)| 0.5 * (row.iter().sum::<f32>() - 6.0 * s[0]) / (e * e) * radius)
                .collect()
        } else {
            Vec::new()
        };
        let thickness = if has(Channel::Thickness) {
            // March inwards along the normal until the field turns positive again.
            let mut q: Vec<V3> = p.iter().zip(&normal).map(|(q, n)| sub(*q, scale(*n, 0.01 * radius))).collect();
            let mut depth = vec![0.01 * radius; chunk.len()];
            let mut active: Vec<usize> = (0..chunk.len()).collect();
            for _ in 0..24 {
                if active.is_empty() {
                    break;
                }
                let d = field.distance(&active.iter().map(|&i| q[i]).collect::<Vec<_>>());
                let mut next = Vec::new();
                for (&i, dd) in active.iter().zip(d) {
                    if dd < 0.0 {
                        let step = (-dd).max(0.02 * radius);
                        depth[i] += step;
                        q[i] = sub(q[i], scale(normal[i], step));
                        next.push(i);
                    }
                }
                active = next;
            }
            depth.iter().map(|d| (d / radius).clamp(0.0, 1.0)).collect()
        } else {
            Vec::new()
        };
        // Custom maps: each wired-in field read at the point on the surface.
        for (k, f) in customs.iter_mut().enumerate() {
            let values = f.distance(&p);
            for (t, v) in chunk.iter().zip(values) {
                custom_images[k][t.index as usize] = v.clamp(0.0, 1.0);
            }
        }
        for (i, t) in chunk.iter().enumerate() {
            valid[t.index as usize] = true;
            let (n_low, tt, bb) = frame[i];
            for (k, &c) in channels.iter().enumerate() {
                let at = t.index as usize * c.components();
                let out = &mut images[k];
                match c {
                    Channel::Normal => {
                        let n = normal[i];
                        let g = dot(n, bb) * if settings.green_up { -1.0 } else { 1.0 };
                        out[at] = 0.5 + 0.5 * dot(n, tt);
                        out[at + 1] = 0.5 + 0.5 * g;
                        out[at + 2] = 0.5 + 0.5 * dot(n, n_low);
                    }
                    Channel::Occlusion => out[at] = occlusion[i],
                    Channel::Curvature => {
                        out[at] = curvature[i];
                        curvatures.push(curvature[i].abs());
                    }
                    Channel::Thickness => out[at] = thickness[i],
                    Channel::Depth => {
                        out[at] = dot(sub(p[i], base[i]), n_low);
                        heights.push(out[at].abs());
                    }
                    Channel::Position => {
                        for a in 0..3 {
                            out[at + a] = ((p[i][a] - lo[a]) / (hi[a] - lo[a]).max(1e-12)).clamp(0.0, 1.0);
                        }
                    }
                }
            }
        }
    }
    // Height: symmetric about 0.5, scaled by the 99.9th percentile so a stray spike does not flatten everything.
    if let Some(k) = channels.iter().position(|&c| c == Channel::Depth) {
        heights.sort_by(|a, b| a.total_cmp(b));
        let range = heights.get((heights.len() as f32 * 0.999) as usize).copied().unwrap_or(1.0).max(1e-9);
        for v in &mut images[k] {
            *v = 0.5 + 0.5 * (*v / range).clamp(-1.0, 1.0);
        }
    }
    // Curvature: symmetric about 0.5, scaled by the 95th percentile of its magnitude (a noisy surface curves far
    // more than a smooth one, and a fixed range would clip it all).
    if let Some(k) = channels.iter().position(|&c| c == Channel::Curvature) {
        curvatures.sort_by(|a, b| a.total_cmp(b));
        let range = curvatures.get((curvatures.len() as f32 * 0.95) as usize).copied().unwrap_or(1.0).max(1e-9);
        for v in &mut images[k] {
            *v = 0.5 + 0.5 * (*v / range).clamp(-1.0, 1.0);
        }
    }
    progress(texels.len(), texels.len());
    // Pad the islands outwards.
    let mut out = Vec::new();
    for (k, &c) in channels.iter().enumerate() {
        let mut data = std::mem::take(&mut images[k]);
        dilate(&mut data, valid.clone(), size, c.components(), settings.padding as usize);
        out.push(Image { channel: Some(c), label: c.name().to_string(), size: size as u32, components: c.components(), data });
    }
    for (k, mut data) in custom_images.into_iter().enumerate() {
        dilate(&mut data, valid.clone(), size, 1, settings.padding as usize);
        out.push(Image { channel: None, label: settings.custom.get(k).cloned().unwrap_or_else(|| format!("Custom{}", k + 1)), size: size as u32, components: 1, data });
    }
    Ok(out)
}

/// Copies valid texels into the invalid ones next to them, `passes` times, averaging when several neighbours are valid.
fn dilate(data: &mut [f32], mut valid: Vec<bool>, size: usize, components: usize, passes: usize) {
    for _ in 0..passes {
        let mut grown = valid.clone();
        let mut writes: Vec<(usize, Vec<f32>)> = Vec::new();
        for y in 0..size {
            for x in 0..size {
                if valid[y * size + x] {
                    continue;
                }
                let mut sum = vec![0.0f32; components];
                let mut n = 0;
                for dy in [-1i32, 0, 1] {
                    for dx in [-1i32, 0, 1] {
                        let (nx, ny) = (x as i32 + dx, y as i32 + dy);
                        if nx < 0 || ny < 0 || nx >= size as i32 || ny >= size as i32 {
                            continue;
                        }
                        let j = ny as usize * size + nx as usize;
                        if valid[j] {
                            for c in 0..components {
                                sum[c] += data[j * components + c];
                            }
                            n += 1;
                        }
                    }
                }
                if n > 0 {
                    writes.push((y * size + x, sum.iter().map(|s| s / n as f32).collect()));
                    grown[y * size + x] = true;
                }
            }
        }
        if writes.is_empty() {
            break;
        }
        for (i, v) in writes {
            data[i * components..(i + 1) * components].copy_from_slice(&v);
        }
        valid = grown;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::surface::RawMesh;

    /// A bumpy sphere: the field of radius 1 plus a bump pattern, with an analytic gradient by differences.
    struct Bumpy(f32);
    impl Bumpy {
        fn f(&self, p: V3) -> f32 {
            (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt() - 1.0 - self.0 * (6.0 * p[0]).sin() * (6.0 * p[1]).sin() * (6.0 * p[2]).sin()
        }
    }
    impl Field for Bumpy {
        fn gradient(&mut self, pts: &[V3]) -> Vec<[f32; 4]> {
            pts.iter()
                .map(|p| {
                    let e = 1e-3;
                    let g = [0, 1, 2].map(|a| {
                        let (mut hi, mut lo) = (*p, *p);
                        hi[a] += e;
                        lo[a] -= e;
                        (self.f(hi) - self.f(lo)) / (2.0 * e)
                    });
                    [self.f(*p), g[0], g[1], g[2]]
                })
                .collect()
        }
        fn distance(&mut self, pts: &[V3]) -> Vec<f32> {
            pts.iter().map(|p| self.f(*p)).collect()
        }
    }

    fn sphere_uv(subdivisions: u32) -> UvMesh {
        let m = crate::mesh::shapes::icosphere(1.0, subdivisions);
        let raw = RawMesh { normals: m.positions.clone(), positions: m.positions, triangles: m.triangles };
        crate::uvgen::generate(&raw, &crate::uvgen::UvParams { resolution: 256, padding: 2, ..Default::default() }).unwrap()
    }

    #[test]
    fn a_smooth_sphere_bakes_a_flat_normal_map_and_open_occlusion() {
        let mesh = sphere_uv(3);
        let settings = Settings { size: 128, padding: 4, channels: vec![Channel::Normal, Channel::Occlusion, Channel::Position, Channel::Thickness, Channel::Curvature], custom: Vec::new(), green_up: false };
        let images = bake(&mesh, &mut Bumpy(0.0), &mut [], &settings, &|_, _| {}).unwrap();
        let get = |c: Channel| images.iter().find(|i| i.channel == Some(c)).unwrap();
        // Normal map: tangent-space z (blue) near 1 on a sphere sampled by a fine low-poly one.
        let n = get(Channel::Normal);
        let valid = n.data.chunks(3).filter(|p| p[2] > 0.0).count();
        assert!(valid > 128 * 128 / 3, "{valid} texels baked");
        let (mut sum, mut count) = (0.0, 0);
        for p in n.data.chunks(3) {
            if p[2] > 0.0 {
                sum += p[2];
                count += 1;
            }
        }
        assert!(sum / count as f32 > 0.97, "blue averages {}", sum / count as f32);
        // Occlusion: a convex sphere is open everywhere.
        let ao = get(Channel::Occlusion);
        let covered: Vec<f32> = ao.data.iter().cloned().filter(|v| *v > 0.0).collect();
        let mean = covered.iter().sum::<f32>() / covered.len() as f32;
        assert!(covered.len() > 128 * 128 / 3 && mean > 0.9, "occlusion {mean} over {} texels", covered.len());
        // Thickness: the diameter of the unit sphere, 2, in units of the radius 1, clamped to 1.
        let th = get(Channel::Thickness);
        assert!(th.data.iter().filter(|v| **v > 0.0).all(|v| *v > 0.9), "thickness");
        // Curvature of a sphere is the same and positive everywhere: after scaling by its own magnitude, close to 1.
        let cv = get(Channel::Curvature);
        let covered: Vec<f32> = cv.data.iter().cloned().filter(|v| *v > 0.0).collect();
        let mean_curv = covered.iter().sum::<f32>() / covered.len() as f32;
        assert!(mean_curv > 0.85, "curvature {mean_curv}");
    }

    #[test]
    fn bumps_show_up_in_the_normal_map_and_the_height() {
        let mesh = sphere_uv(2);
        let settings = Settings { size: 128, padding: 4, channels: vec![Channel::Normal, Channel::Depth], custom: Vec::new(), green_up: false };
        let images = bake(&mesh, &mut Bumpy(0.05), &mut [], &settings, &|_, _| {}).unwrap();
        let n = images.iter().find(|i| i.channel == Some(Channel::Normal)).unwrap();
        let blues: Vec<f32> = n.data.chunks(3).map(|p| p[2]).filter(|b| *b > 0.0).collect();
        let mean = blues.iter().sum::<f32>() / blues.len() as f32;
        let steepest = blues.iter().cloned().fold(1.0, f32::min);
        assert!(mean < 0.998 && steepest < 0.985, "the bumps tilt the normals: blue mean {mean}, lowest {steepest}");
        let h = images.iter().find(|i| i.channel == Some(Channel::Depth)).unwrap();
        let (lo, hi) = (h.data.iter().cloned().fold(1.0, f32::min), h.data.iter().cloned().fold(0.0, f32::max));
        assert!(lo < 0.35 && hi > 0.65, "height spans {lo}..{hi}");
    }
}
