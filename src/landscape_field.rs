//! A terrain heightmap as a field for the Mesh graph: a Load Mesh node given the PNG the Landscape tab exports (with its
//! `.json` sidecar next to it) is the ground, `field = (z - height(x, y)) / sqrt(1 + slope^2)`, in the units of the node.
//! The heights live in a 3D texture of 2048 x 2048 tiles (3D textures are limited to 2048 per side), and the shader
//! reads them bilinearly (`land_dist` in `meshexport.rs`).

use crate::mesh::Mesh;
use std::path::Path;

/// Texels per tile side.
pub const TILE: usize = 2048;

pub struct Heights {
    /// Samples per side.
    pub n: usize,
    /// Distance between samples, in units.
    pub cell: f32,
    /// Row major (x fastest), in units.
    pub h: Vec<f32>,
    pub z: (f32, f32),
    /// The height of the water in the node's units, when the heightmap says there is some.
    pub water: Option<f32>,
    /// Paint layer weightmaps (0 to 1, `n` x `n`) travelling with the heights: extra layers of the same texture, after
    /// the height tiles (see `texels`). Names as in the sidecar's `layers`.
    pub maps: Vec<(String, Vec<f32>)>,
}

/// A terrain file: a heightmap PNG from the Landscape tab, or a Landscape project (a Load Landscape node).
pub fn is_heightmap(path: &str) -> bool {
    Path::new(path).extension().is_some_and(|e| e.eq_ignore_ascii_case("png")) || is_landscape(Path::new(path))
}

pub fn is_landscape(path: &Path) -> bool {
    path.extension().is_some_and(|e| e.eq_ignore_ascii_case("landscape"))
}

/// Where a Landscape project is exported for a Load Landscape node: a heightmap, its sidecar and weightmaps in the temp
/// folder (not beside the project, which may sit in a synced library), one per project path.
pub fn export_path(project: &Path) -> std::path::PathBuf {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    project.hash(&mut h);
    let stem = project.file_stem().map_or_else(|| "landscape".into(), |s| s.to_string_lossy().into_owned());
    std::env::temp_dir().join("noiser_landscapes").join(format!("{stem}_{:016x}.png", h.finish()))
}

/// The heightmap to read for a terrain file: a PNG as it is; a Landscape project exported first, and again whenever the
/// project was saved after the export.
// shortcut: files the project reads (heightmaps, SDFs, noise graphs) changing do not redo the export; resave the project.
pub fn heightmap_file(path: &Path) -> Result<std::path::PathBuf, String> {
    if !is_landscape(path) {
        return Ok(path.to_path_buf());
    }
    let out = export_path(path);
    let modified = |p: &Path| std::fs::metadata(p).and_then(|m| m.modified()).ok();
    let project = modified(path).ok_or_else(|| format!("{}: no such Landscape project", path.display()))?;
    if modified(&out).is_none_or(|done| done < project) {
        std::fs::create_dir_all(out.parent().unwrap_or(Path::new("."))).map_err(|e| e.to_string())?;
        crate::landscape::export_headless(path, &out, false, &mut |_| {}).map_err(|e| format!("{}: {e}", path.display()))?;
    }
    Ok(out)
}

/// For reading a sidecar without exporting (the UI, every frame): a project's last export, if there is one.
fn exported(path: &Path) -> std::path::PathBuf {
    if is_landscape(path) { export_path(path) } else { path.to_path_buf() }
}

/// The heights of an exported heightmap in units of `unit_cm` centimetres.
pub fn load(path: &Path, unit_cm: f32) -> Result<Heights, String> {
    let path = &heightmap_file(path)?;
    let img = image::open(path).map_err(|e| format!("{}: {e}", path.display()))?.to_luma16();
    let n = img.width() as usize;
    if img.height() as usize != n || n < 2 {
        return Err(format!("{}: a heightmap must be square", path.display()));
    }
    let side = std::path::PathBuf::from(format!("{}.json", path.display()));
    let text = std::fs::read_to_string(&side).map_err(|e| format!("{}: {e} (the sidecar the Landscape tab writes next to the heightmap)", side.display()))?;
    let json: serde_json::Value = serde_json::from_str(&text).map_err(|e| format!("{}: {e}", side.display()))?;
    let size_m = json["size_m"].as_f64().ok_or("the sidecar has no size_m")? as f32;
    let step_cm = json["step_cm"].as_f64().ok_or("the sidecar has no step_cm")? as f32;
    let centre_cm = json["unreal_import"]["location_cm"][2].as_f64().ok_or("the sidecar has no unreal_import.location_cm")? as f32;
    // The inverse of `landscape::encode_unreal`: metres = (sample - 32768) * step + centre, in centimetres.
    let k = 1.0 / unit_cm;
    let h: Vec<f32> = img.as_raw().iter().map(|v| ((*v as f32 - 32768.0) * step_cm + centre_cm) * k).collect();
    let z = h.iter().fold((f32::MAX, f32::MIN), |(lo, hi), v| (lo.min(*v), hi.max(*v)));
    let water = json["water_level_m"].as_f64().map(|w| w as f32 * 100.0 * k);
    Ok(Heights { n, cell: size_m * 100.0 * k / (n - 1) as f32, h, z, water, maps: Vec::new() })
}

/// The paint layers the Landscape tab wrote beside a heightmap: (name, weightmap file), from the sidecar's `layers`.
pub fn sidecar_layers(path: &Path) -> Vec<(String, String)> {
    let path = &exported(path);
    let Ok(text) = std::fs::read_to_string(format!("{}.json", path.display())) else { return Vec::new() };
    let Ok(json) = serde_json::from_str::<serde_json::Value>(&text) else { return Vec::new() };
    json["layers"].as_array().map(|a| a.iter().filter_map(|l| Some((l["name"].as_str()?.to_string(), l["file"].as_str()?.to_string()))).collect()).unwrap_or_default()
}

/// The paint layers beside a heightmap with the colour the Landscape tab painted each in (grey when the sidecar has none).
pub fn sidecar_layer_colours(path: &Path) -> Vec<(String, [f32; 3])> {
    let path = &exported(path);
    let Ok(text) = std::fs::read_to_string(format!("{}.json", path.display())) else { return Vec::new() };
    let Ok(json) = serde_json::from_str::<serde_json::Value>(&text) else { return Vec::new() };
    json["layers"].as_array().map(|a| a.iter().filter_map(|l| {
        let c = l["color"].as_array().filter(|c| c.len() == 3).map(|c| [0, 1, 2].map(|i| c[i].as_f64().unwrap_or(0.5) as f32)).unwrap_or([0.5; 3]);
        Some((l["name"].as_str()?.to_string(), c))
    }).collect()).unwrap_or_default()
}

/// `load`, with the paint layers called `names` (from the sidecar's `layers`) as weightmaps. A weightmap of another size
/// is resampled to the heights'.
pub fn load_with_maps(path: &Path, unit_cm: f32, names: &[String]) -> Result<Heights, String> {
    let mut heights = load(path, unit_cm)?;
    if names.is_empty() {
        return Ok(heights);
    }
    let layers = sidecar_layers(path);
    for name in names {
        let (_, file) = layers.iter().find(|(n, _)| n == name).ok_or_else(|| {
            let known: Vec<&str> = layers.iter().map(|(n, _)| n.as_str()).collect();
            format!("{}: no paint layer called {name} (the sidecar lists: {})", path.display(), if known.is_empty() { "none; export the terrain with Layer nodes".to_string() } else { known.join(", ") })
        })?;
        let wpath = exported(path).with_file_name(file);
        let img = image::open(&wpath).map_err(|e| format!("{}: {e}", wpath.display()))?.to_luma16();
        let side = img.width() as usize;
        if img.height() as usize != side || side < 2 {
            return Err(format!("{}: a weightmap must be square", wpath.display()));
        }
        let values: Vec<f32> = img.as_raw().iter().map(|v| *v as f32 / 65535.0).collect();
        let values = if side == heights.n { values } else { crate::landscape_heightmap::shrink_or_grow(&values, side, heights.n) };
        heights.maps.push((name.clone(), values));
    }
    Ok(heights)
}

/// The height of the water a heightmap's sidecar names, in centimetres.
pub fn water_level_cm(path: &Path) -> Option<f32> {
    let path = &exported(path);
    let text = std::fs::read_to_string(format!("{}.json", path.display())).ok()?;
    let json: serde_json::Value = serde_json::from_str(&text).ok()?;
    json["water_level_m"].as_f64().map(|w| w as f32 * 100.0)
}

/// The side of a heightmap's terrain and the middle of its height range in a node's units, from its sidecar alone (no
/// need to read the image).
pub fn extent(path: &Path, unit_cm: f32) -> Option<(f32, f32)> {
    let path = &exported(path);
    let text = std::fs::read_to_string(format!("{}.json", path.display())).ok()?;
    let json: serde_json::Value = serde_json::from_str(&text).ok()?;
    let k = 100.0 / unit_cm;
    let (lo, hi) = (json["height_range_m"][0].as_f64()? as f32, json["height_range_m"][1].as_f64()? as f32);
    Some((json["size_m"].as_f64()? as f32 * k, (lo + hi) * 0.5 * k))
}

impl Heights {
    /// Side of the terrain, in units.
    pub fn world(&self) -> f32 {
        self.cell * (self.n - 1) as f32
    }

    /// The lowest and highest height among the samples that touch the box `x0..x1` by `y0..y1`.
    pub fn z_range(&self, x0: f32, x1: f32, y0: f32, y1: f32) -> (f32, f32) {
        let idx = |v: f32, up: bool| {
            let g = v / self.cell;
            (if up { g.ceil() } else { g.floor() }).clamp(0.0, (self.n - 1) as f32) as usize
        };
        let (i0, i1, j0, j1) = (idx(x0, false), idx(x1, true), idx(y0, false), idx(y1, true));
        let mut r = (f32::MAX, f32::MIN);
        for j in j0..=j1 {
            for v in &self.h[j * self.n + i0..=j * self.n + i1] {
                r = (r.0.min(*v), r.1.max(*v));
            }
        }
        r
    }

    /// Quads a surface-nets mesh of this terrain has per column of lattice cells, on average over the box: one for the top
    /// plus the slope along x and along y (a surface climbing 1 m per metre crosses one more lattice edge per cell).
    pub fn quads_per_column(&self, x0: f32, x1: f32, y0: f32, y1: f32) -> f32 {
        let idx = |v: f32| ((v / self.cell).round().max(0.0) as usize).min(self.n - 2);
        let (i0, i1, j0, j1) = (idx(x0), idx(x1).max(idx(x0) + 1), idx(y0), idx(y1).max(idx(y0) + 1));
        let (mut sum, mut count) = (0.0f64, 0usize);
        for j in j0..j1 {
            for i in i0..i1 {
                let h = self.h[j * self.n + i];
                let (dx, dy) = ((self.h[j * self.n + i + 1] - h).abs(), (self.h[(j + 1) * self.n + i] - h).abs());
                sum += (1.0 + (dx + dy) / self.cell) as f64;
                count += 1;
            }
        }
        (sum / count.max(1) as f64) as f32
    }

    /// The height of the ground at (x, y), bilinear (the nearest edge outside the terrain).
    pub fn ground(&self, x: f32, y: f32) -> f32 {
        let g = |v: f32| (v / self.cell).clamp(0.0, (self.n - 1) as f32);
        let (gx, gy) = (g(x), g(y));
        let (i, j) = ((gx as usize).min(self.n - 2), (gy as usize).min(self.n - 2));
        let (fx, fy) = (gx - i as f32, gy - j as f32);
        let h = |i: usize, j: usize| self.h[j * self.n + i];
        (h(i, j) * (1.0 - fx) + h(i + 1, j) * fx) * (1.0 - fy) + (h(i, j + 1) * (1.0 - fx) + h(i + 1, j + 1) * fx) * fy
    }

    /// The highest ground within `r` of (x, y): the centre and eight points around.
    pub fn highest_ground(&self, x: f32, y: f32, r: f32) -> f32 {
        (0..9).map(|k| if k == 8 { (x, y) } else { let a = k as f32 * std::f32::consts::FRAC_PI_4; (x + r * a.cos(), y + r * a.sin()) }).map(|(px, py)| self.ground(px, py)).fold(f32::MIN, f32::max)
    }

    /// The lowest ground within `r` of (x, y): the centre and eight points around.
    pub fn lowest_ground(&self, x: f32, y: f32, r: f32) -> f32 {
        (0..9).map(|k| if k == 8 { (x, y) } else { let a = k as f32 * std::f32::consts::FRAC_PI_4; (x + r * a.cos(), y + r * a.sin()) }).map(|(px, py)| self.ground(px, py)).fold(f32::MAX, f32::min)
    }

    /// Texture size: a square of `min(n, 2048)` texels, in `tiles()` by `tiles()` layers.
    pub fn tile_size(&self) -> usize {
        self.n.min(TILE)
    }

    pub fn tiles(&self) -> usize {
        self.n.div_ceil(TILE)
    }

    /// The heights as the layers of the texture (tile (tx, ty) is layer ty * tiles + tx), then each paint layer's
    /// weights in the same arrangement (map k starts at layer (k + 1) * tiles * tiles).
    fn texels(&self) -> Vec<f32> {
        let (t, tiles) = (self.tile_size(), self.tiles());
        let per = t * t * tiles * tiles;
        let mut out = vec![self.z.0; per];
        out.resize(per * (1 + self.maps.len()), 0.0);
        for y in 0..self.n {
            for x in 0..self.n {
                let layer = (y / TILE) * tiles + x / TILE;
                let at = (layer * t + y % TILE) * t + x % TILE;
                out[at] = self.h[y * self.n + x];
                for (k, (_, map)) in self.maps.iter().enumerate() {
                    out[(k + 1) * per + at] = map[y * self.n + x];
                }
            }
        }
        out
    }

    pub fn upload(&self, device: &eframe::wgpu::Device, queue: &eframe::wgpu::Queue) -> (eframe::wgpu::Texture, eframe::wgpu::TextureView) {
        use eframe::wgpu;
        let (t, tiles) = (self.tile_size() as u32, self.tiles() as u32);
        let size = wgpu::Extent3d { width: t, height: t, depth_or_array_layers: tiles * tiles * (1 + self.maps.len() as u32) };
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("terrain heights"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo { texture: &texture, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            bytemuck::cast_slice(&self.texels()),
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(t * 4), rows_per_image: Some(t) },
            size,
        );
        let view = texture.create_view(&Default::default());
        (texture, view)
    }

    /// The terrain as a coarse grid mesh (at most 129 x 129 vertices), for nodes that want a mesh from the file.
    pub fn coarse_mesh(&self) -> Mesh {
        let m = 129.min(self.n);
        let at = |i: usize, j: usize| {
            let (x, y) = (i * (self.n - 1) / (m - 1), j * (self.n - 1) / (m - 1));
            [x as f32 * self.cell, y as f32 * self.cell, self.h[y * self.n + x]]
        };
        let mut mesh = Mesh::default();
        for j in 0..m {
            for i in 0..m {
                mesh.positions.push(at(i, j));
            }
        }
        for j in 0..m - 1 {
            for i in 0..m - 1 {
                let v = |a: usize, b: usize| (b * m + a) as u32;
                mesh.triangles.push([v(i, j), v(i + 1, j), v(i + 1, j + 1)]);
                mesh.triangles.push([v(i, j), v(i + 1, j + 1), v(i, j + 1)]);
            }
        }
        mesh
    }
}

/// The Mesh graph that turns an exported heightmap into Nanite chunks: the terrain as a Load Mesh (1 unit = 1 cm, as Unreal
/// has it), Remesh in chunks of about 500 m, planar UVs, normal / occlusion / curvature maps, and the outputs, written
/// next to the heightmap in `chunks/`.
pub fn terrain_graph(png: &Path, world_m: f32) -> crate::graph::Graph {
    use crate::graph::{Graph, NodeKind};
    use crate::meshgraph::{LoadMeshParams, OutParams};
    let mut g = Graph::default();
    let chunks = (world_m / 500.0).round().clamp(1.0, 16.0) as u32;
    let folder = png.parent().unwrap_or(Path::new(".")).join("chunks").to_string_lossy().into_owned();
    // Paint layers beside the heightmap (the Landscape tab writes them for its Layer nodes) are loaded too and baked per chunk.
    let layers: Vec<String> = sidecar_layers(png).into_iter().map(|(name, _)| name).take(8).collect();
    let land = g.add(NodeKind::LoadMesh(LoadMeshParams { path: png.to_string_lossy().into_owned(), unit_cm: 1.0, chunks, maps: layers.clone(), ..Default::default() }), [0.0, 0.0]);
    let remesh = g.add(NodeKind::Remesh { triangles: 250_000, adaptivity: 1.0, mesher: Default::default(), auto_detail: false }, [320.0, 0.0]);
    let uv = g.add(NodeKind::AutoUv(crate::uvgen::UvParams { planar: true, ..Default::default() }), [600.0, 0.0]);
    let out = g.add(NodeKind::OutMesh(OutParams { folder: folder.clone(), name: "terrain".into(), ..Default::default() }), [880.0, 0.0]);
    let bake = g.add(NodeKind::Bake(crate::bake::Settings { channels: vec![crate::bake::Channel::Normal, crate::bake::Channel::Occlusion, crate::bake::Channel::Curvature], custom: layers.clone(), ..Default::default() }), [880.0, 160.0]);
    g.connect(land, remesh, 0);
    g.connect(remesh, uv, 0);
    g.connect(uv, out, 0);
    g.connect(uv, bake, 0);
    g.connect(land, bake, 1);
    for i in 0..3 {
        let img = g.add(NodeKind::OutImage(OutParams { folder: folder.clone(), name: "terrain".into(), ..Default::default() }), [1160.0, 160.0 + 90.0 * i as f32]);
        g.connect_from(bake, i, img, 0);
    }
    if !layers.is_empty() {
        // Each layer's weights as a custom map, the first four packed into one RGBA "splat" image for the material.
        for (k, name) in layers.iter().enumerate() {
            let layer = g.add(NodeKind::TerrainLayer { name: name.clone() }, [600.0, 320.0 + 70.0 * k as f32]);
            g.connect(layer, bake, 2 + k);
        }
        let combine = g.add(NodeKind::ImageCombine, [1160.0, 440.0]);
        for k in 0..layers.len().min(4) {
            g.connect_from(bake, 3 + k, combine, k);
        }
        let splat = g.add(NodeKind::OutImage(OutParams { folder, name: "terrain_splat".into(), ..Default::default() }), [1440.0, 440.0]);
        g.connect(combine, splat, 0);
    }
    g
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_exported_heightmap_reads_back_in_the_nodes_units() {
        let n = 33;
        let metres: Vec<f32> = (0..n * n).map(|k| 100.0 + (k % n) as f32 * 2.0 + (k / n) as f32 * 0.5).collect();
        let enc = crate::landscape::encode_unreal(&metres);
        let dir = std::env::temp_dir().join("noiser_heights_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("h.png");
        crate::landscape::write_unreal(&path, n, 640.0, &enc).unwrap();
        // Centimetres: a hundred times the metres, and 640 m is 64,000 units wide.
        let h = load(&path, 1.0).unwrap();
        assert!((h.world() - 64000.0).abs() < 1.0, "{}", h.world());
        assert!(h.h.iter().zip(&metres).all(|(a, m)| (a - m * 100.0).abs() <= enc.step_cm), "within one height step");
        let (lo, hi) = h.z_range(0.0, 64000.0, 0.0, 64000.0);
        assert!((lo - h.z.0).abs() < 1e-3 && (hi - h.z.1).abs() < 1e-3);
        // A single corner cell sees only its own samples.
        let (clo, chi) = h.z_range(0.0, 1.0, 0.0, 1.0);
        assert!(clo >= h.z.0 && chi <= h.h[n + 1].max(h.h[0]).max(h.h[1]).max(h.h[n]) + 1e-3);
        assert_eq!(h.coarse_mesh().triangles.len(), 2 * 32 * 32);
        assert_eq!(h.texels().len(), n * n);
    }
}
