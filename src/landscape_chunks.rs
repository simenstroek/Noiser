//! The terrain cut into chunks for Nanite: every chunk is a regular grid mesh of up to a triangle budget (about 1M),
//! with planar UVs filling 0..1 (so chunk `i` is UDIM tile 1001 + i for `udim::pack`) and its own normal, ambient
//! occlusion and curvature maps. All chunks sample one global height function at the same positions, so the vertices
//! on a shared border are identical and the normals (from the height function, not the mesh) have no seam either.
//!
//! Files per chunk, named `<name>_<row>_<column>` (row-major, so name order is tile order): the mesh (OBJ or GLB, in
//! centimetres, the chunk's corner at its origin) and `<chunk>_normal_base.png`, `_ambient_occlusion.png`,
//! `_curvature.png`; plus `<name>_chunks.json` with each chunk's location for placing them in Unreal.

use crate::bake::{Channel, Image, ImageFormat};
use crate::export::{Evaluator, Progress};
use crate::landscape::{Mapping, Settings, UNREAL_RES};
use crate::uvgen::UvMesh;
use rayon::prelude::*;
use std::path::PathBuf;
use std::sync::atomic::Ordering;

/// Quads per chunk side in the light stand-ins that make the combined mesh of the UDIM pack.
const PROXY_QUADS: usize = 128;

/// Heights in metres at points given as (x, y, 0) across the tile (x and y in 0..1 over the whole terrain).
pub type Sampler<'a> = &'a dyn Fn(&[[f32; 3]]) -> Vec<f32>;

#[derive(Clone, Debug, PartialEq)]
pub struct Options {
    pub name: String,
    pub folder: String,
    /// Side of a chunk in metres.
    pub chunk_m: f32,
    /// Triangles per chunk, at most.
    pub budget: usize,
    /// Side of the maps in texels.
    pub tex: usize,
    /// How far the ambient occlusion looks, in metres.
    pub ao_m: f32,
    pub glb: bool,
    /// Normal map with green up (OpenGL) instead of down (Unreal).
    pub green_up: bool,
    /// Also run the UDIM pack into `<folder>/UDIM` afterwards.
    pub pack_udim: bool,
}

impl Default for Options {
    fn default() -> Self {
        Options { name: "Terrain".into(), folder: String::new(), chunk_m: 500.0, budget: 1_000_000, tex: 2048, ao_m: 25.0, glb: true, green_up: false, pack_udim: true }
    }
}

/// The chunk grid: chunks per side, the chunk size that makes them tile the terrain exactly, and quads per chunk side.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Plan {
    pub chunks: usize,
    pub chunk_m: f32,
    pub quads: usize,
}

pub fn plan(world_m: f32, chunk_m: f32, budget: usize) -> Plan {
    let chunks = (world_m / chunk_m.max(1.0)).ceil().max(1.0) as usize;
    Plan { chunks, chunk_m: world_m / chunks as f32, quads: ((budget as f32 / 2.0).sqrt().floor() as usize).max(1) }
}

/// Samples a `cols` x `rows` block of points, `coord(col, row)` giving each point (about 262,144 points per call).
fn grid(sampler: Sampler, cols: usize, rows: usize, coord: impl Fn(usize, usize) -> [f32; 3]) -> Vec<f32> {
    let mut out = Vec::with_capacity(cols * rows);
    let per_call = (262_144 / cols).max(1);
    let mut r = 0;
    while r < rows {
        let n = per_call.min(rows - r);
        let pts: Vec<[f32; 3]> = (r..r + n).flat_map(|j| (0..cols).map(move |i| (i, j))).map(|(i, j)| coord(i, j)).collect();
        out.extend(sampler(&pts));
        r += n;
    }
    out
}

/// The mesh of chunk (`cx`, `cy`): positions in centimetres relative to the chunk's corner (z absolute), vertex normals
/// from central differences with a one-vertex margin, planar UVs.
pub fn chunk_mesh(sampler: Sampler, p: &Plan, cx: usize, cy: usize) -> UvMesh {
    let q = p.quads;
    let total = (p.chunks * q) as f32;
    let cell = p.chunk_m / q as f32;
    // Heights with a margin of one vertex: grid index (i + 1, j + 1) is vertex (i, j).
    let w = q + 3;
    let h = grid(sampler, w, w, |i, j| [((cx * q + i) as f32 - 1.0) / total, ((cy * q + j) as f32 - 1.0) / total, 0.0]);
    let at = |i: usize, j: usize| h[j * w + i];
    let mut m = UvMesh::default();
    for j in 0..=q {
        for i in 0..=q {
            let (dx, dy) = ((at(i + 2, j + 1) - at(i, j + 1)) / (2.0 * cell), (at(i + 1, j + 2) - at(i + 1, j)) / (2.0 * cell));
            let l = (dx * dx + dy * dy + 1.0).sqrt();
            m.positions.push([i as f32 * cell * 100.0, j as f32 * cell * 100.0, at(i + 1, j + 1) * 100.0]);
            m.normals.push([-dx / l, -dy / l, 1.0 / l]);
            // UvMesh keeps V pointing down: the file's V (up, along y) is stored as 1 - v.
            m.uvs.push([i as f32 / q as f32, 1.0 - j as f32 / q as f32]);
            m.source.push((j * (q + 1) + i) as u32);
        }
    }
    let v = |i: usize, j: usize| (j * (q + 1) + i) as u32;
    for j in 0..q {
        for i in 0..q {
            m.triangles.push([v(i, j), v(i + 1, j), v(i + 1, j + 1)]);
            m.triangles.push([v(i, j), v(i + 1, j + 1), v(i, j + 1)]);
        }
    }
    m
}

/// The maps of chunk (`cx`, `cy`) from the heights at the texel centres plus a margin wide enough for the occlusion to
/// look past the chunk's edge (so it matches the neighbour's). Rows of the images run from the top, which is +y.
pub fn chunk_maps(sampler: Sampler, world_m: f32, p: &Plan, cx: usize, cy: usize, o: &Options) -> Vec<Image> {
    let (tex, cell) = (o.tex, p.chunk_m / o.tex as f32);
    let margin = (o.ao_m / cell).ceil() as usize + 1;
    let w = tex + 2 * margin;
    let h = grid(sampler, w, w, |i, j| {
        let (x, y) = (cx as f32 * p.chunk_m + (i as f32 - margin as f32 + 0.5) * cell, cy as f32 * p.chunk_m + (j as f32 - margin as f32 + 0.5) * cell);
        [x / world_m, y / world_m, 0.0]
    });
    let at = |i: usize, j: usize| h[j * w + i];
    // Occlusion: how high the horizon rises over each texel along 12 directions, out to the reach.
    const DIRS: usize = 12;
    let mut steps = Vec::new();
    let mut d = cell;
    while d < o.ao_m {
        steps.push(d);
        d *= 1.5;
    }
    let sample = |x: f32, y: f32| {
        let (x, y) = (x.clamp(0.0, (w - 1) as f32 - 1e-3), y.clamp(0.0, (w - 1) as f32 - 1e-3));
        let (i, j, fx, fy) = (x as usize, y as usize, x.fract(), y.fract());
        (at(i, j) * (1.0 - fx) + at(i + 1, j) * fx) * (1.0 - fy) + (at(i, j + 1) * (1.0 - fx) + at(i + 1, j + 1) * fx) * fy
    };
    let mut normal = vec![0.0f32; tex * tex * 3];
    let mut ao = vec![0.0f32; tex * tex];
    let mut lap = vec![0.0f32; tex * tex];
    ao.par_chunks_mut(tex).zip(normal.par_chunks_mut(tex * 3)).zip(lap.par_chunks_mut(tex)).enumerate().for_each(|(k, ((ao_row, n_row), lap_row))| {
        // Image row k from the top is grid row margin + tex - 1 - k.
        let j = margin + tex - 1 - k;
        for c in 0..tex {
            let i = margin + c;
            let (dx, dy) = ((at(i + 1, j) - at(i - 1, j)) / (2.0 * cell), (at(i, j + 1) - at(i, j - 1)) / (2.0 * cell));
            let l = (dx * dx + dy * dy + 1.0).sqrt();
            // Tangent space of the planar UVs: red along +x (u), green along the image's up (+y) when green is up and
            // along its down (-y) for Unreal.
            let green = if o.green_up { -dy } else { dy };
            n_row[c * 3] = 0.5 - 0.5 * dx / l;
            n_row[c * 3 + 1] = 0.5 + 0.5 * green / l;
            n_row[c * 3 + 2] = 0.5 + 0.5 / l;
            lap_row[c] = (at(i + 1, j) + at(i - 1, j) + at(i, j + 1) + at(i, j - 1) - 4.0 * at(i, j)) / (cell * cell);
            let centre = at(i, j);
            let mut open = 0.0;
            for a in 0..DIRS {
                let ang = (a as f32 + 0.5) / DIRS as f32 * std::f32::consts::TAU;
                let (ca, sa) = (ang.cos(), ang.sin());
                let mut slope = 0.0f32;
                for &s in &steps {
                    slope = slope.max((sample(i as f32 + ca * s / cell, j as f32 + sa * s / cell) - centre) / s);
                }
                open += 1.0 - slope / (1.0 + slope * slope).sqrt();
            }
            ao_row[c] = open / DIRS as f32;
        }
    });
    // Curvature: 0.5 on flat ground, towards 1 on convex and 0 on concave parts, scaled by the 95th percentile of its size.
    let mut mags: Vec<f32> = lap.iter().map(|v| v.abs()).collect();
    mags.sort_by(|a, b| a.total_cmp(b));
    let range = mags.get((mags.len() as f32 * 0.95) as usize).copied().unwrap_or(1.0).max(1e-9);
    let curvature: Vec<f32> = lap.iter().map(|v| 0.5 - 0.5 * (v / range).clamp(-1.0, 1.0)).collect();
    let image = |channel: Channel, components: usize, data: Vec<f32>| Image { channel: Some(channel), label: channel.suffix().to_string(), size: tex as u32, components, data };
    vec![image(Channel::Normal, 3, normal), image(Channel::Occlusion, 1, ao), image(Channel::Curvature, 1, curvature)]
}

/// Writes every chunk, the manifest, and optionally the UDIM pack. `log` gets one line per step.
pub fn export(ev: &Evaluator, s: &Settings, o: &Options, progress: &Progress, log: &mut dyn FnMut(&str)) -> Result<String, String> {
    if o.folder.is_empty() {
        return Err("choose a folder".into());
    }
    let name = crate::udim::clean_name(&o.name);
    if name.is_empty() {
        return Err("give the chunks a name".into());
    }
    let folder = PathBuf::from(&o.folder);
    std::fs::create_dir_all(&folder).map_err(|e| format!("{}: {e}", folder.display()))?;
    let p = plan(s.world_m, o.chunk_m, o.budget);
    log(&format!("{} x {} chunks of {} m, {} x {} quads each ({} triangles), maps {} px", p.chunks, p.chunks, p.chunk_m, p.quads, p.quads, 2 * p.quads * p.quads, o.tex));
    // One height mapping for all chunks (and the one the heightmap export uses): the range of the whole terrain.
    log("measuring the height range");
    let mapping = Mapping::from_values(&crate::landscape::sample(ev, UNREAL_RES, &Progress::default()), s);
    let sampler = |pts: &[[f32; 3]]| ev.eval_points(pts).into_iter().map(|v| mapping.apply(v)).collect::<Vec<f32>>();
    let total = p.chunks * p.chunks;
    let (mut meshes, mut entries) = (Vec::new(), Vec::new());
    for cy in 0..p.chunks {
        for cx in 0..p.chunks {
            let index = cy * p.chunks + cx;
            let stem = format!("{name}_{cy:02}_{cx:02}");
            let path = folder.join(format!("{stem}.{}", if o.glb { "glb" } else { "obj" }));
            let mesh = chunk_mesh(&sampler, &p, cx, cy);
            let written = if o.glb { crate::meshformats::write_glb(&path, &mesh.positions, &mesh.normals, Some(&mesh.uvs), &mesh.triangles) } else { mesh.write_obj(&path) };
            written.map_err(|e| format!("{}: {e}", path.display()))?;
            for img in chunk_maps(&sampler, s.world_m, &p, cx, cy, o) {
                let map = folder.join(format!("{stem}_{}.png", img.label));
                img.write(&map, ImageFormat::Png16)?;
            }
            log(&format!("chunk {} of {total}: {stem}", index + 1));
            progress.store(((index + 1) * 1000 / total) as u32, Ordering::Relaxed);
            entries.push(serde_json::json!({
                "file": path.file_name().unwrap_or_default().to_string_lossy(),
                "grid": [cx, cy],
                "location_cm": [cx as f32 * p.chunk_m * 100.0, cy as f32 * p.chunk_m * 100.0, 0.0],
                "udim": crate::udim::udim(index),
            }));
            meshes.push(path);
        }
    }
    let manifest = serde_json::json!({ "name": name, "units": "cm", "world_m": s.world_m, "chunk_m": p.chunk_m, "chunks_per_side": p.chunks, "quads_per_chunk_side": p.quads, "chunks": entries });
    std::fs::write(folder.join(format!("{name}_chunks.json")), serde_json::to_string_pretty(&manifest).unwrap_or_default()).map_err(|e| e.to_string())?;
    if o.pack_udim {
        // The chunks themselves (tile UVs, one material, the maps as .1001 ...) without a combined mesh, which at a million
        // triangles per chunk would be enormous; the combined mesh for texturing in Substance is made from light stand-ins.
        log("packing the UDIM tileset");
        let udim = folder.join("UDIM");
        let packed = crate::udim::pack(&crate::udim::Settings { name: name.clone(), folder: udim.clone(), files: meshes.clone(), no_combined: true, ..Default::default() }, log)?;
        log(&packed);
        let proxy_plan = Plan { quads: p.quads.min(PROXY_QUADS), ..p };
        let proxy_dir = udim.join("proxy");
        std::fs::create_dir_all(&proxy_dir).map_err(|e| format!("{}: {e}", proxy_dir.display()))?;
        let mut proxies = Vec::new();
        for (index, path) in meshes.iter().enumerate() {
            let (cx, cy) = (index % p.chunks, index / p.chunks);
            let proxy = proxy_dir.join(format!("{}.obj", path.file_stem().unwrap_or_default().to_string_lossy()));
            chunk_mesh(&sampler, &proxy_plan, cx, cy).write_obj(&proxy).map_err(|e| format!("{}: {e}", proxy.display()))?;
            proxies.push(proxy);
        }
        let combined = crate::udim::pack(&crate::udim::Settings { name: name.clone(), folder: udim, files: proxies, only_combined: true, ..Default::default() }, log)?;
        log(&format!("combined mesh from {0} x {0} quad stand-ins: {combined}", proxy_plan.quads));
    }
    Ok(format!("Wrote {total} chunk(s) to {}", folder.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A smooth, non-flat height function over the tile (metres): hills.
    fn hills(pts: &[[f32; 3]]) -> Vec<f32> {
        pts.iter().map(|p| 40.0 * (p[0] * 9.0).sin() * (p[1] * 7.0).cos() + 30.0 * p[0]).collect()
    }

    #[test]
    fn the_plan_cuts_the_terrain_evenly_within_the_budget() {
        let p = plan(2000.0, 500.0, 1_000_000);
        assert_eq!((p.chunks, p.chunk_m, p.quads), (4, 500.0, 707));
        assert!(2 * p.quads * p.quads <= 1_000_000);
        // A chunk size that does not divide the world is rounded so the chunks fill it exactly.
        assert_eq!(plan(1000.0, 300.0, 200).chunk_m, 250.0);
    }

    #[test]
    fn neighbouring_chunks_share_border_vertices_and_normals() {
        let p = Plan { chunks: 2, chunk_m: 50.0, quads: 16 };
        let (a, b) = (chunk_mesh(&hills, &p, 0, 0), chunk_mesh(&hills, &p, 1, 0));
        let q = p.quads;
        assert_eq!(a.triangles.len(), 2 * q * q);
        for j in 0..=q {
            let (left, right) = (j * (q + 1) + q, j * (q + 1));
            // Same height and the same normal; x is the chunk's width at one end and 0 at the other (chunks are local to their corner).
            assert_eq!(a.positions[left][2], b.positions[right][2]);
            assert_eq!(a.normals[left], b.normals[right]);
            assert_eq!(a.positions[left][0], p.chunk_m * 100.0);
            assert_eq!(b.positions[right][0], 0.0);
        }
    }

    #[test]
    fn flat_ground_is_open_and_flat_in_the_maps_and_a_pit_is_darker() {
        let p = Plan { chunks: 1, chunk_m: 40.0, quads: 8 };
        let o = Options { tex: 32, ao_m: 10.0, ..Default::default() };
        let flat = chunk_maps(&|pts: &[[f32; 3]]| vec![5.0; pts.len()], 40.0, &p, 0, 0, &o);
        assert!(flat[0].data.chunks(3).all(|t| (t[0] - 0.5).abs() < 1e-5 && (t[1] - 0.5).abs() < 1e-5 && (t[2] - 1.0).abs() < 1e-5));
        assert!(flat[1].data.iter().all(|v| (*v - 1.0).abs() < 1e-5), "flat ground sees the whole sky");
        // A round pit in the middle: the bottom sees less sky than the corner.
        let pit = |pts: &[[f32; 3]]| pts.iter().map(|p| -8.0 * (-(((p[0] - 0.5).powi(2) + (p[1] - 0.5).powi(2)) / 0.004)).exp()).collect::<Vec<f32>>();
        let ao = &chunk_maps(&pit, 40.0, &p, 0, 0, &o)[1].data;
        let (middle, corner) = (ao[16 * 32 + 16], ao[0]);
        assert!(middle < corner - 0.05, "pit {middle}, corner {corner}");
    }
}
