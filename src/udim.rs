//! Packs many textured meshes into one UDIM tileset, after a batch run: every mesh keeps its UV layout but moves to its
//! own tile (rows of 10: tile 1001 is the first, 1011 starts the second row), the maps baked for it are copied under the
//! new name with the UDIM number, and a combined mesh with every part laid out in a grid carries one material.
//!
//! Output folder: `<name>.obj` (all parts, one material called `<name>`), `<name>_<map>.<1001>.<ext>` for every map, and
//! each part again under its own file name (OBJ or binary glTF) with the tile's UVs and the material `<name>`.

use crate::mesh::{Mesh, V3};
use crate::uvgen::UvMesh;
use std::path::{Path, PathBuf};

const IMAGE_EXTENSIONS: [&str; 7] = ["png", "exr", "tga", "jpg", "jpeg", "tif", "tiff"];

pub struct Settings {
    pub name: String,
    pub folder: PathBuf,
    pub files: Vec<PathBuf>,
}

/// The UDIM number of tile `i` (counted from 0): 1001 + column + 10 * row.
pub fn udim(i: usize) -> u32 {
    1001 + i as u32
}

/// The tile's offset in UV space: (column, row).
fn tile_offset(i: usize) -> (f32, f32) {
    ((i % 10) as f32, (i / 10) as f32)
}

/// Letters, digits, `_` and `-` only: the name becomes a file name and a material name.
pub fn clean_name(name: &str) -> String {
    name.trim().chars().map(|c| if c.is_alphanumeric() || c == '_' || c == '-' { c } else { '_' }).collect()
}

/// Smooth vertex normals from the triangles (area weighted).
fn vertex_normals(mesh: &Mesh) -> Vec<V3> {
    let mut n = vec![[0.0f32; 3]; mesh.positions.len()];
    for t in &mesh.triangles {
        let (a, b, c) = (mesh.positions[t[0] as usize], mesh.positions[t[1] as usize], mesh.positions[t[2] as usize]);
        let (u, v) = ([b[0] - a[0], b[1] - a[1], b[2] - a[2]], [c[0] - a[0], c[1] - a[1], c[2] - a[2]]);
        let f = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
        for &i in t {
            for k in 0..3 {
                n[i as usize][k] += f[k];
            }
        }
    }
    n.iter().map(|v| {
        let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-20);
        [v[0] / l, v[1] / l, v[2] / l]
    }).collect()
}

/// The mesh as one vertex per (position, UV) pair, moved by `shift` and with its UVs moved to tile `i`.
/// `UvMesh` keeps V pointing down, so the file's V (up) is stored as `1 - v`.
fn tiled(mesh: &Mesh, i: usize, shift: V3) -> UvMesh {
    let normals = vertex_normals(mesh);
    let (ox, oy) = tile_offset(i);
    let mut out = UvMesh::default();
    let mut seen = std::collections::HashMap::new();
    for (t, uvs) in mesh.triangles.iter().zip(&mesh.uvs) {
        let mut corner = [0u32; 3];
        for k in 0..3 {
            let v = t[k] as usize;
            let (u, w) = (uvs[k][0] + ox, uvs[k][1] + oy);
            corner[k] = *seen.entry((t[k], u.to_bits(), w.to_bits())).or_insert_with(|| {
                let p = mesh.positions[v];
                out.positions.push([p[0] + shift[0], p[1] + shift[1], p[2] + shift[2]]);
                out.normals.push(normals[v]);
                out.uvs.push([u, 1.0 - w]);
                out.source.push(t[k]);
                out.positions.len() as u32 - 1
            });
        }
        out.triangles.push(corner);
    }
    out
}

/// The maps baked for `stem`: files `<stem>_<map>.<ext>` next to it, as (path, `<map>`, extension). A file that
/// starts with the name of another, longer mesh (`rock_big_` for `rock`) belongs to that one.
fn maps_of(dir: &Path, stem: &str, stems: &[String]) -> Vec<(PathBuf, String, String)> {
    let prefix = format!("{stem}_");
    let mut found = Vec::new();
    let Ok(entries) = std::fs::read_dir(dir) else { return found };
    for e in entries.flatten() {
        let path = e.path();
        let Some(ext) = path.extension().map(|x| x.to_string_lossy().to_lowercase()).filter(|x| IMAGE_EXTENSIONS.contains(&x.as_str())) else { continue };
        let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        let Some(rest) = name.strip_prefix(&prefix) else { continue };
        if stems.iter().any(|o| o.len() > stem.len() && name.starts_with(&format!("{o}_"))) {
            continue;
        }
        let map = Path::new(rest).file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
        found.push((path, map, ext));
    }
    found.sort();
    found
}

/// Does the work; `log` gets one line per step. The files are taken in name order, which is the tile order.
pub fn pack(s: &Settings, log: &mut dyn FnMut(&str)) -> Result<String, String> {
    let name = clean_name(&s.name);
    if name.is_empty() {
        return Err("give the set a name".into());
    }
    if s.files.is_empty() {
        return Err("no mesh files selected".into());
    }
    std::fs::create_dir_all(&s.folder).map_err(|e| format!("{}: {e}", s.folder.display()))?;
    let mut files = s.files.clone();
    files.sort();
    let stems: Vec<String> = files.iter().map(|f| f.file_stem().map(|x| x.to_string_lossy().into_owned()).unwrap_or_default()).collect();
    let mut meshes = Vec::new();
    for f in &files {
        let mesh = crate::meshformats::load(f)?;
        if mesh.uvs.is_empty() {
            return Err(format!("{} has no UVs", f.display()));
        }
        meshes.push(mesh);
    }

    // The grid: a cell as large as the largest part's footprint (X and Y, Z up), with a margin, so no two parts meet.
    let bounds = |m: &Mesh| {
        let (mut lo, mut hi) = ([f32::MAX; 3], [f32::MIN; 3]);
        for p in &m.positions {
            for a in 0..3 {
                lo[a] = lo[a].min(p[a]);
                hi[a] = hi[a].max(p[a]);
            }
        }
        (lo, hi)
    };
    let all: Vec<_> = meshes.iter().map(bounds).collect();
    let cell = all.iter().map(|(lo, hi)| (hi[0] - lo[0]).max(hi[1] - lo[1])).fold(0.0, f32::max) * 1.1 + 1e-3;
    let columns = (meshes.len() as f32).sqrt().ceil() as usize;

    let mut combined = UvMesh::default();
    let mut tiles_out_of_range = Vec::new();
    for (i, ((mesh, stem), (lo, hi))) in meshes.iter().zip(&stems).zip(&all).enumerate() {
        if mesh.uvs.iter().flatten().any(|uv| uv[0] < -1e-3 || uv[0] > 1.001 || uv[1] < -1e-3 || uv[1] > 1.001) {
            tiles_out_of_range.push(stem.clone());
        }
        // The part's footprint centred in its cell, standing on Z = 0.
        let (col, row) = ((i % columns) as f32, (i / columns) as f32);
        let shift = [(col + 0.5) * cell - (lo[0] + hi[0]) * 0.5, (row + 0.5) * cell - (lo[1] + hi[1]) * 0.5, -lo[2]];
        let part = tiled(mesh, i, shift);
        let base = combined.positions.len() as u32;
        combined.triangles.extend(part.triangles.iter().map(|t| [t[0] + base, t[1] + base, t[2] + base]));
        combined.positions.extend(part.positions);
        combined.normals.extend(part.normals);
        combined.uvs.extend(part.uvs);
        combined.source.extend(part.source);

        // The part on its own, under its own file name, in its tile.
        let own = tiled(mesh, i, [0.0; 3]);
        let ext = files[i].extension().map(|e| e.to_string_lossy().to_lowercase()).unwrap_or_default();
        let (out_ext, note) = match ext.as_str() {
            "glb" | "gltf" => ("glb", ""),
            "obj" => ("obj", ""),
            _ => ("obj", " (written as OBJ: that format has no material)"),
        };
        let out = s.folder.join(format!("{stem}.{out_ext}"));
        match out_ext {
            "glb" => crate::meshformats::write_glb_as(&out, &own.positions, &own.normals, Some(&own.uvs), &own.triangles, &name),
            _ => own.write_obj_as(&out, &name),
        }
        .map_err(|e| format!("{}: {e}", out.display()))?;
        log(&format!("tile {} {stem}: {}{note}", udim(i), out.file_name().unwrap_or_default().to_string_lossy()));

        // Its maps, copied under the set's name with the tile number.
        let dir = files[i].parent().unwrap_or(Path::new("."));
        let maps = maps_of(dir, stem, &stems);
        if maps.is_empty() {
            log(&format!("  no maps named {stem}_<map> found in {}", dir.display()));
        }
        for (path, map, ext) in maps {
            let to = s.folder.join(format!("{name}_{map}.{}.{ext}", udim(i)));
            std::fs::copy(&path, &to).map_err(|e| format!("{}: {e}", to.display()))?;
            log(&format!("  {} -> {}", path.file_name().unwrap_or_default().to_string_lossy(), to.file_name().unwrap_or_default().to_string_lossy()));
        }
    }
    let combined_path = s.folder.join(format!("{name}.obj"));
    combined.write_obj_as(&combined_path, &name).map_err(|e| format!("{}: {e}", combined_path.display()))?;
    if !tiles_out_of_range.is_empty() {
        log(&format!("warning: UVs outside 0..1 in {}: they spill into other tiles", tiles_out_of_range.join(", ")));
    }
    Ok(format!("Packed {} meshes into {}: {}.obj, {} tiles from {} to {}", meshes.len(), s.folder.display(), name, meshes.len(), udim(0), udim(meshes.len() - 1)))
}

/// The "UDIM pack" window of the Mesh graph tab.
#[derive(Default)]
pub struct Tool {
    pub open: bool,
    name: String,
    folder: String,
    files: Vec<PathBuf>,
    log: std::sync::Arc<std::sync::Mutex<Vec<String>>>,
    job: Option<std::sync::mpsc::Receiver<Result<String, String>>>,
    result: Option<Result<String, String>>,
}

impl Tool {
    pub fn window(&mut self, ctx: &eframe::egui::Context) {
        use eframe::egui;
        if let Some(rx) = &self.job {
            match rx.try_recv() {
                Ok(r) => {
                    self.result = Some(r);
                    self.job = None;
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => ctx.request_repaint(),
                Err(_) => self.job = None,
            }
        }
        let mut open = self.open;
        egui::Window::new("UDIM pack").open(&mut open).default_width(420.0).show(ctx, |ui| {
            ui.weak("After a batch: puts the meshes you pick into one UDIM tileset (rows of 10, in name order), copies each mesh's maps (`<mesh>_<map>.png` next to the mesh) as `<name>_<map>.1001.png` and so on, writes one combined mesh with a single material, and each mesh again under its own name with the tile's UVs and that material.");
            ui.horizontal(|ui| {
                ui.label("name");
                ui.text_edit_singleline(&mut self.name);
            });
            ui.horizontal(|ui| {
                ui.label("folder");
                ui.add(egui::TextEdit::singleline(&mut self.folder).desired_width(260.0));
                if ui.button("Browse…").clicked() && let Some(d) = crate::library::dialog(crate::library::Kind::Udim, rfd::FileDialog::new()).pick_folder() {
                    self.folder = d.display().to_string();
                }
            });
            if ui.button("Select meshes…").clicked() && let Some(files) = rfd::FileDialog::new().add_filter("Mesh", &["obj", "ply", "gltf", "glb"]).pick_files() {
                if self.folder.is_empty() && let Some(dir) = files.first().and_then(|f| f.parent()) {
                    self.folder = crate::library::dir(crate::library::Kind::Udim).filter(|d| d.is_dir()).unwrap_or_else(|| dir.join("UDIM")).display().to_string();
                }
                self.files = files;
                self.files.sort();
            }
            ui.label(format!("{} mesh(es) selected", self.files.len()));
            egui::ScrollArea::vertical().id_salt("udim files").max_height(120.0).show(ui, |ui| {
                for (i, f) in self.files.iter().enumerate() {
                    ui.weak(format!("{}  {}", udim(i), f.file_name().unwrap_or_default().to_string_lossy()));
                }
            });
            let ready = self.job.is_none() && !clean_name(&self.name).is_empty() && !self.folder.is_empty() && !self.files.is_empty();
            if ui.add_enabled(ready, egui::Button::new("Pack")).clicked() {
                let settings = Settings { name: self.name.clone(), folder: PathBuf::from(&self.folder), files: self.files.clone() };
                let (tx, rx) = std::sync::mpsc::channel();
                let log = self.log.clone();
                log.lock().unwrap().clear();
                self.result = None;
                self.job = Some(rx);
                std::thread::spawn(move || {
                    let _ = tx.send(pack(&settings, &mut |s| log.lock().unwrap().push(s.to_string())));
                });
            }
            egui::ScrollArea::vertical().id_salt("udim log").max_height(160.0).stick_to_bottom(true).show(ui, |ui| {
                for l in self.log.lock().unwrap().iter() {
                    ui.weak(l);
                }
            });
            match &self.result {
                Some(Ok(m)) => _ = ui.label(m),
                Some(Err(e)) => _ = ui.colored_label(crate::editor::ORANGE, e),
                None => {}
            }
        });
        self.open = open;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiles_count_along_rows_of_ten() {
        assert_eq!((udim(0), tile_offset(0)), (1001, (0.0, 0.0)));
        assert_eq!((udim(9), tile_offset(9)), (1010, (9.0, 0.0)));
        assert_eq!((udim(10), tile_offset(10)), (1011, (0.0, 1.0)));
        assert_eq!(clean_name(" My Rocks! "), "My_Rocks_");
    }

    #[test]
    fn meshes_and_their_maps_are_packed_into_tiles() {
        let dir = std::env::temp_dir().join("noiser_udim_test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        // Two one-triangle meshes with UVs; "rock_big" starts with "rock_", so its map must not go to "rock".
        for stem in ["rock", "rock_big"] {
            std::fs::write(dir.join(format!("{stem}.obj")), "v 0 0 0\nv 1 0 0\nv 0 1 0\nvt 0 0\nvt 1 0\nvt 0 1\nf 1/1 2/2 3/3\n").unwrap();
            std::fs::write(dir.join(format!("{stem}_normal_base.png")), stem).unwrap();
        }
        let out = dir.join("packed");
        let s = Settings { name: "Rocks".into(), folder: out.clone(), files: vec![dir.join("rock.obj"), dir.join("rock_big.obj")] };
        let msg = pack(&s, &mut |_| {}).unwrap();
        assert!(msg.contains("1001") && msg.contains("1002"), "{msg}");
        assert_eq!(std::fs::read_to_string(out.join("Rocks_normal_base.1001.png")).unwrap(), "rock");
        assert_eq!(std::fs::read_to_string(out.join("Rocks_normal_base.1002.png")).unwrap(), "rock_big");
        let combined = std::fs::read_to_string(out.join("Rocks.obj")).unwrap();
        assert!(combined.lines().any(|l| l == "usemtl Rocks") && combined.matches("\nf ").count() == 2, "{combined}");
        // The second mesh's UVs sit in tile 1002: u from 1 to 2.
        let own = std::fs::read_to_string(out.join("rock_big.obj")).unwrap();
        assert!(own.lines().any(|l| l == "usemtl Rocks"));
        assert!(own.lines().filter(|l| l.starts_with("vt ")).all(|l| l.split(' ').nth(1).unwrap().parse::<f32>().unwrap() >= 1.0), "{own}");
        // The parts do not overlap in the combined mesh: their X ranges are disjoint.
        let xs: Vec<f32> = combined.lines().filter(|l| l.starts_with("v ")).map(|l| l.split(' ').nth(1).unwrap().parse().unwrap()).collect();
        assert!(xs[..3].iter().cloned().fold(f32::MIN, f32::max) < xs[3..].iter().cloned().fold(f32::MAX, f32::min));
    }
}
