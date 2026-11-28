//! The details-panel controls of the mesh graph nodes.

use crate::graph::NodeKind;
use eframe::egui;

thread_local! {
    /// The folder of the mesh graph being edited: relative paths of its nodes are relative to it.
    static BASE: std::cell::RefCell<std::path::PathBuf> = std::cell::RefCell::new(std::path::PathBuf::from("."));
}

thread_local! {
    /// While the Mesh graph draws a node's settings: how many centimetres one unit of its field is, so lengths can show in metres.
    static UNIT_CM: std::cell::Cell<Option<f32>> = const { std::cell::Cell::new(None) };
}

pub fn set_unit_cm(unit: Option<f32>) {
    UNIT_CM.with(|u| u.set(unit));
}

/// A length in the field's units as metres for display ("45000" is "450 m" for a field in centimetres); empty outside the Mesh graph.
pub fn metres(v: f32) -> String {
    let Some(unit) = UNIT_CM.with(|u| u.get()) else { return String::new() };
    let m = v * unit / 100.0;
    if m.abs() >= 1000.0 { format!("= {:.2} km", m / 1000.0) } else if m.abs() >= 10.0 { format!("= {m:.0} m") } else { format!("= {m:.1} m") }
}

/// Tells the node settings which folder relative paths belong to (the mesh graph's).
pub fn set_base(dir: std::path::PathBuf) {
    BASE.with(|b| *b.borrow_mut() = dir);
}

pub(crate) fn base() -> std::path::PathBuf {
    BASE.with(|b| b.borrow().clone())
}

/// A text field with a Browse button; `filter` is the file dialog's name and extensions, empty for a folder.
pub(crate) fn path_row(ui: &mut egui::Ui, id: &str, path: &mut String, hint: &str, filter: Option<(&str, &[&str])>) {
    ui.horizontal(|ui| {
        ui.add(egui::TextEdit::singleline(path).hint_text(hint).desired_width(150.0).id_salt(id));
        if ui.button("Browse…").clicked() {
            let start = std::path::Path::new(path.as_str()).parent().map(|p| p.to_path_buf());
            let mut dialog = rfd::FileDialog::new();
            if let Some(dir) = start.filter(|d| d.is_dir()) {
                dialog = dialog.set_directory(dir);
            } else if filter.is_none() {
                dialog = crate::library::dialog(crate::library::Kind::Bake, dialog);
            }
            let picked = match filter {
                Some((name, extensions)) => dialog.add_filter(name, extensions).pick_file(),
                None => dialog.pick_folder(),
            };
            if let Some(p) = picked {
                *path = p.to_string_lossy().into_owned();
            }
        }
    });
}

/// Controls for the mesh graph node kinds; false when `kind` is not one of them.
pub fn props(ui: &mut egui::Ui, kind: &mut NodeKind) -> bool {
    let image = matches!(kind, NodeKind::OutImage(_));
    match kind {
        NodeKind::LoadMesh(p) if p.landscape || crate::landscape_field::is_heightmap(&p.path) => {
            if p.landscape {
                ui.weak("A terrain from a Landscape project, z up: the project is exported to a heightmap when the graph runs (again after the project is saved). The field is the height above it. Wire it to a Remesh: that makes the terrain in chunks of about the Remesh's triangle count each, cut from one lattice so neighbours meet exactly. Add Load Mesh / Load SDF nodes with a union to put objects on it, and a noise graph to add 3D detail.");
                path_row(ui, "landscape path", &mut p.path, "a .landscape file", Some(("Landscape project", &["landscape"])));
            } else {
                ui.weak("A terrain: a heightmap PNG from the Landscape tab (its .json sidecar must sit next to it) as the ground, z up. The field is the height above it. Wire it to a Remesh: that makes the terrain in chunks of about the Remesh's triangle count each, cut from one lattice so neighbours meet exactly. Add Load Mesh / Load SDF nodes with a union to put objects on it, and a noise graph to add 3D detail.");
                path_row(ui, "terrain path", &mut p.path, "heightmap .png", Some(("Heightmap", &["png"])));
            }
            let d = crate::meshgraph::LoadMeshParams::default();
            crate::row(ui, &mut p.chunks, d.chunks, |ui, v| _ = ui.add(egui::DragValue::new(v).range(1..=32).prefix("chunks ").suffix(" per side")).on_hover_text("The terrain is cut into this many chunks along each side; a run writes one file per chunk and map, named ending _row_column."));
            let layers = crate::landscape_field::sidecar_layers(&crate::meshgraph::resolve_in(&base(), &p.path));
            if layers.is_empty() {
                ui.weak(if p.landscape { "No paint layers yet: they show here after the first run, when the project has Layer nodes." } else { "No paint layers beside this heightmap (the Landscape tab writes them when its graph has Layer nodes)." });
            } else {
                ui.label("paint layers to load:");
                ui.horizontal_wrapped(|ui| {
                    for (name, _) in &layers {
                        let mut on = p.maps.contains(name);
                        if ui.checkbox(&mut on, name).on_hover_text("Loaded beside the heights: Scatter's density map and Terrain layer nodes can read it. Each costs as much video memory as the heights.").changed() {
                            if on { p.maps.push(name.clone()) } else { p.maps.retain(|m| m != name) }
                        }
                    }
                });
            }
            crate::row(ui, &mut p.unit_cm, d.unit_cm, |ui, v| _ = ui.add(egui::DragValue::new(v).range(0.0001..=1.0e6).speed(0.5).prefix("1 unit = ").suffix(" cm")).on_hover_text("The unit of the field: 100 for metres, 1 for centimetres (Unreal). Other meshes in the graph must use the same unit."));
            true
        }
        NodeKind::LoadMesh(p) => {
            if p.sdf || p.path.to_lowercase().ends_with(".sdf") {
                ui.weak("An SDF Modeler composition (made in the SDF Composer tab) as a field: its signed distance, negative inside. It is sampled on a grid and turned into a mesh first, Z up like SDF Modeler's exports, so the file's own detail is limited by the samples below.");
                path_row(ui, "sdf path", &mut p.path, "a .sdf file", Some(("SDF Modeler file", &["sdf"])));
            } else {
                ui.weak("A mesh file as a field: its signed distance, negative inside, in the units of the file.");
                ui.weak("A path of {batch} reads the mesh file of a headless batch run (--mesh-graph with --batch-dir).");
                path_row(ui, "mesh path", &mut p.path, "mesh file or {batch}", Some(("Mesh", &["obj", "ply", "stl", "gltf", "glb", "fbx", "sdf", "png"])));
            }
            let d = crate::meshgraph::LoadMeshParams::default();
            crate::row(ui, &mut p.resolution, d.resolution, |ui, v| _ = ui.add(egui::DragValue::new(v).range(32..=512).suffix(" samples")).on_hover_text("Resolution of the distance grid the mesh is baked into."));
            crate::row(ui, &mut p.padding, d.padding, |ui, v| _ = ui.add(egui::DragValue::new(v).range(0.1..=4.0).speed(0.01).prefix("padding ")).on_hover_text("Room around the mesh in the grid, in units of the mesh's radius: displacement cannot reach past it."));
            crate::row(ui, &mut p.unit_cm, d.unit_cm, |ui, v| _ = ui.add(egui::DragValue::new(v).range(0.0001..=1.0e6).speed(0.5).prefix("1 unit = ").suffix(" cm")).on_hover_text("The size of one unit of this file: 100 for a file in metres, 1 for centimetres (Unreal and many FBX files). Noise Graph nodes work in metres, so their frequency and amplitude are scaled to the file's units."));
            crate::row(ui, &mut p.combine.mode, d.combine.mode, |ui, v| {
                egui::ComboBox::from_id_salt("combine mode").selected_text(v.label()).show_ui(ui, |ui| {
                    for m in crate::meshsdf::CombineMode::ALL {
                        ui.selectable_value(v, m, m.label());
                    }
                });
            });
            if p.combine.mode.is_smooth() {
                crate::row(ui, &mut p.combine.smooth, d.combine.smooth, |ui, v| _ = ui.add(egui::Slider::new(v, 0.0..=0.5).text("blend width")));
            }
            true
        }
        NodeKind::NoiseGraph(p) => {
            ui.weak("A noise graph used as a node. The file is the source of truth; an embedded copy is kept for when it is missing.");
            path_row(ui, "noise graph path", &mut p.file, "noise graph file", Some(("Noise graph", &["noisegraph"])));
            let d = crate::meshgraph::NoiseGraphParams::default();
            crate::row(ui, &mut p.frequency, d.frequency, |ui, v| _ = ui.add(egui::DragValue::new(v).speed(0.01).prefix("frequency × ")).on_hover_text("The noise graph sees the position times this: tiles per unit of the mesh file."));
            crate::row(ui, &mut p.amplitude, d.amplitude, |ui, v| _ = ui.add(egui::DragValue::new(v).speed(0.001).prefix("amplitude × ")).on_hover_text("The graph's density output times this: its size in units of the mesh file. Positive pushes the surface outward where the noise graph's density is positive, as in the noise graph tab."));
            if p.detached {
                ui.colored_label(crate::editor::ORANGE, "Detached: this graph lives in the mesh graph only.");
                if ui.button("Re-attach to the file").clicked() {
                    p.detached = false;
                }
            } else if !p.embedded.is_empty() && ui.button("Detach (keep only the embedded copy)").on_hover_text("Edits to the file no longer change this mesh graph.").clicked() {
                p.detached = true;
            }
            let full = crate::meshgraph::resolve_in(&base(), &p.file);
            match crate::meshgraph::noise_graph_source(p, &base()) {
                Ok(_) if !p.detached && !p.file.is_empty() && std::fs::metadata(&full).is_ok() => {
                    if ui.button("Refresh embedded copy").on_hover_text("Saving the mesh graph does this too.").clicked()
                        && let Ok(text) = std::fs::read_to_string(&full) {
                            p.embedded = text;
                        }
                    ui.weak(if p.embedded.is_empty() { "No embedded copy yet." } else { "An embedded copy is stored in the mesh graph." });
                }
                Ok(_) => _ = ui.colored_label(crate::editor::ORANGE, "The file is missing: using the embedded copy."),
                Err(e) => _ = ui.colored_label(crate::editor::ORANGE, e),
            }
            true
        }
        NodeKind::Remesh { triangles, adaptivity, mesher, auto_detail } => {
            ui.weak("The surface of the field as a triangle mesh. Adaptive (the default) makes even triangles, small where the surface bends and large where it is flat.");
            crate::row(ui, mesher, crate::graph::Mesher::Adaptive, |ui, v| {
                egui::ComboBox::from_id_salt("remesh mesher").selected_text(v.label()).show_ui(ui, |ui| {
                    for m in crate::graph::Mesher::ALL {
                        ui.selectable_value(v, m, m.label());
                    }
                }).response.on_hover_text("Adaptive: curvature-following, well-shaped triangles. Surface nets: a regular grid (what the Composer exports), fast. Marching tetrahedra: a regular grid that is always closed. The grid ones need the surface to be thicker than a cell.");
            });
            crate::row(ui, triangles, 5000, |ui, v| _ = ui.add(egui::DragValue::new(v).range(200..=2_000_000).speed(100.0).suffix(if *auto_detail { " triangles (sphere)" } else { " triangles" })));
            crate::row(ui, auto_detail, false, |ui, v| _ = ui.checkbox(v, "scale by shape complexity").on_hover_text("Measures how much the shape bends on a coarse version of it (fine surface noise left out) and scales the triangle count: a sphere gets the count above, a plainer shape fewer, a lumpy or complex one up to four times more. Each mesh of a batch gets its own count."));
            ui.add_enabled_ui(*mesher == crate::graph::Mesher::Adaptive, |ui| {
                crate::row(ui, adaptivity, 1.0, |ui, v| _ = ui.add(egui::Slider::new(v, 0.0..=1.0).text("adaptivity")).on_hover_text("0 gives triangles of one size everywhere; 1 lets them be a quarter of that size on tight curves and five times larger on flat ground."));
            });
            true
        }
        NodeKind::AutoUv(uv) => {
            let d = crate::uvgen::UvParams::default();
            ui.weak("An overlap-free atlas of islands in the 0 to 1 square.");
            crate::row(ui, &mut uv.resolution, d.resolution, |ui, v| _ = ui.add(egui::DragValue::new(v).range(256..=8192).speed(32.0).suffix(" px atlas")));
            crate::row(ui, &mut uv.padding, d.padding, |ui, v| _ = ui.add(egui::DragValue::new(v).range(0..=64).prefix("padding ").suffix(" px")));
            crate::row(ui, &mut uv.smoothing, d.smoothing, |ui, v| _ = ui.add(egui::DragValue::new(v).range(0..=60).prefix("chart smoothing ")).on_hover_text("Rounds of smoothing of the surface the islands are cut from; the UVs still map the true surface. A few rounds merge small islands on a bumpy surface; many rounds stretch the texture on bumps."));
            crate::row(ui, &mut uv.absorb_tiny, d.absorb_tiny, |ui, v| _ = ui.checkbox(v, "fold tiny islands into neighbours"));
            crate::row(ui, &mut uv.planar, d.planar, |ui, v| _ = ui.checkbox(v, "planar from above (terrain chunks)").on_hover_text("Projects the chunk straight down onto its 0 to 1 square: instant at any size, and chunk i is UDIM tile 1001 + i. Only for a terrain (Load Mesh with a heightmap before the Remesh); on any other mesh the node unwraps as usual."));
            crate::row(ui, &mut uv.brute_force, d.brute_force, |ui, v| _ = ui.checkbox(v, "best packing (slower; up to 50k triangles)"));
            true
        }
        NodeKind::Bake(b) => {
            let d = crate::bake::Settings::default();
            ui.weak("Maps sampled from the field onto the mesh's UVs. Input 1 is the mesh, input 2 the field. One image output per map.");
            egui::ComboBox::from_id_salt("bake size").selected_text(format!("{0} × {0}", b.size)).show_ui(ui, |ui| {
                for n in [512u32, 1024, 2048, 4096] {
                    ui.selectable_value(&mut b.size, n, format!("{n} × {n}"));
                }
            });
            crate::row(ui, &mut b.padding, d.padding, |ui, v| _ = ui.add(egui::DragValue::new(v).range(0..=64).prefix("padding ").suffix(" px")));
            crate::row(ui, &mut b.samples, d.samples, |ui, v| _ = ui.add(egui::DragValue::new(v).range(1..=4).prefix("samples ").suffix(" per side")).on_hover_text("Anti-aliasing: the map is baked at this many times the size and averaged down (not above 4096 px in all)."));
            for c in crate::bake::Channel::ALL {
                let mut on = b.channels.contains(&c);
                if ui.checkbox(&mut on, c.name()).changed() {
                    if on {
                        b.channels.push(c);
                        b.channels.sort_by_key(|c| crate::bake::Channel::ALL.iter().position(|x| x == c));
                    } else {
                        b.channels.retain(|x| *x != c);
                    }
                }
            }
            ui.checkbox(&mut b.green_up, "normal map green up (Blender, Unity, glTF)");
            ui.separator();
            ui.weak("Custom maps: a field wired into an extra input is baked as a map (values 0 to 1), for example a mask you blend surfaces with.");
            let mut remove = None;
            for (i, name) in b.custom.iter_mut().enumerate() {
                ui.horizontal(|ui| {
                    ui.add(egui::TextEdit::singleline(name).desired_width(120.0));
                    if ui.small_button("×").on_hover_text("Remove this custom map").clicked() {
                        remove = Some(i);
                    }
                });
            }
            if let Some(i) = remove {
                b.custom.remove(i);
            }
            if b.custom.len() < 6 && ui.button("Add custom map").clicked() {
                b.custom.push(format!("Custom{}", b.custom.len() + 1));
            }
            true
        }
        NodeKind::TransferUv => {
            ui.weak("Carries the UVs of a mesh file (the Mesh output of a Load Mesh whose file has UVs) or of an unwrapped mesh (Auto UVs, for a lower level of detail that shares the layout and the maps) onto another mesh that lies on the same surface, such as the Remesh output. Corners look for the closest point on the island under their triangle, so triangles are not stretched across seams; the new mesh's seams follow the source's.");
            true
        }
        NodeKind::ImageSplit => {
            ui.weak("Splits an image into four grey images: red, green, blue and alpha (1 when the image has none).");
            true
        }
        NodeKind::ImageCombine => {
            ui.weak("Makes one image from four grey images. Unconnected channels are 0 (alpha 1). For example: a normal map's red and green, then occlusion in blue and a mask in alpha, with the blue of the normal rebuilt in the shader.");
            true
        }
        NodeKind::ImageLevels { scale, offset, invert } => {
            ui.weak("value × scale + offset, then inverted if ticked, limited to 0 to 1.");
            crate::row(ui, scale, 1.0, |ui, v| _ = ui.add(egui::DragValue::new(v).speed(0.01).prefix("scale ")));
            crate::row(ui, offset, 0.0, |ui, v| _ = ui.add(egui::DragValue::new(v).speed(0.01).prefix("offset ")));
            crate::row(ui, invert, false, |ui, v| _ = ui.checkbox(v, "invert"));
            true
        }
        NodeKind::OutMesh(o) | NodeKind::OutImage(o) => {
            ui.weak(if image { "Writes the image." } else { "Writes the mesh (with UVs and normals when it has them)." });
            if image {
                egui::ComboBox::from_id_salt("image format").selected_text(o.image.label()).show_ui(ui, |ui| {
                    for f in crate::bake::ImageFormat::ALL {
                        ui.selectable_value(&mut o.image, f, f.label());
                    }
                });
            } else {
                ui.horizontal(|ui| {
                    let (mut ply, mut glb) = (o.ply, o.glb);
                    if ui.selectable_label(!ply && !glb, "OBJ").clicked() {
                        (ply, glb) = (false, false);
                    }
                    if ui.selectable_label(ply && !glb, "PLY (binary)").clicked() {
                        (ply, glb) = (true, false);
                    }
                    if ui.selectable_label(glb, "glTF (.glb)").clicked() {
                        (ply, glb) = (false, true);
                    }
                    (o.ply, o.glb) = (ply, glb);
                });
            }
            path_row(ui, "output folder", &mut o.folder, "folder (empty: library Bakes, else next to the graph)", None);
            ui.horizontal(|ui| {
                ui.label("name");
                ui.add(egui::TextEdit::singleline(&mut o.name).desired_width(140.0));
            });
            ui.weak(if image { "A map's name is added to the file name: name_normal_base.png." } else { "An existing file with the same name is overwritten." });
            ui.weak(format!("Smart tags in the name: {}.", crate::meshgraph::TAG_HELP)).on_hover_text("Tags are replaced when the graph runs. In a batch (Run on files, or Run several times) {index}, {seed} or {timestamp} keep each run's files apart.");
            true
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use egui_kittest::{kittest::Queryable, Harness};

    fn harness(kind: NodeKind) -> Harness<'static, NodeKind> {
        Harness::builder().with_size(egui::Vec2::new(500.0, 400.0)).with_step_dt(1.0 / 60.0).build_ui_state(|ui, kind: &mut NodeKind| {
            egui_extras::install_image_loaders(ui.ctx());
            props(ui, kind);
        }, kind)
    }

    /// Typing into a node's number field (double-click, type, Enter) and moving a slider with the keyboard change the node.
    #[test]
    fn remesh_fields_can_be_typed_into_and_the_slider_moved() {
        let mut h = harness(NodeKind::Remesh { triangles: 20_000, adaptivity: 1.0, mesher: Default::default(), auto_detail: false });
        h.run();
        h.get_by_label("triangles").click();
        h.run();
        h.get_by_label("triangles").type_text("5000");
        h.key_press(egui::Key::Enter);
        h.run();
        let NodeKind::Remesh { triangles, .. } = h.state() else { panic!() };
        assert_eq!(*triangles, 5000, "typing into the field sets it");
        // Pressing the slider's rail towards its left end lowers it.
        let r = h.get_by_role_and_label(egui::accesskit::Role::SpinButton, "adaptivity").rect();
        let at = egui::Pos2::new(r.left() - 90.0, r.center().y);
        h.hover_at(at);
        h.step();
        h.event(egui::Event::PointerButton { pos: at, button: egui::PointerButton::Primary, pressed: true, modifiers: Default::default() });
        h.step();
        h.event(egui::Event::PointerButton { pos: at, button: egui::PointerButton::Primary, pressed: false, modifiers: Default::default() });
        h.step();
        let NodeKind::Remesh { adaptivity, .. } = h.state() else { panic!() };
        assert!(*adaptivity < 0.9, "the slider moved: {adaptivity}");
    }
}
