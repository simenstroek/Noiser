//! The details-panel controls of the mesh graph nodes.

use crate::graph::NodeKind;
use eframe::egui;

thread_local! {
    /// The folder of the mesh graph being edited: relative paths of its nodes are relative to it.
    static BASE: std::cell::RefCell<std::path::PathBuf> = std::cell::RefCell::new(std::path::PathBuf::from("."));
}

/// Tells the node settings which folder relative paths belong to (the mesh graph's).
pub fn set_base(dir: std::path::PathBuf) {
    BASE.with(|b| *b.borrow_mut() = dir);
}

fn base() -> std::path::PathBuf {
    BASE.with(|b| b.borrow().clone())
}

/// A text field with a Browse button; `filter` is the file dialog's name and extensions, empty for a folder.
fn path_row(ui: &mut egui::Ui, id: &str, path: &mut String, hint: &str, filter: Option<(&str, &[&str])>) {
    ui.horizontal(|ui| {
        ui.add(egui::TextEdit::singleline(path).hint_text(hint).desired_width(150.0).id_salt(id));
        if ui.button("Browse…").clicked() {
            let start = std::path::Path::new(path.as_str()).parent().map(|p| p.to_path_buf());
            let mut dialog = rfd::FileDialog::new();
            if let Some(dir) = start.filter(|d| d.is_dir()) {
                dialog = dialog.set_directory(dir);
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
        NodeKind::LoadMesh(p) => {
            ui.weak("A mesh file as a field: its signed distance, negative inside, in the units of the file.");
            ui.weak("A path of {batch} reads the mesh file of a headless batch run (--mesh-graph with --batch-dir).");
            path_row(ui, "mesh path", &mut p.path, "mesh file or {batch}", Some(("Mesh", &["obj", "ply", "stl", "gltf", "glb"])));
            let d = crate::meshgraph::LoadMeshParams::default();
            crate::row(ui, &mut p.resolution, d.resolution, |ui, v| _ = ui.add(egui::DragValue::new(v).range(32..=512).suffix(" samples")).on_hover_text("Resolution of the distance grid the mesh is baked into."));
            crate::row(ui, &mut p.padding, d.padding, |ui, v| _ = ui.add(egui::DragValue::new(v).range(0.1..=4.0).speed(0.01).prefix("padding ")).on_hover_text("Room around the mesh in the grid, in units of the mesh's radius: displacement cannot reach past it."));
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
            crate::row(ui, &mut p.amplitude, d.amplitude, |ui, v| _ = ui.add(egui::DragValue::new(v).speed(0.001).prefix("amplitude × ")).on_hover_text("The graph's density output times this: its size in units of the mesh file."));
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
                    if ui.button("Refresh embedded copy").on_hover_text("Saving the mesh graph does this too.").clicked() {
                        if let Ok(text) = std::fs::read_to_string(&full) {
                            p.embedded = text;
                        }
                    }
                    ui.weak(if p.embedded.is_empty() { "No embedded copy yet." } else { "An embedded copy is stored in the mesh graph." });
                }
                Ok(_) => _ = ui.colored_label(crate::editor::ORANGE, "The file is missing: using the embedded copy."),
                Err(e) => _ = ui.colored_label(crate::editor::ORANGE, e),
            }
            true
        }
        NodeKind::Remesh { triangles, adaptivity } => {
            ui.weak("The surface of the field as an adaptive triangle mesh: small even triangles where it bends, large ones where it is flat.");
            crate::row(ui, triangles, 20_000, |ui, v| _ = ui.add(egui::DragValue::new(v).range(200..=2_000_000).speed(100.0).suffix(" triangles")));
            crate::row(ui, adaptivity, 1.0, |ui, v| _ = ui.add(egui::Slider::new(v, 0.0..=1.0).text("adaptivity")).on_hover_text("0 gives triangles of one size everywhere; 1 lets them be a quarter of that size on tight curves and five times larger on flat ground."));
            true
        }
        NodeKind::AutoUv(uv) => {
            let d = crate::uvgen::UvParams::default();
            ui.weak("An overlap-free atlas of islands in the 0 to 1 square.");
            crate::row(ui, &mut uv.resolution, d.resolution, |ui, v| _ = ui.add(egui::DragValue::new(v).range(256..=8192).speed(32.0).suffix(" px atlas")));
            crate::row(ui, &mut uv.padding, d.padding, |ui, v| _ = ui.add(egui::DragValue::new(v).range(0..=64).prefix("padding ").suffix(" px")));
            crate::row(ui, &mut uv.smoothing, d.smoothing, |ui, v| _ = ui.add(egui::DragValue::new(v).range(0..=60).prefix("chart smoothing ")).on_hover_text("Rounds of smoothing of the surface the islands are cut from; the UVs still map the true surface."));
            crate::row(ui, &mut uv.absorb_tiny, d.absorb_tiny, |ui, v| _ = ui.checkbox(v, "fold tiny islands into neighbours"));
            crate::row(ui, &mut uv.brute_force, d.brute_force, |ui, v| _ = ui.checkbox(v, "best packing (slower)"));
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
                    ui.selectable_value(&mut o.ply, false, "OBJ");
                    ui.selectable_value(&mut o.ply, true, "PLY (binary)");
                });
            }
            path_row(ui, "output folder", &mut o.folder, "folder (empty: next to the graph)", None);
            ui.horizontal(|ui| {
                ui.label("name");
                ui.add(egui::TextEdit::singleline(&mut o.name).desired_width(140.0));
            });
            ui.weak(if image { "A map's name is added to the file name: name_Normal.png. {name} in the name is the mesh file's name in a batch." } else { "Existing files are not overwritten: a number is added. {name} in the name is the mesh file's name in a batch." });
            true
        }
        _ => false,
    }
}
