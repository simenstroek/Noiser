//! The details-panel controls of the mesh graph nodes.

use crate::graph::NodeKind;
use eframe::egui;

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
            path_row(ui, "mesh path", &mut p.path, "mesh file", Some(("Mesh", &["obj", "ply", "stl", "gltf", "glb"])));
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
            match crate::meshgraph::noise_graph_source(p) {
                Ok(_) if !p.file.is_empty() && std::fs::metadata(&p.file).is_ok() => {
                    if ui.button("Refresh embedded copy").on_hover_text("Saving the mesh graph does this too.").clicked() {
                        if let Ok(text) = std::fs::read_to_string(&p.file) {
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
        NodeKind::Remesh { triangles } => {
            ui.weak("The surface of the field as an adaptive triangle mesh: small even triangles where it bends, large ones where it is flat.");
            crate::row(ui, triangles, 20_000, |ui, v| _ = ui.add(egui::DragValue::new(v).range(200..=2_000_000).speed(100.0).suffix(" triangles")));
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
            true
        }
        NodeKind::OutMesh(o) | NodeKind::OutImage(o) => {
            ui.weak(if image { "Writes the image as a 16-bit PNG." } else { "Writes the mesh as an OBJ (with UVs and normals when it has them)." });
            path_row(ui, "output folder", &mut o.folder, "folder (empty: next to the graph)", None);
            ui.horizontal(|ui| {
                ui.label("name");
                ui.add(egui::TextEdit::singleline(&mut o.name).desired_width(140.0));
            });
            ui.weak(if image { "Written as name.png; a node fed by a Bake gets the map's name added: name_Normal.png." } else { "Written as name.obj. Existing files are not overwritten: a number is added." });
            true
        }
        _ => false,
    }
}
