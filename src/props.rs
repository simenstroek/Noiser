//! The side-panel property editors: node parameters, preview settings, curve and pattern widgets.

use crate::graph::{self, Graph, MathOp, NodeKind, Tiling};
use crate::{editor, meshapp, meshgraph_ui, Lighting, Preview, PRIMS};
use eframe::egui;

/// Small icon button that puts `value` back to `default`; disabled while it already is.
pub(crate) fn reset<T: PartialEq>(ui: &mut egui::Ui, value: &mut T, default: T) {
    let icon = egui::Image::new(egui::include_image!("../assets/icons/restart-alt.svg"))
        .fit_to_exact_size(egui::vec2(14.0, 14.0))
        .tint(ui.visuals().text_color());
    let btn = ui.add_enabled(*value != default, egui::Button::image(icon).frame(false));
    if btn.on_hover_text("Reset to default").clicked() {
        *value = default;
    }
}

/// A row with a reset button in front of the widget.
pub(crate) fn row<T: PartialEq>(ui: &mut egui::Ui, value: &mut T, default: T, widget: impl FnOnce(&mut egui::Ui, &mut T)) {
    ui.horizontal(|ui| {
        reset(ui, value, default);
        widget(ui, value);
    });
}

/// The output multiplier of a Noise or Pattern node (saves a Multiply node, for example for a warp offset).
pub(crate) fn amount_row(ui: &mut egui::Ui, amount: &mut f32) {
    row(ui, amount, 1.0, |ui, v| {
        ui.add(egui::DragValue::new(v).range(-1000.0..=1000.0).speed(0.005).prefix("amplitude × "))
            .on_hover_text("The node's output is multiplied by this: 0.01 gives a small offset for domain warping without a Multiply node.");
    });
}

/// The tabs of the side panel.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub(crate) enum SideTab {
    #[default]
    View3d,
    Material,
    Lighting,
    Slice,
    Mesh,
}

impl SideTab {
    pub(crate) const ALL: [(SideTab, &'static str); 5] = [(SideTab::View3d, "3D view"), (SideTab::Material, "Material"), (SideTab::Lighting, "Lighting"), (SideTab::Slice, "Slice"), (SideTab::Mesh, "Mesh")];
}

/// The viewport settings of one tab (the Mesh tab is drawn by `App::mesh_props`).
pub(crate) fn preview_props(ui: &mut egui::Ui, pv: &mut Preview, mesh_loaded: bool, tab: SideTab) {
    let d = Preview::default();
    match tab {
        SideTab::View3d => {
    row(ui, &mut pv.cloud, d.cloud, |ui, v| {
        ui.selectable_value(v, false, "Solid");
        ui.selectable_value(v, true, "Cloud");
    });
    row(ui, &mut pv.prim, d.prim, |ui, v| {
        let shown = PRIMS.get(*v as usize).copied().unwrap_or("Mesh");
        egui::ComboBox::from_label("shape").selected_text(shown).show_ui(ui, |ui| {
            for (i, name) in PRIMS.iter().enumerate() {
                ui.selectable_value(v, i as u32, *name);
            }
            if mesh_loaded {
                ui.selectable_value(v, meshapp::MESH_PRIM, "Mesh");
            }
        });
    });
    row(ui, &mut pv.prim_size, d.prim_size, |ui, v| _ = ui.add(egui::Slider::new(v, 0.25..=2.0).text("shape size")));
    row(ui, &mut pv.amount, d.amount, |ui, v| _ = ui.add(egui::Slider::new(v, 0.0..=1.0).clamping(egui::SliderClamping::Never).text("displacement")));
    row(ui, &mut pv.tile_size, d.tile_size, |ui, v| _ = ui.add(egui::Slider::new(v, 0.25..=4.0).logarithmic(true).clamping(egui::SliderClamping::Never).text("tile size")));
    row(ui, &mut pv.seed, d.seed, |ui, v| _ = ui.add(egui::Slider::new(v, 0..=100).text("global seed")));
    if pv.cloud {
        row(ui, &mut pv.cloud_density, d.cloud_density, |ui, v| _ = ui.add(egui::Slider::new(v, 0.5..=64.0).logarithmic(true).text("cloud density")));
    } else {
        row(ui, &mut pv.step_scale, d.step_scale, |ui, v| _ = ui.add(egui::Slider::new(v, 0.1..=1.0).text("step safety")));
        row(ui, &mut pv.render_scale, d.render_scale, |ui, v| {
            ui.horizontal(|ui| {
                ui.label("render resolution");
                for (s, label) in [(1.0, "Full"), (0.5, "Half"), (0.25, "Quarter")] {
                    ui.selectable_value(v, s, label);
                }
            });
        });
        row(ui, &mut pv.max_steps, d.max_steps, |ui, v| _ = ui.add(egui::Slider::new(v, 32..=600).logarithmic(true).text("max march steps")));
        }
        }
        SideTab::Material => {
            ui.weak("Viewport only: never exported.");
        row(ui, &mut pv.color, d.color, |ui, v| _ = ui.color_edit_button_rgb(v));
        row(ui, &mut pv.specular, d.specular, |ui, v| _ = ui.add(egui::Slider::new(v, 0.0..=1.0).text("specular")));
        row(ui, &mut pv.roughness, d.roughness, |ui, v| _ = ui.add(egui::Slider::new(v, 0.05..=1.0).text("roughness")));
        row(ui, &mut pv.shadows, d.shadows, |ui, v| _ = ui.checkbox(v, "soft shadows"));
        }
        SideTab::Lighting => {
            ui.weak("Viewport only: never exported.");
        ui.horizontal(|ui| {
            for (name, preset) in Lighting::PRESETS {
                if ui.button(name).clicked() {
                    pv.lighting = preset();
                }
            }
        });
        let l = &mut pv.lighting;
        let ld = Lighting::default();
        row(ui, &mut l.azimuth, ld.azimuth, |ui, v| _ = ui.add(egui::Slider::new(v, 0.0..=360.0).suffix("°").text("sun direction")));
        row(ui, &mut l.elevation, ld.elevation, |ui, v| _ = ui.add(egui::Slider::new(v, -10.0..=90.0).suffix("°").text("sun height")));
        row(ui, &mut l.sun, ld.sun, |ui, v| _ = ui.add(egui::Slider::new(v, 0.0..=3.0).text("sun strength")));
        row(ui, &mut l.sun_color, ld.sun_color, |ui, v| {
            ui.color_edit_button_rgb(v);
            ui.label("sun colour");
        });
        row(ui, &mut l.ambient, ld.ambient, |ui, v| _ = ui.add(egui::Slider::new(v, 0.0..=3.0).text("sky light")));
        row(ui, &mut l.sky_top, ld.sky_top, |ui, v| {
            ui.color_edit_button_rgb(v);
            ui.label("sky top");
        });
        row(ui, &mut l.sky_bottom, ld.sky_bottom, |ui, v| {
            ui.color_edit_button_rgb(v);
            ui.label("sky bottom");
        });
        }
        SideTab::Slice => {
    row(ui, &mut pv.slice_axis, d.slice_axis, |ui, v| {
        ui.label("axis");
        for (i, name) in ["X", "Y", "Z"].iter().enumerate() {
            ui.selectable_value(v, i as u32, *name);
        }
    });
    row(ui, &mut pv.slice_pos, d.slice_pos, |ui, v| _ = ui.add(egui::Slider::new(v, 0.0..=1.0).text("position")));
    row(ui, &mut pv.heatmap, d.heatmap, |ui, v| {
        ui.selectable_value(v, false, "Grayscale");
        ui.selectable_value(v, true, "Heatmap");
    });
    row(ui, &mut pv.tiles_2x2, d.tiles_2x2, |ui, v| _ = ui.checkbox(v, "show 2×2 tiles"));
        }
        SideTab::Mesh => {}
    }
}

/// The per-node cache controls: preview-only, so they sit apart from the node's own parameters.
pub(crate) fn cache_props(ui: &mut egui::Ui, graph: &mut Graph, id: graph::NodeId, tiling: &std::collections::BTreeMap<graph::NodeId, Tiling>, total_bytes: u64) {
    ui.separator();
    let eligible = graph.cache_eligibility(id, tiling);
    let auto = {
        // Resolution "Auto" would pick, whatever is stored now.
        let mut probe = graph.clone();
        probe.nodes.get_mut(&id).unwrap().cache = Some(0);
        probe.cache_resolution(id)
    };
    let res = graph.cache_resolution(id);
    let node = graph.nodes.get_mut(&id).unwrap();
    let mut on = node.cache.is_some();
    ui.horizontal(|ui| {
        reset(ui, &mut node.cache, None);
        let r = ui.add_enabled(eligible.is_ok(), egui::Checkbox::new(&mut on, "Cache in preview"));
        let tip = match &eligible {
            Ok(()) => "Bakes everything upstream of this node into a 3D texture so the preview stays fast.\nExport always uses the full graph.".to_string(),
            Err(why) => format!("Can't cache: {why}"),
        };
        r.on_hover_text(tip);
    });
    if on != node.cache.is_some() {
        node.cache = on.then_some(0);
    }
    if let (Some(cur), Ok(())) = (node.cache, &eligible) {
        let label = |r: u32| if r == 0 { format!("Auto ({auto}³)") } else { format!("{r}³") };
        let mut sel = cur;
        ui.horizontal(|ui| {
            reset(ui, &mut sel, 0);
            egui::ComboBox::from_label("resolution").selected_text(label(sel)).show_ui(ui, |ui| {
                for r in [0, 32, 64, 128, 256, 512, 1024] {
                    ui.selectable_value(&mut sel, r, label(r));
                }
            });
        });
        node.cache = Some(sel);
        let mb = |b: u64| b as f64 / (1 << 20) as f64;
        ui.weak(format!("{:.0} MB for this cache · {:.0} MB for all", mb((res as u64).pow(3) * 4), mb(total_bytes)));
        if res >= 1024 {
            ui.colored_label(editor::ORANGE, "⚠ 1024³ needs 4 GB of GPU memory");
        }
    }
    if let Err(why) = &eligible {
        ui.weak(format!("Not cacheable: {why}"));
    }
}

/// Properties of one node.
pub(crate) fn node_props(ui: &mut egui::Ui, kind: &mut NodeKind, names: &[String]) {
    match kind {
        NodeKind::Float(_) | NodeKind::Vec3(_) => {
            let is_vec = matches!(kind, NodeKind::Vec3(_));
            ui.horizontal(|ui| {
                if ui.selectable_label(!is_vec, "Float").clicked() && is_vec
                    && let NodeKind::Vec3(v) = kind {
                        *kind = NodeKind::Float(v[0]);
                    }
                if ui.selectable_label(is_vec, "Vector").clicked() && !is_vec
                    && let NodeKind::Float(v) = kind {
                        *kind = NodeKind::Vec3([*v; 3]);
                    }
            });
            let drag = |ui: &mut egui::Ui, x: &mut f32| _ = ui.add(egui::DragValue::new(x).speed(0.01));
            match kind {
                NodeKind::Float(v) => row(ui, v, 1.0, drag),
                NodeKind::Vec3(v) => row(ui, v, [1.0; 3], |ui, v| v.iter_mut().for_each(|x| drag(ui, x))),
                _ => {}
            }
        }
        NodeKind::Math { op, .. } => {
            row(ui, op, MathOp::Add, |ui, op| {
                egui::ComboBox::from_label("operation").selected_text(op.label()).show_ui(ui, |ui| {
                    for o in MathOp::ALL {
                        ui.selectable_value(op, o, o.label());
                    }
                });
            });
            if *op == MathOp::Ridge {
                ui.weak("1 − |a|. Ridge of noise gives ridged noise; Absolute gives billow.");
            }
        }
        NodeKind::Separate => _ = ui.weak("Splits a vector into X, Y and Z floats: drag from the pin of the component you want. An unconnected vector is 0."),
        NodeKind::Combine { .. } => _ = ui.weak("A vector from three floats. Unconnected inputs use the values on the node."),
        NodeKind::Mix { clamp, .. } => {
            row(ui, clamp, true, |ui, c| _ = ui.checkbox(c, "limit the factor to 0..1"));
            ui.weak("A + (B - A) x Factor. The input values sit on the node.");
        }
        NodeKind::Remap { clamp, .. } => {
            row(ui, clamp, false, |ui, c| _ = ui.checkbox(c, "clamp to the output range"));
            ui.weak("The input values sit on the node.");
        }
        NodeKind::Curve { points, mode } => {
            ui.horizontal(|ui| {
                ui.label("Interpolation");
                for m in graph::CurveMode::ALL {
                    ui.selectable_value(mode, m, m.label());
                }
            });
            curve_plot(ui, points, *mode);
            let mut remove = None;
            for (i, q) in points.iter_mut().enumerate() {
                ui.horizontal(|ui| {
                    ui.add(egui::DragValue::new(&mut q[0]).speed(0.01).prefix("x "));
                    ui.add(egui::DragValue::new(&mut q[1]).speed(0.01).prefix("y "));
                    if ui.small_button("x").on_hover_text("Remove point").clicked() {
                        remove = Some(i);
                    }
                });
            }
            if let Some(i) = remove.filter(|_| points.len() > 1) {
                points.remove(i);
            }
            if points.len() < graph::MAX_CURVE_POINTS && ui.button("Add point").clicked() {
                let x = points.iter().map(|q| q[0]).fold(f32::MIN, f32::max) + 0.25;
                let y = points.last().map_or(0.0, |q| q[1]);
                points.push([x, y]);
            }
            ui.weak("Drag the points in the plot, double-click to add one, right-click a point to remove it. Before the first point and after the last, the output holds the end value.");
        }
        NodeKind::Pattern(p) => {
            pattern_props(ui, p);
            amount_row(ui, &mut p.amount);
        }
        NodeKind::Grid { counts, corners, cubic, amount } => {
            row(ui, counts, [8.0; 3], |ui, v| {
                for (c, label) in v.iter_mut().zip(["X ", "Y ", "Z "]) {
                    ui.add(egui::DragValue::new(c).range(1.0..=64.0).speed(0.1).prefix(label));
                }
            });
            row(ui, corners, false, |ui, v| {
                ui.selectable_value(v, false, "Cell centres");
                ui.selectable_value(v, true, "Cell corners");
            });
            row(ui, cubic, false, |ui, v| {
                ui.selectable_value(v, false, "Round");
                ui.selectable_value(v, true, "Cubic").on_hover_text("Sharp-cornered boxes: the largest offset along any axis, abs(0.5 - fract(position * cells))");
            });
            ui.weak("Cells per tile on each axis; whole numbers tile. Outputs the distance to the nearest cell centre (or corner): -1 there, +1 as far from it as the grid gets. Cubic measures it as the largest offset along any axis, so the cells are boxes rather than spheres.");
            amount_row(ui, amount);
        }
        NodeKind::Normal { flip } => row(ui, flip, false, |ui, f| _ = ui.checkbox(f, "flip")),
        NodeKind::Noise(n) => {
            let d = graph::NoiseParams::default();
            // Changing the type keeps the other settings; it has no reset since it defines the node.
            egui::ComboBox::from_label("type").selected_text(n.kind.name()).show_ui(ui, |ui| {
                for k in graph::NoiseKind::ALL {
                    ui.selectable_value(&mut n.kind, k, k.name());
                }
            });
            if n.kind == graph::NoiseKind::Worley {
                row(ui, &mut n.feature, d.feature, |ui, v| {
                    use graph::WorleyFeature::*;
                    for (f, name) in [(F1, "F1"), (F2, "F2"), (F2MinusF1, "F2 − F1"), (CellValue, "Cell value"), (Edges, "Cell edges")] {
                        ui.selectable_value(v, f, name);
                    }
                });
                row(ui, &mut n.jitter, d.jitter, |ui, v| _ = ui.add(egui::Slider::new(v, 0.0..=1.0).text("jitter")).on_hover_text("0 puts every point in the middle of its cell (a perfect lattice), 1 anywhere in it"));
                if n.feature != graph::WorleyFeature::Edges {
                    row(ui, &mut n.metric, d.metric, |ui, v| {
                        use graph::WorleyMetric::*;
                        ui.selectable_value(v, Euclidean, "Euclidean");
                        ui.selectable_value(v, Manhattan, "Manhattan").on_hover_text("Diamond-shaped cells");
                        ui.selectable_value(v, Chebyshev, "Chebyshev").on_hover_text("Square cells");
                    });
                }
                row(ui, &mut n.fractal, d.fractal, |ui, v| _ = ui.checkbox(v, "fractal (octaves)"));
            }
            row(ui, &mut n.scale, d.scale, |ui, v| _ = ui.add(egui::Slider::new(v, 0.5..=32.0).text("frequency (cells/tile)")));
            row(ui, &mut n.normalize, d.normalize, |ui, v| _ = ui.checkbox(v, "normalize 0-1"));
            if n.uses_octaves() {
                row(ui, &mut n.octaves, d.octaves, |ui, v| _ = ui.add(egui::Slider::new(v, 1..=8).text("octaves")));
                row(ui, &mut n.lacunarity, d.lacunarity, |ui, v| _ = ui.add(egui::Slider::new(v, 1.0..=4.0).text("lacunarity")));
                row(ui, &mut n.roughness, d.roughness, |ui, v| _ = ui.add(egui::Slider::new(v, 0.0..=1.0).text("roughness")));
            }
            row(ui, &mut n.distortion, d.distortion, |ui, v| _ = ui.add(egui::Slider::new(v, 0.0..=2.0).text("distortion")));
            row(ui, &mut n.seed, d.seed, |ui, seed| {
                let mut locked = seed.is_some();
                ui.checkbox(&mut locked, "seed override");
                if locked {
                    ui.add(egui::DragValue::new(seed.get_or_insert(0)));
                } else {
                    *seed = None;
                }
            });
            amount_row(ui, &mut n.amount);
        }
        NodeKind::Position => _ = ui.label("World position. Feeds noise inputs by default."),
        NodeKind::Radial(r) => {
            let d = graph::RadialParams::default();
            row(ui, &mut r.mode, d.mode, |ui, v| {
                ui.selectable_value(v, graph::RadialMode::Cylindrical, "Cylindrical").on_hover_text("(distance from the axis, angle around it, height along it)");
                ui.selectable_value(v, graph::RadialMode::Spherical, "Spherical").on_hover_text("(distance from the centre, angle around the axis, angle from the axis)");
            });
            row(ui, &mut r.center, d.center, |ui, v| {
                for (c, label) in v.iter_mut().zip(["Centre X ", "Y ", "Z "]) {
                    ui.add(egui::DragValue::new(c).speed(0.01).prefix(label));
                }
            });
            row(ui, &mut r.axis, d.axis, |ui, v| {
                for (c, label) in v.iter_mut().zip(["Axis X ", "Y ", "Z "]) {
                    ui.add(egui::DragValue::new(c).speed(0.01).prefix(label));
                }
            });
            ui.weak("The centre line runs through the centre along the axis. Angles are in radians; the angle around the axis jumps from 3.14 to -3.14 on one half plane. Breaks tiling.");
        }
        NodeKind::Reroute => _ = ui.weak("A waypoint on a wire: it passes its input through unchanged."),
        NodeKind::NamedReroute { name } => {
            ui.add(egui::TextEdit::singleline(name).hint_text("name"));
            ui.weak("Passes its input on and gives it a name: add a Use named reroute node anywhere and pick this name, no wire needed.");
        }
        NodeKind::NamedRerouteUse { name } => {
            ui.weak("Reads the named reroute with this name, without a wire.");
            egui::ComboBox::from_label("name").selected_text(name.clone()).show_ui(ui, |ui| {
                for n in names {
                    ui.selectable_value(name, n.clone(), n);
                }
            });
            if names.is_empty() {
                ui.colored_label(editor::ORANGE, "No named reroute exists yet: add one and name it.");
            }
        }
        NodeKind::OutDensity | NodeKind::OutNormal => {}
        NodeKind::LoadMesh(_) | NodeKind::NoiseGraph(_) | NodeKind::Remesh { .. } | NodeKind::AutoUv(_) | NodeKind::Bake(_) | NodeKind::OutMesh(_) | NodeKind::OutImage(_) | NodeKind::ImageSplit | NodeKind::ImageCombine | NodeKind::ImageLevels { .. } | NodeKind::TransferUv => _ = meshgraph_ui::props(ui, kind),
    }
}

/// Draws a curve with draggable points: drag a point to move it, double-click empty space to add one,
/// right-click a point to remove it.
pub(crate) fn curve_plot(ui: &mut egui::Ui, points: &mut Vec<[f32; 2]>, mode: graph::CurveMode) {
    let size = egui::Vec2::new(ui.available_width().min(300.0), 150.0);
    let (rect, resp) = ui.allocate_exact_size(size, egui::Sense::click_and_drag());
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 4.0, egui::Color32::from_gray(28));
    let sorted = graph::curve_points(points);
    // View: the points extent plus a margin, always including 0..1 so the usual range is visible.
    let (mut x0, mut x1, mut y0, mut y1) = (0.0f32, 1.0f32, 0.0f32, 1.0f32);
    for p in &sorted {
        (x0, x1, y0, y1) = (x0.min(p[0]), x1.max(p[0]), y0.min(p[1]), y1.max(p[1]));
    }
    let (mx, my) = ((x1 - x0) * 0.08, (y1 - y0) * 0.08);
    (x0, x1, y0, y1) = (x0 - mx, x1 + mx, y0 - my, y1 + my);
    let to_screen = |p: [f32; 2]| egui::Pos2::new(rect.left() + (p[0] - x0) / (x1 - x0) * rect.width(), rect.bottom() - (p[1] - y0) / (y1 - y0) * rect.height());
    let to_curve = |s: egui::Pos2| [x0 + (s.x - rect.left()) / rect.width() * (x1 - x0), y0 + (rect.bottom() - s.y) / rect.height() * (y1 - y0)];
    let grid = egui::Stroke::new(1.0, egui::Color32::from_gray(50));
    for v in [0.0f32, 1.0] {
        painter.line_segment([to_screen([x0, v]), to_screen([x1, v])], grid);
        painter.line_segment([to_screen([v, y0]), to_screen([v, y1])], grid);
    }
    let line: Vec<egui::Pos2> = (0..=120)
        .map(|i| {
            let x = x0 + (x1 - x0) * i as f32 / 120.0;
            to_screen([x, graph::curve_value(&sorted, mode, x)])
        })
        .collect();
    painter.add(egui::Shape::line(line, egui::Stroke::new(1.5, egui::Color32::from_rgb(120, 190, 255))));
    let hovered = resp.hover_pos().and_then(|m| points.iter().position(|p| to_screen(*p).distance(m) < 8.0));
    for (i, p) in points.iter().enumerate() {
        let radius = if hovered == Some(i) { 6.0 } else { 4.5 };
        painter.circle(to_screen(*p), radius, egui::Color32::from_rgb(255, 210, 90), egui::Stroke::new(1.0, egui::Color32::BLACK));
    }
    // Which point a drag moves is decided when it starts.
    let id = resp.id.with("drag point");
    if resp.drag_started() {
        let at = resp.interact_pointer_pos().and_then(|m| points.iter().position(|p| to_screen(*p).distance(m) < 10.0));
        ui.memory_mut(|mem| mem.data.insert_temp(id, at));
    }
    if resp.dragged()
        && let (Some(Some(i)), Some(m)) = (ui.memory(|mem| mem.data.get_temp::<Option<usize>>(id)), resp.interact_pointer_pos())
            && let Some(p) = points.get_mut(i) {
                *p = to_curve(m);
            }
    if resp.double_clicked() && points.len() < graph::MAX_CURVE_POINTS
        && let Some(m) = resp.interact_pointer_pos() {
            points.push(to_curve(m));
        }
    if resp.secondary_clicked() && points.len() > 1
        && let Some(i) = resp.interact_pointer_pos().and_then(|m| points.iter().position(|p| to_screen(*p).distance(m) < 10.0)) {
            points.remove(i);
        }
}

/// A row of cell counts for the axes a pattern uses, labelled by what they count.
pub(crate) fn pattern_counts(ui: &mut egui::Ui, p: &mut graph::PatternParams, labels: [&str; 3]) {
    let d = graph::PatternParams::for_kind(p.kind);
    let used = p.used_axes();
    let names = ["X", "Y", "Z"];
    row(ui, &mut p.counts, d.counts, |ui, c| {
        for a in used {
            ui.add(egui::DragValue::new(&mut c[a]).range(1.0..=64.0).speed(0.1).prefix(format!("{} ({}) ", labels[a], names[a])));
        }
    });
}

pub(crate) fn pattern_2d(ui: &mut egui::Ui, p: &mut graph::PatternParams) {
    let d = graph::PatternParams::for_kind(p.kind);
    row(ui, &mut p.mode2d, d.mode2d, |ui, v| _ = ui.checkbox(v, "2D (extruded)"));
    if p.mode2d {
        row(ui, &mut p.axis, d.axis, |ui, v| {
            ui.label("extrude along");
            for (a, name) in [(0, "X"), (1, "Y"), (2, "Z")] {
                ui.selectable_value(v, a, name);
            }
        });
    }
}

/// The controls of a Pattern node: the kind, then the settings that kind has.
pub(crate) fn pattern_props(ui: &mut egui::Ui, p: &mut graph::PatternParams) {
    use graph::{PatternKind, PatternOutput::*};
    let before = p.kind;
    egui::ComboBox::from_label("pattern").selected_text(p.kind.name()).show_ui(ui, |ui| {
        for k in PatternKind::ALL {
            ui.selectable_value(&mut p.kind, k, k.name());
        }
    });
    if p.kind != before {
        // A new kind starts from its own defaults, keeping only the seed override.
        let (seed, amount) = (p.seed, p.amount);
        *p = graph::PatternParams::for_kind(p.kind);
        p.seed = seed;
        p.amount = amount;
    }
    let d = graph::PatternParams::for_kind(p.kind);
    match p.kind {
        PatternKind::Bricks => {
            pattern_2d(ui, p);
            let (row_axis, stack_axis, layer_axis) = p.axes();
            let names = ["X", "Y", "Z"];
            row(ui, &mut p.counts, d.counts, |ui, c| {
                let mut count = |ui: &mut egui::Ui, axis: usize, what: &str| {
                    ui.add(egui::DragValue::new(&mut c[axis]).range(1.0..=64.0).speed(0.1).prefix(format!("{what} ({}) ", names[axis])));
                };
                count(ui, row_axis, "per row");
                count(ui, stack_axis, "rows");
                if let Some(l) = layer_axis {
                    count(ui, l, "layers");
                }
            });
            row(ui, &mut p.steps, d.steps, |ui, v| {
                ui.add(egui::Slider::new(v, 1..=4).text("row offset steps"))
                    .on_hover_text("1 stacks the bricks in a grid, 2 is a running bond (each row shifted half a brick), 3 shifts by thirds.\nThe number of rows must be a multiple of this to tile.");
            });
            row(ui, &mut p.output, d.output, |ui, v| {
                ui.selectable_value(v, Distance, "Distance");
                ui.selectable_value(v, Shape, "Shape");
                ui.selectable_value(v, CellValue, "Cell value");
            });
            match p.output {
                Distance => _ = ui.weak("Distance to the centre of the brick: -1 at the centre, +1 at the corners."),
                Shape => {
                    row(ui, &mut p.gap, d.gap, |ui, v| _ = ui.add(egui::Slider::new(v, 0.0..=0.5).text("mortar gap")));
                    row(ui, &mut p.bevel, d.bevel, |ui, v| _ = ui.add(egui::Slider::new(v, 0.0..=0.5).text("edge softness")));
                    ui.weak("-1 on the brick face, +1 in the mortar.");
                }
                CellValue => {
                    pattern_seed(ui, p, &d);
                    ui.weak("A flat random value per brick.");
                }
            }
        }
        PatternKind::Dimples => {
            pattern_counts(ui, p, ["cells", "cells", "cells"]);
            row(ui, &mut p.variant, d.variant, |ui, v| {
                ui.selectable_value(v, 0, "Cubic");
                ui.selectable_value(v, 1, "Body-centred");
                ui.selectable_value(v, 2, "Face-centred");
            });
            row(ui, &mut p.output, d.output, |ui, v| {
                ui.selectable_value(v, Distance, "Distance");
                ui.selectable_value(v, Shape, "Dimples");
            });
            if p.output == Shape {
                row(ui, &mut p.radius, d.radius, |ui, v| _ = ui.add(egui::Slider::new(v, 0.1..=1.2).text("radius")).on_hover_text("1 makes neighbouring dimples touch"));
                row(ui, &mut p.profile, d.profile, |ui, v| {
                    ui.selectable_value(v, 0, "Sphere cap");
                    ui.selectable_value(v, 1, "Cone");
                    ui.selectable_value(v, 2, "Smooth bowl");
                    ui.selectable_value(v, 3, "Flat disc");
                });
                ui.weak("-1 at the centre of a dimple, +1 on the flat between them. A body-centred or face-centred lattice with a radius near 0.9 is a golf ball.");
            } else {
                ui.weak("Distance to the nearest lattice point: -1 at a point, +1 as far from any as the lattice gets.");
            }
        }
        PatternKind::Checker => {
            pattern_2d(ui, p);
            pattern_counts(ui, p, ["cells", "cells", "cells"]);
            row(ui, &mut p.softness, d.softness, |ui, v| _ = ui.add(egui::Slider::new(v, 0.0..=1.0).text("edge softness")));
            ui.weak("+1 and -1 cells. Every used count must be even to tile.");
        }
        PatternKind::Stripes => {
            row(ui, &mut p.counts[0], d.counts[0], |ui, v| _ = ui.add(egui::DragValue::new(v).range(1.0..=64.0).speed(0.1).prefix("stripes per tile ")));
            row(ui, &mut p.dir, d.dir, |ui, v| {
                for (c, name) in v.iter_mut().zip(["X ", "Y ", "Z "]) {
                    ui.add(egui::DragValue::new(c).range(-8.0..=8.0).speed(0.1).prefix(name));
                }
            });
            ui.weak("Direction the stripes run along; whole numbers tile, (1, 1, 0) is a diagonal.");
            row(ui, &mut p.variant, d.variant, |ui, v| {
                ui.selectable_value(v, 0, "Sine");
                ui.selectable_value(v, 1, "Triangle");
                ui.selectable_value(v, 2, "Sawtooth");
                ui.selectable_value(v, 3, "Square");
            });
            if p.variant == 3 {
                row(ui, &mut p.width, d.width, |ui, v| _ = ui.add(egui::Slider::new(v, 0.0..=1.0).text("width")));
                row(ui, &mut p.softness, d.softness, |ui, v| _ = ui.add(egui::Slider::new(v, 0.0..=1.0).text("edge softness")));
            }
        }
        PatternKind::Fractal => {
            pattern_counts(ui, p, ["cells", "cells", "cells"]);
            row(ui, &mut p.variant, d.variant, |ui, v| {
                ui.selectable_value(v, 0, "Menger sponge");
                ui.selectable_value(v, 1, "Sierpinski tetrahedron");
            });
            row(ui, &mut p.iterations, d.iterations, |ui, v| _ = ui.add(egui::Slider::new(v, 1..=6).text("iterations")));
            ui.weak("-1 in the solid, +1 in the empty space, from a distance estimate. Changing the iterations does not recompile the shader.");
        }
        PatternKind::Truchet => {
            row(ui, &mut p.variant, d.variant, |ui, v| {
                ui.selectable_value(v, 0, "Arcs");
                ui.selectable_value(v, 1, "Diagonals");
                ui.selectable_value(v, 2, "Pipes (3D)");
            });
            if p.variant < 2 {
                pattern_2d(ui, p);
            }
            pattern_counts(ui, p, ["cells", "cells", "cells"]);
            row(ui, &mut p.width, d.width, |ui, v| _ = ui.add(egui::Slider::new(v, 0.0..=1.0).text("line width")));
            row(ui, &mut p.softness, d.softness, |ui, v| _ = ui.add(egui::Slider::new(v, 0.0..=1.0).text("edge softness")));
            pattern_seed(ui, p, &d);
            ui.weak("-1 on the line, +1 away from it. Each cell is flipped at random; the lines join across cells.");
        }
        PatternKind::Honeycomb => {
            let before = p.variant;
            row(ui, &mut p.variant, d.variant, |ui, v| {
                ui.selectable_value(v, 0, "Hexagons (2D)");
                ui.selectable_value(v, 1, "Body-centred cells");
                ui.selectable_value(v, 2, "Face-centred cells");
                ui.selectable_value(v, 3, "Triangles (2D)");
            });
            if p.variant != before {
                p.mode2d = p.variant == 0 || p.variant == 3; // hexagons and triangles are flat; the lattice cells are 3D
            }
            if p.variant == 0 || p.variant == 3 {
                row(ui, &mut p.axis, d.axis, |ui, v| {
                    ui.label("extrude along");
                    for (a, name) in [(0, "X"), (1, "Y"), (2, "Z")] {
                        ui.selectable_value(v, a, name);
                    }
                });
            }
            pattern_counts(ui, p, ["cells", "cells", "cells"]);
            if p.variant == 0 {
                ui.weak("Hexagons are regular when the cell counts are in the ratio 1 : 0.577: 7 × 4, 12 × 7, 26 × 15.");
            }
            if p.variant == 3 {
                ui.weak("Triangles are equilateral when the second count is 1.155 × the first and even: 6 × 7 is close, 12 × 14, 26 × 30.");
            }
            row(ui, &mut p.output, d.output, |ui, v| {
                ui.selectable_value(v, Distance, "Distance");
                ui.selectable_value(v, Shape, "Walls");
                ui.selectable_value(v, CellValue, "Cell value");
            });
            match p.output {
                Distance => _ = ui.weak("-1 at a wall, +1 at the centre of a cell."),
                Shape => {
                    row(ui, &mut p.gap, d.gap, |ui, v| _ = ui.add(egui::Slider::new(v, 0.0..=1.0).text("wall width")));
                    row(ui, &mut p.bevel, d.bevel, |ui, v| _ = ui.add(egui::Slider::new(v, 0.0..=1.0).text("edge softness")));
                    ui.weak("+1 on the walls, -1 inside the cells.");
                }
                CellValue => {
                    pattern_seed(ui, p, &d);
                    ui.weak("A flat random value per cell.");
                }
            }
        }
        PatternKind::Gyroid => {
            pattern_counts(ui, p, ["periods", "periods", "periods"]);
            row(ui, &mut p.variant, d.variant, |ui, v| {
                ui.selectable_value(v, 0, "Gyroid");
                ui.selectable_value(v, 1, "Schwarz P");
                ui.selectable_value(v, 2, "Diamond");
            });
            row(ui, &mut p.output, d.output, |ui, v| {
                ui.selectable_value(v, Distance, "Field");
                ui.selectable_value(v, Shape, "Shell");
            });
            if p.output == Shape {
                row(ui, &mut p.width, d.width, |ui, v| _ = ui.add(egui::Slider::new(v, 0.0..=0.95).text("thickness")));
                ui.weak("-1 in a thin sheet around the surface, +1 away from it.");
            } else {
                ui.weak("The field itself, scaled to -1..1; its zero level is the surface.");
            }
        }
    }
}

pub(crate) fn pattern_seed(ui: &mut egui::Ui, p: &mut graph::PatternParams, d: &graph::PatternParams) {
    row(ui, &mut p.seed, d.seed, |ui, seed| {
        let mut locked = seed.is_some();
        ui.checkbox(&mut locked, "seed override");
        if locked {
            ui.add(egui::DragValue::new(seed.get_or_insert(0)));
        } else {
            *seed = None;
        }
    });
}
