//! The Scatter on Surface node (Mesh graph): plants copies of an SDF shape on the surface of any field (barnacles on a
//! rock), where a mask allows. Space is cut into cubic cells; each cell may hold one copy (a hash of the cell and the seed
//! decides), at a random point of the cell moved onto the surface. A copy is kept when that point stays in its own cell
//! (so none is counted twice) and the mask there, 0 to 1, lets it.
//!
//! Inputs: 0 the surface field, 1 an optional displacement that roughens only the copies, 2 an optional mask. The surface
//! and the mask are read at the copies' points, not at the sample, so a mask is any graph: the surface's Normal dotted
//! with a direction (tops only), another Load SDF (inside a shape), Position (a gradient), multiplied together.

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ScatterSurfaceParams {
    /// The SDF Composer file to plant.
    pub file: String,
    pub seed: u32,
    /// Cell size: about the distance between copies, in the field's units. Copies should not be much bigger.
    pub cell: f32,
    /// The chance a cell the surface passes through holds a copy, 0 to 1 (times the mask).
    pub density: f32,
    /// A copy's longest side is a random size in this range.
    pub size: [f32; 2],
    /// How far a copy's starting point strays from the middle of its cell, 0 to 1 of the cell.
    pub jitter: f32,
    /// Stand each copy along the surface's normal (barnacles); off, copies stay upright (+Z), like trees on a boulder.
    pub align: bool,
    /// Turn every copy a random angle about its up.
    pub spin: bool,
    /// Copies shrink with the mask, not only thin out.
    pub size_by_mask: bool,
    /// How far into the surface a copy is pushed, as a fraction of its size.
    pub sink: f32,
    /// Width of the smooth blend between the copies and the surface.
    pub blend: f32,
    /// Cut the copies out of the surface (pits, holes) instead of adding them.
    pub subtract: bool,
    /// How much of the displacement input roughens a copy (the input times this, in field units).
    pub roughen: f32,
}

impl Default for ScatterSurfaceParams {
    fn default() -> Self {
        ScatterSurfaceParams { file: String::new(), seed: 1, cell: 0.1, density: 0.6, size: [0.03, 0.06], jitter: 0.8, align: true, spin: true, size_by_mask: false, sink: 0.1, blend: 0.005, subtract: false, roughen: 1.0 }
    }
}

/// How many shader parameters the node uses (see `shader_params`).
pub const SLOTS: usize = 20;

impl ScatterSurfaceParams {
    pub fn shader_params(&self) -> [f32; SLOTS] {
        let (inv, centre, half) = crate::scatter::shape_fit(&self.file);
        [
            self.seed as f32,
            self.cell.max(1e-6),
            self.density,
            self.size[0],
            self.size[1].max(self.size[0]),
            self.jitter,
            if self.align { 1.0 } else { 0.0 },
            if self.spin { 1.0 } else { 0.0 },
            if self.size_by_mask { 1.0 } else { 0.0 },
            self.sink,
            self.blend,
            if self.subtract { 1.0 } else { 0.0 },
            self.roughen,
            inv,
            centre[0],
            centre[1],
            centre[2],
            half,
            // The 27 cells around a sample, read from here so the shader compiler cannot unroll the loop.
            27.0,
            // Four rounds of five surface readings (three steps onto the surface, then its value and normal there).
            20.0,
        ]
    }

    /// How far the copies reach out of the surface: the bounds of a mesh made from it grow by this much.
    pub fn reach(&self) -> f32 {
        if self.subtract { 0.0 } else { self.size[1].max(self.size[0]) * 1.9 + self.blend }
    }
}

/// The shader function `ssurf{id}(p, base, displace)` for a node whose parameters start at `slot`. `surf` and `mask`
/// name functions of a point: the surface field and the mask; `eps` is the step its gradient is taken over. The shape's interpreter
/// (`sdf{id}_at`) must be embedded with z up.
pub fn wgsl(id: u32, slot: usize, surf: &str, mask: &str, eps: f32) -> String {
    let p = |i: usize| format!("P[{}]", slot + i);
    let (seed, cell, density, size_lo, size_hi, jitter, align, spin, by_mask, sink, blend, subtract, roughen, inv) =
        (p(0), p(1), p(2), p(3), p(4), p(5), p(6), p(7), p(8), p(9), p(10), p(11), p(12), p(13));
    let (cx, cy, cz, half, cells, steps) = (p(14), p(15), p(16), p(17), p(18), p(19));
    format!(
        "
fn ssurf{id}_smin(a: f32, b: f32, k: f32) -> f32 {{
    if (k <= 0.0) {{ return min(a, b); }}
    let h = max(k - abs(a - b), 0.0) / k;
    return min(a, b) - h * h * k * 0.25;
}}
// The distance from `p` to the copy standing at surface point `q` with normal `n`.
fn ssurf{id}_one(p: vec3f, q: vec3f, n: vec3f, size: f32, spin: f32) -> f32 {{
    let up = select(vec3f(0.0, 0.0, 1.0), n, {align} > 0.5);
    let sc = size * {inv};
    let centre = q + up * ({half} * sc - {sink} * size);
    let rel = p - centre;
    let reach = size * 0.9;
    let dist = length(rel);
    if (dist > reach + {blend}) {{ return dist - reach; }}
    // A frame round `up`, turned by the copy's spin: the shape's up (+Y in the Composer, +Z here) goes along `up`.
    let helper = select(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 1.0, 0.0), abs(up.x) > 0.9);
    let t0 = normalize(cross(helper, up));
    let b0 = cross(up, t0);
    let a = spin * 6.2831853 * {spin};
    let t = cos(a) * t0 + sin(a) * b0;
    let b = cross(up, t);
    let local = vec3f(dot(rel, t), dot(rel, b), dot(rel, up));
    return sdf{id}_at(centre + local, centre, sc, vec3f({cx}, {cy}, {cz}));
}}
fn ssurf{id}(p: vec3f, base: f32, displace: f32) -> f32 {{
    // A copy is never further from the surface than this, so further out the surface alone decides (the result stays a
    // lower bound of the distance, which keeps ray marching safe).
    let extent = {size_hi} * 1.9 + {blend};
    if (abs(base) > extent + {cell}) {{
        if ({subtract} > 0.5 || base < 0.0) {{ return base; }}
        return base - extent;
    }}
    let seed = u32({seed});
    let cell = {cell};
    let c0 = vec3i(floor(p / cell));
    let cells = i32({cells});
    var near = 1.0e6;
    for (var k = 0; k < cells; k++) {{
        let id3 = c0 + vec3i(k % 3 - 1, (k / 3) % 3 - 1, k / 9 - 1);
        let r = hash01(id3, seed);
        if (r.x > {density}) {{ continue; }}
        let r2 = hash01(id3, seed + 1u);
        // Rounds of five readings (the value, then a tetrahedron round it for the gradient), each round a Newton step onto
        // the surface. The surface is read at this one place: the shader compiler copies a shape's interpreter into every
        // call, and with a call per step the compile took minutes.
        var q = (vec3f(id3) + vec3f(0.5) + (r2 - vec3f(0.5)) * {jitter}) * cell;
        var d = 0.0;
        var g = vec3f(0.0);
        var off = false;
        let steps = i32({steps});
        for (var j = 0; j < steps; j++) {{
            let s = j % 5;
            let o = tetra(u32(max(s - 1, 0)));
            let v = {surf}(select(q + o * {eps:e}, q, s == 0));
            if (s == 0) {{
                d = v;
                g = vec3f(0.0);
                // The start is in the cell, so further than its diagonal the surface does not pass through it.
                if (j == 0 && abs(v) > 1.74 * cell) {{ off = true; break; }}
            }} else {{
                g += o * v;
            }}
            if (s == 4 && j < steps - 1) {{ q -= d * safe_normalize(g); }}
        }}
        if (off || abs(d) > 0.02 * cell || any(vec3i(floor(q / cell)) != id3)) {{ continue; }}
        let m = clamp({mask}(q), 0.0, 1.0);
        let r3 = hash01(id3, seed + 2u);
        if (r3.x >= m) {{ continue; }}
        var size = mix({size_lo}, {size_hi}, r3.y);
        if ({by_mask} > 0.5) {{ size = size * m; }}
        near = min(near, ssurf{id}_one(p, q, safe_normalize(g), size, r3.z));
    }}
    let shapes = near + displace * {roughen};
    if ({subtract} > 0.5) {{ return -ssurf{id}_smin(-base, shapes, {blend}); }}
    return ssurf{id}_smin(base, shapes, {blend});
}}
"
    )
}

pub fn props(ui: &mut eframe::egui::Ui, p: &mut ScatterSurfaceParams) {
    use eframe::egui;
    let d = ScatterSurfaceParams::default();
    ui.weak("Plants copies of an SDF shape on the surface of any field: barnacles on a rock. Input 1 is the surface (a Load SDF, a mesh, a union of them); input 2 optionally roughens the copies; input 3 is a mask, 0 to 1, read where each copy would stand. Build masks from nodes: the surface's Normal, Math Dot with a direction and a Select for tops only; another Load SDF and a Select for inside a shape; Position for a gradient; Math Multiply to combine. Units are the field's.");
    ui.horizontal(|ui| {
        ui.add(egui::TextEdit::singleline(&mut p.file).hint_text("a .sdf file").desired_width(150.0).id_salt("scatter surface sdf path"));
        if ui.button("Browse…").clicked() {
            let mut dialog = crate::library::dialog(crate::library::Kind::Sdf, rfd::FileDialog::new().add_filter("SDF Modeler file", &["sdf"]));
            if let Some(dir) = std::path::Path::new(&p.file).parent().filter(|d| d.is_dir()) {
                dialog = dialog.set_directory(dir);
            }
            if let Some(picked) = dialog.pick_file() {
                p.file = picked.to_string_lossy().into_owned();
            }
        }
    });
    match crate::sdf_node::load(&p.file).as_ref() {
        Err(e) if !p.file.is_empty() => _ = ui.colored_label(crate::editor::ORANGE, e),
        Err(_) => _ = ui.colored_label(crate::editor::ORANGE, "No file chosen: the node passes its input through."),
        Ok(_) => {}
    }
    crate::row(ui, &mut p.seed, d.seed, |ui, v| _ = ui.add(egui::DragValue::new(v).prefix("seed ")).on_hover_text("Another seed plants the copies somewhere else."));
    let step = (p.cell * 0.01).max(1e-4);
    crate::row(ui, &mut p.cell, d.cell, |ui, v| {
        ui.add(egui::DragValue::new(v).range(1e-4..=1.0e7).speed(step).prefix("cell ")).on_hover_text("Space is cut into cubes this big, each holding at most one copy: about the distance between copies. Keep the largest copy under about 0.7 of a cell, or copies are cut off at the cell edges.");
        ui.weak(crate::meshgraph_ui::metres(*v));
    });
    crate::row(ui, &mut p.density, d.density, |ui, v| _ = ui.add(egui::Slider::new(v, 0.0..=1.0).text("density")).on_hover_text("The chance a cell the surface passes through holds a copy, times the mask."));
    crate::row(ui, &mut p.jitter, d.jitter, |ui, v| _ = ui.add(egui::Slider::new(v, 0.0..=1.0).text("jitter")).on_hover_text("How far a copy strays from the middle of its cell."));
    ui.horizontal(|ui| {
        ui.label("size");
        let step = (p.size[1] * 0.01).max(1e-4);
        ui.add(egui::DragValue::new(&mut p.size[0]).range(1e-4..=1.0e7).speed(step));
        ui.add(egui::DragValue::new(&mut p.size[1]).range(1e-4..=1.0e7).speed(step));
        ui.weak(format!("{} to {}", crate::meshgraph_ui::metres(p.size[0]).trim_start_matches("= "), crate::meshgraph_ui::metres(p.size[1]).trim_start_matches("= ")));
    })
    .response
    .on_hover_text("Each copy's longest side is a random size between these.");
    crate::row(ui, &mut p.align, d.align, |ui, v| _ = ui.checkbox(v, "stand along the surface").on_hover_text("On: each copy's up (+Y in the Composer) follows the surface's normal, like barnacles. Off: copies stay upright (+Z), like trees on a boulder."));
    crate::row(ui, &mut p.spin, d.spin, |ui, v| _ = ui.checkbox(v, "random turn").on_hover_text("Turn every copy a random angle about its up."));
    crate::row(ui, &mut p.size_by_mask, d.size_by_mask, |ui, v| _ = ui.checkbox(v, "mask shrinks copies").on_hover_text("Where the mask is between 0 and 1, copies are smaller as well as fewer."));
    crate::row(ui, &mut p.sink, d.sink, |ui, v| _ = ui.add(egui::DragValue::new(v).range(-0.5..=0.5).speed(0.005).prefix("sink ").suffix(" x size")));
    let step = (p.cell * 0.002).max(1e-5);
    crate::row(ui, &mut p.blend, d.blend, |ui, v| _ = ui.add(egui::DragValue::new(v).range(0.0..=1.0e6).speed(step).prefix("blend ")).on_hover_text("Width of the smooth join between the copies and the surface."));
    crate::row(ui, &mut p.subtract, d.subtract, |ui, v| _ = ui.checkbox(v, "cut out of the surface").on_hover_text("Subtract the copies (pits, holes) instead of adding them."));
    crate::row(ui, &mut p.roughen, d.roughen, |ui, v| _ = ui.add(egui::DragValue::new(v).range(0.0..=100.0).speed(0.01).prefix("roughen × ")).on_hover_text("How much of input 2 (the displacement) is added to the copies' field."));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{Graph, MathOp, NodeKind};

    /// Barnacles on the top of a ball: copies stand on the surface where the mask (made of nodes: the ball's Normal, its Z,
    /// greater than 0.5) lets them, and nowhere else.
    #[test]
    fn copies_stand_on_the_surface_only_where_the_mask_lets_them() {
        let Some(gpu) = crate::gpu_tests::gpu() else { return eprintln!("no GPU adapter; skipping") };
        let sdf = crate::sdf_node::tests::temp_sphere("noiser_scatter_surface.sdf");
        let mut g = Graph::default();
        let rock = g.add(NodeKind::LoadSdf(crate::sdf_node::LoadSdfParams { file: sdf.clone(), size: 1.0, offset: [0.0; 3], ..Default::default() }), [0.0; 2]);
        let normal = g.add(NodeKind::Normal { flip: false }, [0.0; 2]);
        let split = g.add(NodeKind::Separate, [0.0; 2]);
        let top = g.add(NodeKind::Math { op: MathOp::Greater, a: 0.0, b: 0.5, c: 0.0 }, [0.0; 2]);
        let params = ScatterSurfaceParams { file: sdf, cell: 0.15, density: 1.0, size: [0.08, 0.08], sink: 0.0, blend: 0.0, ..Default::default() };
        let scatter = g.add(NodeKind::ScatterSurface(params), [0.0; 2]);
        let out = g.add(NodeKind::OutDensity, [0.0; 2]);
        g.connect(rock, normal, 0);
        g.connect(normal, split, 0);
        g.connect_from(split, 2, top, 0);
        g.connect(rock, scatter, 0);
        g.connect(top, scatter, 2);
        g.connect(scatter, out, 0);
        // Points 0.04 above the ball (radius 0.5), the middle height of a copy 0.08 across standing on it.
        let n = 3000;
        let dirs: Vec<[f32; 3]> = (0..n).map(|i| {
            let z = 1.0 - 2.0 * (i as f32 + 0.5) / n as f32;
            let r = (1.0 - z * z).sqrt();
            let a = i as f32 * 2.399_963;
            [r * a.cos(), r * a.sin(), z]
        }).collect();
        let points: Vec<[f32; 3]> = dirs.iter().map(|d| d.map(|c| c * 0.54)).collect();
        let field = crate::gpu_tests::eval(&gpu, &g, &points);
        let inside = |keep: &dyn Fn(f32) -> bool| dirs.iter().zip(&field).filter(|(d, f)| keep(d[2]) && **f < 0.0).count();
        let on_top = inside(&|z| z > 0.7);
        let below = inside(&|z| z < 0.3);
        assert!(on_top > 30, "copies on the top: {on_top} points inside one");
        assert_eq!(below, 0, "the mask keeps copies off the sides and bottom");
        // Away from the copies the field is the ball's.
        let far = crate::gpu_tests::eval(&gpu, &g, &[[0.0, 0.0, -0.8], [0.0, 0.0, 0.0]]);
        assert!((far[0] - 0.3).abs() < 0.02 && (far[1] + 0.5).abs() < 0.02, "{far:?}");
    }
}
