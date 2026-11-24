//! The Scatter SDF node (Mesh graph): plants copies of an SDF shape across a terrain by a rule, instead of one node per
//! copy. The field is `union(base, shapes)` (or `base` minus the shapes), where the shapes are found without a list: the
//! terrain is cut into square cells, each cell may hold one copy (a hash of the cell and the seed decides whether, where
//! in the cell, how big and which way it faces), and a sample looks at its own cell and the eight around it.
//!
//! A copy stands on the ground under it (the terrain is the Load Mesh with a heightmap), and is dropped where the ground is
//! too steep or too high or low for it. Inputs: 0 the field to scatter onto (usually the terrain), 1 an optional
//! displacement field (a Noise Graph node) that roughens only the copies.

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ScatterParams {
    /// The SDF Composer file to plant.
    pub file: String,
    pub seed: u32,
    /// Cell size: about the distance between copies, in the field's units. Copies may be about this big.
    pub cell: f32,
    /// The chance a cell holds a copy, 0 to 1.
    pub density: f32,
    /// A copy's longest side is a random size in this range.
    pub size: [f32; 2],
    /// How far a copy may stray from the middle of its cell, 0 to 1 of the cell.
    pub jitter: f32,
    /// Turn every copy a random angle about the vertical.
    pub yaw: bool,
    /// Turn every copy so the shape's back (its +Z in the Composer) faces uphill: a cave's chamber goes into the hillside and
    /// its mouth opens downhill. Replaces the random turn where the ground has a slope.
    pub face_uphill: bool,
    /// Where the ground may be: slope as rise over run, and height as a fraction of the terrain's height range.
    pub slope: [f32; 2],
    pub elevation: [f32; 2],
    /// How far into the ground a copy is pushed, as a fraction of its size.
    pub sink: f32,
    /// Width of the smooth blend between the copies and the base.
    pub blend: f32,
    /// Cut the copies out of the base (caves, pits) instead of adding them.
    pub subtract: bool,
    /// How much of the displacement input roughens a copy (the input times this, in field units).
    pub roughen: f32,
}

impl Default for ScatterParams {
    fn default() -> Self {
        ScatterParams { file: String::new(), seed: 1, cell: 4000.0, density: 0.5, size: [1500.0, 2500.0], jitter: 0.8, yaw: true, face_uphill: false, slope: [0.0, 0.6], elevation: [0.0, 1.0], sink: 0.05, blend: 100.0, subtract: false, roughen: 1.0 }
    }
}

/// How many shader parameters the node uses (see `shader_params`).
pub const SLOTS: usize = 22;

impl ScatterParams {
    /// The shader's numbers: seed, cell, density, size low and high, jitter, slope low and high, elevation low and high,
    /// sink, blend, subtract, yaw, roughen, then of the shape (in its file's own units) 1 / its longest side, its centre,
    /// and how far its centre is above its lowest point.
    pub fn shader_params(&self) -> [f32; SLOTS] {
        let (inv, centre, half) = match crate::sdf_node::load(&self.file).as_ref() {
            Ok(data) => match data.scene.bounds {
                Some((lo, hi)) => {
                    let longest = (0..3).map(|a| hi[a] - lo[a]).fold(0.0, f64::max).max(1e-9);
                    let c = [0, 1, 2].map(|a| ((lo[a] + hi[a]) * 0.5) as f32);
                    ((1.0 / longest) as f32, c, c[1] - lo[1] as f32)
                }
                None => (1.0, [0.0; 3], 0.0),
            },
            Err(_) => (1.0, [0.0; 3], 0.0),
        };
        [
            self.seed as f32,
            self.cell.max(1e-3),
            self.density,
            self.size[0],
            self.size[1].max(self.size[0]),
            self.jitter,
            self.slope[0],
            self.slope[1],
            self.elevation[0],
            self.elevation[1],
            self.sink,
            self.blend,
            if self.subtract { 1.0 } else { 0.0 },
            if self.yaw { 1.0 } else { 0.0 },
            self.roughen,
            inv,
            centre[0],
            centre[1],
            centre[2],
            half,
            // The nine cells around a sample. The loop's length is read from here, not written as 9, so the shader compiler
            // cannot unroll it (nine copies of the shape interpreter per node take minutes to compile).
            9.0,
            if self.face_uphill { 1.0 } else { 0.0 },
        ]
    }

    /// How far above or below the ground the copies reach: the surface can leave the terrain's range by this much.
    pub fn reach(&self) -> f32 {
        self.size[1].max(self.size[0]) + self.blend
    }
}

/// The shader function `scatter{id}(p, base, displace)` for a node whose parameters start at `slot`, scattering onto the
/// terrain in mesh slot `land`. The shape's interpreter (`sdf{id}_at`) must be embedded as for a Load SDF node.
pub fn wgsl(id: u32, slot: usize, land: usize) -> String {
    let p = |i: usize| format!("P[{}]", slot + i);
    format!(
        "
fn scatter{id}_smin(a: f32, b: f32, k: f32) -> f32 {{
    if (k <= 0.0) {{ return min(a, b); }}
    let h = max(k - abs(a - b), 0.0) / k;
    return min(a, b) - h * h * k * 0.25;
}}
fn scatter{id}(p: vec3f, base: f32, displace: f32) -> f32 {{
    let seed = u32({seed});
    let cell = {cell};
    let c0 = vec2i(floor(p.xy / cell));
    let zr = land_range_{land}();
    var near = 1.0e6;
    let cells = i32({cells});
    for (var k = 0; k < cells; k++) {{
        {{
            let id2 = c0 + vec2i(k % 3 - 1, k / 3 - 1);
            let r = hash01(vec3i(id2, 0), seed);
            if (r.x > {density}) {{ continue; }}
            let r2 = hash01(vec3i(id2, 1), seed);
            let ctr = (vec2f(id2) + vec2f(0.5) + (r.yz - vec2f(0.5)) * {jitter}) * cell;
            let size = mix({size_lo}, {size_hi}, r2.x);
            let g = land_ground_{land}(ctr);
            let slope = length(g.yz);
            let e = (g.x - zr.x) / max(zr.y - zr.x, 1.0e-6);
            if (slope < {slope_lo} || slope > {slope_hi} || e < {elev_lo} || e > {elev_hi}) {{ continue; }}
            let rad = 0.4 * size;
            var low = g.x;
            low = min(low, land_ground_{land}(ctr + vec2f(rad, 0.0)).x);
            low = min(low, land_ground_{land}(ctr - vec2f(rad, 0.0)).x);
            low = min(low, land_ground_{land}(ctr + vec2f(0.0, rad)).x);
            low = min(low, land_ground_{land}(ctr - vec2f(0.0, rad)).x);
            let sc = size * {inv};
            // Standing shapes rest on the lowest ground under them; cut-outs (caves) start at the ground in the middle, so the mouth
            // opens where the slope falls away in front of them.
            let floor = select(low, g.x, {subtract} > 0.5);
            let centre = vec3f(ctr, floor + {half} * sc - {sink} * size);
            let rel = p - centre;
            let reach = size * 0.9;
            let dist = length(rel);
            if (dist > reach + {blend}) {{ near = min(near, dist - reach); continue; }}
            let uphill = normalize(g.yz + vec2f(1.0e-9));
            let yaw = select(r2.y * 6.2831853 * {yaw}, atan2(uphill.x, -uphill.y), {face} > 0.5 && slope > 0.02);
            let cy = cos(yaw);
            let sy = sin(yaw);
            let turned = centre + vec3f(cy * rel.x + sy * rel.y, -sy * rel.x + cy * rel.y, rel.z);
            near = min(near, sdf{id}_at(turned, centre, sc, vec3f({cx}, {cy_}, {cz})));
        }}
    }}
    let shapes = near + displace * {roughen};
    if ({subtract} > 0.5) {{ return -scatter{id}_smin(-base, shapes, {blend}); }}
    return scatter{id}_smin(base, shapes, {blend});
}}
",
        seed = p(0),
        cell = p(1),
        density = p(2),
        size_lo = p(3),
        size_hi = p(4),
        jitter = p(5),
        slope_lo = p(6),
        slope_hi = p(7),
        elev_lo = p(8),
        elev_hi = p(9),
        sink = p(10),
        blend = p(11),
        subtract = p(12),
        yaw = p(13),
        roughen = p(14),
        inv = p(15),
        cx = p(16),
        cy_ = p(17),
        cz = p(18),
        half = p(19),
        cells = p(20),
        face = p(21),
    )
}

pub fn props(ui: &mut eframe::egui::Ui, p: &mut ScatterParams) {
    use eframe::egui;
    let d = ScatterParams::default();
    ui.weak("Plants copies of an SDF shape across the terrain by a rule. Needs a terrain: a Load Mesh node with a heightmap. Input 1 is the field to scatter onto (the terrain, or what earlier nodes made of it); input 2 optionally roughens the copies (a Noise Graph node). Wire several Scatter nodes in a row for different shapes. Units are the field's (centimetres for Unreal).");
    ui.horizontal(|ui| {
        ui.add(egui::TextEdit::singleline(&mut p.file).hint_text("a .sdf file").desired_width(150.0).id_salt("scatter sdf path"));
        if ui.button("Browse…").clicked() {
            let mut dialog = rfd::FileDialog::new().add_filter("SDF Modeler file", &["sdf"]);
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
    let step = (p.cell * 0.01).max(0.01);
    crate::row(ui, &mut p.cell, d.cell, |ui, v| _ = ui.add(egui::DragValue::new(v).range(1.0..=1.0e7).speed(step).prefix("cell ")).on_hover_text("The terrain is cut into squares this big, each holding at most one copy. Copies should not be much bigger than a cell."));
    crate::row(ui, &mut p.density, d.density, |ui, v| _ = ui.add(egui::Slider::new(v, 0.0..=1.0).text("density")).on_hover_text("The chance a cell holds a copy."));
    crate::row(ui, &mut p.jitter, d.jitter, |ui, v| _ = ui.add(egui::Slider::new(v, 0.0..=1.0).text("jitter")).on_hover_text("How far a copy strays from the middle of its cell."));
    ui.horizontal(|ui| {
        ui.label("size");
        let step = (p.size[1] * 0.01).max(0.01);
        ui.add(egui::DragValue::new(&mut p.size[0]).range(1.0..=1.0e7).speed(step));
        ui.add(egui::DragValue::new(&mut p.size[1]).range(1.0..=1.0e7).speed(step));
    })
    .response
    .on_hover_text("Each copy's longest side is a random size between these.");
    crate::row(ui, &mut p.yaw, d.yaw, |ui, v| _ = ui.checkbox(v, "random facing"));
    crate::row(ui, &mut p.face_uphill, d.face_uphill, |ui, v| _ = ui.checkbox(v, "face uphill").on_hover_text("Turns each copy so its back (+Z in the Composer) faces uphill: a cave goes into the hillside and opens downhill. Used instead of the random turn where there is a slope."));
    ui.horizontal(|ui| {
        ui.label("slope");
        ui.add(egui::DragValue::new(&mut p.slope[0]).range(0.0..=10.0).speed(0.01));
        ui.add(egui::DragValue::new(&mut p.slope[1]).range(0.0..=10.0).speed(0.01));
    })
    .response
    .on_hover_text("Rise over run of the ground under a copy: 0.5 is a 27 degree slope, 1 is 45.");
    ui.horizontal(|ui| {
        ui.label("height");
        ui.add(egui::DragValue::new(&mut p.elevation[0]).range(0.0..=1.0).speed(0.01));
        ui.add(egui::DragValue::new(&mut p.elevation[1]).range(0.0..=1.0).speed(0.01));
    })
    .response
    .on_hover_text("Where on the terrain's height range a copy may stand: 0 is the lowest ground, 1 the highest.");
    crate::row(ui, &mut p.sink, d.sink, |ui, v| _ = ui.add(egui::DragValue::new(v).range(-0.5..=0.5).speed(0.005).prefix("sink ").suffix(" x size")));
    let step = (p.cell * 0.002).max(0.01);
    crate::row(ui, &mut p.blend, d.blend, |ui, v| _ = ui.add(egui::DragValue::new(v).range(0.0..=1.0e6).speed(step).prefix("blend ")).on_hover_text("Width of the smooth join between the copies and the terrain."));
    crate::row(ui, &mut p.subtract, d.subtract, |ui, v| _ = ui.checkbox(v, "cut out of the base").on_hover_text("Subtract the copies from the terrain (caves, pits) instead of adding them."));
    crate::row(ui, &mut p.roughen, d.roughen, |ui, v| _ = ui.add(egui::DragValue::new(v).range(0.0..=100.0).speed(0.01).prefix("roughen × ")).on_hover_text("How much of input 2 (the displacement) is added to the copies' field. Keep the noise gentle: an amplitude over about a sixth of its wavelength tears the surface into pieces."));
}
