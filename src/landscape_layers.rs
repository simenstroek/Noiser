//! Paint layers in the Landscape graph. A Layer node is an output: its input is the layer's weight (0 to 1) over the
//! terrain. The exports normalise all layers to sum to 1 per cell (a cell no layer covers goes to the first layer) and
//! write one 8-bit weightmap each, named after the layer, listed in the heightmap's sidecar; the 3D view colours the
//! terrain with the layers' colours. See docs/terrain-maps-plan.md.

use crate::export::Progress;
use crate::graph::{Graph, NodeId, NodeKind};
use crate::landscape::{build_evaluator, sample, Settings};
use crate::landscape_terrain::Prepared;
use std::path::Path;

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct LayerParams {
    /// The layer's name: the weightmap's file name and Unreal's layer name.
    pub name: String,
    /// The colour the 3D view paints the layer in (0 to 1 RGB).
    pub color: [f32; 3],
}

impl Default for LayerParams {
    fn default() -> Self {
        LayerParams { name: "layer".into(), color: [0.5, 0.5, 0.5] }
    }
}

/// One layer's weights over the working grid.
#[derive(Clone, Debug)]
pub struct Layer {
    pub name: String,
    pub color: [f32; 3],
    pub weights: Vec<f32>,
}

/// A name safe for a file and for Unreal's layer names: letters, digits and underscores.
pub fn file_name(name: &str) -> String {
    let s: String = name.trim().chars().map(|c| if c.is_ascii_alphanumeric() || c == '_' { c } else { '_' }).collect();
    if s.is_empty() { "layer".into() } else { s }
}

/// The graph's layers (not muted): node, file-safe name made unique (a second `rock` becomes `rock_2`), colour. In node order.
pub fn layer_nodes(graph: &Graph) -> Vec<(NodeId, String, [f32; 3])> {
    let mut out: Vec<(NodeId, String, [f32; 3])> = Vec::new();
    for (id, n) in &graph.nodes {
        let NodeKind::Layer(l) = &n.kind else { continue };
        if n.muted {
            continue;
        }
        let base = file_name(&l.name);
        let mut name = base.clone();
        let mut k = 2;
        while out.iter().any(|(_, o, _)| *o == name) {
            name = format!("{base}_{k}");
            k += 1;
        }
        out.push((*id, name, l.color));
    }
    out
}

pub fn has_layers(graph: &Graph) -> bool {
    !layer_nodes(graph).is_empty()
}

/// Clamps every layer to 0..1 and scales the cells where they add up to more than 1 so they sum to exactly 1; a cell
/// with no weight at all goes to the first layer.
pub fn normalise(layers: &mut [Layer]) {
    let Some(cells) = layers.first().map(|l| l.weights.len()) else { return };
    for k in 0..cells {
        let sum: f32 = layers.iter().map(|l| l.weights[k].clamp(0.0, 1.0)).sum();
        for l in layers.iter_mut() {
            l.weights[k] = l.weights[k].clamp(0.0, 1.0) / sum.max(1e-6);
        }
        if sum <= 1e-6 {
            layers[0].weights[k] = 1.0;
        }
    }
}

/// Weights as 8-bit values that add up to exactly 255 in every cell (the largest layer takes the rounding).
pub fn quantise(layers: &[Layer]) -> Vec<Vec<u8>> {
    let Some(cells) = layers.first().map(|l| l.weights.len()) else { return Vec::new() };
    let mut out = vec![vec![0u8; cells]; layers.len()];
    for k in 0..cells {
        let mut total = 0i32;
        let mut biggest = 0;
        for (i, l) in layers.iter().enumerate() {
            let v = (l.weights[k] * 255.0).round().clamp(0.0, 255.0) as u8;
            out[i][k] = v;
            total += v as i32;
            if l.weights[k] > layers[biggest].weights[k] {
                biggest = i;
            }
        }
        out[biggest][k] = (out[biggest][k] as i32 + 255 - total).clamp(0, 255) as u8;
    }
    out
}

/// The layers' normalised weights on an `n` x `n` grid, from a prepared (Erosion nodes lowered) graph.
pub fn sample_layers(device: &eframe::wgpu::Device, queue: &eframe::wgpu::Queue, prepared: &Prepared, s: &Settings, base: &Path, n: usize, progress: &Progress) -> Result<Vec<Layer>, String> {
    let nodes = layer_nodes(&prepared.graph);
    let mut layers = Vec::new();
    for (k, (id, name, color)) in nodes.iter().enumerate() {
        let weights = match prepared.graph.input_from(*id, 0) {
            Some((src, port)) => {
                let mut sub = prepared.graph.clone();
                sub.nodes.retain(|_, node| !matches!(node.kind, NodeKind::OutDensity | NodeKind::OutNormal | NodeKind::Layer(_)));
                sub.prune_edges();
                let out = sub.add(NodeKind::OutDensity, [0.0; 2]);
                sub.connect_from(src, port, out, 0);
                let ev = build_evaluator(device, queue, &sub, s, &prepared.maps, base)?;
                let w = sample(&ev, n, &Progress::default());
                progress.store(((k + 1) * 1000 / nodes.len()) as u32, std::sync::atomic::Ordering::Relaxed);
                w
            }
            None => vec![0.0; n * n],
        };
        layers.push(Layer { name: name.clone(), color: *color, weights });
    }
    normalise(&mut layers);
    Ok(layers)
}

/// The colour of each cell: the layers' colours mixed by their weights (RGBA8, for the 3D view).
pub fn colours(layers: &[Layer]) -> Vec<u8> {
    let Some(cells) = layers.first().map(|l| l.weights.len()) else { return Vec::new() };
    let mut out = Vec::with_capacity(cells * 4);
    for k in 0..cells {
        let mut c = [0.0f32; 3];
        for l in layers {
            for a in 0..3 {
                c[a] += l.color[a] * l.weights[k];
            }
        }
        out.extend(c.map(|v| (v.clamp(0.0, 1.0) * 255.0).round() as u8));
        out.push(255);
    }
    out
}

/// Writes one 8-bit grayscale PNG per layer, `<stem>_<layer>.png` beside `heightmap`, and returns (layer, file name).
pub fn write_weightmaps(heightmap: &Path, n: usize, layers: &[Layer]) -> Result<Vec<(String, String)>, String> {
    let stem = heightmap.file_stem().map_or_else(|| "terrain".to_string(), |s| s.to_string_lossy().into_owned());
    let mut written = Vec::new();
    for (l, bytes) in layers.iter().zip(quantise(layers)) {
        let file = format!("{stem}_{}.png", l.name);
        let path = heightmap.with_file_name(&file);
        let img = image::GrayImage::from_raw(n as u32, n as u32, bytes).ok_or("bad weightmap size")?;
        img.save(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        written.push((l.name.clone(), file));
    }
    Ok(written)
}

/// A Layer node's settings: its name and preview colour.
pub fn node_ui(ui: &mut eframe::egui::Ui, l: &mut LayerParams) {
    use eframe::egui;
    let d = LayerParams::default();
    crate::row(ui, &mut l.name, d.name, |ui, v| {
        ui.label("name");
        ui.add(egui::TextEdit::singleline(v).desired_width(120.0));
    });
    crate::row(ui, &mut l.color, d.color, |ui, v| {
        ui.label("colour");
        ui.color_edit_button_rgb(v);
    });
    ui.weak("A paint layer: the input is its weight (0 to 1). The exports write one 8-bit weightmap per layer, named after it, normalised so the layers sum to 1 in every cell; the 3D view paints the terrain in the layers' colours. Make a weight with Select (from a height, slope or map).");
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layer(name: &str, w: &[f32]) -> Layer {
        Layer { name: name.into(), color: [1.0, 0.0, 0.0], weights: w.to_vec() }
    }

    #[test]
    fn layers_sum_to_one_and_an_empty_cell_goes_to_the_first() {
        let mut layers = [layer("a", &[1.0, 0.2, 0.0, 2.0]), layer("b", &[1.0, 0.2, 0.0, 0.0])];
        normalise(&mut layers);
        let sums: Vec<f32> = (0..4).map(|k| layers.iter().map(|l| l.weights[k]).sum()).collect();
        assert!(sums.iter().all(|s| (s - 1.0).abs() < 1e-5), "{sums:?}");
        assert_eq!((layers[0].weights[0], layers[1].weights[0]), (0.5, 0.5));
        assert_eq!((layers[0].weights[2], layers[1].weights[2]), (1.0, 0.0));
        assert_eq!(layers[0].weights[3], 1.0, "clamped to 1 first");
    }

    #[test]
    fn eight_bit_weights_add_up_to_255() {
        let mut layers = [layer("a", &[1.0 / 3.0, 0.5]), layer("b", &[1.0 / 3.0, 0.5]), layer("c", &[1.0 / 3.0, 0.0])];
        normalise(&mut layers);
        for k in 0..2 {
            assert_eq!(quantise(&layers).iter().map(|l| l[k] as u32).sum::<u32>(), 255);
        }
    }

    #[test]
    fn names_are_file_safe_and_unique() {
        let mut g = Graph::default();
        for name in ["rock", "rock", "wet sand!", ""] {
            g.add(NodeKind::Layer(LayerParams { name: name.into(), ..Default::default() }), [0.0; 2]);
        }
        let names: Vec<String> = layer_nodes(&g).into_iter().map(|(_, n, _)| n).collect();
        assert_eq!(names, ["rock", "rock_2", "wet_sand_", "layer"]);
    }

    #[test]
    fn colours_mix_by_weight() {
        let layers = [Layer { name: "a".into(), color: [1.0, 0.0, 0.0], weights: vec![0.5] }, Layer { name: "b".into(), color: [0.0, 0.0, 1.0], weights: vec![0.5] }];
        assert_eq!(colours(&layers), [128, 0, 128, 255]);
    }

    /// height -> Select -> Layer, twice (low ground and high ground), sampled on a small grid.
    #[test]
    fn layers_from_a_graph_are_normalised_weights_and_select_picks_a_range() {
        use crate::landscape_terrain::{prepare, ErosionCache};
        let Some(gpu) = crate::gpu_tests::gpu() else { return eprintln!("no GPU adapter; skipping") };
        let mut g = Graph::default();
        let pos = g.add(NodeKind::position(), [0.0; 2]);
        let sep = g.add(NodeKind::Separate, [0.0; 2]);
        g.connect(pos, sep, 0);
        // The height is x across the terrain, 0 to 1.
        let low = g.add(NodeKind::Select { lo: -1.0, hi: 0.3, soft: 0.0 }, [0.0; 2]);
        let high = g.add(NodeKind::Select { lo: 0.7, hi: 2.0, soft: 0.0 }, [0.0; 2]);
        g.connect_from(sep, 0, low, 0);
        g.connect_from(sep, 0, high, 0);
        let a = g.add(NodeKind::Layer(LayerParams { name: "grass".into(), color: [0.0, 1.0, 0.0] }), [0.0; 2]);
        let b = g.add(NodeKind::Layer(LayerParams { name: "snow".into(), color: [1.0, 1.0, 1.0] }), [0.0; 2]);
        g.connect(low, a, 0);
        g.connect(high, b, 0);
        let s = Settings { resolution: 33, ..Default::default() };
        let prepared = prepare(&gpu.device, &gpu.queue, &g, &s, &Default::default(), 0, Path::new("."), &ErosionCache::default(), &Default::default(), &Default::default()).unwrap();
        let layers = sample_layers(&gpu.device, &gpu.queue, &prepared, &s, Path::new("."), 33, &Default::default()).unwrap();
        assert_eq!(layers.iter().map(|l| l.name.as_str()).collect::<Vec<_>>(), ["grass", "snow"]);
        let at = |l: usize, x: usize| layers[l].weights[x]; // row 0
        assert_eq!((at(0, 0), at(1, 0)), (1.0, 0.0), "west is grass");
        assert_eq!((at(0, 32), at(1, 32)), (0.0, 1.0), "east is snow");
        // In the middle neither covers: the first layer takes it.
        assert_eq!((at(0, 16), at(1, 16)), (1.0, 0.0));
        assert!((0..33 * 33).all(|k| (layers[0].weights[k] + layers[1].weights[k] - 1.0).abs() < 1e-4));
    }

    #[test]
    fn weightmaps_are_written_beside_the_heightmap() {
        let dir = std::env::temp_dir().join(format!("noiser_layers_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mut layers = [layer("a b", &[1.0, 0.0, 0.5, 0.25]), layer("c", &[0.0, 1.0, 0.5, 0.75])];
        layers[0].name = file_name(&layers[0].name);
        normalise(&mut layers);
        let files = write_weightmaps(&dir.join("t.png"), 2, &layers).unwrap();
        assert_eq!(files, [("a_b".to_string(), "t_a_b.png".to_string()), ("c".to_string(), "t_c.png".to_string())]);
        let img = image::open(dir.join("t_a_b.png")).unwrap().to_luma8();
        assert_eq!(img.as_raw(), &[255, 0, 127, 64], "a tie: the first layer takes the rounding so the cell adds up to 255");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
