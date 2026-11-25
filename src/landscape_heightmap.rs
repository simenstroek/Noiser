//! The Heightmap node of the Landscape graph: an image as a terrain map. A PNG the Landscape tab exported (its `.json`
//! sidecar next to it) comes back in metres, as heights in the graph's units; any other PNG (8 or 16 bits, grey) is
//! 0 to 1 of the height setting; an EXR's first channel is taken as it is (units). The image is read when the graph is
//! prepared and becomes an ordinary terrain map (see `landscape_terrain`), so it works anywhere a Map does.
//! See docs/terrain-maps-plan.md.

use std::path::{Path, PathBuf};

/// The biggest side an imported map keeps; bigger images are shrunk to it.
pub const MAX_SIDE: usize = 2049;

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct HeightmapParams {
    /// The image: relative to the project's folder, or absolute.
    pub file: String,
    /// The heights times this (1 keeps them).
    pub scale: f32,
    /// Added after scaling, in the graph's units (1 = the full height setting).
    pub offset: f32,
}

impl Default for HeightmapParams {
    fn default() -> Self {
        HeightmapParams { file: String::new(), scale: 1.0, offset: 0.0 }
    }
}

/// A heightmap read into the graph's units: `side` x `side` values, rows along y (top row first, as the Landscape
/// tab's own exports have it).
#[derive(Debug)]
pub struct Loaded {
    pub side: usize,
    pub data: Vec<f32>,
    /// What was read, for the node's panel ("exported terrain, 2000 m", "16-bit PNG", ...).
    pub kind: &'static str,
}

fn resolve(base: &Path, file: &str) -> PathBuf {
    let p = Path::new(file);
    if p.is_absolute() { p.to_path_buf() } else { base.join(p) }
}

/// Reads the image and puts it in the graph's units (`height_m` is the height setting in metres).
pub fn load(p: &HeightmapParams, base: &Path, height_m: f32) -> Result<Loaded, String> {
    if p.file.trim().is_empty() {
        return Err("no file chosen".into());
    }
    let path = resolve(base, &p.file);
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("").to_ascii_lowercase();
    let (side, mut data, kind) = match ext.as_str() {
        "exr" => {
            let (side, data) = read_exr(&path)?;
            (side, data, "EXR")
        }
        "png" => {
            if Path::new(&format!("{}.json", path.display())).exists() {
                // One of our own exports: metres back, then the graph's units.
                let h = crate::landscape_field::load(&path, 100.0)?;
                let k = 1.0 / height_m.max(1e-3);
                (h.n, h.h.iter().map(|m| m * k).collect(), "exported terrain")
            } else {
                let img = image::open(&path).map_err(|e| format!("{}: {e}", path.display()))?;
                let (deep, luma) = (img.color().bits_per_pixel() / img.color().channel_count() as u16 > 8, img.to_luma16());
                let side = luma.width() as usize;
                if luma.height() as usize != side {
                    return Err(format!("{}: a heightmap must be square ({} x {})", path.display(), luma.width(), luma.height()));
                }
                (side, luma.as_raw().iter().map(|v| *v as f32 / 65535.0).collect(), if deep { "16-bit PNG" } else { "8-bit PNG" })
            }
        }
        _ => return Err(format!("{}: expected a .png or .exr file", path.display())),
    };
    if side < 2 {
        return Err("the image is too small".into());
    }
    let (side, shrunk) = if side > MAX_SIDE { (MAX_SIDE, shrink(&data, side, MAX_SIDE)) } else { (side, data.clone()) };
    data = shrunk;
    for v in &mut data {
        *v = *v * p.scale + p.offset;
    }
    Ok(Loaded { side, data, kind })
}

/// An EXR's first channel as `side` x `side` floats.
fn read_exr(path: &Path) -> Result<(usize, Vec<f32>), String> {
    use exr::prelude::*;
    let image = read_first_flat_layer_from_file(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let layer = &image.layer_data;
    let (w, h) = (layer.size.width(), layer.size.height());
    if w != h {
        return Err(format!("{}: a heightmap must be square ({w} x {h})", path.display()));
    }
    // Prefer a channel that is called height or Y or R, else the first.
    let channel = layer.channel_data.list.iter().find(|c| matches!(c.name.to_string().as_str(), "Y" | "R" | "height" | "H")).or_else(|| layer.channel_data.list.first()).ok_or("the EXR has no channels")?;
    Ok((w, (0..w * h).map(|k| channel.sample_data.value_by_flat_index(k).to_f32()).collect()))
}

/// `data` (`n` x `n`) averaged down to `m` x `m` over the same area (bilinear in the source).
pub fn shrink(data: &[f32], n: usize, m: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(m * m);
    let k = (n - 1) as f32 / (m - 1) as f32;
    for y in 0..m {
        for x in 0..m {
            let (fx, fy) = (x as f32 * k, y as f32 * k);
            let (i, j) = ((fx as usize).min(n - 2), (fy as usize).min(n - 2));
            let (tx, ty) = (fx - i as f32, fy - j as f32);
            let a = data[j * n + i] * (1.0 - tx) + data[j * n + i + 1] * tx;
            let b = data[(j + 1) * n + i] * (1.0 - tx) + data[(j + 1) * n + i + 1] * tx;
            out.push(a * (1.0 - ty) + b * ty);
        }
    }
    out
}

/// `data` (`n` x `n`) resampled bilinearly to `m` x `m` over the same area, larger or smaller.
pub fn shrink_or_grow(data: &[f32], n: usize, m: usize) -> Vec<f32> {
    shrink(data, n, m)
}

/// A stamp of the files the graph's Heightmap nodes read (modified time and size), so an edited image is seen.
pub fn files_stamp(graph: &crate::graph::Graph, base: &Path) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    for n in graph.nodes.values() {
        if let crate::graph::NodeKind::Heightmap(p) = &n.kind {
            let path = resolve(base, &p.file);
            for file in [path.clone(), PathBuf::from(format!("{}.json", path.display()))] {
                if let Ok(m) = std::fs::metadata(&file) {
                    (m.len(), m.modified().ok()).hash(&mut h);
                }
            }
        }
    }
    h.finish()
}

/// A Heightmap node's settings.
pub fn node_ui(ui: &mut eframe::egui::Ui, p: &mut HeightmapParams) {
    use eframe::egui;
    ui.weak("An image as terrain. A PNG exported by the Landscape tab (with its .json beside it) comes back in metres; another PNG (8 or 16 bit grey) is 0 to 1 of the height setting; an EXR is taken as it is. Read when the terrain is made, shrunk to 2049 a side at most, and stretched over the whole terrain. Wire a position in to move or tile it.");
    crate::meshgraph_ui::path_row(ui, "heightmap path", &mut p.file, "heightmap .png or .exr", Some(("Heightmap", &["png", "exr"])));
    let d = HeightmapParams::default();
    crate::row(ui, &mut p.scale, d.scale, |ui, v| _ = ui.add(egui::DragValue::new(v).speed(0.01).prefix("heights × ")));
    crate::row(ui, &mut p.offset, d.offset, |ui, v| _ = ui.add(egui::DragValue::new(v).speed(0.005).prefix("offset ")).on_hover_text("Added after the scale, in the graph's units (1 = the full height setting)."));
    if !p.file.trim().is_empty() {
        let (height_m, base) = (crate::landscape::HEIGHT_FOR_UI.with(|h| h.get()), crate::meshgraph_ui::base());
        match load(p, &base, height_m) {
            Ok(l) => _ = ui.weak(format!("{} x {} ({})", l.side, l.side, l.kind)),
            Err(e) => _ = ui.colored_label(crate::editor::ORANGE, e),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("noiser_heightmap_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir.join(name)
    }

    #[test]
    fn a_plain_16_bit_png_is_zero_to_one_and_scale_and_offset_apply() {
        let path = temp("plain.png");
        let img = image::ImageBuffer::<image::Luma<u16>, _>::from_raw(2, 2, vec![0u16, 65535, 32768, 65535]).unwrap();
        img.save(&path).unwrap();
        let p = HeightmapParams { file: path.to_string_lossy().into_owned(), scale: 2.0, offset: 0.5 };
        let l = load(&p, Path::new("."), 400.0).unwrap();
        assert_eq!((l.side, l.kind), (2, "16-bit PNG"));
        assert!((l.data[0] - 0.5).abs() < 1e-5 && (l.data[1] - 2.5).abs() < 1e-5 && (l.data[2] - 1.5).abs() < 1e-3);
    }

    #[test]
    fn our_own_export_comes_back_in_the_graphs_units() {
        let path = temp("own.png");
        let metres: Vec<f32> = (0..9).map(|k| k as f32 * 10.0).collect();
        crate::landscape::write_unreal(&path, 3, 300.0, &crate::landscape::encode_unreal(&metres)).unwrap();
        let l = load(&HeightmapParams { file: path.to_string_lossy().into_owned(), ..Default::default() }, Path::new("."), 80.0).unwrap();
        assert_eq!((l.side, l.kind), (3, "exported terrain"));
        for (m, v) in metres.iter().zip(&l.data) {
            assert!((m / 80.0 - v).abs() < 0.01, "{m} m -> {v} units");
        }
    }

    #[test]
    fn a_missing_or_non_square_file_says_so_and_big_images_shrink() {
        assert!(load(&HeightmapParams { file: "nope.png".into(), ..Default::default() }, Path::new("."), 100.0).is_err());
        assert!(load(&HeightmapParams::default(), Path::new("."), 100.0).unwrap_err().contains("no file"));
        let path = temp("wide.png");
        image::GrayImage::new(4, 2).save(&path).unwrap();
        assert!(load(&HeightmapParams { file: path.to_string_lossy().into_owned(), ..Default::default() }, Path::new("."), 100.0).unwrap_err().contains("square"));
        let big: Vec<f32> = (0..9 * 9).map(|k| (k % 9) as f32).collect();
        let small = shrink(&big, 9, 3);
        assert_eq!((small.len(), small[0], small[2], small[8]), (9, 0.0, 8.0, 8.0));
    }

    #[test]
    fn an_exr_is_read_as_it_is() {
        let path = temp("h.exr");
        exr::prelude::write_rgba_file(&path, 4, 4, |x, y| ((x + 4 * y) as f32 * 0.25, 0.0, 0.0, 1.0)).unwrap();
        let l = load(&HeightmapParams { file: path.to_string_lossy().into_owned(), ..Default::default() }, Path::new("."), 100.0).unwrap();
        assert_eq!((l.side, l.kind), (4, "EXR"));
        assert_eq!((l.data[0], l.data[5], l.data[15]), (0.0, 1.25, 3.75));
    }
}
