//! Painted masks for the Landscape tab: a `RES` x `RES` grid of values from 0 to 1 over the terrain (rows along y), edited
//! with a round brush and read by the graph's Mask nodes (see `codegen::MASK_BASE`). Masks are saved inside the project as
//! 16-bit PNGs, base64 encoded.

use crate::codegen::MASK_RES as RES;

#[derive(Clone, Debug, PartialEq)]
pub struct Mask {
    pub name: String,
    pub data: Vec<f32>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Tool {
    Paint,
    Erase,
    Smooth,
}

/// A round brush: `radius` is a fraction of the terrain's side, `hardness` 0 (fades from the middle) to 1 (hard edge).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Brush {
    pub radius: f32,
    pub strength: f32,
    pub hardness: f32,
}

impl Default for Brush {
    fn default() -> Self {
        Brush { radius: 0.05, strength: 0.5, hardness: 0.3 }
    }
}

impl Mask {
    pub fn new(name: &str) -> Mask {
        Mask { name: name.to_string(), data: vec![0.0; RES * RES] }
    }

    /// One dab of the brush centred at (`x`, `y`) across the terrain (0 to 1).
    pub fn dab(&mut self, x: f32, y: f32, b: &Brush, tool: Tool) {
        let r = (b.radius * RES as f32).max(1.0);
        let (cx, cy) = (x * (RES - 1) as f32, y * (RES - 1) as f32);
        let (x0, x1) = ((cx - r).floor().max(0.0) as usize, ((cx + r).ceil() as usize).min(RES - 1));
        let (y0, y1) = ((cy - r).floor().max(0.0) as usize, ((cy + r).ceil() as usize).min(RES - 1));
        // Smoothing reads the values as they were before this dab.
        let before = if tool == Tool::Smooth { Some(self.data.clone()) } else { None };
        for j in y0..=y1 {
            for i in x0..=x1 {
                let d = ((i as f32 - cx).powi(2) + (j as f32 - cy).powi(2)).sqrt() / r;
                if d >= 1.0 {
                    continue;
                }
                let t = ((1.0 - d) / (1.0 - b.hardness).max(0.001)).clamp(0.0, 1.0);
                let f = t * t * (3.0 - 2.0 * t) * b.strength;
                let v = &mut self.data[j * RES + i];
                match tool {
                    Tool::Paint => *v += (1.0 - *v) * f,
                    Tool::Erase => *v -= *v * f,
                    Tool::Smooth => {
                        let src = before.as_ref().unwrap();
                        let (mut sum, mut n) = (0.0, 0.0);
                        for dj in -2i32..=2 {
                            for di in -2i32..=2 {
                                let (a, c) = (i as i32 + di, j as i32 + dj);
                                if a >= 0 && c >= 0 && (a as usize) < RES && (c as usize) < RES {
                                    sum += src[c as usize * RES + a as usize];
                                    n += 1.0;
                                }
                            }
                        }
                        *v += (sum / n - *v) * f;
                    }
                }
            }
        }
    }

    /// A stroke from one point to another: dabs a fifth of the brush radius apart.
    pub fn stroke(&mut self, from: (f32, f32), to: (f32, f32), b: &Brush, tool: Tool) {
        let length = ((to.0 - from.0).powi(2) + (to.1 - from.1).powi(2)).sqrt();
        let steps = (length / (b.radius * 0.2).max(1.0 / RES as f32)).ceil().max(1.0) as usize;
        // Several dabs make a stronger mark than one: the strength is shared out over them.
        let each = Brush { strength: 1.0 - (1.0 - b.strength).powf(1.0 / (steps as f32).sqrt().max(1.0)), ..*b };
        for k in 1..=steps {
            let t = k as f32 / steps as f32;
            self.dab(from.0 + (to.0 - from.0) * t, from.1 + (to.1 - from.1) * t, &each, tool);
        }
    }

    pub fn fill(&mut self, v: f32) {
        self.data.fill(v);
    }

    pub fn invert(&mut self) {
        for v in &mut self.data {
            *v = 1.0 - *v;
        }
    }

    /// A box blur of `radius` texels, twice (close to a Gaussian).
    pub fn blur(&mut self, radius: usize) {
        for _ in 0..2 {
            for horizontal in [true, false] {
                let src = self.data.clone();
                for j in 0..RES {
                    for i in 0..RES {
                        let (mut sum, mut n) = (0.0, 0.0);
                        for k in -(radius as i32)..=radius as i32 {
                            let (a, c) = if horizontal { (i as i32 + k, j as i32) } else { (i as i32, j as i32 + k) };
                            if a >= 0 && c >= 0 && (a as usize) < RES && (c as usize) < RES {
                                sum += src[c as usize * RES + a as usize];
                                n += 1.0;
                            }
                        }
                        self.data[j * RES + i] = sum / n;
                    }
                }
            }
        }
    }

    /// The mask as a 16-bit grayscale PNG.
    pub fn to_png(&self) -> Vec<u8> {
        let samples: Vec<u16> = self.data.iter().map(|v| (v.clamp(0.0, 1.0) * 65535.0).round() as u16).collect();
        let img = image::ImageBuffer::<image::Luma<u16>, _>::from_raw(RES as u32, RES as u32, samples).expect("mask size");
        let mut out = Vec::new();
        image::DynamicImage::ImageLuma16(img).write_to(&mut std::io::Cursor::new(&mut out), image::ImageFormat::Png).expect("png");
        out
    }

    /// A mask from a PNG of any size (resampled to the mask size) and bit depth.
    pub fn from_png(name: &str, bytes: &[u8]) -> Option<Mask> {
        let img = image::load_from_memory(bytes).ok()?.to_luma16();
        let (w, h) = (img.width() as usize, img.height() as usize);
        let at = |x: usize, y: usize| img.get_pixel(x.min(w - 1) as u32, y.min(h - 1) as u32).0[0] as f32 / 65535.0;
        let data = (0..RES * RES)
            .map(|k| {
                let (gx, gy) = ((k % RES) as f32 / (RES - 1) as f32 * (w - 1) as f32, (k / RES) as f32 / (RES - 1) as f32 * (h - 1) as f32);
                let (i, j, fx, fy) = (gx as usize, gy as usize, gx.fract(), gy.fract());
                (at(i, j) * (1.0 - fx) + at(i + 1, j) * fx) * (1.0 - fy) + (at(i, j + 1) * (1.0 - fx) + at(i + 1, j + 1) * fx) * fy
            })
            .collect();
        Some(Mask { name: name.to_string(), data })
    }
}

const B64: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

pub fn base64_encode(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let n = (chunk[0] as u32) << 16 | (*chunk.get(1).unwrap_or(&0) as u32) << 8 | *chunk.get(2).unwrap_or(&0) as u32;
        for k in 0..4 {
            if k <= chunk.len() {
                out.push(B64[(n >> (18 - 6 * k) & 63) as usize] as char);
            } else {
                out.push('=');
            }
        }
    }
    out
}

pub fn base64_decode(text: &str) -> Option<Vec<u8>> {
    let mut out = Vec::with_capacity(text.len() / 4 * 3);
    let (mut acc, mut bits) = (0u32, 0);
    for c in text.bytes().filter(|c| *c != b'=' && !c.is_ascii_whitespace()) {
        acc = acc << 6 | B64.iter().position(|b| *b == c)? as u32;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits & 255) as u8);
        }
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_stroke_paints_inside_the_brush_and_erase_and_smooth_work() {
        let mut m = Mask::new("cliffs");
        let b = Brush { radius: 0.05, strength: 1.0, hardness: 1.0 };
        m.stroke((0.2, 0.5), (0.8, 0.5), &b, Tool::Paint);
        let at = |m: &Mask, x: f32, y: f32| m.data[(y * (RES - 1) as f32) as usize * RES + (x * (RES - 1) as f32) as usize];
        assert!(at(&m, 0.5, 0.5) > 0.99, "along the stroke");
        assert!(at(&m, 0.5, 0.6) < 0.01 && at(&m, 0.1, 0.5) < 0.01, "outside it");
        m.stroke((0.4, 0.5), (0.6, 0.5), &b, Tool::Erase);
        assert!(at(&m, 0.5, 0.5) < 0.01 && at(&m, 0.3, 0.5) > 0.99, "erased in the middle only");
        // Smoothing spreads a lone bright texel out.
        let mut lone = Mask::new("lone");
        let centre = RES / 2 * RES + RES / 2;
        lone.data[centre] = 1.0;
        lone.dab(0.5, 0.5, &Brush { radius: 0.01, strength: 1.0, hardness: 1.0 }, Tool::Smooth);
        assert!(lone.data[centre] < 0.5 && lone.data[centre + 1] > 0.0, "{} {}", lone.data[centre], lone.data[centre + 1]);
        m.invert();
        assert!(at(&m, 0.5, 0.5) > 0.99);
    }

    #[test]
    fn masks_survive_a_png_round_trip_and_base64() {
        let mut m = Mask::new("dunes");
        m.dab(0.3, 0.6, &Brush::default(), Tool::Paint);
        let back = Mask::from_png("dunes", &m.to_png()).unwrap();
        assert!(m.data.iter().zip(&back.data).all(|(a, b)| (a - b).abs() < 2e-5), "16 bits keep the values");
        let text = base64_encode(&m.to_png());
        assert_eq!(base64_decode(&text).unwrap(), m.to_png());
        assert_eq!(base64_decode(&base64_encode(b"ab")).unwrap(), b"ab");
        assert_eq!(base64_decode(&base64_encode(b"abcd")).unwrap(), b"abcd");
    }
}
