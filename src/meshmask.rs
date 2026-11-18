//! Displacement masks for meshes: how strongly the noise displaces each part of the surface.
//!
//! The displaced field is `mesh_distance - amount * mask * noise`. With no terms the mask is 1 (the
//! whole surface moves equally). Otherwise it is a weighted sum of terms, each measured on the mesh's
//! own distance field (never on the displaced one, so the mask cannot feed back on itself):
//! a ramp along an axis, how much the surface faces a direction, how open or enclosed it is, how thick
//! the solid is behind it, or its curvature. Every term is first remapped from an input range to
//! 0 to 1 (clamped), optionally inverted, then multiplied by its weight.

/// The most terms a mask can have (the shaders hold them in a fixed array).
pub const MAX_TERMS: usize = 6;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MaskKind {
    /// Always 1: a base strength to add the other terms to.
    Constant,
    /// A ramp along an axis, 0 at one end of the mesh's bounds and 1 at the other.
    Position,
    /// How much the surface faces along an axis: the dot product of its normal and the axis, from -1 to 1.
    Normal,
    /// How open the surface is: 1 on open ground, towards 0 in creases and hollows (ambient occlusion).
    Occlusion,
    /// How deep the solid is behind the surface, along the inward normal, in mesh units (radius 1).
    Thickness,
    /// Mean curvature, in 1 / mesh unit: positive where the surface is convex, negative where concave.
    Curvature,
    /// A colour painted on the mesh's vertices (OBJ `v x y z r g b` or PLY red/green/blue): one channel, 0 to 1.
    VertexColor,
}

impl MaskKind {
    pub const ALL: [MaskKind; 7] = [MaskKind::Constant, MaskKind::Position, MaskKind::Normal, MaskKind::Occlusion, MaskKind::Thickness, MaskKind::Curvature, MaskKind::VertexColor];

    pub fn name(self) -> &'static str {
        match self {
            MaskKind::Constant => "Constant",
            MaskKind::Position => "Position ramp",
            MaskKind::Normal => "Normal direction",
            MaskKind::Occlusion => "Occlusion",
            MaskKind::Thickness => "Thickness",
            MaskKind::Curvature => "Curvature",
            MaskKind::VertexColor => "Vertex colour",
        }
    }

    /// The number the shaders switch on.
    pub fn code(self) -> u32 {
        self as u32 + 1
    }

    /// A one-line explanation for the panel.
    pub fn help(self) -> &'static str {
        match self {
            MaskKind::Constant => "A base strength.",
            MaskKind::Position => "0 at one end of the mesh along the axis, 1 at the other.",
            MaskKind::Normal => "Dot product of the surface normal and the direction, -1 to 1.",
            MaskKind::Occlusion => "1 on open surface, towards 0 in creases and hollows.",
            MaskKind::Thickness => "Depth of solid behind the surface, in mesh units (the mesh is scaled to radius 1).",
            MaskKind::Curvature => "Mean curvature: positive on convex parts, negative in concave ones.",
            MaskKind::VertexColor => "A colour channel painted on the mesh's vertices (0 to 1). The mesh file needs vertex colours.",
        }
    }

    /// The input range that maps to 0 to 1 by default.
    pub fn default_range(self) -> (f32, f32) {
        match self {
            MaskKind::Constant => (0.0, 1.0),
            MaskKind::Position => (0.0, 1.0),
            MaskKind::Normal => (0.0, 1.0),
            MaskKind::Occlusion => (0.0, 1.0),
            MaskKind::Thickness => (0.0, 0.5),
            MaskKind::Curvature => (-2.0, 2.0),
            MaskKind::VertexColor => (0.0, 1.0),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct MaskTerm {
    pub kind: MaskKind,
    pub weight: f32,
    /// The input values that map to 0 and 1 (clamped); ignored by Constant.
    pub lo: f32,
    pub hi: f32,
    pub invert: bool,
    /// Position and Normal: the axis (0 X, 1 Y, 2 Z) and whether the direction is along its negative side.
    pub axis: u32,
    pub negative: bool,
}

impl MaskTerm {
    pub fn new(kind: MaskKind) -> MaskTerm {
        let (lo, hi) = kind.default_range();
        MaskTerm { kind, weight: 1.0, lo, hi, invert: false, axis: 1, negative: false }
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct MaskSettings {
    pub terms: Vec<MaskTerm>,
}

/// A mask packed for the shaders: two vec4 per term, the number of terms, and the range of its value.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PackedMask {
    /// Per term: (kind code, weight, lo, hi) and (direction x, y, z, invert).
    pub terms: [[f32; 4]; 2 * MAX_TERMS],
    pub count: u32,
    pub range: (f32, f32),
}

impl PackedMask {
    /// No mask: the noise displaces everything equally.
    pub const NONE: PackedMask = PackedMask { terms: [[0.0; 4]; 2 * MAX_TERMS], count: 0, range: (1.0, 1.0) };
}

impl MaskSettings {
    pub fn pack(&self) -> PackedMask {
        let mut packed = PackedMask::NONE;
        for (i, t) in self.terms.iter().take(MAX_TERMS).enumerate() {
            let mut dir = [0.0; 3];
            if t.kind == MaskKind::VertexColor {
                // The channel as weights on (r, g, b): red, green, blue or luminance.
                dir = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.2126, 0.7152, 0.0722]][(t.axis as usize).min(3)];
            } else {
                dir[(t.axis as usize).min(2)] = if t.negative { -1.0 } else { 1.0 };
            }
            packed.terms[2 * i] = [t.kind.code() as f32, t.weight, t.lo, t.hi];
            packed.terms[2 * i + 1] = [dir[0], dir[1], dir[2], if t.invert { 1.0 } else { 0.0 }];
        }
        packed.count = self.terms.len().min(MAX_TERMS) as u32;
        packed.range = self.range();
        packed
    }

    /// The smallest and largest value the mask can take: every term is 0 to 1 before its weight.
    pub fn range(&self) -> (f32, f32) {
        if self.terms.is_empty() {
            return (1.0, 1.0);
        }
        self.terms.iter().take(MAX_TERMS).fold((0.0, 0.0), |(lo, hi), t| (lo + t.weight.min(0.0), hi + t.weight.max(0.0)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packing_and_range() {
        assert_eq!(MaskSettings::default().pack(), PackedMask::NONE);
        assert_eq!(MaskSettings::default().range(), (1.0, 1.0));
        let mut m = MaskSettings::default();
        m.terms.push(MaskTerm::new(MaskKind::Constant));
        let mut ramp = MaskTerm::new(MaskKind::Position);
        ramp.weight = -0.5;
        ramp.axis = 2;
        ramp.negative = true;
        ramp.invert = true;
        m.terms.push(ramp);
        let p = m.pack();
        assert_eq!(p.count, 2);
        assert_eq!(p.range, (-0.5, 1.0));
        assert_eq!(p.terms[0], [1.0, 1.0, 0.0, 1.0]);
        assert_eq!(p.terms[2], [2.0, -0.5, 0.0, 1.0]);
        assert_eq!(p.terms[3], [0.0, 0.0, -1.0, 1.0]);
        // More terms than fit are ignored, not a panic.
        let many = MaskSettings { terms: vec![MaskTerm::new(MaskKind::Curvature); MAX_TERMS + 3] };
        assert_eq!(many.pack().count, MAX_TERMS as u32);
    }
}
