//! Terrain maps: named grids over the terrain (x and y across it, 0 to 1), read by the graph's Map nodes (the Mask
//! node, labelled "Map"). Painted masks are maps; so will be imported heightmaps, baked stamps, erosion outputs and layer
//! weights (see docs/terrain-maps-plan.md). Each map has its own side, and all of them go into the parameter buffer
//! from `codegen::MASK_BASE` on, each starting on a multiple of `ALIGN` floats so its offset (passed to the shader in
//! units of `ALIGN`) stays a small whole number that f32 holds exactly.

#![allow(dead_code)] // the other sources and lookups arrive with phases 1 to 4 of docs/terrain-maps-plan.md

use std::collections::BTreeMap;
use std::sync::Arc;

/// Maps start on multiples of this many floats; offsets are passed in these units.
pub const ALIGN: usize = 256;

/// Where a map came from (the Map node's menu groups them by it).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Source {
    Painted,
    Imported,
    Stamp,
    Erosion,
    Layer,
}

impl Source {
    pub fn label(self) -> &'static str {
        match self {
            Source::Painted => "Painted masks",
            Source::Imported => "Imported",
            Source::Stamp => "Stamps",
            Source::Erosion => "Erosion",
            Source::Layer => "Layers",
        }
    }
}

#[derive(Clone, Debug)]
pub struct Map {
    /// Samples along a side; `data` is `side * side` values, rows along y.
    pub side: usize,
    pub data: Arc<Vec<f32>>,
    pub source: Source,
}

#[derive(Clone, Debug, Default)]
pub struct TerrainMaps {
    maps: BTreeMap<String, Map>,
}

/// Maps packed for the shader: the floats to put at `codegen::MASK_BASE`, and per name its offset (in units of `ALIGN`)
/// and side.
pub struct Packed {
    pub data: Vec<f32>,
    pub slots: BTreeMap<String, (f32, f32)>,
}

impl TerrainMaps {
    /// Adds or replaces a map. `data` must hold `side * side` values and `side` must be at least 2.
    pub fn insert(&mut self, name: impl Into<String>, side: usize, data: Arc<Vec<f32>>, source: Source) {
        assert!(side >= 2 && data.len() == side * side, "a terrain map is side x side values");
        self.maps.insert(name.into(), Map { side, data, source });
    }

    /// Painted masks, each `side` x `side`.
    pub fn painted(masks: &BTreeMap<String, Vec<f32>>, side: usize) -> TerrainMaps {
        let mut maps = TerrainMaps::default();
        for (name, data) in masks {
            if data.len() == side * side {
                maps.insert(name.clone(), side, Arc::new(data.clone()), Source::Painted);
            }
        }
        maps
    }

    pub fn get(&self, name: &str) -> Option<&Map> {
        self.maps.get(name)
    }

    pub fn is_empty(&self) -> bool {
        self.maps.is_empty()
    }

    /// The names with their sources, grouped by source and then by name (for menus).
    pub fn names(&self) -> Vec<(String, Source)> {
        let mut out: Vec<(String, Source)> = self.maps.iter().map(|(n, m)| (n.clone(), m.source)).collect();
        out.sort_by(|a, b| (a.1, &a.0).cmp(&(b.1, &b.0)));
        out
    }

    /// Packs the maps called `names` (a name with no map reads 0 everywhere).
    pub fn pack(&self, names: &[String]) -> Packed {
        let mut data = Vec::new();
        let mut slots = BTreeMap::new();
        let empty = Map { side: 2, data: Arc::new(vec![0.0; 4]), source: Source::Painted };
        for name in names {
            if slots.contains_key(name) {
                continue;
            }
            let map = self.maps.get(name).unwrap_or(&empty);
            debug_assert_eq!(data.len() % ALIGN, 0);
            slots.insert(name.clone(), ((data.len() / ALIGN) as f32, map.side as f32));
            data.extend_from_slice(&map.data);
            data.resize(data.len().div_ceil(ALIGN) * ALIGN, 0.0);
        }
        Packed { data, slots }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_of_different_sides_pack_on_aligned_offsets_and_a_missing_one_reads_zero() {
        let mut maps = TerrainMaps::default();
        maps.insert("big", 300, Arc::new(vec![1.0; 300 * 300]), Source::Painted);
        maps.insert("small", 3, Arc::new((0..9).map(|v| v as f32).collect()), Source::Erosion);
        let packed = maps.pack(&["small".into(), "big".into(), "gone".into(), "small".into()]);
        let (small, big, gone) = (packed.slots["small"], packed.slots["big"], packed.slots["gone"]);
        assert_eq!((small, big.1, gone.1), ((0.0, 3.0), 300.0, 2.0));
        // `small` takes one aligned block, `big` starts after it and takes ceil(90000 / 256) blocks.
        assert_eq!(big.0, 1.0);
        assert_eq!(gone.0, 1.0 + (90000f32 / ALIGN as f32).ceil());
        let at = |slot: (f32, f32), k: usize| packed.data[slot.0 as usize * ALIGN + k];
        assert_eq!((at(small, 8), at(big, 89999), at(gone, 3)), (8.0, 1.0, 0.0));
        assert_eq!(packed.data.len() % ALIGN, 0);
    }

    #[test]
    fn map_nodes_get_their_map_s_offset_and_side_and_too_much_is_refused() {
        use crate::graph::{Graph, NodeKind};
        let mut g = Graph::default();
        let pos = g.add(NodeKind::position(), [0.0; 2]);
        let a = g.add(NodeKind::Mask { name: "a".into() }, [0.0; 2]);
        let out = g.add(NodeKind::OutDensity, [0.0; 2]);
        g.connect(pos, a, 0);
        g.connect(a, out, 0);
        let c = crate::codegen::compile(&g).unwrap();
        let mut maps = TerrainMaps::default();
        maps.insert("a", 64, Arc::new(vec![0.5; 64 * 64]), Source::Painted);
        let params = crate::export::map_params(&c, &maps, 1 << 30).unwrap();
        let (at, _) = &c.map_params[0];
        assert_eq!((params[*at], params[*at + 1]), (0.0, 64.0));
        assert_eq!(params[crate::codegen::MASK_BASE + 100], 0.5);
        let err = crate::export::map_params(&c, &maps, 1 << 20).unwrap_err();
        assert!(err.contains("more than the GPU allows"), "{err}");
    }

    #[test]
    fn names_are_grouped_by_source() {
        let mut maps = TerrainMaps::default();
        let grid = Arc::new(vec![0.0; 4]);
        maps.insert("zeta", 2, grid.clone(), Source::Painted);
        maps.insert("alpha", 2, grid.clone(), Source::Erosion);
        maps.insert("beta", 2, grid, Source::Painted);
        let names: Vec<String> = maps.names().into_iter().map(|(n, _)| n).collect();
        assert_eq!(names, ["beta", "zeta", "alpha"]);
    }
}
