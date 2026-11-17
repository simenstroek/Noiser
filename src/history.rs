//! Undo / redo as whole-graph snapshots. Graphs are small (a few KB), so a snapshot per settled
//! edit is simpler and safer than inverse operations.

use crate::graph::Graph;

const LIMIT: usize = 200;

pub struct History {
    undo: Vec<Graph>,
    redo: Vec<Graph>,
    /// The graph as of the last commit; the next commit pushes it onto `undo`.
    last: Graph,
}

impl History {
    pub fn new(g: &Graph) -> Self {
        Self { undo: Vec::new(), redo: Vec::new(), last: g.clone() }
    }

    /// Records `g` if it differs from the last commit. Call with `settled = false` while a drag is in
    /// progress so one drag is one undo step.
    pub fn commit(&mut self, g: &Graph, settled: bool) {
        if !settled || *g == self.last {
            return;
        }
        self.undo.push(std::mem::replace(&mut self.last, g.clone()));
        if self.undo.len() > LIMIT {
            self.undo.remove(0);
        }
        self.redo.clear();
    }

    pub fn undo(&mut self, cur: &Graph) -> Option<Graph> {
        self.commit(cur, true);
        let prev = self.undo.pop()?;
        self.redo.push(std::mem::replace(&mut self.last, prev.clone()));
        Some(prev)
    }

    pub fn redo(&mut self, cur: &Graph) -> Option<Graph> {
        self.commit(cur, true);
        let next = self.redo.pop()?;
        self.undo.push(std::mem::replace(&mut self.last, next.clone()));
        Some(next)
    }

    pub fn can_undo(&self) -> bool {
        !self.undo.is_empty()
    }

    pub fn can_redo(&self) -> bool {
        !self.redo.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::NodeKind;

    #[test]
    fn undo_and_redo_walk_the_edits_and_a_new_edit_clears_redo() {
        let mut g = Graph::default();
        let mut h = History::new(&g);
        let a = g.clone();
        g.add(NodeKind::Position, [0.0; 2]);
        h.commit(&g, false); // mid-drag: nothing recorded
        assert!(!h.can_undo());
        h.commit(&g, true);
        let b = g.clone();
        g.add(NodeKind::OutDensity, [0.0; 2]);
        h.commit(&g, true);
        let c = g.clone();
        g = h.undo(&g).unwrap();
        assert_eq!(g, b);
        g = h.undo(&g).unwrap();
        assert_eq!(g, a);
        assert!(h.undo(&g).is_none());
        g = h.redo(&g).unwrap();
        assert_eq!(g, b);
        // An uncommitted edit is kept by undo (it commits first) and a new edit drops the redo branch.
        g.add(NodeKind::Position, [1.0; 2]);
        let d = g.clone();
        g = h.undo(&g).unwrap();
        assert_eq!(g, b);
        g = h.redo(&g).unwrap();
        assert_eq!(g, d);
        assert_ne!(g, c);
        g.add(NodeKind::Position, [2.0; 2]);
        h.commit(&g, true);
        assert!(!h.can_redo());
    }
}
