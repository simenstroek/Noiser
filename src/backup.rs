//! Crash recovery. Each running Noiser owns a session folder holding a `lock` file it keeps
//! exclusively locked, plus rolling backups of the graph. A clean exit deletes the folder. A crash
//! leaves it behind with the lock released by the OS, so the next start can tell crashed sessions
//! (lock takeable) from other running instances (lock held) and offer the latest backup.

use crate::graph::Graph;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Backups kept per session; older ones are deleted.
const KEEP: u64 = 10;
/// At most one backup per this interval while the graph keeps changing (e.g. dragging a node).
pub const INTERVAL: Duration = Duration::from_secs(1);

#[derive(Debug, PartialEq, Serialize, Deserialize)]
pub struct Backup {
    /// File the graph was opened from / saved to, if any.
    pub path: Option<PathBuf>,
    pub saved_unix: u64,
    pub graph: Graph,
}

pub struct Session {
    dir: PathBuf,
    _lock: File,
    last: Option<Graph>,
    last_write: Option<Instant>,
    counter: u64,
}

/// Where session folders live.
pub fn root() -> Option<PathBuf> {
    eframe::storage_dir("Noiser").map(|d| d.join("sessions"))
}

fn unix_now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs())
}

impl Session {
    pub fn start(root: &Path) -> std::io::Result<Self> {
        let nanos = SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_nanos());
        let dir = root.join(format!("{}-{nanos}", std::process::id()));
        std::fs::create_dir_all(&dir)?;
        let lock = File::create(dir.join("lock"))?;
        lock.try_lock().map_err(std::io::Error::other)?;
        Ok(Self { dir, _lock: lock, last: None, last_write: None, counter: 0 })
    }

    /// Backs up `graph` if it changed since the last backup. Returns true when a change is waiting
    /// for the interval to pass, so the caller should run another frame after `INTERVAL`.
    pub fn backup(&mut self, graph: &Graph, path: Option<&Path>) -> bool {
        if self.last.as_ref() == Some(graph) {
            return false;
        }
        if self.last_write.is_some_and(|t| t.elapsed() < INTERVAL) {
            return true;
        }
        let backup = Backup { path: path.map(Path::to_path_buf), saved_unix: unix_now(), graph: graph.clone() };
        let file = self.dir.join(format!("backup-{:08}.json", self.counter));
        // Write then rename, so a crash mid-write never leaves a truncated newest backup.
        let tmp = self.dir.join("backup.tmp");
        let ok = serde_json::to_vec(&backup).ok().is_some_and(|bytes| std::fs::write(&tmp, bytes).is_ok() && std::fs::rename(&tmp, &file).is_ok());
        if ok {
            if self.counter >= KEEP {
                let _ = std::fs::remove_file(self.dir.join(format!("backup-{:08}.json", self.counter - KEEP)));
            }
            self.counter += 1;
            self.last = Some(graph.clone());
        }
        self.last_write = Some(Instant::now());
        false
    }

    /// Clean exit: nothing to recover, so the folder goes.
    pub fn end(self) {
        let dir = self.dir.clone();
        drop(self); // releases the lock first
        let _ = std::fs::remove_dir_all(dir);
    }
}

/// Session folders under `root` left behind by crashed instances (their lock is takeable).
pub fn crashed(root: &Path) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(root) else { return Vec::new() };
    entries
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|dir| dir.is_dir())
        .filter(|dir| match File::open(dir.join("lock")) {
            Ok(f) => f.try_lock().is_ok(), // dropping `f` releases it again
            Err(_) => true,                // no lock file: a session that never finished starting
        })
        .collect()
}

/// The newest readable backup across the given session folders.
pub fn latest(dirs: &[PathBuf]) -> Option<Backup> {
    dirs.iter()
        .filter_map(|d| std::fs::read_dir(d).ok())
        .flatten()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "json"))
        .filter_map(|p| Some((serde_json::from_slice::<Backup>(&std::fs::read(&p).ok()?).ok()?, p)))
        // Seconds can tie within a session; the zero-padded counter in the name breaks ties.
        .max_by(|(a, pa), (b, pb)| (a.saved_unix, pa).cmp(&(b.saved_unix, pb)))
        .map(|(b, _)| b)
}

pub fn remove(dirs: &[PathBuf]) {
    for d in dirs {
        let _ = std::fs::remove_dir_all(d);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::NodeKind;

    fn temp_root(name: &str) -> PathBuf {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target").join("test-sessions").join(name);
        let _ = std::fs::remove_dir_all(&root);
        root
    }

    #[test]
    fn running_sessions_are_not_crashed_but_abandoned_ones_are() {
        let root = temp_root("crash");
        let mut a = Session::start(&root).unwrap();
        let g = Graph::default_graph();
        a.backup(&g, Some(Path::new("x.noisegraph")));
        // A live session holds its lock.
        assert!(crashed(&root).is_empty());
        // Simulate a crash: the lock is released but the folder stays.
        let dir = a.dir.clone();
        drop(a);
        let found = crashed(&root);
        assert_eq!(found, vec![dir]);
        let b = latest(&found).unwrap();
        assert_eq!(b.graph, g);
        assert_eq!(b.path.as_deref(), Some(Path::new("x.noisegraph")));
        remove(&found);
        assert!(crashed(&root).is_empty());
    }

    #[test]
    fn clean_exit_leaves_nothing_and_backups_roll() {
        let root = temp_root("roll");
        let mut s = Session::start(&root).unwrap();
        let mut g = Graph::default_graph();
        for i in 0..15 {
            g.add(NodeKind::Float(i as f32), [0.0; 2]);
            s.last_write = None; // skip the 1 s throttle
            assert!(!s.backup(&g, None));
        }
        let files = std::fs::read_dir(&s.dir).unwrap().filter(|e| e.as_ref().unwrap().path().extension().is_some_and(|x| x == "json")).count();
        assert_eq!(files, KEEP as usize);
        // Unchanged graph: no new backup and nothing pending.
        assert!(!s.backup(&g, None));
        // A change inside the interval is held back and reported as pending.
        g.add(NodeKind::position(), [0.0; 2]);
        assert!(s.backup(&g, None));
        s.end();
        assert!(std::fs::read_dir(&root).unwrap().next().is_none());
    }
}
