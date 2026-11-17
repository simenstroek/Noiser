//! `noiser --headless graph.noisegraph --batch-out DIR (--batch-file F ... | --batch-dir SRC) [options]`:
//! displaces meshes with a graph without opening a window, for scripts and render farms.
//! Progress goes to the console (when started from one) and to `noiser-batch.log` in the output folder.

use crate::graph::Graph;
use eframe::wgpu;
use crate::meshexport::{Displace, FileJob, Format, Progress, SeedMode, Settings};
use crate::meshmask::{MaskKind, MaskSettings, MaskTerm};
use std::io::Write;
use std::path::PathBuf;

pub const USAGE: &str = "usage: noiser --headless GRAPH.noisegraph --batch-out DIR (--batch-file F.obj ... | --batch-dir SRC)
  [--amount 0.25] [--tile 1.0] [--size 1.0] [--seed 0] [--seed-mode same|increment|random]
  [--res 256] [--grid-res 128] [--format obj|ply] [--cull 0.01] [--mask KIND ...]";

#[derive(Debug)]
struct Options {
    graph: PathBuf,
    out: PathBuf,
    files: Vec<PathBuf>,
    amount: f32,
    tile: f32,
    size: f32,
    seed: u32,
    mode: SeedMode,
    settings: Settings,
    grid_res: u32,
    mask: MaskSettings,
}

fn parse(args: &[String]) -> Result<Options, String> {
    let mut o = Options {
        graph: PathBuf::new(),
        out: PathBuf::new(),
        files: Vec::new(),
        amount: 0.25,
        tile: 1.0,
        size: 1.0,
        seed: 0,
        mode: SeedMode::Same,
        settings: Settings::default(),
        grid_res: 128,
        mask: MaskSettings::default(),
    };
    let mut it = args.iter();
    while let Some(a) = it.next() {
        let mut value = |what: &str| it.next().cloned().ok_or(format!("{a} needs {what}"));
        let float = |s: String| s.parse::<f32>().map_err(|_| format!("{a}: {s} is not a number"));
        let whole = |s: String| s.parse::<u32>().map_err(|_| format!("{a}: {s} is not a whole number"));
        match a.as_str() {
            "--headless" => {}
            "--batch-out" => o.out = value("a folder")?.into(),
            "--batch-file" => o.files.push(value("a file")?.into()),
            "--batch-dir" => {
                let dir = PathBuf::from(value("a folder")?);
                let mut found: Vec<PathBuf> = std::fs::read_dir(&dir)
                    .map_err(|e| format!("{}: {e}", dir.display()))?
                    .filter_map(|e| e.ok().map(|e| e.path()))
                    .filter(|p| p.extension().is_some_and(|e| e.eq_ignore_ascii_case("obj")))
                    .collect();
                found.sort();
                o.files.extend(found);
            }
            "--amount" => o.amount = float(value("a number")?)?,
            "--tile" => o.tile = float(value("a number")?)?,
            "--size" => o.size = float(value("a number")?)?,
            "--cull" => o.settings.cull = float(value("a number")?)?,
            "--seed" => o.seed = whole(value("a number")?)?,
            "--res" => o.settings.res = whole(value("a number")?)?,
            "--grid-res" => o.grid_res = whole(value("a number")?)?,
            "--seed-mode" => {
                o.mode = match value("same, increment or random")?.as_str() {
                    "same" => SeedMode::Same,
                    "increment" => SeedMode::Increment,
                    "random" => SeedMode::Random,
                    other => return Err(format!("unknown seed mode {other}")),
                }
            }
            "--format" => {
                o.settings.format = match value("obj or ply")?.as_str() {
                    "obj" => Format::Obj,
                    "ply" => Format::Ply,
                    other => return Err(format!("unknown format {other}")),
                }
            }
            "--mask" => {
                let name = value("a term name")?.to_lowercase();
                let kind = MaskKind::ALL.iter().find(|k| k.name().to_lowercase().starts_with(&name)).ok_or(format!("unknown mask term {name}"))?;
                o.mask.terms.push(MaskTerm::new(*kind));
            }
            other if !other.starts_with("--") && o.graph.as_os_str().is_empty() => o.graph = other.into(),
            other => return Err(format!("unexpected argument {other}")),
        }
    }
    if o.graph.as_os_str().is_empty() {
        return Err("no graph file given".into());
    }
    if o.out.as_os_str().is_empty() {
        return Err("--batch-out is required".into());
    }
    if o.files.is_empty() {
        return Err("no meshes: use --batch-file or --batch-dir".into());
    }
    Ok(o)
}

/// Runs the batch when `--headless` is among the arguments; returns the process exit code.
pub fn try_run() -> Option<i32> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if !args.iter().any(|a| a == "--headless") {
        return None;
    }
    attach_console();
    let mut log = Log::default();
    let code = match run(&args, &mut log) {
        Ok(failed) => i32::from(failed > 0),
        Err(e) => {
            log.line(&format!("error: {e}\n{USAGE}"));
            2
        }
    };
    Some(code)
}

fn run(args: &[String], log: &mut Log) -> Result<usize, String> {
    let o = parse(args)?;
    let graph = Graph::from_json(&std::fs::read_to_string(&o.graph).map_err(|e| format!("{}: {e}", o.graph.display()))?)?;
    std::fs::create_dir_all(&o.out).map_err(|e| format!("{}: {e}", o.out.display()))?;
    log.file = std::fs::OpenOptions::new().create(true).append(true).open(o.out.join("noiser-batch.log")).ok();
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default())).map_err(|e| format!("no GPU adapter: {e}"))?;
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).map_err(|e| format!("no GPU device: {e}"))?;
    let d = Displace { amount: o.amount, tile_size: o.tile, prim_size: o.size, mask: o.mask.pack() };
    let total = o.files.len();
    let mut failed = 0;
    for (i, input) in o.files.iter().enumerate() {
        let stem = input.file_stem().map_or("mesh".into(), |s| s.to_string_lossy().into_owned());
        let seed = o.mode.seed_for(o.seed, i);
        let label = if o.mode == SeedMode::Same { stem.clone() } else { format!("{stem}_seed{seed}") };
        let output = crate::meshapp::unique_path(&o.out, &format!("{label}_displaced"), o.settings.format.extension());
        let job = FileJob { device: &device, queue: &queue, graph: &graph, seed, d, settings: o.settings.clone(), grid_res: o.grid_res, grid_pad: 0.5 };
        let started = std::time::Instant::now();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| job.run(input, &output, &Progress::default())));
        let n = i + 1;
        match result {
            Ok(Ok(info)) => log.line(&format!("[{n}/{total}] {} -> {} ({info}, {:.1} s)", input.display(), output.display(), started.elapsed().as_secs_f32())),
            Ok(Err(e)) => {
                failed += 1;
                log.line(&format!("[{n}/{total}] {} FAILED: {e}", input.display()));
            }
            Err(_) => {
                failed += 1;
                log.line(&format!("[{n}/{total}] {} FAILED: the export crashed", input.display()));
            }
        }
    }
    log.line(&format!("done: {} of {total} meshes written to {}", total - failed, o.out.display()));
    Ok(failed)
}

/// Where progress lines go: the console if there is one, and the log file once known.
#[derive(Default)]
struct Log {
    file: Option<std::fs::File>,
}

impl Log {
    fn line(&mut self, s: &str) {
        let console = if cfg!(windows) { "CONOUT$" } else { "/dev/stdout" };
        if let Ok(mut con) = std::fs::OpenOptions::new().write(true).open(console) {
            let _ = writeln!(con, "{s}");
        }
        if let Some(f) = &mut self.file {
            let _ = writeln!(f, "{s}");
        }
    }
}

/// The release build is a GUI-subsystem program with no console of its own; borrow the one it was started from.
fn attach_console() {
    #[cfg(windows)]
    {
        unsafe extern "system" {
            fn AttachConsole(process_id: u32) -> i32;
        }
        // SAFETY: a plain Win32 call with the documented "parent process" argument.
        unsafe { AttachConsole(u32::MAX) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(s: &str) -> Vec<String> {
        s.split_whitespace().map(String::from).collect()
    }

    #[test]
    fn options_are_parsed_and_checked() {
        let o = parse(&args("--headless g.noisegraph --batch-out out --batch-file a.obj --seed 7 --seed-mode random --format ply --mask thickness")).unwrap();
        assert_eq!((o.seed, o.mode, o.settings.format, o.mask.terms.len(), o.files.len()), (7, SeedMode::Random, Format::Ply, 1, 1));
        assert!(parse(&args("--headless g.noisegraph --batch-file a.obj")).unwrap_err().contains("--batch-out"));
        assert!(parse(&args("--headless --batch-out out --batch-file a.obj")).unwrap_err().contains("graph"));
        assert!(parse(&args("--headless g.noisegraph --batch-out out --batch-file a.obj --seed x")).is_err());
        assert!(parse(&args("--headless g.noisegraph --batch-out out --batch-file a.obj --bogus")).is_err());
    }
}
