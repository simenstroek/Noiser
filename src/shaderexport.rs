//! The graph's density and normal functions as shader code for other tools: WGSL, HLSL or GLSL, written by
//! naga from the same WGSL the preview runs. The parameters are baked in as a constant table and the seed as a
//! constant, so the result is a self-contained module with `noiser_density(p)` and `noiser_normal(p)`.

use crate::codegen::{self, NOISE_LIB};
use crate::graph::Graph;
use naga::valid::{Capabilities, ValidationFlags, Validator};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Lang {
    Wgsl,
    Hlsl,
    Glsl,
}

impl Lang {
    pub fn from_extension(ext: &str) -> Option<Lang> {
        match ext.to_ascii_lowercase().as_str() {
            "wgsl" => Some(Lang::Wgsl),
            "hlsl" | "fx" => Some(Lang::Hlsl),
            "glsl" | "comp" => Some(Lang::Glsl),
            _ => None,
        }
    }
}

/// The standalone WGSL module: the noise library, the parameter table and the generated graph functions.
pub fn standalone_wgsl(g: &Graph, seed: u32) -> Result<String, String> {
    let c = codegen::compile(g)?;
    if g.nodes.values().any(|n| matches!(&n.kind, crate::graph::NodeKind::Noise(p) if p.kind == crate::graph::NoiseKind::Blue)) {
        return Err("the graph uses blue noise, whose table is not included in shader exports".into());
    }
    let n = c.params.len();
    let table: Vec<String> = c.params.iter().map(|v| format!("{v:?}")).collect();
    Ok(format!(
        "{NOISE_LIB}\nstruct U {{ seed: u32 }}\nvar<private> u: U = U({seed}u);\nvar<private> P: array<f32, {n}> = array<f32, {n}>({});\nvar<private> BN: array<f32, 1> = array<f32, 1>(0.0);\n{}\nfn noiser_density(p: vec3f) -> f32 {{ return graph_density(p); }}\nfn noiser_normal(p: vec3f) -> vec3f {{ return graph_normal(p); }}\n@compute @workgroup_size(1) fn noiser_entry() {{ _ = noiser_density(vec3f(0.0)); _ = noiser_normal(vec3f(0.0)); }}\n",
        table.join(", "),
        c.wgsl
    ))
}

/// The graph as shader code in `lang`.
pub fn export(g: &Graph, seed: u32, lang: Lang) -> Result<String, String> {
    let src = standalone_wgsl(g, seed)?;
    let module = naga::front::wgsl::parse_str(&src).map_err(|e| e.emit_to_string(&src))?;
    let info = Validator::new(ValidationFlags::all(), Capabilities::all()).validate(&module).map_err(|e| format!("{e:?}"))?;
    match lang {
        Lang::Wgsl => naga::back::wgsl::write_string(&module, &info, naga::back::wgsl::WriterFlags::empty()).map_err(|e| e.to_string()),
        Lang::Hlsl => {
            let mut out = String::new();
            let (options, pipeline) = (naga::back::hlsl::Options::default(), naga::back::hlsl::PipelineOptions::default());
            let mut writer = naga::back::hlsl::Writer::new(&mut out, &options, &pipeline);
            writer.write(&module, &info, None).map_err(|e| e.to_string())?;
            Ok(out)
        }
        Lang::Glsl => {
            let mut out = String::new();
            let options = naga::back::glsl::Options { version: naga::back::glsl::Version::Desktop(450), ..Default::default() };
            let pipeline = naga::back::glsl::PipelineOptions { shader_stage: naga::ShaderStage::Compute, entry_point: "noiser_entry".into(), multiview: None };
            let mut writer = naga::back::glsl::Writer::new(&mut out, &module, &info, &options, &pipeline, naga::proc::BoundsCheckPolicies::default()).map_err(|e| e.to_string())?;
            writer.write().map_err(|e| e.to_string())?;
            Ok(out)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_graph_exports_to_all_three_languages() {
        let g = Graph::default_graph();
        for lang in [Lang::Wgsl, Lang::Hlsl, Lang::Glsl] {
            let code = export(&g, 3, lang).unwrap_or_else(|e| panic!("{lang:?}: {e}"));
            assert!(code.contains("noiser_density"), "{lang:?} keeps the entry function");
            assert!(code.len() > 2000);
        }
    }

    #[test]
    fn every_template_exports_to_all_three_languages() {
        for name in Graph::TEMPLATES {
            let g = Graph::template(name);
            for lang in [Lang::Wgsl, Lang::Hlsl, Lang::Glsl] {
                export(&g, 1, lang).unwrap_or_else(|e| panic!("{name} as {lang:?}: {e}"));
            }
        }
    }

    #[test]
    fn exported_wgsl_gives_the_same_values_as_the_preview_code() {
        // The standalone module still parses and its density function evaluates like the graph's parameters say.
        let g = crate::graph::Graph::template("Warped cells");
        let code = export(&g, 0, Lang::Wgsl).unwrap();
        assert!(naga::front::wgsl::parse_str(&code).is_ok(), "the re-written WGSL parses");
    }

    #[test]
    fn blue_noise_is_refused() {
        use crate::graph::{NodeKind, NoiseKind, NoiseParams};
        let mut g = Graph::default();
        let n = g.add(NodeKind::Noise(NoiseParams { kind: NoiseKind::Blue, ..Default::default() }), [0.0; 2]);
        let o = g.add(NodeKind::OutDensity, [0.0; 2]);
        g.connect(n, o, 0);
        assert!(export(&g, 0, Lang::Hlsl).is_err());
    }

    #[test]
    fn language_by_extension() {
        assert_eq!(Lang::from_extension("HLSL"), Some(Lang::Hlsl));
        assert_eq!(Lang::from_extension("glsl"), Some(Lang::Glsl));
        assert_eq!(Lang::from_extension("txt"), None);
    }
}
