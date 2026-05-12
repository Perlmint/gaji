use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::rc::Rc;

use anyhow::{Context, Result};
use oxc_allocator::Allocator;
use oxc_codegen::Codegen;
use oxc_parser::Parser;
use oxc_semantic::SemanticBuilder;
use oxc_span::SourceType;
use oxc_transformer::{TransformOptions, Transformer};
use rquickjs::{function::Func, Context as JsContext, Runtime as JsRuntime};

/// Output from a single __gha_build call
#[derive(Debug, Clone)]
pub struct BuildOutput {
    pub id: String,
    pub json: String,
    /// "workflow" or "action"
    pub output_type: String,
}

/// Strip TypeScript types from source code, producing plain JavaScript.
/// Uses the oxc pipeline: Parser -> SemanticBuilder -> Transformer -> Codegen
pub fn strip_typescript(source: &str, filename: &str) -> Result<String> {
    let allocator = Allocator::default();
    let source_type =
        SourceType::from_path(Path::new(filename)).unwrap_or_else(|_| SourceType::tsx());

    let parser_ret = Parser::new(&allocator, source, source_type).parse();
    if !parser_ret.errors.is_empty() {
        let errors: Vec<String> = parser_ret.errors.iter().map(|e| e.to_string()).collect();
        return Err(anyhow::anyhow!("Parse errors:\n{}", errors.join("\n")));
    }

    let mut program = parser_ret.program;

    let semantic_ret = SemanticBuilder::new().build(&program);
    let scoping = semantic_ret.semantic.into_scoping();

    let transform_options = TransformOptions::default();
    let _transformer_ret = Transformer::new(&allocator, Path::new(filename), &transform_options)
        .build_with_scoping(scoping, &mut program);

    let code = Codegen::new().build(&program).code;
    Ok(code)
}

struct LoadedModule {
    script: String,
    deps: Vec<PathBuf>,
}

/// Resolves and preprocesses JS/TS modules for QuickJS evaluation.
///
/// Each module is stored with exports converted to `var` declarations so they
/// persist as globals across separate `eval()` calls. Import lines are stripped
/// (dependencies are tracked and eval'd first instead).
///
/// When a module or any of its dependencies cannot be loaded or preprocessed,
/// it is marked QuickJS N/A (`None`). Callers can detect this via the `bool`
/// return of `load()` and bail out to a fallback executor.
#[derive(Default)]
pub struct ModuleResolver {
    modules: HashMap<PathBuf, Result<LoadedModule, String>>,
}

impl ModuleResolver {
    /// Load a module and its transitive dependencies.
    ///
    /// Returns `Ok(true)` if the module is ready for QuickJS evaluation.
    /// Returns `Ok(false)` if the module (or any dependency) is QuickJS N/A —
    /// i.e., a dependency file doesn't exist on disk, or preprocessing failed.
    pub fn load(&mut self, script_path: &Path) -> Result<bool> {
        let canonicalized_path = script_path
            .canonicalize()
            .context("Failed to canonicalize script path")?;

        match self.modules.get(&canonicalized_path) {
            Some(Err(_)) => return Ok(false),
            Some(Ok(_)) => return Ok(true),
            None => {}
        }

        let script = std::fs::read_to_string(script_path)
            .with_context(|| format!("Failed to read JS: {}", script_path.display()))?;

        let source = if script_path.extension().is_some_and(|e| e == "ts") {
            let filename = script_path
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string();
            match strip_typescript(&script, &filename) {
                Ok(js) => js,
                Err(e) => {
                    let reason = format!("{}: {}", script_path.display(), e);
                    self.modules.insert(canonicalized_path, Err(reason));
                    return Ok(false);
                }
            }
        } else {
            script
        };

        let mut dependencies = Vec::new();
        let mut result = Vec::new();

        for (line_no, line) in source.lines().enumerate() {
            let trimmed = line.trim();

            if trimmed.starts_with("import ") || trimmed.starts_with("import{") {
                if let Some((_, from_part)) = trimmed.split_once(" from ") {
                    let module_path = from_part
                        .trim()
                        .trim_end_matches(';')
                        .trim_matches(|c: char| c.is_whitespace() || c == '\'' || c == '"');
                    let joined = script_path
                        .parent()
                        .unwrap_or(Path::new("."))
                        .join(module_path);
                    let loc = format!("{}:{}", script_path.display(), line_no + 1);
                    match joined.canonicalize() {
                        Ok(resolved_path) => match self.load(&resolved_path)? {
                            true => dependencies.push(resolved_path),
                            false => {
                                let dep_reason = self
                                    .modules
                                    .get(&resolved_path)
                                    .and_then(|r| r.as_ref().err())
                                    .cloned()
                                    .unwrap_or_else(|| resolved_path.display().to_string());
                                let reason = format!("{}: {}", loc, dep_reason);
                                self.modules.insert(canonicalized_path, Err(reason));
                                return Ok(false);
                            }
                        },
                        Err(_) => {
                            let reason = format!("{}: cannot resolve '{}'", loc, module_path);
                            self.modules.insert(canonicalized_path, Err(reason));
                            return Ok(false);
                        }
                    }
                }
                // No 'from' clause (side-effect import) → skip line
                continue;
            }

            if trimmed.starts_with("export {") {
                continue;
            }

            if let Some(rest) = trimmed.strip_prefix("export default ") {
                result.push(rest.to_string());
                continue;
            }

            if let Some(rest) = trimmed.strip_prefix("export async function ") {
                if let Some(paren) = rest.find('(') {
                    let name = rest[..paren].trim();
                    result.push(format!("var {} = async function {}", name, rest));
                } else {
                    result.push(rest.to_string());
                }
                continue;
            }

            if let Some(rest) = trimmed.strip_prefix("export function ") {
                if let Some(paren) = rest.find('(') {
                    let name = rest[..paren].trim();
                    result.push(format!("var {} = function {}", name, rest));
                } else {
                    result.push(rest.to_string());
                }
                continue;
            }

            if let Some(rest) = trimmed.strip_prefix("export class ") {
                let name_end = rest.find([' ', '{']).unwrap_or(rest.len());
                let name = &rest[..name_end];
                let rest_of_line = &rest[name_end..];
                result.push(format!("var {} = class {}{}", name, name, rest_of_line));
                continue;
            }

            if let Some(rest) = trimmed.strip_prefix("export const ") {
                result.push(format!("var {}", rest));
                continue;
            }

            if let Some(rest) = trimmed.strip_prefix("export let ") {
                result.push(format!("var {}", rest));
                continue;
            }

            if let Some(rest) = trimmed.strip_prefix("export var ") {
                result.push(format!("var {}", rest));
                continue;
            }

            // Other export forms (re-exports, etc.) — keep the non-export part
            if let Some(rest) = trimmed.strip_prefix("export ") {
                result.push(rest.to_string());
                continue;
            }

            result.push(line.to_string());
        }

        self.modules.insert(
            canonicalized_path,
            Ok(LoadedModule {
                script: result.join("\n"),
                deps: dependencies,
            }),
        );

        Ok(true)
    }

    /// Like [`load`] but treats unresolvable imports as side-effect imports (skip them)
    /// instead of marking the module N/A. Use for files whose imports are injected
    /// synthetically at eval time (e.g. `defineConfig` for config files).
    ///
    /// Always succeeds unless the file cannot be read or TypeScript stripping fails.
    pub fn load_lenient(&mut self, script_path: &Path) -> Result<()> {
        let canonicalized_path = script_path
            .canonicalize()
            .context("Failed to canonicalize script path")?;

        if matches!(self.modules.get(&canonicalized_path), Some(Ok(_))) {
            return Ok(());
        }

        let script = std::fs::read_to_string(script_path)
            .with_context(|| format!("Failed to read: {}", script_path.display()))?;

        let source = if script_path.extension().is_some_and(|e| e == "ts") {
            let filename = script_path
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string();
            strip_typescript(&script, &filename)
                .with_context(|| format!("Failed to strip TypeScript: {}", script_path.display()))?
        } else {
            script
        };

        let mut dependencies = Vec::new();
        let mut result = Vec::new();

        for line in source.lines() {
            let trimmed = line.trim();

            if trimmed.starts_with("import ") || trimmed.starts_with("import{") {
                if let Some((_, from_part)) = trimmed.split_once(" from ") {
                    let module_path = from_part
                        .trim()
                        .trim_end_matches(';')
                        .trim_matches(|c: char| c.is_whitespace() || c == '\'' || c == '"');
                    let joined = script_path
                        .parent()
                        .unwrap_or(Path::new("."))
                        .join(module_path);
                    if let Ok(resolved_path) = joined.canonicalize() {
                        if self.load(&resolved_path)? {
                            dependencies.push(resolved_path);
                        }
                        // Ok(false) → dep is N/A → skip silently
                    }
                    // Err → unresolvable → skip silently
                }
                continue;
            }

            if trimmed.starts_with("export {") {
                continue;
            }
            if let Some(rest) = trimmed.strip_prefix("export default ") {
                result.push(rest.to_string());
                continue;
            }
            if let Some(rest) = trimmed.strip_prefix("export async function ") {
                if let Some(paren) = rest.find('(') {
                    result.push(format!(
                        "var {} = async function {}",
                        rest[..paren].trim(),
                        rest
                    ));
                } else {
                    result.push(rest.to_string());
                }
                continue;
            }
            if let Some(rest) = trimmed.strip_prefix("export function ") {
                if let Some(paren) = rest.find('(') {
                    result.push(format!("var {} = function {}", rest[..paren].trim(), rest));
                } else {
                    result.push(rest.to_string());
                }
                continue;
            }
            if let Some(rest) = trimmed.strip_prefix("export class ") {
                let name_end = rest.find([' ', '{']).unwrap_or(rest.len());
                let name = &rest[..name_end];
                result.push(format!(
                    "var {} = class {}{}",
                    name,
                    name,
                    &rest[name_end..]
                ));
                continue;
            }
            if let Some(rest) = trimmed.strip_prefix("export const ") {
                result.push(format!("var {}", rest));
                continue;
            }
            if let Some(rest) = trimmed.strip_prefix("export let ") {
                result.push(format!("var {}", rest));
                continue;
            }
            if let Some(rest) = trimmed.strip_prefix("export var ") {
                result.push(format!("var {}", rest));
                continue;
            }
            if let Some(rest) = trimmed.strip_prefix("export ") {
                result.push(rest.to_string());
                continue;
            }

            result.push(line.to_string());
        }

        self.modules.insert(
            canonicalized_path,
            Ok(LoadedModule {
                script: result.join("\n"),
                deps: dependencies,
            }),
        );
        Ok(())
    }

    pub fn get_preprocessed_script(&self, path: &Path) -> Option<&str> {
        let canonical = path.canonicalize().ok()?;
        self.modules
            .get(&canonical)?
            .as_ref()
            .ok()
            .map(|m| m.script.as_str())
    }

    fn na_reason(&self, path: &Path) -> &str {
        path.canonicalize()
            .ok()
            .and_then(|p| self.modules.get(&p))
            .and_then(|r| r.as_ref().err())
            .map(String::as_str)
            .unwrap_or("unresolvable dependency")
    }

    fn collect_ordered(
        &self,
        canonical: &PathBuf,
        visited: &mut HashSet<PathBuf>,
        order: &mut Vec<PathBuf>,
    ) {
        if visited.contains(canonical) {
            return;
        }
        visited.insert(canonical.clone());
        if let Some(Ok(module)) = self.modules.get(canonical) {
            for dep in &module.deps {
                self.collect_ordered(dep, visited, order);
            }
        }
        order.push(canonical.clone());
    }

    /// Evaluate this module and its transitive dependencies in the given QuickJS context.
    ///
    /// Dependencies are eval'd before the module that imports them, so each module's
    /// exported names (converted to `var`) are available as globals for subsequent evals.
    ///
    /// The caller is responsible for pre-configuring `ctx` before this call:
    ///   - Register native Rust callbacks (e.g. `__gha_build`)
    ///   - Eval any JS preamble (e.g. config values, helper functions)
    pub fn execute_module(&self, path: &Path, ctx: &rquickjs::Ctx<'_>) -> Result<()> {
        let canonical = path
            .canonicalize()
            .context("Failed to canonicalize path for execute_module")?;

        match self.modules.get(&canonical) {
            Some(Ok(_)) => {}
            Some(Err(reason)) => {
                return Err(anyhow::anyhow!(
                    "Module {} is not available for QuickJS: {}",
                    path.display(),
                    reason
                ))
            }
            None => return Err(anyhow::anyhow!("Module {} was not loaded", path.display())),
        }

        let mut visited = HashSet::new();
        let mut order = Vec::new();
        self.collect_ordered(&canonical, &mut visited, &mut order);

        for module_path in &order {
            if let Some(Ok(module)) = self.modules.get(module_path) {
                ctx.eval::<(), _>(module.script.as_bytes()).map_err(|e| {
                    anyhow::anyhow!("QuickJS eval error in {}: {}", module_path.display(), e)
                })?;
            }
        }

        Ok(())
    }
}

/// Execute a workflow TypeScript file in QuickJS via module resolution.
///
/// Loads the workflow file and its transitive dependencies (including the runtime).
/// If the workflow or any dependency is QuickJS N/A, returns Err so the caller
/// can fall back to a Node.js-based executor.
///
/// The caller provides a shared `resolver` so common modules (e.g. generated/index.js)
/// are processed only once across multiple workflow builds in a single run.
pub fn execute_workflow(
    resolver: &mut ModuleResolver,
    workflow_path: &Path,
) -> Result<Vec<BuildOutput>> {
    if !resolver.load(workflow_path)? {
        let reason = resolver.na_reason(workflow_path);
        return Err(anyhow::anyhow!(
            "Workflow {} cannot be processed by QuickJS: {}",
            workflow_path.display(),
            reason
        ));
    }

    let outputs: Rc<RefCell<Vec<BuildOutput>>> = Rc::new(RefCell::new(Vec::new()));

    {
        let rt = JsRuntime::new().context("Failed to create QuickJS runtime")?;
        let ctx = JsContext::full(&rt).context("Failed to create QuickJS context")?;

        ctx.with(|ctx| {
            let outputs_clone = outputs.clone();
            let build_fn = Func::from(
                move |id: String, json: String, output_type: rquickjs::function::Opt<String>| {
                    outputs_clone.borrow_mut().push(BuildOutput {
                        id,
                        json,
                        output_type: output_type.0.unwrap_or_else(|| "workflow".to_string()),
                    });
                },
            );
            ctx.globals()
                .set("__gha_build", build_fn)
                .map_err(|e| anyhow::anyhow!("Failed to set __gha_build: {}", e))?;

            resolver.execute_module(workflow_path, &ctx)
        })?;
    }

    let result = Rc::try_unwrap(outputs)
        .map_err(|_| anyhow::anyhow!("Failed to unwrap Rc - references still held"))?
        .into_inner();

    Ok(result)
}

/// Register __gha_build host function and evaluate JavaScript with QuickJS.
/// Uses Rc/RefCell pattern to capture build outputs from JS callbacks.
pub fn execute_js(code: &str) -> Result<Vec<BuildOutput>> {
    let outputs: Rc<RefCell<Vec<BuildOutput>>> = Rc::new(RefCell::new(Vec::new()));

    {
        let rt = JsRuntime::new().context("Failed to create QuickJS runtime")?;
        let ctx = JsContext::full(&rt).context("Failed to create QuickJS context")?;

        let code_owned = code.to_string();

        ctx.with(|ctx| {
            let outputs_clone = outputs.clone();

            // Register __gha_build(id, json, type) host function
            let build_fn = Func::from(
                move |id: String, json: String, output_type: rquickjs::function::Opt<String>| {
                    outputs_clone.borrow_mut().push(BuildOutput {
                        id,
                        json,
                        output_type: output_type.0.unwrap_or_else(|| "workflow".to_string()),
                    });
                },
            );

            ctx.globals()
                .set("__gha_build", build_fn)
                .map_err(|e| anyhow::anyhow!("Failed to set __gha_build: {}", e))?;

            // Evaluate the bundled JavaScript
            ctx.eval::<(), _>(code_owned.as_bytes())
                .map_err(|e| anyhow::anyhow!("QuickJS evaluation error: {}", e))?;

            Ok::<_, anyhow::Error>(())
        })?;

        // ctx and rt are dropped here, releasing the Rc clone held by the Func
    }

    // Extract the outputs - Rc::try_unwrap succeeds because ctx/rt are dropped
    let result = Rc::try_unwrap(outputs)
        .map_err(|_| anyhow::anyhow!("Failed to unwrap Rc - references still held"))?
        .into_inner();

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Strip export/import for tests that use execute_js with a single bundled string.
    /// In single-eval mode class/function/const all work without var conversion.
    fn strip_for_eval(source: &str) -> String {
        let mut result = Vec::new();
        for line in source.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with("import ") || trimmed.starts_with("import{") {
                continue;
            }
            if trimmed.starts_with("export type ") || trimmed.starts_with("export {") {
                continue;
            }
            if trimmed.starts_with("export ") {
                result.push(trimmed.replacen("export ", "", 1));
                continue;
            }
            result.push(line.to_string());
        }
        result.join("\n")
    }

    #[test]
    fn test_strip_typescript_basic() {
        let ts_source = "const x: number = 42;\nconst y: string = \"hello\";";
        let result = strip_typescript(ts_source, "test.ts").unwrap();
        assert!(result.contains("const x = 42"));
        assert!(result.contains("const y = \"hello\""));
        assert!(!result.contains(": number"));
        assert!(!result.contains(": string"));
    }

    #[test]
    fn test_module_export_as_global() {
        let dir = tempfile::tempdir().unwrap();
        let mod_path = dir.path().join("test.js");
        std::fs::write(
            &mod_path,
            "export const x = 42;\nexport function greet() { return \"hello\"; }\n",
        )
        .unwrap();

        let mut resolver = ModuleResolver::default();
        assert!(resolver.load(&mod_path).unwrap());

        let rt = JsRuntime::new().unwrap();
        let ctx = JsContext::full(&rt).unwrap();
        ctx.with(|ctx| {
            resolver.execute_module(&mod_path, &ctx).unwrap();
            let x: i32 = ctx.globals().get("x").unwrap();
            assert_eq!(x, 42);
            let greeting: String = ctx
                .eval("greet()")
                .map_err(|e| anyhow::anyhow!("{}", e))
                .unwrap();
            assert_eq!(greeting, "hello");
        });
    }

    #[test]
    fn test_module_na_on_missing_dep() {
        let dir = tempfile::tempdir().unwrap();
        let mod_path = dir.path().join("test.js");
        std::fs::write(
            &mod_path,
            "import { X } from \"./nonexistent.js\";\nvar y = 1;\n",
        )
        .unwrap();

        let mut resolver = ModuleResolver::default();
        assert!(!resolver.load(&mod_path).unwrap());
    }

    #[test]
    fn test_module_dep_ordering() {
        let dir = tempfile::tempdir().unwrap();
        let dep_path = dir.path().join("dep.js");
        let main_path = dir.path().join("main.js");

        std::fs::write(&dep_path, "export const BASE = 10;\n").unwrap();
        std::fs::write(
            &main_path,
            "import { BASE } from \"./dep.js\";\nexport const VALUE = BASE + 5;\n",
        )
        .unwrap();

        let mut resolver = ModuleResolver::default();
        assert!(resolver.load(&main_path).unwrap());

        let rt = JsRuntime::new().unwrap();
        let ctx = JsContext::full(&rt).unwrap();
        ctx.with(|ctx| {
            resolver.execute_module(&main_path, &ctx).unwrap();
            let value: i32 = ctx.globals().get("VALUE").unwrap();
            assert_eq!(value, 15);
        });
    }

    #[test]
    fn test_execute_js_basic() {
        let code = r#"
            function __test() {
                __gha_build("test-workflow", '{"name":"test","on":{"push":{}},"jobs":{}}', "workflow");
            }
            __test();
        "#;
        let outputs = execute_js(code).unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].id, "test-workflow");
        assert_eq!(outputs[0].output_type, "workflow");
    }

    #[test]
    fn test_execute_js_multiple_outputs() {
        let code = r#"
            __gha_build("wf1", '{"name":"first"}', "workflow");
            __gha_build("wf2", '{"name":"second"}', "workflow");
            __gha_build("act1", '{"name":"action1"}', "action");
        "#;
        let outputs = execute_js(code).unwrap();
        assert_eq!(outputs.len(), 3);
        assert_eq!(outputs[0].output_type, "workflow");
        assert_eq!(outputs[2].output_type, "action");
    }

    /// End-to-end test: runtime JS + Job/Workflow classes → QuickJS → JSON output
    #[test]
    fn test_job_workflow_pipeline() {
        use crate::generator::templates::JOB_WORKFLOW_RUNTIME_TEMPLATE;

        let runtime_js = format!(
            r#"function getAction(ref) {{
    return function(config) {{
        if (config === undefined) config = {{}};
        var step = {{ uses: ref }};
        if (config.name !== undefined) step.name = config.name;
        if (config.with !== undefined) step.with = config.with;
        return step;
    }};
}}
{}"#,
            JOB_WORKFLOW_RUNTIME_TEMPLATE
        );

        let workflow_js = r#"
var checkout = getAction("actions/checkout@v5");

new Workflow({
    name: "CI",
    on: { push: { branches: ["main"] } },
}).jobs(j => j
    .add("build",
        new Job("ubuntu-latest")
            .steps(s => s
                .add(checkout({ name: "Checkout", with: { "fetch-depth": 1 } }))
                .add({ name: "Test", run: "npm test" })
            )
    )
).build("ci");
"#;

        let runtime_stripped = strip_for_eval(&runtime_js);
        let bundled = format!("{}\n\n{}", runtime_stripped, workflow_js);

        let outputs = execute_js(&bundled).unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].id, "ci");
        assert_eq!(outputs[0].output_type, "workflow");

        let json: serde_json::Value = serde_json::from_str(&outputs[0].json).unwrap();
        assert_eq!(json["name"], "CI");
        assert!(json["on"]["push"]["branches"].is_array());
        assert_eq!(json["jobs"]["build"]["runs-on"], "ubuntu-latest");

        let steps = json["jobs"]["build"]["steps"].as_array().unwrap();
        assert_eq!(steps.len(), 2);
        assert_eq!(steps[0]["uses"], "actions/checkout@v5");
        assert_eq!(steps[0]["name"], "Checkout");
        assert_eq!(steps[1]["run"], "npm test");
    }

    /// Test Action (formerly CompositeAction) through QuickJS
    #[test]
    fn test_composite_action_pipeline() {
        use crate::generator::templates::JOB_WORKFLOW_RUNTIME_TEMPLATE;

        let runtime_js = format!(
            "function getAction(ref) {{ return function(config) {{ return {{ uses: ref }}; }}; }}\n{}",
            JOB_WORKFLOW_RUNTIME_TEMPLATE
        );

        let workflow_js = r#"
new Action({
    name: "My Action",
    description: "A composite action",
})
    .steps(s => s
        .add({ name: "Step 1", run: "echo hello", shell: "bash" })
    )
    .build("my-action");
"#;

        let runtime_stripped = strip_for_eval(&runtime_js);
        let bundled = format!("{}\n\n{}", runtime_stripped, workflow_js);

        let outputs = execute_js(&bundled).unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].id, "my-action");
        assert_eq!(outputs[0].output_type, "action");

        let json: serde_json::Value = serde_json::from_str(&outputs[0].json).unwrap();
        assert_eq!(json["name"], "My Action");
        assert_eq!(json["runs"]["using"], "composite");
        assert_eq!(json["runs"]["steps"][0]["run"], "echo hello");
    }

    /// Test full TS→JS strip + QuickJS execution
    #[test]
    fn test_strip_then_execute() {
        use crate::generator::templates::JOB_WORKFLOW_RUNTIME_TEMPLATE;

        let runtime_js = format!(
            "function getAction(ref) {{ return function(config) {{ return {{ uses: ref }}; }}; }}\n{}",
            JOB_WORKFLOW_RUNTIME_TEMPLATE
        );

        // TypeScript source with type annotations
        let ts_source = r#"
const wf: Workflow = new Workflow({
    name: "Typed",
    on: { push: {} },
}).jobs(j => j
    .add("job1",
        new Job("ubuntu-latest")
            .steps(s => s
                .add({ name: "Hello", run: "echo hi" })
            )
    )
);

wf.build("typed-wf");
"#;

        let js = strip_typescript(ts_source, "test.ts").unwrap();

        let runtime_stripped = strip_for_eval(&runtime_js);
        let bundled = format!("{}\n\n{}", runtime_stripped, js);

        let outputs = execute_js(&bundled).unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].id, "typed-wf");

        let json: serde_json::Value = serde_json::from_str(&outputs[0].json).unwrap();
        assert_eq!(json["name"], "Typed");
        assert_eq!(json["jobs"]["job1"]["steps"][0]["run"], "echo hi");
    }
}
