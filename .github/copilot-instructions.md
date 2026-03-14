# GitHub Copilot Instructions

These instructions are strict. Generated code and documentation must follow them.

## Documentation Source of Truth

Keep this file for normative rules only.

- Use this file for must and must-not guidance covering architecture boundaries, typing, testing, documentation, and quality gates.
- Use `docs/architecture_module_map.md` as the architecture source of truth for package boundaries, entry points, ownership, and architectural guardrails. In this repository, that document serves the role that some projects call `docs/copilot_architecture.md`.
- Use `README.md` and `docs/quickstart.md` for user-facing setup, run, and demo instructions.

When adding or changing functionality:

- Update this file only if a rule changes.
- Update `docs/architecture_module_map.md` if package structure, module ownership, boundaries, or public entry points change.
- Update `README.md` or `docs/quickstart.md` if setup, runtime behavior, or user workflows change.
- Do not duplicate long package maps or architecture prose in this file.

## Documentation Coverage and Maintenance

Goal: no undocumented public surface, with minimal maintenance overhead.

### Required Documentation Coverage

- Every new or modified module must include a top-level docstring.
- Module docstrings must explain:
  - purpose,
  - where the module fits in the package architecture,
  - key inputs and outputs,
  - non-obvious implementation decisions,
  - what the module does not do.
- If the module is a common integration point or entry point, include a minimal working example or short usage note.
- Every public function, class, and method must include a Google-style docstring.
- Every public dataclass must document its fields in an `Attributes:` section.
- Every package `__init__.py` should include a short package purpose and curated `__all__` exports where useful.
- Every runnable script or entry point, including CLI modules, server modules, and app launchers, must include a module docstring and either a short usage section or a clear pointer to `README.md` or `docs/quickstart.md`.

### Minimum Documentation Gate

A change is incomplete if any of the following are true:

- A new public symbol has no docstring.
- A touched module has no top-level docstring or its docstring no longer reflects behavior.
- A public dataclass, request model, or response model changed without updating its documented fields.
- A public package surface or architectural boundary changed without updating `docs/architecture_module_map.md`.

## Architecture and Design Rules

- Preserve the current package split documented in `docs/architecture_module_map.md`.
- Keep ingestion, graph construction, retrieval, agent orchestration, evaluation, observability, and app surface concerns separated.
- Prefer explicit data flow between modules over hidden global state.
- Prefer deterministic heuristics and typed helpers over opaque shortcuts when both are viable.
- Keep retrieval code focused on evidence collection and ranking; answer generation belongs in the agent or app layer.
- Keep graph-specific traversal and ontology logic in graph or retrieval layers, not in UI code.
- Do not add cross-module coupling that bypasses documented entry points without updating the architecture doc.

## Language and Style

- Target Python 3.13+.
- Use native typing only: `list[str]`, `dict[str, float]`, `X | None`.
- Do not use `typing.List`, `Optional`, or `Union` in new code.
- Use `from __future__ import annotations` in new Python modules.
- Prefer clarity over cleverness.
- Avoid one-letter variable names except in very local, obvious contexts.
- Use logging instead of `print` in library code.

### Docstrings

- All public functions, classes, and methods require Google-style docstrings.
- Document data shape expectations when returning structured payloads, especially citation metadata, graph payloads, evaluation outputs, and API response models.
- For modules that are likely to be opened in isolation, optimize the first 30 seconds: the docstring should let a reader understand the module without chasing multiple files.

## Code Quality and Linting

This repository uses Ruff and tests in CI. All generated code must be clean in edited files.

### Must Avoid

- Unused variables.
- Unused imports.
- Imports outside the top of the file unless there is a clear lazy-import reason.
- Bare `except:` blocks.
- Shadowing builtins.
- Commented-out dead code.

### Formatting Rules

- Max line length: 130.
- Organize imports: standard library, third-party, local.
- Keep public APIs and naming stable unless the task explicitly requires a breaking change.

## Pylance and Typing Quality Gate

- Generated or modified Python code must be Pylance-clean in edited files.
- Prefer fixing root typing issues with narrowing, typed helpers, and explicit return types over suppressions.
- Avoid blanket `# type: ignore` usage. If a suppression is unavoidable, keep it narrow and justified.

## Testing Expectations

- Add or update tests when behavior changes.
- Prefer focused unit tests near the changed area over broad incidental test churn.
- Do not rewrite unrelated tests to fit new code if the production change can be made compatible.
- If behavior cannot be tested in the current environment, state that clearly in the final response.