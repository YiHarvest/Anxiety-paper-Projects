# HemoZero

HemoZero is a derivation-aware multi-view research system for small-sample anxiety classification
from six measured blood biomarkers and nine deterministic ratios.

## Architecture

![HemoZero end-to-end project workflow](docs/assets/hemozero_project_flow.svg)

```text
Pi Agent                 decides the next admissible stage
  -> Tools               validate parameters and return typed ToolResult objects
    -> src/hemzero       performs data, graph, modelling, fusion, and statistics
      -> artifacts/runs  stores run-scoped evidence and a frozen SHA-256 manifest
```

Core dependencies are managed by uv. NetworkX owns the feature-lineage graph, InterpretML owns EBM,
MLflow records experiment metadata in SQLite, and optional adapters integrate TabPFN, TabDistill,
PySR, and TaskWeaver. Backend identity is enforced: unavailable TabPFN cannot be silently replaced
by another estimator under the TabPFN name.

An optional Knowledge-Constrained Distillation Layer keeps NetworkX as the deterministic lineage
calculator and uses Neo4j only for evidence/provenance persistence. `PolicyEngine`, not Neo4j,
classifies each TabDistill interaction before confirmatory and discovery EBM students are trained.
The editable Graphviz source and detailed architecture notes are available in
[`docs/assets/hemozero_project_flow.dot`](docs/assets/hemozero_project_flow.dot) and
[`docs/PROJECT_OVERVIEW.md`](docs/PROJECT_OVERVIEW.md).

## Setup and smoke test

```bash
uv sync --python 3.12
uv run hemozero smoke
uv run pytest -q
```

Optional backends:

```bash
UV_PROJECT_ENVIRONMENT=.venv-hemozero uv sync \
  --extra foundation --extra symbolic --extra agent --extra distill --python 3.12
```

Add `--extra knowledge-graph` when running the Neo4j workflow.

This project pins `torch==2.9.1+cpu` to the official PyTorch CPU index. For the current 336-row
training set, TabPFN is configured with four estimators, one preprocessing job, and CPU execution.
Local model inference still requires one-time Prior Labs license acceptance. Set the key only in
your shell (do not commit it):

```bash
export TABPFN_TOKEN="<key copied after license acceptance>"
.venv-hemozero/bin/python scripts/verify_optional_backends.py --tabpfn-fit
```

Alternatively, fill the root `.env`, then load it before starting TaskWeaver:

```bash
set -a
source .env
set +a
.venv-hemozero/bin/taskweaver -p taskweaver_project chat
```

PySR uses Julia 1.10.11 at `~/.local/bin/julia`; `pysr_student.configure_julia()` selects it without
requiring a global PATH change. A real backend smoke result is stored under
`artifacts/backend-verification/pysr-smoke/`. It uses actual baseline OOF probabilities only to
verify SymbolicRegression.jl and is not the final HemoZero score.

TaskWeaver is pinned to an official repository commit and initialized in `taskweaver_project/`.
The `hemozero_tool` plugin validates run paths and exposes the deterministic HemoZero registry.
An LLM-backed chat additionally requires a provider/model key in TaskWeaver's local configuration.

Configure the local `Clouddelta/tab-distill` checkout in `configs/models.yaml`. The adapter runs its
SPEX implementation in a separate process because that repository exposes a top-level package named
`src`, which would otherwise collide with this project's src layout.

## Main directories

- `pi_agent/`: policy, state, tool registry, and evidence-only reader.
- `tools/`: thin validated interfaces; scientific calculations do not live here.
- `src/hemzero/`: pure algorithm and statistics modules.
- `configs/`: dataset, formula, model, experiment, workflow, and audit SSOT.
- `artifacts/runs/<run_id>/`: immutable evidence after `_FROZEN` is created.
- `scripts/`: full run, smoke test, and isolated TabDistill worker.
- `taskweaver_project/`: initialized TaskWeaver application and HemoZero tool bridge.
- `src/hemzero/knowledge_graph/`: Neo4j schema, lineage sync, evidence, policy inputs, snapshots and audit.

Only the layered `hemzero` implementation is retained. Historical proxy-model code and outputs have
been removed to prevent accidental reuse or confusion with the real TabPFN workflow.

## Complete primary experiment

The recommended primary analysis uses the leak-free files under
`dataset/processed/hemozero_kg_20260719/`. The current `configs/dataset.yaml` intentionally points
to the 336-row raw sensitivity input and records authorized exact feature-vector overlap; it must not
be interpreted as an independent external-validation setup. Both profiles use 5 repeated outer CV
runs × 5 folds, four TabPFN ensemble members, fold-local multi-index TabDistill, repeated EBM
students and 100-iteration PySR searches.

```bash
.venv-hemozero/bin/python scripts/run_complete_experiment.py
```

Resume an interrupted run without reusing completed stages:

```bash
.venv-hemozero/bin/python scripts/run_complete_experiment.py --resume <run_id>
```

## Knowledge-constrained experiment

Configure `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`, and optionally `NEO4J_DATABASE` in the
shell or local `.env`. External evidence is never fetched automatically: populate
`configs/evidence_registry.yaml` only with manually reviewed assertions. Then run:

```bash
.venv-hemozero/bin/python scripts/run_kg_experiment.py --preflight-only
.venv-hemozero/bin/python scripts/run_kg_experiment.py
```

The KG workflow writes a portable snapshot under `artifacts/runs/<run_id>/knowledge_graph/` before
the existing run-level SHA-256 manifest is created. TaskWeaver is restricted to five read-only
queries over audit-passed, frozen snapshots and cannot invoke graph writers or change decisions.
