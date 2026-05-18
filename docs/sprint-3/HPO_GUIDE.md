# HPO trial run (uncommitted)

A short run-book for the optional ClearML `HyperParameterOptimizer` trial over
`OOS_RERANK_THRESHOLD`. Local-only. Nothing committed. Children run serially
on Ollama. Total wall-clock ~2 hours.

Decision criteria after the run live in
`team/assignments/submissions/may-24/HPO-OPTIONS.md`. This guide just covers
how to drive the scripts.

## What gets created

```
ism-cyberrag/
  scripts/
    hpo_subset.py
    hpo_base_task.py
    hpo_controller.py
  evaluations/sprint-3/hpo/
    hpo_subset.json                          (built by hpo_subset.py; deterministic, seed=42)
    results/                                 (one CSV + one JSON per child run)
      thrm5_00_<utc-ts>.csv                  per-question RAGAS scores for threshold=-5.0
      thrm5_00_<utc-ts>_metrics.json         aggregate metrics + composite for that run
      ...                                    one pair per swept threshold value
    run_<utc-ts>/summary.json                (written at the end of the controller run; top-k summary)
  docs/sprint-3/
    HPO_GUIDE.md                             (this file)
```

The `results/` files are written by `hpo_base_task.py` itself so the docs and
reports can be assembled from local files alone, without pulling artifacts
from ClearML. Filenames encode the threshold (`thrm5_00` = -5.00) and the run
timestamp.

ClearML side: tasks land in the existing `ISM-CyberRAG` project, all tagged
`hpo-sweep` so they're easy to filter from the headline Sprint 3 tasks. One
controller task (timestamped name so you can tell sweeps apart) plus five
child tasks. Archive the lot from the UI if the run is bad.

The base task script also forces `LLM_PROVIDER=ollama` inside its own process
via `os.environ.setdefault`, so the generation step uses local Ollama even
though your `.env` keeps `LLM_PROVIDER=groq` for the production app. No shell
export needed.

## Prerequisites checklist

Run these in the `ism-cyberrag` directory.

```bash
# 1. clearml-agent must be installed
pip show clearml-agent | head -2
#    Missing? pip install clearml-agent

# 2. llama3.1:8b must be pulled in Ollama
ollama list | grep llama3.1:8b
#    Missing? ollama pull llama3.1:8b

# 3. .env must have ClearML credentials
grep CLEARML_API_ACCESS_KEY .env
#    Missing? Add CLEARML_API_ACCESS_KEY, CLEARML_API_SECRET_KEY,
#             CLEARML_API_HOST, CLEARML_WEB_HOST, CLEARML_FILES_HOST.
#             Get these from the ClearML web UI: Profile -> Workspace -> Create new credentials.

# 4. Supabase + Ollama config must already work (they do if Sprint 3 eval runs)
grep -E "SUPABASE_URL|OLLAMA_BASE_URL" .env
```

## Loading ClearML credentials

No `clearml-agent init` or `~/clearml.conf` needed. ClearML reads its
credentials from environment variables (`CLEARML_API_ACCESS_KEY`,
`CLEARML_API_SECRET_KEY`, `CLEARML_API_HOST`, `CLEARML_WEB_HOST`,
`CLEARML_FILES_HOST`). Python scripts pick these up automatically via
`src/config.py` and python-dotenv.

The `clearml-agent` CLI is a separate process and does not auto-load `.env`,
so before running it (step 4) you need to export the vars into your shell:

```bash
set -a && source .env && set +a
```

Do this once per terminal session. After that, `clearml-agent daemon` works
without any extra setup.

## Step 1 - build the subset (~1 second)

```bash
python scripts/hpo_subset.py
```

Writes `evaluations/sprint-3/hpo/hpo_subset.json` (30 questions, stratified,
seed=42). Re-runnable; use `--force` to overwrite.

## Step 2 - register the base task (~22 minutes)

```bash
python scripts/hpo_base_task.py --dry-run    # sanity, no API calls
python scripts/hpo_base_task.py              # one full run with default threshold (-5.0)
```

This creates a regular ClearML task in `ISM-CyberRAG` named
`HPO base - OOS threshold sweep`. The controller looks it up by name. Without
this step, the controller exits with a clear error.

After it finishes, open the ClearML UI and confirm:
- The task is in the `ISM-CyberRAG` project, named `HPO base - OOS threshold sweep`.
- Tag is `hpo-sweep`.
- A scalar `composite/score` is reported.
- The artifact `eval_results` is attached.

The base task name is intentionally static. The controller looks it up by name
and picks the most recent match, so re-registering (re-running this step)
just creates a new latest base task; old ones become harmless duplicates you
can archive at leisure.

## Step 3 - start a worker (separate terminal)

```bash
source .venv/bin/activate
set -a && source .env && set +a
clearml-agent daemon --queue default --foreground
```

Leave this running. It prints "ClearML Worker ready" and then waits. Each
spawned child will appear here as `Running task <id>`.

## Step 4 - run the controller (~1h 50m)

```bash
python scripts/hpo_controller.py
```

The controller dispatches five children to the `default` queue, one at a
time. Heartbeat lines every 60 seconds print `completed / in_progress /
failed` counts.

When all five are done, the controller prints a JSON summary and writes
`evaluations/sprint-3/hpo/run_<utc-ts>/summary.json`. The controller task in
the ClearML UI is named `HPO sweep - OOS threshold - <UTC timestamp>` so
multiple sweeps stay distinguishable without overwriting each other.

Interrupt at any point with Ctrl+C - the controller stops the optimizer
cleanly and closes the task.

## Step 5 - inspect

Open the ClearML web UI, filter by tag `hpo-sweep`:

- The controller task (type `optimizer`) has:
  - **PLOTS** tab: optimisation-objective scatter plot (5 dots), parallel
    coordinates.
  - **ARTIFACTS** tab: `summary` artifact.
- Each child task has:
  - **SCALARS** tab: composite score, all RAGAS metrics, diagnostics.
  - **ARTIFACTS** tab: `eval_results.csv` (per-question RAGAS scores) and
    `subset_used.json`.

Take three screenshots and drop them into `evaluations/sprint-3/hpo/`:

- `hpo_optimization_objective.png`
- `hpo_parallel_coordinates.png`
- `hpo_summary_table.png`

## Step 6 - decide

| Outcome | Action |
| --- | --- |
| Our `-5.0` wins (or ties within 0.01) | Cite as confirmation. Embed the three screenshots in the completion report + release note. |
| A different value wins by more than 0.02 | Either: (a) update `OOS_RERANK_THRESHOLD` in `src/config.py` and re-run the full 100-question eval, OR (b) keep `-5.0` with the calibration argument and report HPO as exploration. Both are honest. |
| One or more children failed | Archive every task in this run (right-click -> Archive). Existing manual-grid story stands; no doc changes. |

## Failure cookbook

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `subset file missing` | Step 1 not run | `python scripts/hpo_subset.py` |
| `no base task named ... found` | Step 2 not run yet | `python scripts/hpo_base_task.py` |
| `Ollama unreachable` | Ollama not running | `ollama serve &` (or open the desktop app) |
| `model llama3.1:8b not pulled` | Model not local | `ollama pull llama3.1:8b` |
| Controller starts but no child progresses | No `clearml-agent` listening on the queue | Open the agent terminal from Step 3 |
| Heartbeat warns `no progress for 600s` | Ollama hanging on one question | Check the agent terminal; if generation has stalled, Ctrl+C the agent, restart Ollama, restart the agent |
| `ragas_failed` scalar = 1.0 in a child | RAGAS judge call failed | Confirm `EVAL_LLM_PROVIDER=ollama` and `EVAL_LLM_MODEL=llama3.1:8b` in `.env` |
| Child reports `composite = -1.0` | More than 50% of questions errored, or no OOS questions in subset | Inspect the child's logs in ClearML for the underlying error |

## Cleanup

To wipe everything from this trial:

1. ClearML UI -> filter by tag `hpo-sweep` -> select all -> Archive.
2. `rm -rf evaluations/sprint-3/hpo/run_*` (keep `hpo_subset.json` if you might re-run later).
3. Stop the agent terminal with Ctrl+C.

No source code is committed by these steps. Nothing in the regular project
state is touched.
