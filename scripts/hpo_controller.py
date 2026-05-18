"""
HPO controller: clones the registered `HPO base - OOS threshold sweep` task
five times, each with a different `OOS_RERANK_THRESHOLD` value, and tracks the
composite optimisation objective.

Prerequisites (one-time, in order):
  1. python scripts/hpo_subset.py             (build the 30-question subset)
  2. python scripts/hpo_base_task.py          (register the base task in ClearML)
  3. clearml-agent daemon --queue default     (in a separate terminal, foreground)

Then run this:
  python scripts/hpo_controller.py

The controller spawns children one at a time (Ollama on a single Mac is serial,
so concurrency > 1 just wastes resources). Each child takes ~22 minutes on the
30-question subset. Total wall clock: ~1h 50m for 5 children.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLEARML_PROJECT = "ISM-CyberRAG"
BASE_TASK_NAME = "HPO base - OOS threshold sweep"
CLEARML_TAGS = ["hpo-sweep"]
HPO_OUT_DIR = PROJECT_ROOT / "evaluations" / "sprint-3" / "hpo"


def _controller_task_name() -> str:
    """Timestamped controller task name so successive sweeps are distinguishable."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    return f"HPO sweep - OOS threshold - {ts}"

THRESHOLD_VALUES = [-7.0, -6.0, -5.0, -4.0, -3.0]
DEFAULT_TIME_LIMIT_PER_JOB_S = 45 * 60      # 45 minutes per child
DEFAULT_OVERALL_TIME_LIMIT_S = 4 * 3600     # 4 hours total
HEARTBEAT_INTERVAL_S = 60
STUCK_WARN_AFTER_S = 600


sys.path.insert(0, str(PROJECT_ROOT))


def _pre_flight_clearml() -> bool:
    keys = ["CLEARML_API_ACCESS_KEY", "CLEARML_API_SECRET_KEY"]
    missing = [k for k in keys if not os.getenv(k)]
    if missing:
        print(
            f"ERROR: ClearML credentials missing: {missing}\n"
            f"       Set them in your shell env or in ism-cyberrag/.env",
            file=sys.stderr,
        )
        return False
    return True


def _find_base_task(Task):
    # Task.get_task with name + project returns the most recent matching task.
    try:
        chosen = Task.get_task(
            project_name=CLEARML_PROJECT,
            task_name=BASE_TASK_NAME,
        )
    except Exception as exc:
        print(
            f"ERROR: failed to query ClearML for the base task: {exc}\n"
            f"       Check your CLEARML_API_* credentials.",
            file=sys.stderr,
        )
        return None

    if chosen is None:
        print(
            f"ERROR: no base task named '{BASE_TASK_NAME}' found in project "
            f"'{CLEARML_PROJECT}'.\n"
            f"       Run `python scripts/hpo_base_task.py` once first to register it.",
            file=sys.stderr,
        )
        return None

    print(f"Using base task id={chosen.id}.")
    return chosen


def _summarise(optimizer, controller_task) -> dict[str, Any]:
    """Return a JSON-serialisable summary of the run."""
    summary: dict[str, Any] = {
        "controller_task_id": controller_task.id,
        "controller_task_url": controller_task.get_output_log_web_page(),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "top_tasks": [],
    }
    try:
        top = optimizer.get_top_experiments(top_k=len(THRESHOLD_VALUES))
    except Exception as exc:
        summary["warning"] = f"get_top_experiments failed: {exc}"
        return summary

    for t in top:
        params = {}
        try:
            params = t.get_parameters() or {}
        except Exception:
            pass
        threshold = params.get("General/OOS_RERANK_THRESHOLD") or params.get(
            "OOS_RERANK_THRESHOLD"
        )
        scalars = {}
        try:
            scalars = t.get_last_scalar_metrics() or {}
        except Exception:
            pass
        composite = None
        try:
            composite = scalars.get("composite", {}).get("score", {}).get("last")
        except AttributeError:
            pass
        summary["top_tasks"].append({
            "task_id": t.id,
            "threshold": threshold,
            "composite_score": composite,
            "status": getattr(t, "status", None),
        })
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--queue",
        default="default",
        help="ClearML queue children are dispatched to. A `clearml-agent` must be listening on this queue.",
    )
    parser.add_argument(
        "--time-limit-per-job",
        type=int,
        default=DEFAULT_TIME_LIMIT_PER_JOB_S,
        help="Hard timeout per child task in seconds.",
    )
    parser.add_argument(
        "--overall-time-limit",
        type=int,
        default=DEFAULT_OVERALL_TIME_LIMIT_S,
        help="Hard timeout for the whole sweep in seconds.",
    )
    args = parser.parse_args()

    if not _pre_flight_clearml():
        return 1

    try:
        from clearml import Task
        from clearml.automation import (
            DiscreteParameterRange,
            HyperParameterOptimizer,
        )
        from clearml.automation.optimization import GridSearch
    except ImportError as exc:
        print(
            f"ERROR: clearml is not installed or too old: {exc}\n"
            f"       Run `pip install -U clearml`.",
            file=sys.stderr,
        )
        return 1

    base_task = _find_base_task(Task)
    if base_task is None:
        return 1

    controller_task = None
    optimizer = None
    interrupted = False

    def _on_sigint(signum, frame):
        nonlocal interrupted
        interrupted = True
        print("\nKeyboardInterrupt received. Stopping optimizer cleanly...", file=sys.stderr)

    signal.signal(signal.SIGINT, _on_sigint)

    try:
        controller_task = Task.init(
            project_name=CLEARML_PROJECT,
            task_name=_controller_task_name(),
            tags=CLEARML_TAGS,
            task_type=Task.TaskTypes.optimizer,
            reuse_last_task_id=False,
        )

        controller_task.connect({
            "base_task_id": base_task.id,
            "threshold_values": THRESHOLD_VALUES,
            "execution_queue": args.queue,
            "time_limit_per_job_s": args.time_limit_per_job,
            "overall_time_limit_s": args.overall_time_limit,
        })

        optimizer = HyperParameterOptimizer(
            base_task_id=base_task.id,
            hyper_parameters=[
                DiscreteParameterRange(
                    "General/OOS_RERANK_THRESHOLD",
                    values=THRESHOLD_VALUES,
                ),
            ],
            objective_metric_title="composite",
            objective_metric_series="score",
            objective_metric_sign="max",
            max_number_of_concurrent_tasks=1,
            optimizer_class=GridSearch,
            execution_queue=args.queue,
            total_max_jobs=len(THRESHOLD_VALUES),
            time_limit_per_job=args.time_limit_per_job,
            compute_time_limit=args.overall_time_limit,
            save_top_k_tasks_only=len(THRESHOLD_VALUES),
        )

        print(f"\nStarting optimizer. Children dispatched to queue '{args.queue}'.")
        print(f"Threshold values: {THRESHOLD_VALUES}")
        print(f"Make sure `clearml-agent daemon --queue {args.queue}` is running somewhere.\n")

        optimizer.start_locally()

        start = time.time()
        last_heartbeat = 0.0
        last_status_change = time.time()
        last_completed = 0

        while True:
            if interrupted:
                break
            if not optimizer.is_active():
                break
            elapsed = time.time() - start
            if elapsed > args.overall_time_limit:
                print(
                    f"WARN: overall time limit ({args.overall_time_limit}s) exceeded. Stopping.",
                    file=sys.stderr,
                )
                break
            if time.time() - last_heartbeat >= HEARTBEAT_INTERVAL_S:
                last_heartbeat = time.time()
                try:
                    top = optimizer.get_top_experiments(top_k=len(THRESHOLD_VALUES))
                    completed = sum(1 for t in top if getattr(t, "status", "") == "completed")
                    in_progress = sum(1 for t in top if getattr(t, "status", "") == "in_progress")
                    failed = sum(1 for t in top if getattr(t, "status", "") in {"failed", "aborted"})
                    print(
                        f"[{int(elapsed):>5}s] completed={completed} in_progress={in_progress} "
                        f"failed={failed} (target={len(THRESHOLD_VALUES)})"
                    )
                    if completed != last_completed:
                        last_status_change = time.time()
                        last_completed = completed
                    elif time.time() - last_status_change > STUCK_WARN_AFTER_S and in_progress > 0:
                        print(
                            f"WARN: no progress for {STUCK_WARN_AFTER_S}s while a task is in_progress. "
                            f"Check Ollama and the agent terminal.",
                            file=sys.stderr,
                        )
                        last_status_change = time.time()  # don't spam
                except Exception as exc:
                    print(f"WARN: heartbeat query failed: {exc}")
            time.sleep(5)

        # Summary
        summary = _summarise(optimizer, controller_task)
        print("\n=== Final summary ===")
        print(json.dumps(summary, indent=2, default=str))

        run_dir = HPO_OUT_DIR / f"run_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        run_dir.mkdir(parents=True, exist_ok=True)
        summary_path = run_dir / "summary.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"Summary written to {summary_path.relative_to(PROJECT_ROOT)}")

        try:
            controller_task.upload_artifact("summary", artifact_object=summary)
        except Exception as exc:
            print(f"WARN: failed to upload summary artifact: {exc}")

        return 0 if not interrupted else 130

    except Exception as exc:
        print(f"\nUNHANDLED ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1
    finally:
        if optimizer is not None:
            try:
                optimizer.stop()
            except Exception:
                pass
        if controller_task is not None:
            try:
                controller_task.close()
            except Exception:
                pass


if __name__ == "__main__":
    sys.exit(main())
