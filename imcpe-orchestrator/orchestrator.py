"""
iMED Pose Estimation orchestrator.

Polls a Synapse evaluation queue, runs each new submission's Docker container
in a sealed environment, validates output, scores against held-out GT, and
writes per-submission status + annotations back to Synapse.

Usage:
    export SYNAPSE_AUTH_TOKEN=eyJ0...
    export IMCPE_QUEUE_ID=9614xxxx
    python orchestrator.py

Stop with Ctrl-C. Safe to restart -- in-progress submissions are picked up by
status, not memory.
"""
from __future__ import annotations

import json
import logging
import shutil
import signal
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import config
from validate_output import validate_submission

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("orchestrator")

SYNAPSE_STRING_ANNO_MAX = 500

class SubStatus:
    PULL_FAILED = "pull_failed"
    TIMEOUT = "timeout"
    CONTAINER_ERROR = "container_error"
    OUTPUT_INVALID = "output_invalid"
    SCORING_ERROR = "scoring_error"
    SCORED = "scored"


@dataclass
class RunResult:
    status: str
    metrics: dict
    error_message: str = ""
    container_log_tail: str = ""


# ----------------------------------------------------------------------------
# Synapse helpers
# ----------------------------------------------------------------------------
def _synapse_login():
    import synapseclient
    if not config.SYNAPSE_AUTH_TOKEN:
        raise RuntimeError("SYNAPSE_AUTH_TOKEN environment variable is not set.")
    syn = synapseclient.Synapse()
    syn.login(authToken=config.SYNAPSE_AUTH_TOKEN)
    return syn


def _list_received(syn, queue_id: int):
    for bundle in syn.getSubmissionBundles(queue_id, status="RECEIVED"):
        submission, status = bundle
        yield submission, status


def _set_status(syn, status_obj, new_status: str, annotations: Optional[dict] = None):
    status_obj.status = new_status
    if annotations:
        ann_list = []
        for key, (value, is_private) in annotations.items():
            ann_list.append({
                "key": key,
                "value": value,
                "isPrivate": bool(is_private),
            })
        status_obj.submissionAnnotations = _format_annotations(ann_list)
    return syn.store(status_obj)


def _format_annotations(ann_list):
    out = {
        "stringAnnos": [],
        "doubleAnnos": [],
        "longAnnos": [],
    }
    for a in ann_list:
        v = a["value"]
        if isinstance(v, bool):
            out["stringAnnos"].append({
                "key": a["key"],
                "value": "true" if v else "false",
                "isPrivate": a["isPrivate"],
            })
        elif isinstance(v, int):
            out["longAnnos"].append({
                "key": a["key"],
                "value": int(v),
                "isPrivate": a["isPrivate"],
            })
        elif isinstance(v, float):
            out["doubleAnnos"].append({
                "key": a["key"],
                "value": float(v),
                "isPrivate": a["isPrivate"],
            })
        else:
            s = str(v)
            if len(s) > SYNAPSE_STRING_ANNO_MAX:
                s = s[: SYNAPSE_STRING_ANNO_MAX - 12] + "...[truncated]"
            out["stringAnnos"].append({
                "key": a["key"],
                "value": s,
                "isPrivate": a["isPrivate"],
            })
    return out


# ----------------------------------------------------------------------------
# Per-submission pipeline
# ----------------------------------------------------------------------------
def _run_container(image_ref: str, work_dir: Path) -> RunResult:
    output_dir = work_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = work_dir / "container.log"

    log.info("Pulling %s", image_ref)
    pull = subprocess.run(
        ["docker", "pull", image_ref],
        capture_output=True, text=True,
    )
    if pull.returncode != 0:
        return RunResult(
            status=SubStatus.PULL_FAILED,
            metrics={},
            error_message=f"docker pull failed: {pull.stderr.strip()[:512]}",
        )

    cmd = [
        "timeout", "--kill-after=30s", str(config.TIMEOUT_S),
        "docker", "run", "--rm",
        "--gpus", "all",
        "--network=none",
        f"--memory={config.MEMORY_LIMIT}",
        f"--pids-limit={config.PIDS_LIMIT}",
        "-v", f"{config.TEST_DATA_ROOT}:/input:ro",
        "-v", f"{output_dir}:/output",
        image_ref,
    ]
    log.info("Running container (timeout=%ds)", config.TIMEOUT_S)

    with log_path.open("wb") as logf:
        proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT)

    log_tail = ""
    if log_path.exists():
        raw = log_path.read_bytes()
        log_tail = raw[-config.STDERR_FORWARD_LIMIT:].decode("utf-8", errors="replace")

    if proc.returncode == 124:
        return RunResult(
            status=SubStatus.TIMEOUT,
            metrics={},
            error_message=f"Container exceeded {config.TIMEOUT_S}s wall-clock budget.",
            container_log_tail=log_tail,
        )
    if proc.returncode != 0:
        return RunResult(
            status=SubStatus.CONTAINER_ERROR,
            metrics={},
            error_message=f"Container exited with code {proc.returncode}.",
            container_log_tail=log_tail,
        )

    return RunResult(
        status="container_ok",
        metrics={},
        container_log_tail=log_tail,
    )


def _validate(work_dir: Path):
    return validate_submission(work_dir / "output", config.TEST_DATA_ROOT)


def _score(work_dir: Path) -> dict:
    metrics_path = work_dir / "metrics.json"
    cmd = [
        *config.HOST_PYTHON.split(),
        str(config.EVALUATE_SCRIPT),
        "--gt-root", str(config.GT_ROOT),
        "--pred-root", str(work_dir / "output"),
        "--json-out", str(metrics_path),
    ]
    log.info("Scoring: %s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Scoring failed (exit {proc.returncode}): {proc.stderr.strip()[:512]}"
        )
    return json.loads(metrics_path.read_text())


def process_submission(syn, submission, status_obj) -> None:
    sub_id = submission.id

    # Claim the submission FIRST so any crash below doesn't loop forever.
    # This transitions RECEIVED -> EVALUATION_IN_PROGRESS so subsequent
    # polls won't pick it up again.
    status_obj.status = "EVALUATION_IN_PROGRESS"
    status_obj = syn.store(status_obj)

    work_dir = config.WORK_ROOT / f"sub_{sub_id}"
    try:
        if work_dir.exists():
            shutil.rmtree(work_dir)
        work_dir.mkdir(parents=True)
    except OSError as e:
        log.error("[sub %s] cannot create work dir: %s", sub_id, e)
        _set_status(syn, status_obj, "INVALID", {
            "submission_status": (SubStatus.SCORING_ERROR, False),
            "error_message": (f"Orchestrator setup error: {e}", True),
        })
        return

    image_ref = submission.dockerRepositoryName + "@" + submission.dockerDigest
    log.info("[sub %s] image=%s", sub_id, image_ref)

    annotations: dict[str, tuple[object, bool]] = {}

    # 1. Run the container.
    run_result = _run_container(image_ref, work_dir)
    if run_result.status != "container_ok":
        log.warning("[sub %s] %s: %s", sub_id, run_result.status, run_result.error_message)
        annotations["submission_status"] = (run_result.status, False)
        annotations["error_message"] = (run_result.error_message, True)
        if run_result.container_log_tail:
            annotations["container_log_tail"] = (run_result.container_log_tail, True)
        _set_status(syn, status_obj, "INVALID", annotations)
        return

    # 2. Validate output.
    ok, errs = _validate(work_dir)
    if not ok:
        msg = "; ".join(errs[:10])
        if len(errs) > 10:
            msg += f" (+{len(errs) - 10} more)"
        log.warning("[sub %s] output_invalid: %s", sub_id, msg)
        annotations["submission_status"] = (SubStatus.OUTPUT_INVALID, False)
        annotations["error_message"] = (msg, True)
        _set_status(syn, status_obj, "INVALID", annotations)
        return

    # 3. Score.
    try:
        metrics = _score(work_dir)
    except Exception as e:
        log.exception("[sub %s] scoring_error", sub_id)
        annotations["submission_status"] = (SubStatus.SCORING_ERROR, False)
        annotations["error_message"] = (str(e)[:512], True)
        _set_status(syn, status_obj, "INVALID", annotations)
        return

    # 4. Build annotations.
    agg = metrics["aggregate"]
    annotations["submission_status"] = (SubStatus.SCORED, False)
    annotations["mean_ate_rmse_mm"] = (
        float(agg["mean_ate_rmse_mm"]) if agg["mean_ate_rmse_mm"] == agg["mean_ate_rmse_mm"]
        else float("inf"),
        False,
    )
    annotations["mean_registered_pct"] = (float(agg["mean_registered_pct"]), False)
    annotations["n_scored_sequences"] = (int(agg["n_scored_sequences"]), False)
    annotations["n_sequences"] = (int(agg["n_sequences"]), False)

    for seq_name, m in metrics["per_sequence"].items():
        rmse = m["ate_rmse_mm"]
        annotations[f"seq__{seq_name}__ate_rmse_mm"] = (
            float(rmse) if rmse == rmse else float("inf"),
            True,
        )
        annotations[f"seq__{seq_name}__registered_pct"] = (
            float(m["registered_pct"]),
            True,
        )

    annotations = {
        k: (v, k not in config.PUBLIC_ANNOTATIONS or is_private)
        for k, (v, is_private) in annotations.items()
    }

    log.info(
        "[sub %s] SCORED: ATE=%.4fmm reg=%.2f%% (%d/%d seqs)",
        sub_id,
        agg["mean_ate_rmse_mm"],
        agg["mean_registered_pct"],
        agg["n_scored_sequences"],
        agg["n_sequences"],
    )
    _set_status(syn, status_obj, "SCORED", annotations)


# ----------------------------------------------------------------------------
# Polling loop
# ----------------------------------------------------------------------------
_stop = False


def _handle_signal(signum, frame):
    global _stop
    log.info("Signal %s received, stopping after current submission.", signum)
    _stop = True


def main():
    if config.EVAL_QUEUE_ID == 0:
        sys.exit("Set IMCPE_QUEUE_ID environment variable.")
    if not config.TEST_DATA_ROOT.is_dir():
        sys.exit(f"TEST_DATA_ROOT does not exist: {config.TEST_DATA_ROOT}")
    if not config.GT_ROOT.is_dir():
        sys.exit(f"GT_ROOT does not exist: {config.GT_ROOT}")
    config.WORK_ROOT.mkdir(parents=True, exist_ok=True)

    leaked = list(config.TEST_DATA_ROOT.rglob("pose.txt"))
    if leaked:
        sys.exit(
            f"REFUSING TO START: {len(leaked)} pose.txt file(s) found under "
            f"TEST_DATA_ROOT. These would be visible to participant containers. "
            f"Move them to GT_ROOT first. First few: {leaked[:3]}"
        )

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    syn = _synapse_login()
    log.info("Polling queue %s every %ds.", config.EVAL_QUEUE_ID, config.POLL_INTERVAL_S)

    while not _stop:
        try:
            for submission, status_obj in _list_received(syn, config.EVAL_QUEUE_ID):
                if _stop:
                    break
                try:
                    process_submission(syn, submission, status_obj)
                except Exception:
                    log.exception("Unhandled error processing submission %s", submission.id)
        except KeyboardInterrupt:
            break
        except Exception:
            log.exception("Polling error; backing off 30s.")
            time.sleep(30)
            continue

        # ALWAYS sleep between polls. Without this, transient errors (auth,
        # network, etc.) on the iteration above cause a tight retry loop that
        # hammers Synapse.
        time.sleep(config.POLL_INTERVAL_S)

    log.info("Stopped.")


if __name__ == "__main__":
    main()