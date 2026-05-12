"""
Orchestrator configuration. Edit values to match your workstation.

Sensitive values (Synapse PAT) come from environment variables, never hardcoded.
"""
from __future__ import annotations

import os
from pathlib import Path

# --- Paths on this workstation ----------------------------------------------

# Read-only test data mounted into every participant container at /input.
# IMPORTANT: must NOT contain any pose.txt files (those live in GT_ROOT).
TEST_DATA_ROOT = Path("/srv/imedpe/test")

# Ground-truth pose.txt files, one per sequence directory. Never mounted into
# any container -- only read by the scoring step that runs on the host.
GT_ROOT = Path("/home/juseonghan/imed/sample_data/iMED_pe/test")

# Workspace for per-submission scratch dirs (output mounts, logs, metrics JSON).
# Will be created if missing.
WORK_ROOT = Path("/srv/imedpe/orchestrator_work")

# Path to the host-side evaluate_ate.py script (run outside the container).
EVALUATE_SCRIPT = Path("/home/juseonghan/imed/imedpe/scripts/evaluate_ate.py")

# Python interpreter to use for the host-side scoring step.
HOST_PYTHON = "/usr/bin/env python3"

# --- Synapse settings -------------------------------------------------------

# Numeric evaluation queue ID, e.g. 9614xxxx. Set after creating the queue.
EVAL_QUEUE_ID = int(os.environ.get("IMCPE_QUEUE_ID", "0"))

# Synapse personal access token. NEVER hardcode -- export before running:
#   export SYNAPSE_AUTH_TOKEN=eyJ0...
SYNAPSE_AUTH_TOKEN = os.environ.get("SYNAPSE_AUTH_TOKEN", "")

# Poll interval (seconds) between queue checks.
POLL_INTERVAL_S = 60

# --- Container runtime constraints ------------------------------------------

# Wall-clock cap per submission, including container startup. Slightly over
# the 10-min budget we promise participants.
TIMEOUT_S = 660

STDERR_FORWARD_LIMIT = 480

# Hard memory cap inside the container.
MEMORY_LIMIT = "20g"

# Process count cap (defends against fork bombs).
PIDS_LIMIT = 512

# --- Result handling --------------------------------------------------------

# How much stderr to forward to participants on failure (bytes). Cap prevents
# a malicious container from exfiltrating data via the log channel.
STDERR_FORWARD_LIMIT = 8192

# Annotations exposed publicly on the leaderboard. All others are private
# (submitter-team-only).
PUBLIC_ANNOTATIONS = {
    "mean_ate_rmse_mm",
    "mean_registered_pct",
    "n_scored_sequences",
    "submission_status",
}