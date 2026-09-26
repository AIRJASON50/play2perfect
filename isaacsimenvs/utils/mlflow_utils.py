"""MLflow observer for rl_games Runner — the sole experiment tracker.

Enabled with ``--mlflow_activate`` (server + credentials read from the standard
MLFLOW_TRACKING_URI / MLFLOW_TRACKING_USERNAME / MLFLOW_TRACKING_PASSWORD env
vars by mlflow itself; nothing hardcoded).

rl_games writes all scalars (rewards, losses, env stats) to its TensorBoard
SummaryWriter. This observer wraps ``algo.writer.add_scalar`` to buffer every
scalar and flush one batched ``log_metrics`` per iteration (from
``after_print_stats``). Mirrors the GMT rsl_rl MLflowSummaryWriter behaviour on
the rl_games side.
"""

from __future__ import annotations

import os
import re
from datetime import datetime

from rl_games.common.algo_observer import AlgoObserver

# MLflow metric keys allow only alphanumerics + a small punctuation set; coerce
# TensorBoard tags like "losses/a_loss" so uploads never fail on the key.
_ALLOWED_METRIC_KEY = re.compile(r"[^A-Za-z0-9_\-. :/]")


def _sanitize(name: str) -> str:
    return _ALLOWED_METRIC_KEY.sub("_", str(name))


class MlflowAlgoObserver(AlgoObserver):
    """Log rl_games scalars to an MLflow tracking server."""

    def __init__(self, experiment: str, run_name: str):
        super().__init__()
        self.experiment = experiment
        self.run_name = run_name
        self._buffer: dict[str, float] = {}
        self._active = False

    def before_init(self, base_name, config, experiment_name):
        import mlflow

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{self.run_name}_{ts}"
        try:
            mlflow.set_experiment(self.experiment)
            mlflow.start_run(run_name=run_name)
            self._active = True
            print(
                f"[MLflow] run '{run_name}' -> "
                f"{os.environ.get('MLFLOW_TRACKING_URI')} (exp={self.experiment})"
            )
        except Exception as exc:  # noqa: BLE001 — never let logging kill training
            print(f"[MLflow] init failed: {exc}")

    def after_init(self, algo):
        if not self._active:
            return
        writer = algo.writer
        orig_add_scalar = writer.add_scalar
        buffer = self._buffer

        def wrapped(tag, value, *args, **kwargs):
            orig_add_scalar(tag, value, *args, **kwargs)
            try:
                buffer[_sanitize(tag)] = float(value)
            except (TypeError, ValueError):
                pass

        writer.add_scalar = wrapped

    def after_print_stats(self, frame, epoch_num, total_time):
        if not self._active or not self._buffer:
            return
        import mlflow

        try:
            mlflow.log_metrics(self._buffer, step=int(frame))
        except Exception as exc:  # noqa: BLE001
            print(f"[MLflow] log_metrics failed: {exc}")
        self._buffer = {}

    def after_clear_stats(self):
        self._buffer = {}
