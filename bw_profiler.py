import csv
import os
import threading
import time
from contextlib import contextmanager
from datetime import datetime


class BWProfiler:
    def __init__(self, csv_path="/tmp/bookworm_latency.csv"):
        self.csv_path = csv_path
        self.lock = threading.Lock()
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="") as file_obj:
                writer = csv.writer(file_obj)
                writer.writerow(
                    [
                        "timestamp_iso",
                        "utterance_id",
                        "sst_ms",
                        "llm_ms",
                        "tts_ms",
                        "e2e_ms",
                    ]
                )

    def new_utterance(self, utterance_id=None):
        return {
            "id": utterance_id or f"utt-{int(time.time() * 1000)}",
            "t0": time.perf_counter(),
            "sst": None,
            "llm": None,
            "tts": None,
        }

    def mark(self, ctx, stage, elapsed_ms):
        ctx[stage] = elapsed_ms

    @contextmanager
    def timer(self, ctx, stage):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.mark(ctx, stage, (time.perf_counter() - start) * 1000.0)

    def finish(self, ctx):
        e2e_ms = (time.perf_counter() - ctx["t0"]) * 1000.0
        row = [
            datetime.utcnow().isoformat(),
            ctx["id"],
            round(ctx.get("sst") or 0, 1),
            round(ctx.get("llm") or 0, 1),
            round(ctx.get("tts") or 0, 1),
            round(e2e_ms, 1),
        ]
        with self.lock:
            with open(self.csv_path, "a", newline="") as file_obj:
                csv.writer(file_obj).writerow(row)
        return e2e_ms


profiler = BWProfiler()

