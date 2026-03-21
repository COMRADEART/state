# EXPERIMENTAL — not used by the production engine
from __future__ import annotations
import json
import threading
import queue
import os


class StateWriter(threading.Thread):
    def __init__(self, instance_path="dimensional_core/state/instance_point.json"):
        super().__init__(daemon=True)
        self.instance_path = instance_path
        self.queue = queue.Queue()
        os.makedirs(os.path.dirname(self.instance_path), exist_ok=True)
        self.start()

    def submit_instance_point(self, data):
        self.queue.put(data)

    def run(self):
        while True:
            data = self.queue.get()
            tmp = self.instance_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.instance_path)
            self.queue.task_done()