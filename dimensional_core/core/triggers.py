from dataclasses import dataclass

@dataclass
class TriggerConfig:
    commit_every_n_events: int = 25      # commit cursor every N events
    commit_on_cycle_done: bool = True    # also commit when CYCLE_DONE occurs
    rotate_every_n_commits: int = 5      # rotate events.jsonl after N commits
    keep_archives: int = 10              # keep last N rotated files
