from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TriggerConfig:
    """
    Runtime durability / checkpoint policy.

    commit_every_n_events:
        Save instance point after this many emitted events.

    rotate_every_n_commits:
        Rotate events.jsonl after this many instance-point commits.
        Set to 0 to disable rotation.

    keep_archives:
        Number of rotated archive files to keep.
    """

    commit_every_n_events: int = 32
    rotate_every_n_commits: int = 50
    keep_archives: int = 10

    @classmethod
    def normalized(
        cls,
        commit_every_n_events: int = 32,
        rotate_every_n_commits: int = 50,
        keep_archives: int = 10,
    ) -> "TriggerConfig":
        return cls(
            commit_every_n_events=max(1, int(commit_every_n_events)),
            rotate_every_n_commits=max(0, int(rotate_every_n_commits)),
            keep_archives=max(1, int(keep_archives)),
        )