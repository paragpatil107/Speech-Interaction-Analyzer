"""Pure functions for speech timeline metrics; kept separate for easy testing."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable


@dataclass(frozen=True)
class Interval:
    start: float
    end: float
    speaker: str | None = None

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


def merge_intervals(intervals: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    cleaned = sorted((float(s), float(e)) for s, e in intervals if e > s)
    if not cleaned:
        return []
    merged: list[list[float]] = [[cleaned[0][0], cleaned[0][1]]]
    for start, end in cleaned[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(s, e) for s, e in merged]


def overlap_intervals(turns: list[Interval], min_duration: float = 0.15) -> list[dict]:
    """Find time spans covered by at least two distinct speakers.

    Uses a sweep-line event list, so a span is counted once even if three speakers overlap.
    """
    events: list[tuple[float, int, str | None]] = []
    for turn in turns:
        if turn.end <= turn.start:
            continue
        events.append((turn.start, 1, turn.speaker))
        events.append((turn.end, -1, turn.speaker))
    # End events before start events at the same timestamp: touching turns aren't overlap.
    events.sort(key=lambda x: (x[0], x[1]))
    active: dict[str | None, int] = {}
    spans: list[dict] = []
    previous_time: float | None = None
    i = 0
    while i < len(events):
        time = events[i][0]
        if previous_time is not None and time > previous_time and len(active) >= 2:
            speakers = sorted(str(s) for s, count in active.items() if count > 0)
            if time - previous_time >= min_duration:
                spans.append({"start": previous_time, "end": time, "duration": time - previous_time, "speakers": speakers})
        while i < len(events) and events[i][0] == time:
            _, delta, speaker = events[i]
            active[speaker] = active.get(speaker, 0) + delta
            if active[speaker] <= 0:
                active.pop(speaker, None)
            i += 1
        previous_time = time
    # Coalesce adjacent spans even when the speaker pair changes at the boundary.
    coalesced: list[dict] = []
    for span in spans:
        if coalesced and abs(coalesced[-1]["end"] - span["start"]) < 1e-9:
            coalesced[-1]["end"] = span["end"]
            coalesced[-1]["duration"] = coalesced[-1]["end"] - coalesced[-1]["start"]
            coalesced[-1]["speakers"] = sorted(set(coalesced[-1]["speakers"]) | set(span["speakers"]))
        else:
            coalesced.append(dict(span))
    return coalesced


def union_duration(intervals: Iterable[tuple[float, float]]) -> float:
    return sum(end - start for start, end in merge_intervals(intervals))


def word_error_rate(reference: str, hypothesis: str) -> dict:
    """Return WER and edit counts using case-insensitive word-level Levenshtein alignment."""
    ref = reference.lower().split()
    hyp = hypothesis.lower().split()
    rows, cols = len(ref) + 1, len(hyp) + 1
    dp = [[0] * cols for _ in range(rows)]
    back = [[""] * cols for _ in range(rows)]
    for i in range(1, rows): dp[i][0], back[i][0] = i, "D"
    for j in range(1, cols): dp[0][j], back[0][j] = j, "I"
    for i in range(1, rows):
        for j in range(1, cols):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            options = [(dp[i - 1][j - 1] + cost, "M" if cost == 0 else "S"),
                       (dp[i - 1][j] + 1, "D"), (dp[i][j - 1] + 1, "I")]
            dp[i][j], back[i][j] = min(options, key=lambda x: x[0])
    i, j = len(ref), len(hyp)
    subs = dels = ins = 0
    while i or j:
        op = back[i][j]
        if op == "S": subs += 1; i -= 1; j -= 1
        elif op == "D": dels += 1; i -= 1
        elif op == "I": ins += 1; j -= 1
        elif op == "M": i -= 1; j -= 1
        else: break
    errors = subs + dels + ins
    return {"wer": errors / len(ref) if ref else None, "substitutions": subs, "deletions": dels,
            "insertions": ins, "errors": errors, "reference_words": len(ref), "hypothesis_words": len(hyp)}
