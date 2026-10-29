from speech_analyzer.metrics import Interval, merge_intervals, overlap_intervals, union_duration, word_error_rate


def test_merge_intervals_merges_touching_and_overlapping():
    assert merge_intervals([(1, 2), (1.5, 3), (3, 4), (7, 8)]) == [(1.0, 4.0), (7.0, 8.0)]


def test_overlap_counts_only_simultaneous_distinct_speakers():
    turns = [Interval(0, 3, "A"), Interval(2, 4, "B"), Interval(3, 5, "C")]
    result = overlap_intervals(turns, min_duration=0.1)
    assert result[0]["start"] == 2
    assert result[0]["end"] == 4
    assert union_duration((x["start"], x["end"]) for x in result) == 2


def test_touching_turns_are_not_overlap():
    assert overlap_intervals([Interval(0, 1, "A"), Interval(1, 2, "B")]) == []


def test_wer_counts_substitutions_deletions_insertions():
    result = word_error_rate("the cat sat", "the dog sat today")
    assert result["substitutions"] == 1
    assert result["insertions"] == 1
    assert result["deletions"] == 0
    assert result["wer"] == 2 / 3
