from multiagent import TokenUsageTracker, calculate, detect_task_mix


def test_calculate_handles_operator_precedence():
    assert calculate("2 + 3 * 4") == "14"
    assert calculate("(10 - 2) / 4") == "2.0"


def test_detect_task_mix_handles_search_and_math():
    query = "Search for the current price of iPhone 15 and calculate 18% tax on it."
    assert detect_task_mix(query) == ["search", "math"]


def test_token_tracker_accumulates_usage():
    tracker = TokenUsageTracker()
    tracker.add_usage({"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150})
    tracker.add_usage({"prompt_tokens": 25, "completion_tokens": 10, "total_tokens": 35})
    assert tracker.prompt_tokens == 125
    assert tracker.completion_tokens == 60
    assert tracker.total_tokens == 185
