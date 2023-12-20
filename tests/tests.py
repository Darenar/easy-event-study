import pandas as pd
import pytest

from easy_es import EventStudy, calculate_car_stats


@pytest.mark.parametrize("input_file, output_file, estimator_type", [
    ("tests/test_one/input_events.csv", "tests/test_one/gt_mam.csv", "mam"),
    ("tests/test_one/input_events.csv", "tests/test_one/gt_capm.csv", "capm"),
    ("tests/test_one/input_events.csv", "tests/test_one/gt_ff3.csv", "ff3"),
    ("tests/test_one/input_events.csv", "tests/test_one/gt_ff5.csv", "ff5"),
    ("tests/test_two/input_events.csv", "tests/test_two/gt_mam.csv", "mam"),
    ("tests/test_two/input_events.csv", "tests/test_two/gt_capm.csv", "capm"),
    ("tests/test_two/input_events.csv", "tests/test_two/gt_ff3.csv", "ff3"),
    ("tests/test_two/input_events.csv", "tests/test_two/gt_ff5.csv", "ff5")
])
def test_results(input_file, output_file, estimator_type, max_diff: float = 0.0005):
    events_df = pd.read_csv(input_file)
    event_study = EventStudy(
        estimation_days=255,
        gap_days=50,
        window_after=10,
        window_before=10,
        min_estimation_days=100,
        estimator_type=estimator_type)
    res_df = calculate_car_stats(event_study.run_study(events_df))
    output_df = pd.read_csv(output_file)
    difference = (res_df['mean'].reset_index(drop=True) - output_df['car'])
    assert difference.mean() < max_diff
