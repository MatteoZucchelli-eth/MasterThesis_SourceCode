# MIT License
"""Unit tests for the perception-error mean/std aggregation."""

import numpy as np
import pytest

from g1_control.utils.perception_stats import (
    format_stats,
    perception_error_stats,
    pick_group_column,
    stats_rows,
)


def _columns(n=10, **overrides):
    """A minimal metrics-CSV-shaped column dict with zero error everywhere."""
    cols = {
        "t": np.arange(n, dtype=float),
        "waypoint_idx": np.zeros(n),
        "rel_err_x": np.zeros(n),
        "rel_err_y": np.zeros(n),
        "rel_err_z": np.zeros(n),
        "rel_err_xaxis_deg": np.zeros(n),
        "dist_target_actual": np.full(n, 0.2),
    }
    cols.update({k: np.asarray(v, dtype=float) for k, v in overrides.items()})
    return cols


def test_mean_and_std_are_reported_in_mm():
    # 1 mm and 3 mm on x -> mean 2 mm, population std 1 mm.
    stats = perception_error_stats(_columns(2, rel_err_x=[0.001, 0.003]))
    x = stats[0].channels["x"]
    assert x.unit == "mm"
    assert x.mean == pytest.approx(2.0)
    assert x.std == pytest.approx(1.0)
    assert x.abs_mean == pytest.approx(2.0)
    assert x.max_abs == pytest.approx(3.0)
    assert x.count == 2


def test_l2_channel_is_derived_from_the_axis_columns():
    stats = perception_error_stats(
        _columns(1, rel_err_x=[0.003], rel_err_y=[0.004], rel_err_z=[0.0])
    )
    assert stats[0].channels["L2"].mean == pytest.approx(5.0)


def test_nan_samples_are_dropped_not_propagated():
    # A tick with no prediction yet records NaN; it must not poison the mean.
    stats = perception_error_stats(_columns(3, rel_err_x=[0.001, np.nan, 0.003]))
    x = stats[0].channels["x"]
    assert x.count == 2
    assert x.mean == pytest.approx(2.0)


def test_all_nan_channel_is_omitted_rather_than_reported_as_nan():
    stats = perception_error_stats(_columns(3, rel_err_xaxis_deg=[np.nan] * 3))
    assert "x-axis" not in stats[0].channels
    assert "x" in stats[0].channels


def test_grouped_per_waypoint_plus_an_overall_row():
    cols = _columns(
        4,
        waypoint_idx=[0, 0, 1, 1],
        rel_err_x=[0.001, 0.001, 0.005, 0.005],
        dist_target_actual=[0.3, 0.3, 0.1, 0.1],
    )
    stats = perception_error_stats(cols)
    assert [s.group for s in stats] == ["all", "waypoint 0", "waypoint 1"]
    assert stats[0].channels["x"].mean == pytest.approx(3.0)
    assert stats[1].channels["x"].mean == pytest.approx(1.0)
    assert stats[1].channels["x"].std == pytest.approx(0.0)
    assert stats[2].channels["x"].mean == pytest.approx(5.0)
    # Each group reports the viewpoint distance it was measured from.
    assert stats[2].distance_mean == pytest.approx(0.1)


def test_a_real_vla_run_has_no_waypoints_so_only_the_overall_row():
    # waypoint_idx is -1 outside the scripted sweep.
    stats = perception_error_stats(_columns(4, waypoint_idx=[-1] * 4))
    assert [s.group for s in stats] == ["all"]


def test_missing_rel_err_columns_yield_no_stats():
    cols = _columns(3)
    for axis in "xyz":
        del cols[f"rel_err_{axis}"]
    del cols["rel_err_xaxis_deg"]
    assert perception_error_stats(cols) == []
    assert "no rel_err_* samples" in format_stats([])


def test_group_column_prefers_the_sweep_waypoints_then_the_planner_phase():
    sweep = _columns(4, waypoint_idx=[0, 0, 1, 1], phase=[np.nan] * 4)
    assert pick_group_column(sweep) == "waypoint_idx"

    planner = _columns(4, waypoint_idx=[-1] * 4, phase=[0, 0, 1, 1])
    assert pick_group_column(planner) == "phase"
    assert [s.group for s in perception_error_stats(planner, "phase")] == [
        "all",
        "phase 0",
        "phase 1",
    ]


def test_segments_keep_an_all_nan_grouping_column_as_one_run():
    # A plain VLA run records NaN phase on every sample; a naive `!=` walk would
    # split it into 1-sample segments, whose std is a meaningless zero.
    from g1_control.plot_vla_metrics import _segments

    t = np.arange(5, dtype=float)
    assert len(_segments(t, np.full(5, np.nan))) == 1
    assert len(_segments(t, None)) == 1
    assert [(s[0], s[1]) for s in _segments(t, np.array([0, 0, 1, 1, 1.0]))] == [
        (0.0, 1.0),
        (2.0, 4.0),
    ]


def test_stats_rows_flatten_one_row_per_group_and_channel():
    stats = perception_error_stats(_columns(4, waypoint_idx=[0, 0, 1, 1]))
    header, rows = stats_rows(stats)
    assert header[:4] == ["group", "samples", "dist_target_actual_mean_m", "channel"]
    # 3 groups (all + 2 waypoints) x 5 channels (x, y, z, L2, x-axis).
    assert len(rows) == 15
    assert all(len(r) == len(header) for r in rows)


def test_metric_csv_scores_both_sources_with_matching_columns():
    """Every drv_* column must have a primary twin to be compared against."""
    from g1_control.states.metric_recorder import MetricRecorder

    header = set(MetricRecorder._CSV_HEADER)
    driver = {c for c in header if c.startswith("drv_")}
    assert driver, "the driving model's columns have gone missing"
    for column in driver:
        twin = column.removeprefix("drv_")
        assert twin in header, f"{column} has no primary counterpart {twin}"


def test_driver_stats_score_the_driver_not_the_receiver():
    """The prefix remap must overwrite the primary's columns, not merely add."""
    import numpy as np

    from g1_control.states.metric_recorder import MetricRecorder

    header = MetricRecorder._CSV_HEADER
    row = dict.fromkeys(header, float("nan"))
    row.update(rel_err_x=0.001, rel_err_y=0.0, rel_err_z=0.0)
    row.update(drv_rel_err_x=0.100, drv_rel_err_y=0.0, drv_rel_err_z=0.0)

    recorder = MetricRecorder.__new__(MetricRecorder)
    recorder._records = [[row[c] for c in header]]

    receiver = recorder.perception_stats()[0]
    driver = recorder.perception_stats("drv_")[0]
    # Channel stats are already scaled to mm.
    assert np.isclose(receiver.channels["L2"].mean, 1.0)
    assert np.isclose(driver.channels["L2"].mean, 100.0)
