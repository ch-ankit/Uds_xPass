"""Offline regression checks: synthetic passes, real feature building and XGBoost.
Run: python -m unittest discover -s tests -v
"""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import all_seasons as pipeline


def passes():
    rows = []
    for match in range(6):
        for i in range(50):
            rows.append({
                "id": f"{match}-{i}", "match_id": match, "player_id": 1,
                "location": [20.0 + i % 30, 25.0],
                "pass_end_location": [40.0 + i % 30, 30.0],
                "pass_length": 20.6, "pass_angle": 0.245,
                "pass_height": "Ground Pass", "under_pressure": i % 3 == 0,
                "pass_body_part": "Left Foot" if i % 5 == 0 else "Right Foot",
                "pass_outcome": "Incomplete" if i % 4 == 0 else None,
            })
    return pd.DataFrame(rows)


class PipelineRegressionTests(unittest.TestCase):
    def test_array_coordinates_match_lists(self):
        data = pipeline.add_weak_foot_flag(passes())
        expected = pipeline.build_features(data)
        for column in ("location", "pass_end_location"):
            data[column] = data[column].map(np.asarray)
        actual = pipeline.build_features(data)
        self.assertEqual(len(actual), 300)
        pd.testing.assert_frame_equal(actual[pipeline.BASE_FEATS], expected[pipeline.BASE_FEATS])

    def test_bonus_trains_and_writes_figure(self):
        data = passes()

        def frames(match):
            return pd.DataFrame([
                {"id": f"{match}-{i}", "freeze_frame": [
                    {"location": [35.0, 26.0], "teammate": False, "actor": False},
                    {"location": [40.0, 30.0], "teammate": True, "actor": False},
                ]} for i in range(50)
            ])

        seasons = pd.DataFrame([{"season_name": "2020/2021", "season_id": 90}])
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(pipeline, "OUT_DIR", Path(tmp)), \
                patch.object(pipeline, "get_laliga_seasons", return_value=seasons), \
                patch.object(pipeline, "pull_season_passes", return_value=data), \
                patch.object(pipeline.sb, "matches", return_value=pd.DataFrame({"match_id": range(6)})), \
                patch.object(pipeline, "fetch_360", side_effect=frames):
            pipeline.run_2020_21_event_vs_360_bonus()
            self.assertTrue((Path(tmp) / "event_vs_360_2020_21.png").is_file())

    def test_missing_freeze_frame(self):
        self.assertEqual(pipeline.extract_360_features(np.nan, [20, 25], [40, 30]), {})


if __name__ == "__main__":
    unittest.main()
