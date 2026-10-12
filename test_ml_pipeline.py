import json
import os
import unittest

import joblib
import pandas as pd


class TestMLPipeline(unittest.TestCase):
    def test_output_files_exist(self):
        for name in ["odi_batsman_model.pkl", "metrics.json", "odi_batting_processed.csv"]:
            self.assertTrue(os.path.exists(name), f"{name} is missing")

    def test_metrics_content(self):
        with open("metrics.json") as file:
            metrics = json.load(file)
        for key in ["accuracy", "training_records", "testing_records"]:
            self.assertIn(key, metrics)
        self.assertTrue(0 <= metrics["accuracy"] <= 1)

    def test_model_predicts(self):
        model = joblib.load("odi_batsman_model.pkl")
        sample = pd.DataFrame(
            [[300, 90.0, 20, 60, 10, 1500, 150]],
            columns=["Mat", "SR", "100", "50", "0", "4s", "6s"],
        )
        self.assertIn(int(model.predict(sample)[0]), [0, 1])


if __name__ == "__main__":
    unittest.main()
