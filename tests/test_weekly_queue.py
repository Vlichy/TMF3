"""Pruebas de causalidad de la cola semanal y recálculo de sus evidencias."""
import hashlib
import json
import unittest

import numpy as np
import pandas as pd

from src.machine_learning import ROOT, split
from src.weekly_queue import (MODELS, evaluate, historical_rates, rule_scores,
                              select_weekly, weekly_cohorts)


class TestWeeklyQueue(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gold = pd.read_csv(ROOT / "data/processed/hotel_bookings_prepared.csv")
        cls.raw = pd.read_csv(ROOT / "data/raw/hotel_bookings.csv")
        cls.rows = pd.read_csv(ROOT / "reports/semanal_predicciones.csv")
        cls.cohort = pd.read_csv(ROOT / "reports/semanal_elegibilidad.csv")
        cls.metrics = json.loads((ROOT / "reports/semanal_metricas.json").read_text())
        cls.rows["desempate"] = cls.rows.id.map(
            lambda value: hashlib.sha256(f"42|{value}".encode()).hexdigest())

    def test_future_reservations_do_not_change_previous_selection(self):
        first = pd.DataFrame({"id": range(5), "decision": ["2017-01-09"] * 5,
                              "score": [0.1, 0.2, 0.9, 0.6, 0.3],
                              "desempate": list("abcde")})
        future = pd.DataFrame({"id": range(5, 10), "decision": ["2017-01-16"] * 5,
                               "score": [1.0] * 5, "desempate": list("abcde")})
        combined = pd.concat([first, future], ignore_index=True)
        pd.testing.assert_series_equal(select_weekly(first, "score"),
                                       select_weekly(combined, "score").loc[first.index])

    def test_monday_boundary_and_outcome_availability(self):
        meta = pd.DataFrame({
            "meta_booking_date_proxy": ["2017-01-08", "2017-01-09", "2017-01-08", "2017-01-08"],
            "meta_arrival_date": ["2017-01-09", "2017-01-16", "2017-01-08", "2017-01-12"]})
        result = weekly_cohorts(meta, pd.Series(
            ["2017-01-09", "2017-01-20", "2017-01-10", "2017-01-08"]))
        self.assertEqual(result.decision.dt.strftime("%Y-%m-%d").tolist(),
                         ["2017-01-09", "2017-01-16", "2017-01-09", "2017-01-09"])
        self.assertEqual(result.elegible.tolist(), [True, True, False, False])

    def test_rule_train_only_and_unseen_group(self):
        train = split(self.gold).eq("train")
        rates, prior = historical_rates(self.gold[train], self.gold.is_canceled[train])
        recorded = pd.read_csv(ROOT / "reports/semanal_tasas_train.csv").set_index(["hotel", "tramo"])
        expected = rates.reset_index()
        expected["tramo"] = expected.tramo.astype("str")
        pd.testing.assert_frame_equal(expected, recorded.reset_index())
        self.assertAlmostEqual(prior, self.gold.is_canceled[train].mean())
        unseen = pd.DataFrame({"hotel": ["Unknown Hotel"], "lead_time": [5]})
        self.assertAlmostEqual(rule_scores(unseen, rates, prior).iloc[0], prior)
        self.assertEqual(int(rates["count"].sum()), int(train.sum()))

    def test_real_cohorts_capacity_and_metrics(self):
        reconstructed = weekly_cohorts(self.gold, self.raw.reservation_status_date)
        selected = reconstructed.loc[self.cohort.id]
        np.testing.assert_array_equal(selected.elegible, self.cohort.elegible)
        self.assertTrue(self.rows.id.is_unique)
        self.assertEqual(set(self.rows.id), set(self.cohort.loc[self.cohort.elegible, "id"]))
        self.assertTrue((pd.to_datetime(self.rows.fecha_alta_proxy) <
                         pd.to_datetime(self.rows.decision)).all())
        self.assertTrue((pd.to_datetime(self.rows.llegada) >=
                         pd.to_datetime(self.rows.decision)).all())
        np.testing.assert_array_equal(self.rows.etiqueta,
                                      self.gold.set_index("meta_row_id").loc[self.rows.id, "is_canceled"])
        for stage, rows in self.rows.groupby("particion"):
            for model in MODELS:
                with self.subTest(stage=stage, model=model):
                    pd.testing.assert_series_equal(select_weekly(rows, model),
                                                   rows[f"prioridad_{model}"], check_names=False)
                    expected_capacity = rows.groupby("decision").size().floordiv(5)
                    pd.testing.assert_series_equal(rows.groupby("decision")[f"prioridad_{model}"].sum(),
                                                   expected_capacity, check_names=False)
                    actual = self.metrics["evaluacion"][stage]["modelos"][model]
                    for key, value in evaluate(rows, model).items():
                        self.assertAlmostEqual(actual[key], value)


if __name__ == "__main__":
    unittest.main()

