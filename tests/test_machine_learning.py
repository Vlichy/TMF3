"""Comprueba las evidencias del experimento sobre el Gold de la entrega."""
import hashlib
import json
import unittest

import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from src.machine_learning import INPUTS, ROOT, TEST, VALIDATION, prioritize


class TestEntrega03(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        reports = ROOT / "reports"
        cls.metrics = json.loads((reports / "ml_metricas.json").read_text(encoding="utf-8"))
        gold = ROOT / "data/processed/hotel_bookings_prepared.csv"
        cls.hash = hashlib.sha256(gold.read_bytes()).hexdigest()
        cls.gold = pd.read_csv(gold).set_index("meta_row_id")
        cls.parts = pd.read_csv(reports / "ml_particiones.csv").set_index("id")
        cls.pred = pd.read_csv(reports / "ml_predicciones.csv")

    def test_temporal_and_gold(self):
        self.assertEqual(self.hash, self.metrics["sha256_gold"])
        self.assertTrue(self.parts.index.is_unique)
        pd.testing.assert_index_equal(self.gold.index, self.parts.index, check_names=False)
        self.assertEqual(len(self.gold), 119390)
        booking = self.gold.meta_booking_date_proxy
        maturity = self.gold.meta_label_available_after
        expected = pd.Series("purga", index=self.gold.index)
        expected[(booking < VALIDATION) & (maturity < VALIDATION)] = "train"
        expected[(booking >= VALIDATION) & (booking < TEST) & (maturity < TEST)] = "validacion"
        expected[booking >= TEST] = "prueba"
        pd.testing.assert_series_equal(self.parts.particion, expected, check_names=False)
        active = self.parts.particion[self.parts.particion.ne("purga")]
        self.assertEqual(active.groupby(self.gold.meta_duplicate_group).nunique().max(), 1)
        self.assertEqual(set(INPUTS) & {"country", "is_canceled", "reservation_status",
                                       "reservation_status_date"}, set())
        self.assertFalse(any(column.startswith("meta_") for column in INPUTS))

    def test_predictions_and_metrics(self):
        self.assertTrue(self.pred.id.is_unique)
        held_out = self.parts.index[self.parts.particion.isin(["validacion", "prueba"])]
        self.assertEqual(set(self.pred.id), set(held_out))
        self.assertTrue((self.pred.etiqueta.to_numpy() == self.gold.loc[self.pred.id, "is_canceled"]).all())
        self.assertTrue((self.pred.particion.to_numpy() == self.parts.loc[self.pred.id, "particion"]).all())
        for stage, rows in self.pred.groupby("particion"):
            for model in ["baseline", "logistica"]:
                with self.subTest(stage=stage, model=model):
                    score = rows[model]
                    self.assertTrue(score.between(0, 1).all())
                    selected = rows[f"prioridad_{model}"]
                    monthly = rows.groupby("mes")[f"prioridad_{model}"]
                    pd.testing.assert_series_equal(monthly.sum(), monthly.size() // 5)
                    actual = self.metrics["evaluacion"][stage][model]
                    for metric, function in [("ap", average_precision_score),
                                             ("roc_auc", roc_auc_score), ("brier", brier_score_loss)]:
                        self.assertAlmostEqual(actual[metric], function(rows.etiqueta, score))
                    matrix = pd.crosstab(rows.etiqueta, selected)
                    self.assertEqual(actual["tp"], matrix.loc[1, True])
                    self.assertEqual(actual["fp"], matrix.loc[0, True])
                    self.assertEqual(actual["fn"], matrix.loc[1, False])
                    self.assertEqual(actual["tn"], matrix.loc[0, False])

    def test_priority_with_ties_and_partial_month(self):
        frame = pd.DataFrame({"id": [1, 2, 3, 4, 5, 6], "mes": ["A"] * 5 + ["B"],
                              "desempate": ["a", "c", "b", "d", "e", "a"],
                              "score": [0.4, 0.8, 0.8, 0.5, 0.6, 1.0]})
        self.assertEqual(frame.loc[prioritize(frame, "score"), "id"].tolist(), [3])


if __name__ == "__main__":
    unittest.main()

