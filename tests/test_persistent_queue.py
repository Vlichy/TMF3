"""Pruebas de persistencia, exclusión temporal y comparación justa de cupos."""
import hashlib
import json
import unittest

import pandas as pd

from src.machine_learning import ROOT, counts
from src.persistent_queue import POLICIES, active_at, pick_candidates, simulate


def sample_queue():
    rows = pd.DataFrame({
        "id": range(10), "hotel": ["City Hotel"] * 5 + ["Resort Hotel"] * 5,
        "fecha_alta_proxy": pd.Timestamp("2016-12-01"),
        "llegada": pd.Timestamp("2017-01-25"),
        "fecha_estado": pd.Timestamp("2017-01-26"),
        "score": [0.99, 0.98, 0.97, 0.96, 0.95, 0.8, 0.7, 0.6, 0.5, 0.4],
        "etiqueta": [1, 0] * 5})
    rows["desempate"] = rows.id.map(lambda value: hashlib.sha256(str(value).encode()).hexdigest())
    return rows


class TestPersistentQueue(unittest.TestCase):
    def test_carryover_and_one_contact_per_reservation(self):
        frame = sample_queue()
        events, capacities = simulate(frame, pd.to_datetime(["2017-01-02", "2017-01-09"]))
        self.assertEqual(capacities.cupo_comun.tolist(), [2, 2])
        for policy in POLICIES:
            history = events[events.politica.eq(policy)]
            selected = history[history.seleccionada]
            self.assertTrue(selected.id.is_unique)
            self.assertEqual(len(selected), 4)
            self.assertTrue((selected.fecha_alta_proxy < pd.Timestamp("2016-12-26")).all())
            first_contacts = selected[selected.decision.eq("2017-01-02")].id
            self.assertFalse(history[history.decision.eq("2017-01-09")].id.isin(first_contacts).any())
            self.assertGreater(history.id.value_counts().max(), 1)

    def test_decision_boundaries_and_future_independence(self):
        frame = sample_queue()
        decision = pd.Timestamp("2017-01-02")
        frame.loc[0, "fecha_alta_proxy"] = decision
        frame.loc[1, "fecha_estado"] = decision - pd.Timedelta(days=1)
        frame.loc[2, "llegada"] = decision + pd.Timedelta(days=28)
        frame.loc[3, "llegada"] = decision
        frame.loc[4, "fecha_estado"] = decision
        mask = active_at(frame, decision)
        self.assertEqual(mask.head(5).tolist(), [False, False, False, True, True])
        first, _ = simulate(frame, [decision])
        future = sample_queue()
        future.id += 100
        future.fecha_alta_proxy = pd.Timestamp("2017-01-03")
        future.score = 1.0
        expanded, _ = simulate(pd.concat([frame, future], ignore_index=True), [decision])
        pd.testing.assert_frame_equal(first, expanded)

    def test_allocation_exact_capacity_and_ties(self):
        frame = sample_queue()
        for capacity in range(len(frame) + 1):
            for policy in POLICIES:
                selected = pick_candidates(frame, capacity, policy)
                self.assertEqual(len(selected), capacity)
                self.assertTrue(selected.id.is_unique)
        chosen = pick_candidates(frame, 2, "por_hotel")
        self.assertEqual(chosen.groupby("hotel").size().to_dict(),
                         {"City Hotel": 1, "Resort Hotel": 1})
        self.assertEqual(pick_candidates(frame, 2, "global").id.tolist(), [0, 1])
        events, capacity = simulate(frame, pd.date_range("2017-01-02", periods=4, freq="D"))
        self.assertTrue(events[events.seleccionada].groupby("politica").id.apply(
            lambda ids: ids.is_unique).all())

    def test_real_evidence_has_no_duplicates_and_same_weekly_total(self):
        reports = ROOT / "reports"
        events = pd.read_csv(reports / "cola_persistente_eventos.csv")
        capacities = pd.read_csv(reports / "cola_persistente_cupos.csv")
        metrics = json.loads((reports / "cola_persistente_metricas.json").read_text())
        universe = events.drop_duplicates("id")[["id", "hotel", "etiqueta"]].copy()
        for policy in POLICIES:
            history = events[events.politica.eq(policy)]
            selected = history[history.seleccionada]
            self.assertTrue(selected.id.is_unique)
            total = selected.groupby("decision").size().reindex(capacities.decision, fill_value=0)
            self.assertEqual(total.tolist(), capacities.cupo_comun.tolist())
            selected_dates = selected.set_index("id").decision
            known = history.id.map(selected_dates)
            self.assertTrue((history.decision.le(known) | known.isna()).all())
            for decision, group in history.groupby("decision"):
                self.assertTrue(active_at(group.assign(
                    fecha_alta_proxy=pd.to_datetime(group.fecha_alta_proxy),
                    llegada=pd.to_datetime(group.llegada),
                    fecha_estado=pd.to_datetime(group.fecha_estado)),
                    pd.Timestamp(decision)).all())
            universe["prioridad_policy"] = universe.id.isin(selected.id)
            for name, value in counts(universe, "policy").items():
                self.assertAlmostEqual(metrics["evaluacion"][policy][name], value)
        self.assertTrue((capacities.cupo_comun <= capacities.cupo_nominal).all())
        self.assertTrue((capacities.cupo_nominal == capacities.activas // 5).all())


if __name__ == "__main__":
    unittest.main()
