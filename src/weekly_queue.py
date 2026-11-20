"""Colas semanales retrospectivas con información disponible al cierre."""
import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline

from src.machine_learning import INPUTS, ROOT, TEST, VALIDATION, counts, split
from src.preprocessing import build_preprocessor

MODELS = ["baseline", "regla_hotel_antelacion", "logistica"]


def historical_rates(x, y):
    """Tasas suavizadas, ajustadas exclusivamente con las etiquetas de train."""
    rows = x[["hotel"]].copy()
    rows["tramo"] = pd.cut(x.lead_time, [-1, 7, 30, np.inf],
                           labels=["0-7", "8-30", "31+"], right=True)
    rows["y"] = y
    prior = float(y.mean())
    table = rows.groupby(["hotel", "tramo"], observed=True).y.agg(["sum", "count"])
    table["score"] = (table["sum"] + 20 * prior) / (table["count"] + 20)
    return table, prior


def rule_scores(x, table, prior):
    bands = pd.cut(x.lead_time, [-1, 7, 30, np.inf], labels=["0-7", "8-30", "31+"])
    keys = pd.MultiIndex.from_arrays([x.hotel, bands], names=["hotel", "tramo"])
    return pd.Series(table.score.reindex(keys).fillna(prior).to_numpy(), index=x.index)


def weekly_cohorts(meta, status_date):
    booking = pd.to_datetime(meta.meta_booking_date_proxy)
    # Lunes 00:00 posterior al día de alta: el día del cierre queda para la próxima cola.
    decision = booking.dt.normalize() + pd.to_timedelta(7 - booking.dt.weekday, unit="D")
    arrival = pd.to_datetime(meta.meta_arrival_date)
    status = pd.to_datetime(status_date)
    # Estado final observado antes del cierre permite retirar reservas ya resueltas.
    pending_arrival = arrival.ge(decision)
    known_outcome = status.lt(decision)
    eligible = pending_arrival & ~known_outcome
    return pd.DataFrame({"fecha_alta_proxy": booking, "decision": decision,
                         "llegada": arrival, "estado_final_conocido": known_outcome,
                         "llegada_pendiente": pending_arrival, "elegible": eligible})


def select_weekly(frame, model):
    ranked = frame.sort_values([model, "desempate", "id"], ascending=[False, True, True])
    groups = ranked.groupby("decision")
    flag = groups.cumcount().lt(groups.id.transform("size").floordiv(5))
    return flag.reindex(frame.index)


def evaluate(rows, model):
    return {**counts(rows, model),
            "alertas": int(rows[f"prioridad_{model}"].sum()),
            "ap": float(average_precision_score(rows.etiqueta, rows[model])),
            "roc_auc": float(roc_auc_score(rows.etiqueta, rows[model])),
            "brier": float(brier_score_loss(rows.etiqueta, rows[model]))}


def run():
    reports = ROOT / "reports"
    gold_path = ROOT / "data/processed/hotel_bookings_prepared.csv"
    raw_path = ROOT / "data/raw/hotel_bookings.csv"
    expected = json.loads((reports / "preparacion_resumen.json").read_text(encoding="utf-8"))
    for path, key in [(gold_path, "sha256_preparado"), (raw_path, "sha256_original")]:
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected[key]:
            raise ValueError(f"Huella inesperada: {path.name}")
    gold = pd.read_csv(gold_path)
    raw = pd.read_csv(raw_path)
    if not gold.meta_row_id.eq(np.arange(len(raw))).all():
        raise ValueError("La correspondencia de filas raw/Gold ha cambiado.")
    parts = split(gold)
    train = parts.eq("train")
    x = gold[INPUTS].copy()
    y = gold.is_canceled
    preprocessor = build_preprocessor()
    preprocessor.transformers[1] = (
        "categorical", preprocessor.transformers[1][1],
        [column for column in preprocessor.transformers[1][2] if column in INPUTS])
    model = make_pipeline(preprocessor, LogisticRegression(max_iter=1000, random_state=42))
    model.fit(x[train], y[train])
    table, prior = historical_rates(x[train], y[train])
    table.to_csv(reports / "semanal_tasas_train.csv")
    frame = weekly_cohorts(gold, raw.reservation_status_date)
    frame["id"] = gold.meta_row_id
    frame["hotel"] = gold.hotel
    frame["etiqueta"] = y
    # Cohortes completas cuyos intervalos empiezan después de cada corte.
    # Validación solo por cierre previo a TEST, sin filtrar por desenlace futuro.
    frame["particion"] = "fuera"
    frame.loc[frame.fecha_alta_proxy.ge(VALIDATION) &
              frame.decision.lt(TEST), "particion"] = "validacion"
    frame.loc[frame.fecha_alta_proxy.ge(TEST) &
              frame.decision.le(pd.Timestamp("2017-08-28")), "particion"] = "prueba"
    # El primer lunes de cada bloque contiene días anteriores al corte: se retira entero.
    start = frame.decision - pd.Timedelta(days=7)
    boundary = ((frame.particion.eq("validacion") & start.lt(VALIDATION)) |
                (frame.particion.eq("prueba") & start.lt(TEST)))
    frame.loc[boundary, "particion"] = "fuera"
    cohort = frame[frame.particion.ne("fuera")].copy()
    cohort.to_csv(reports / "semanal_elegibilidad.csv", index=False)
    rows = cohort[cohort.elegible].copy()
    rows["baseline"] = prior
    rows["regla_hotel_antelacion"] = rule_scores(x.loc[rows.index], table, prior)
    rows["logistica"] = model.predict_proba(x.loc[rows.index])[:, 1]
    rows["desempate"] = rows.id.map(lambda value: hashlib.sha256(f"42|{value}".encode()).hexdigest())
    for name in MODELS:
        rows[f"prioridad_{name}"] = select_weekly(rows, name)
    rows.drop(columns="desempate").to_csv(reports / "semanal_predicciones.csv", index=False)
    metrics = {"protocolo": "semanal_v1", "cupo": "floor(0.20 * elegibles de la semana)",
               "suavizado_regla": 20, "train": int(train.sum()),
               "sha256_gold": expected["sha256_preparado"],
               "sha256_raw": expected["sha256_original"], "evaluacion": {}}
    segments = []
    for stage, stage_rows in rows.groupby("particion"):
        group = cohort[cohort.particion.eq(stage)]
        metrics["evaluacion"][stage] = {
            "altas_cohorte": len(group), "elegibles": len(stage_rows),
            "excluidas": int((~group.elegible).sum()),
            "llegada_pasada": int((~group.llegada_pendiente).sum()),
            "desenlace_conocido": int(group.estado_final_conocido.sum()),
            "semanas": stage_rows.decision.nunique(),
            "modelos": {name: evaluate(stage_rows, name) for name in MODELS}}
        for dimension in ["hotel", "decision"]:
            for value, subset in stage_rows.groupby(dimension):
                for name in MODELS:
                    segments.append({"particion": stage, "dimension": dimension,
                                     "grupo": str(value), "modelo": name, **counts(subset, name),
                                     "alertas": int(subset[f"prioridad_{name}"].sum())})
    pd.DataFrame(segments).to_csv(reports / "semanal_segmentos.csv", index=False)
    (reports / "semanal_metricas.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    run()
