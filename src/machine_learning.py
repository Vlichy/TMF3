"""Experimento retrospectivo de priorización de cancelaciones."""
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline

from src.features import FEATURES, separate_model_data
from src.preprocessing import build_preprocessor

ROOT = Path(__file__).resolve().parents[1]
VALIDATION = "2016-07-01"
TEST = "2017-01-01"
INPUTS = [column for column in FEATURES if column != "country"]
MODELS = ["baseline", "logistica"]


def split(meta):
    booking = pd.to_datetime(meta.meta_booking_date_proxy, errors="raise")
    mature = pd.to_datetime(meta.meta_label_available_after, errors="raise")
    if booking.isna().any() or mature.isna().any() or not meta.meta_row_id.is_unique:
        raise ValueError("Metadatos temporales o identificadores inválidos.")
    masks = {
        "train": booking.lt(VALIDATION) & mature.lt(VALIDATION),
        "validacion": booking.ge(VALIDATION) & booking.lt(TEST) & mature.lt(TEST),
        "prueba": booking.ge(TEST),
    }
    partition = pd.Series("purga", index=meta.index)
    for name, mask in masks.items():
        partition.loc[mask] = name
    kept = partition.ne("purga")
    spread = partition[kept].groupby(meta.meta_duplicate_group[kept]).nunique()
    if spread.gt(1).any():
        raise ValueError("Una huella de duplicación exacta cruza particiones.")
    return partition


def prioritize(frame, score):
    ranked = frame.sort_values([score, "desempate", "id"], ascending=[False, True, True])
    monthly = ranked.groupby("mes")
    selected = monthly.cumcount().lt(monthly["id"].transform("size").mul(20).floordiv(100))
    return selected.reindex(frame.index)


def counts(frame, model):
    positive = frame.etiqueta.eq(1)
    flagged = frame[f"prioridad_{model}"]
    tp = int((positive & flagged).sum())
    fp = int((~positive & flagged).sum())
    fn = int((positive & ~flagged).sum())
    tn = int((~positive & ~flagged).sum())
    return {
        "n": len(frame), "prevalencia": float(positive.mean()), "tp": tp, "fp": fp,
        "fn": fn, "tn": tn, "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "fpr": fp / (fp + tn) if fp + tn else None,
    }


def run():
    gold = ROOT / "data/processed/hotel_bookings_prepared.csv"
    reports = ROOT / "reports"
    inherited = json.loads((reports / "preparacion_resumen.json").read_text(encoding="utf-8"))
    digest = hashlib.sha256(gold.read_bytes()).hexdigest()
    if digest != inherited["sha256_preparado"]:
        raise ValueError("La huella del Gold difiere de la Entrega 02.")
    x, y, meta = separate_model_data(pd.read_csv(gold))
    partition = split(meta)
    train = partition.eq("train")
    transform = build_preprocessor()
    transform.transformers[1] = ("categorical", transform.transformers[1][1],
                                 [col for col in INPUTS if col not in transform.transformers[0][2]])
    pipeline = make_pipeline(transform, LogisticRegression(max_iter=1000, random_state=42))
    pipeline.fit(x.loc[train, INPUTS], y[train])
    prior = float(y[train].mean())
    audit = pd.DataFrame({"particion": partition, "id": meta.meta_row_id})
    audit.to_csv(reports / "ml_particiones.csv", index=False)
    metrics = {"sha256_gold": digest, "versiones": {"python": platform.python_version(),
               "pandas": pd.__version__, "numpy": np.__version__, "sklearn": sklearn.__version__},
               "predictores": INPUTS, "filas": partition.value_counts().to_dict(),
               "prevalencia_train": prior, "evaluacion": {}}
    quality = meta[["meta_zero_nights", "meta_zero_guests", "meta_guests_unknown"]].any(axis=1)
    segments = []
    predictions = []
    for stage in ["validacion", "prueba"]:
        idx = partition.eq(stage)
        subset = pd.DataFrame({
            "particion": stage, "id": meta.meta_row_id[idx], "etiqueta": y[idx],
            "mes": meta.meta_booking_date_proxy[idx].str[:7], "hotel": x.hotel[idx],
            "canal": x.distribution_channel[idx], "pais": x.country[idx].fillna("Sin dato")
                .where(x.country[idx].eq("PRT") | x.country[idx].isna(), "Otros"),
            "calidad": quality[idx],
            "antelacion": pd.cut(x.lead_time[idx], [-1, 7, 30, np.inf], labels=["0–7", "8–30", "31+"]),
        })
        subset["desempate"] = subset.id.map(lambda row: hashlib.sha256(f"42|{row}".encode()).hexdigest())
        subset["baseline"] = prior
        subset["logistica"] = pipeline.predict_proba(x.loc[idx, INPUTS])[:, 1]
        metrics["evaluacion"][stage] = {}
        for model in MODELS:
            subset[f"prioridad_{model}"] = prioritize(subset, model)
            metrics["evaluacion"][stage][model] = {
                **counts(subset, model),
                "ap": average_precision_score(subset.etiqueta, subset[model]),
                "roc_auc": roc_auc_score(subset.etiqueta, subset[model]),
                "brier": brier_score_loss(subset.etiqueta, subset[model]),
            }
            for dimension in ["mes", "hotel", "canal", "pais", "calidad", "antelacion"]:
                for group, rows in subset.groupby(dimension, observed=True, dropna=False):
                    segments.append({"particion": stage, "modelo": model, "dimension": dimension,
                                     "grupo": group, **counts(rows, model)})
        predictions.append(subset.drop(columns=["desempate"]))
    pd.concat(predictions).to_csv(reports / "ml_predicciones.csv", index=False)
    pd.DataFrame(segments).to_csv(reports / "ml_segmentos.csv", index=False)
    (reports / "ml_metricas.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    run()

