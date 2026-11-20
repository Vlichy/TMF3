"""Cola persistente para llegadas próximas, con cupo comparable entre políticas."""
import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from src.machine_learning import INPUTS, ROOT, counts, split
from src.preprocessing import build_preprocessor

POLICIES = ["global", "por_hotel"]
DECISIONS = pd.date_range("2017-01-02", "2017-08-28", freq="W-MON")
HORIZON_DAYS = 28


def active_at(frame, decision):
    """Información conocida el lunes 00:00, con fechas de día sin hora."""
    return (frame.fecha_alta_proxy.lt(decision) &
            frame.llegada.ge(decision) &
            frame.llegada.lt(decision + pd.Timedelta(days=HORIZON_DAYS)) &
            frame.fecha_estado.ge(decision))


def pick_candidates(queue, capacity, policy):
    ranked = queue.sort_values(["score", "desempate", "id"], ascending=[False, True, True])
    if capacity < 0 or capacity > len(queue):
        raise ValueError("El cupo excede las reservas pendientes.")
    if policy == "global":
        return ranked.head(capacity)
    if policy != "por_hotel":
        raise ValueError("Política desconocida.")
    # Método de restos mayores, con desempate por nombre del hotel.
    # Las cuotas proporcionales nunca superan el número pendiente de cada hotel.
    sizes = queue.groupby("hotel").size().sort_index()
    if sizes.empty:
        return ranked
    numerator = sizes * capacity
    quotas = numerator.floordiv(len(queue))
    remainder = capacity - int(quotas.sum())
    residual = numerator.mod(len(queue)).sort_values(ascending=False, kind="stable")
    quotas.loc[residual.index[:remainder]] += 1
    position = ranked.groupby("hotel").cumcount()
    return ranked[position.lt(ranked.hotel.map(quotas))]


def simulate(frame, decisions=DECISIONS):
    if not frame.id.is_unique or frame.id.isna().any():
        raise ValueError("Se requiere un ID técnico único por fila.")
    if frame[["fecha_alta_proxy", "llegada", "fecha_estado"]].isna().any().any():
        raise ValueError("Faltan fechas para reconstruir la cola.")
    contacted = {policy: set() for policy in POLICIES}
    events = []
    capacities = []
    for decision in decisions:
        active = frame[active_at(frame, decision)]
        pending = {policy: active[~active.id.isin(contacted[policy])] for policy in POLICIES}
        nominal = len(active) // 5
        # Misma capacidad realizada: cualquier ajuste se conoce en el instante de decisión.
        capacity = min(nominal, *(len(queue) for queue in pending.values()))
        capacities.append({"decision": decision, "activas": len(active),
                           "cupo_nominal": nominal, "cupo_comun": capacity,
                           **{f"pendientes_{policy}": len(queue) for policy, queue in pending.items()}})
        for policy in POLICIES:
            queue = pending[policy].copy()
            selected = pick_candidates(queue, capacity, policy)
            queue["seleccionada"] = queue.id.isin(selected.id)
            queue["decision"] = decision
            queue["politica"] = policy
            events.append(queue)
            contacted[policy].update(selected.id)
    return pd.concat(events, ignore_index=True), pd.DataFrame(capacities)


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
        raise ValueError("Se ha perdido la correspondencia de filas raw/Gold.")
    # Mismo ajuste que en la ampliación anterior, sin modificar hiperparámetros.
    # Se calculan también los scores de las reservas antiguas activas en 2017.
    train = split(gold).eq("train")
    transformer = build_preprocessor()
    transformer.transformers[1] = (
        "categorical", transformer.transformers[1][1],
        [column for column in transformer.transformers[1][2] if column in INPUTS])
    model = make_pipeline(transformer, LogisticRegression(max_iter=1000, random_state=42))
    model.fit(gold.loc[train, INPUTS], gold.is_canceled[train])
    frame = pd.DataFrame({
        "id": gold.meta_row_id, "hotel": gold.hotel,
        "fecha_alta_proxy": pd.to_datetime(gold.meta_booking_date_proxy),
        "llegada": pd.to_datetime(gold.meta_arrival_date),
        "fecha_estado": pd.to_datetime(raw.reservation_status_date),
        "etiqueta": gold.is_canceled,
    })
    ever_active = pd.Series(False, index=frame.index)
    for decision in DECISIONS:
        ever_active |= active_at(frame, decision)
    if (train & ever_active).any():
        raise ValueError("Reservas del entrenamiento aparecen activas en evaluación.")
    frame = frame[ever_active].copy()
    frame["score"] = model.predict_proba(gold.loc[frame.index, INPUTS])[:, 1]
    frame["desempate"] = frame.id.map(lambda value: hashlib.sha256(f"42|{value}".encode()).hexdigest())
    events, capacities = simulate(frame)
    events.drop(columns="desempate").to_csv(reports / "cola_persistente_eventos.csv", index=False)
    capacities.to_csv(reports / "cola_persistente_cupos.csv", index=False)
    metrics = {"protocolo": "cola_persistente_v1", "horizonte_llegada_dias": HORIZON_DAYS,
               "primer_lunes": str(DECISIONS.min().date()), "ultimo_lunes": str(DECISIONS.max().date()),
               "semanas": len(DECISIONS), "train": int(train.sum()), "evaluacion": {},
               "semanas_cupo_limitado": int(capacities.cupo_comun.lt(capacities.cupo_nominal).sum()),
               "sha256_gold": expected["sha256_preparado"], "sha256_raw": expected["sha256_original"]}
    results = []
    for policy in POLICIES:
        selected = events[events.politica.eq(policy) & events.seleccionada]
        audited = frame[["id", "hotel", "etiqueta"]].copy()
        audited["prioridad_politica"] = audited.id.isin(selected.id)
        metrics["evaluacion"][policy] = {**counts(audited, "politica"),
                                         "seguimientos": len(selected)}
        for hotel, group in audited.groupby("hotel"):
            results.append({"politica": policy, "hotel": hotel, **counts(group, "politica"),
                            "seguimientos": int(group.prioridad_politica.sum())})
    pd.DataFrame(results).to_csv(reports / "cola_persistente_hoteles.csv", index=False)
    (reports / "cola_persistente_metricas.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    run()

