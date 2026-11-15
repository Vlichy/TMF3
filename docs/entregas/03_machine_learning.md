# Entrega 03 · Machine Learning

## 1. Qué problema resuelve el modelo

El equipo de reservas necesita decidir qué reservas revisar primero cuando no puede revisar todas. El modelo aborda una clasificación binaria supervisada y ordena reservas por riesgo de cancelación. El equipo puede comprobar datos o decidir si procede un contacto de seguimiento. La decisión queda en manos del equipo. El experimento mide detección de cancelaciones históricas y carece de datos de intervención para estimar cuántas se evitarían.

## 2. Datos de entrada y salida

Se parte del [Gold de la Entrega 02](../../data/processed/hotel_bookings_prepared.csv) con 119.390 filas y la huella registrada en su [informe de preparación](../../reports/preparacion_resumen.json). La etiqueta `is_canceled` vale 1 si la reserva se canceló y 0 en caso contrario. Se reutiliza el código de la [preparación anterior](02_datos_y_preparacion.md).

| Entradas | Campos |
| --- | --- |
| Antelación, estancia y ocupación declarada | `lead_time`, `total_nights`, `adults`, `children`, `babies`, `children_missing` |
| Contexto de reserva | `hotel`, `arrival_date_month`, `meal`, `market_segment`, `distribution_channel`, `reserved_room_type`, `customer_type` |

Estos campos deberían estar disponibles al registrar una reserva. La fuente carece de instantáneas iniciales, fecha real de alta e ID de reserva estable. Las modificaciones posteriores podrían afectar tanto a los campos como a la fecha de alta estimada. Se excluyen país, resultado, fecha de estado, atributos de disponibilidad incierta y todos los metadatos de auditoría. El país se conserva para estudiar diferencias de desempeño. Hotel y canal pueden actuar como variables indirectamente asociadas al origen.

El Gold conserva 31.994 repeticiones exactas adicionales porque podrían corresponder a reservas diferentes. Se comprueba que sus huellas no crucen bloques activos. La mediana aprendida en entrenamiento imputa los cuatro valores desconocidos de niños cuando corresponda y el indicador conserva la distinción entre ausencia y cero. Se mantienen las reservas con noches o huéspedes a cero para auditar sensibilidad. La tarifa queda fuera del modelo y sus anomalías se conservan en la auditoría de la Entrega 02.

## 3. Baseline y selección de modelos

- **Baseline de tasa histórica.** Todas las reservas reciben la tasa de cancelación del entrenamiento (35,81%). Una permutación reproducible por hash rompe empates y representa un seguimiento sin información individual.
- **Regresión logística.** Aprende diferencias de riesgo a partir de las 13 entradas. Se usa regularización L2 por defecto, escalado numérico y one-hot con categorías desconocidas ignoradas. Imputación, escalado, categorías y coeficientes se ajustan exclusivamente en entrenamiento. Se fija una configuración sin búsqueda de hiperparámetros para limitar complejidad y facilitar reproducción.

Ambos enfoques comparten cupo por mes de alta estimada, con los dos hoteles juntos. Se marca el 20% redondeado hacia abajo, ordenando por puntuación descendente y resolviendo empates con SHA-256 de `42|meta_row_id` y después ID. El ID técnico solo interviene en los empates. El 20% representa una capacidad hipotética común a los dos enfoques.

## 4. Validación y evaluación

La fecha de alta estimada se calcula como llegada prevista menos antelación. Se aplican los cortes de la Entrega 02. La fecha conservadora de madurez de la etiqueta es un día después del máximo entre la salida prevista y la fecha del estado final.

| Bloque | Fecha de alta estimada y elegibilidad | Registros |
| --- | --- | ---: |
| Entrenamiento | Antes del 01/07/2016, con etiqueta madura antes del corte | 48.428 |
| Validación | Desde el 01/07/2016 hasta antes del 01/01/2017, con etiqueta madura antes del segundo corte | 14.011 |
| Prueba | Desde el 01/01/2017, con desenlaces históricos observados en la extracción | 26.565 |
| Purga | Alta anterior a prueba con madurez posterior o igual al corte de su bloque | 30.386 |

Las filas purgadas quedan fuera del ajuste y de la evaluación. Validación y prueba se puntúan con el mismo modelo entrenado en el primer bloque. La madurez sirve como regla offline y el protocolo aproxima el orden temporal. La purga modifica la composición de los primeros periodos, el histórico solo contiene llegadas previstas hasta agosto de 2017 y la prueba subrepresenta reservas tardías con mucha antelación. La falta de ID real impide detectar con certeza versiones distintas de la misma reserva.

La precisión y el recall miden cuántos seguimientos se dirigen a cancelaciones reales y qué proporción de cancelaciones se cubre con el cupo. Average Precision (AP) evalúa el ranking bajo desbalance. ROC-AUC mide discriminación general y Brier mide el error cuadrático de los scores.

| Bloque | Enfoque | AP | ROC-AUC | Brier ↓ | Precisión | Recall | TP / FP / FN / TN |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Validación | Baseline | 0,2617 | 0,5000 | 0,2025 | 25,51% | 19,47% | 714 / 2.085 / 2.953 / 8.259 |
| Validación | Logística | 0,3647 | 0,6165 | 0,1902 | 35,83% | 27,35% | 1.003 / 1.796 / 2.664 / 8.548 |
| Prueba | Baseline | 0,3163 | 0,5000 | 0,2180 | 31,66% | 20,01% | 1.681 / 3.629 / 6.721 / 14.534 |
| Prueba | Logística | 0,4789 | 0,6813 | 0,2014 | 49,91% | 31,54% | 2.650 / 2.660 / 5.752 / 15.503 |

- **Errores operativos.** De 5.310 alertas en prueba, 2.660 corresponden a reservas que no se cancelaron. Quedan 5.752 cancelaciones sin alerta. El 64,01% de los FN procede de City Hotel (3.682) por volumen; la proporción de cancelaciones cubiertas es 39,40% allí y 11,01% en Resort Hotel, frente al 20,13% y 19,69% del baseline. El cupo conjunto se concentra en City Hotel, que recibe 4.851 alertas.
- **Canal y antelación.** En canal directo el recall baja de 21,86% a 3,75% frente al baseline, con 590 cancelaciones omitidas. Con 0–7 días de antelación queda en 6,32% y con 31 días o más alcanza 39,53%. Los datos describen qué reservas quedan fuera del ranking, con etiología y efectos de contacto sin medir.
- **Origen y calidad.** La precisión es 77,57% para PRT y 38,15% para otros países, con prevalencias y mezcla de reservas distintas. En 79 registros sin país se omiten las 9 cancelaciones observadas. En el subgrupo de 156 casos con alguna anomalía de ocupación o estancia se detectan 2 de 13 cancelaciones. Estos conteos pequeños requieren cautela y las diferencias descriptivas no prueban causalidad.
- **Variación temporal.** La precisión logística oscila del 60,75% en febrero al 16,44% en agosto. Agosto contiene 730 reservas y está sujeto al fin de cobertura por llegada. El cambio de prevalencia entre validación (26,17%) y prueba (31,63%) también afecta a la comparación entre periodos.

Evidencias reproducibles en [métricas](../../reports/ml_metricas.json), [particiones](../../reports/ml_particiones.csv), [predicciones](../../reports/ml_predicciones.csv) y [segmentos](../../reports/ml_segmentos.csv).

## 5. Decisión y uso en el producto

Se elige la logística para un MVP retrospectivo y supervisado. Con el mismo cupo mejora la precisión en 18,25 puntos porcentuales y localiza 969 cancelaciones adicionales en prueba. La concentración en el hotel urbano y la baja cobertura del canal directo desaconsejan aplicar la política global como única cola de seguimiento en operación. Para un piloto operativo se necesitan snapshots de alta, evaluación prospectiva por hotel y canal y una medición de la capacidad y los costes reales. El histórico de dos hoteles portugueses de 2015–2017 limita la generalización a la operación actual.

El frontal consumiría un identificador de reserva interno, una puntuación de riesgo de 0 a 1, una posición dentro del lote y un indicador de prioridad. El score ordena la revisión y carece de calibración acreditada como probabilidad individual. La prioridad depende del lote disponible y del cupo. La simulación usa meses completos retrospectivos. Para ofrecer una cola durante el mes habría que validar una cadencia de lotes y un cupo disponibles en cada decisión. Una alerta sirve para revisión humana y no desencadena cancelaciones, cambios de precio ni contactos automáticos.

## Verificación y reproducción

Desde la raíz, con Python 3.14.5 y las [dependencias fijadas](../../requirements.txt):

```powershell
python -m pip install -r requirements.txt
python -B -m src.machine_learning
python -B -m unittest discover -s tests -v
```

| Requisito del enunciado | Evidencia |
| --- | --- |
| 1. Problema, usuario, tipo y acción | Apartado 1 |
| 2. Gold, entradas, objetivo, disponibilidad y riesgos | Apartado 2 |
| 3. Baseline y alternativa justificados y comparados | Apartados 3 y 4 |
| 4. Validación, métricas y errores fuera de entrenamiento | Apartado 4 y archivos `reports/ml_*` |
| 5. Selección, mejora, suficiencia y salida del frontal | Apartado 5 |
| Documento en la ruta obligatoria y código reproducible | Este documento y [script](../../src/machine_learning.py) |
| Comprobaciones ejecutables | [Pruebas](../../tests/test_machine_learning.py) de integridad del Gold, separación temporal, empates, cupo, predicciones y métricas |

