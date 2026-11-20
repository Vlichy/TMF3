# Entrega 03 · Machine Learning

La ampliación semanal del 21/11/2026, al final de este documento, responde al feedback del profesor y actualiza la decisión para el MVP. Los cinco apartados iniciales conservan el experimento mensual original.

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


## Ampliación · Cola semanal solicitada por el profesor

Experimento adicional ejecutado el 21/11/2026 mediante [src/weekly_queue.py](../../src/weekly_queue.py). El conjunto de prueba ya había sido consultado en la entrega original. Esta ampliación constituye un análisis retrospectivo exploratorio. Se fija el protocolo, sin optimizar sus parámetros con estos resultados.

### Momento de decisión y reservas disponibles

Cada lunes a las 00:00 se revisan las altas estimadas del intervalo lunes anterior incluido a lunes actual excluido. Se admite una reserva si su llegada prevista es ese lunes o posterior y su estado final no consta fechado antes del cierre. Una fecha de estado del propio lunes se considera aún desconocida por la falta de hora. La fecha del estado solo permite reconstruir la elegibilidad y queda fuera de los predictores y del ranking. Es necesario disponer en operación del estado vigente y de una instantánea real de la reserva.

Se ordenan exclusivamente las reservas elegibles de esa semana. El cupo es floor(n_elegibles / 5), común para los dos hoteles. Los empates usan la misma huella reproducible de la entrega. Las reservas sin prioridad salen de este lote y no se vuelven a revisar. Cada reserva tiene una única oportunidad de seguimiento, con espera de hasta siete días. El cupo se calcula al cerrar la semana, cuando su volumen ya es conocido. Representa una capacidad proporcional hipotética y tendría que contrastarse con horas de trabajo reales. Las reservas que se resuelven o llegan antes del cierre quedan fuera del alcance de esta cadencia.

Se reutiliza el entrenamiento original de 48.428 filas con etiquetas maduras antes del 01/07/2016. El modelo y las tasas se mantienen fijos en todas las semanas, sin realimentación de validación o prueba. Para evaluar validación se recuperan también las altas que la entrega mensual purgaba por tener desenlace posterior al 01/01/2017. Al mantener fijo el modelo, sus resultados futuros se pueden medir retrospectivamente sin usarlos para entrenar. Así se evita seleccionar las reservas semanales de validación por el momento en que terminaron cancelándose.

Se incluyen semanas completas dentro de cada bloque. Los cierres de validación abarcan del 11/07/2016 al 26/12/2016 y los de prueba del 09/01/2017 al 28/08/2017. Las semanas que cruzan un corte y el último fragmento de agosto quedan fuera. El Gold solo cubre llegadas hasta agosto de 2017, lo que limita la representatividad de las últimas semanas.

| Bloque | Semanas | Altas de las cohortes | Elegibles al cierre | Excluidas antes de ordenar | Cupo total |
| --- | ---: | ---: | ---: | ---: | ---: |
| Validación | 25 | 25.663 | 19.646 | 6.017 | 3.920 |
| Prueba | 34 | 26.427 | 18.929 | 7.498 | 3.771 |

En prueba, 3.952 reservas ya habían llegado y 6.570 tenían desenlace conocido antes del cierre. Ambos motivos se solapan y su unión da las 7.498 exclusiones. La prevalencia de cancelación entre las reservas elegibles de prueba es 23,85%. Estas poblaciones y oportunidades de seguimiento difieren de las del experimento mensual, por lo que sus métricas no permiten aislar causalmente el efecto de cambiar de cadencia.

### Regla sencilla y comparación

La nueva regla utiliza seis celdas, combinando hotel con los tramos de antelación 0–7, 8–30 y 31 o más días, fijados antes de ejecutar. Su score es (cancelaciones_celda + 20 × tasa_global_train) / (reservas_celda + 20). El suavizado reduce la inestabilidad de celdas pequeñas. Una combinación desconocida recibe la tasa global. Las tasas y sus recuentos se guardan en [semanal_tasas_train.csv](../../reports/semanal_tasas_train.csv).

Se comparan tres enfoques con exactamente las mismas reservas y el mismo cupo semanal. Se conserva el baseline global con desempate por hash para distinguir el valor de la regla sencilla y el de la logística.

| Bloque | Enfoque | Precisión | Recall | TP | FP | FN | AP | ROC-AUC | Brier ↓ |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Validación | Baseline global | 30,66% | 19,97% | 1.202 | 2.718 | 4.818 | 0,3064 | 0,5000 | 0,2152 |
| Validación | Hotel y antelación | 38,57% | 25,12% | 1.512 | 2.408 | 4.508 | 0,3757 | 0,6257 | 0,2127 |
| Validación | Logística | 55,08% | 35,86% | 2.159 | 1.761 | 3.861 | 0,5198 | 0,6939 | 0,2034 |
| Prueba | Baseline global | 23,60% | 19,71% | 890 | 2.881 | 3.625 | 0,2385 | 0,5000 | 0,1959 |
| Prueba | Hotel y antelación | 28,48% | 23,79% | 1.074 | 2.697 | 3.441 | 0,2845 | 0,6025 | 0,2040 |
| Prueba | Logística | 35,06% | 29,28% | 1.322 | 2.449 | 3.193 | 0,3273 | 0,6264 | 0,1996 |

La logística localiza 248 cancelaciones adicionales frente a la regla por hotel y antelación, con 6,58 puntos más de precisión y 5,49 puntos más de recall. Supera a esa regla en TP en 24 de las 34 semanas de prueba, empata en 3 y queda por debajo en 7. Su Brier empeora frente al baseline global en prueba, aunque mejora la priorización. Los scores requieren validación de calibración antes de interpretarlos como probabilidades.

### Qué ocurre en el hotel vacacional

El cupo conjunto permite que cada enfoque distribuya el trabajo entre hoteles según su ranking.

| Hotel en prueba | Enfoque | Elegibles | Alertas | TP | FP | FN | Precisión | Recall |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| City Hotel | Baseline global | 12.816 | 2.567 | 620 | 1.947 | 2.490 | 24,15% | 19,94% |
| City Hotel | Hotel y antelación | 12.816 | 3.727 | 1.061 | 2.666 | 2.049 | 28,47% | 34,12% |
| City Hotel | Logística | 12.816 | 3.386 | 1.136 | 2.250 | 1.974 | 33,55% | 36,53% |
| Resort Hotel | Baseline global | 6.113 | 1.204 | 270 | 934 | 1.135 | 22,43% | 19,22% |
| Resort Hotel | Hotel y antelación | 6.113 | 44 | 13 | 31 | 1.392 | 29,55% | 0,93% |
| Resort Hotel | Logística | 6.113 | 385 | 186 | 199 | 1.219 | 48,31% | 13,24% |

En Resort Hotel la logística mejora frente a la regla sencilla, con 173 TP adicionales. La regla concede casi todo el cupo al hotel urbano porque su celda City Hotel con 31 o más días tiene una tasa de entrenamiento del 52,29%, frente al 35,33% de la celda análoga vacacional. La logística concede más alertas al vacacional, pero su recall sigue por debajo del baseline global. En validación el recall vacacional de la logística era 27,14%, frente al 13,24% de prueba, mostrando variación temporal relevante.

### Decisión actualizada y límites

La evidencia semanal mantiene la elección de la logística para explorar una cola supervisada orientada a detectar cancelaciones con capacidad limitada. La política global pierde cobertura en el hotel vacacional frente al seguimiento sin información individual. Si el producto debe garantizar cobertura en ambos hoteles, hace falta evaluar explícitamente una asignación de capacidad por hotel antes de decidir su uso operativo. La regla por hotel y antelación resulta insuficiente para corregir esa concentración.

La simulación reduce el uso de información de reservas futuras al ordenar, pero siguen pendientes la fecha real de alta, los snapshots iniciales y el historial de estados. El último estado fechado aproxima cuándo se conoció un desenlace y podría reflejar una actualización tardía. La espera semanal excluye cancelaciones rápidas y reservas de llegada inmediata. Las métricas describen cancelaciones detectables entre los elegibles al cierre, con efectos de contacto y rentabilidad aún sin medir.

### Evidencias y comprobaciones

- [Métricas globales y tamaños](../../reports/semanal_metricas.json).
- [Altas y elegibilidad de cada reserva](../../reports/semanal_elegibilidad.csv).
- [Scores y prioridades de los tres enfoques](../../reports/semanal_predicciones.csv).
- [Resultados por semana y hotel](../../reports/semanal_segmentos.csv).
- [Pruebas de la cola semanal](../../tests/test_weekly_queue.py).

Se ejecutaron siete pruebas en total, incluyendo las tres anteriores. Las cuatro nuevas comprueban que añadir reservas futuras conserve decisiones previas, los límites del lunes y los desenlaces conocidos, las tasas ajustadas con train y el fallback de grupos desconocidos, y la elegibilidad, el cupo y las métricas sobre las evidencias reales.

Para reproducir la ampliación desde la raíz:

```powershell
python -B -m src.weekly_queue
python -B -m unittest discover -s tests -v
```

