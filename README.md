# Cancelaciones hoteleras · Entrega 03

Continuación de la [Entrega 02](docs/entregas/02_datos_y_preparacion.md). La [memoria de Machine Learning](docs/entregas/03_machine_learning.md) contiene los cinco apartados, las dos ampliaciones y el cierre académico de las observaciones del profesor.

## Demo de un lunes

Descarga o clona el repositorio y abre [docs/demo/lunes_2017-01-02.html](docs/demo/lunes_2017-01-02.html) en un navegador. La demo permite alternar entre cupo global y reparto por hotel y consultar las reservas que cambian de lista. Los desenlaces históricos se muestran al activar su casilla.

Se utiliza el primer lunes de la simulación, el 02/01/2017, porque ambas políticas parten de las mismas 1.152 reservas pendientes. Con 230 llamadas por política, 164 reservas se mantienen y 66 se sustituyen. El hotel vacacional gana cinco detecciones y el urbano pierde nueve, con un balance total de cuatro detecciones menos. Son resultados retrospectivos del ejemplo y las llamadas son simuladas.

Los datos proceden de `reports/cola_persistente_eventos.csv`. La demo incluye la huella SHA-256 de esa fuente y conserva su [fragmento editable](docs/demo/lunes_2017-01-02.fragment.html).

**OBSERVACIÓN DEL PROFESOR RESUELTA**, dentro del alcance de las simulaciones retrospectivas documentadas. Las once pruebas pasan y los 13 informes de resultados se han reproducido con SHA-256 idéntico.

## Experimentos conservados

| Experimento | Entrada reproducible | Informes | Población de prueba |
| --- | --- | --- | --- |
| Comparación mensual de baseline y logística | `python -B -m src.machine_learning` | Cuatro `reports/ml_*` | 26.565 reservas |
| Cohortes semanales con baseline global, tasa por hotel y antelación y logística | `python -B -m src.weekly_queue` | Cinco `reports/semanal_*` | 18.929 reservas elegibles, 34 lunes |
| Cola persistente, llegadas en 28 días y cupo global o proporcional por hotel | `python -B -m src.persistent_queue` | Cuatro `reports/cola_persistente_*` | 24.097 IDs accesibles, 35 lunes |

Cada experimento tiene su propia población y capacidad. La primera ampliación obtiene 1.322 cancelaciones detectadas con logística frente a 1.074 de la regla por hotel y antelación con los mismos 3.771 seguimientos simulados.

En la cola persistente, ambas políticas realizan 15.416 seguimientos únicos. El cupo global detecta 1.867 cancelaciones y el reparto por hotel 1.796. El recall de Resort Hotel pasa del 31,31% al 63,43%, mientras el global baja del 64,16% al 61,72%. La precisión global del 12,11% está cerca de la prevalencia del 12,08%, con una cobertura acumulada del 63,97% de los IDs. El cierre conserva estos resultados y delimita su interpretación.

## Caso de uso y conclusión

El producto original plantea revisar a las 09:00 las reservas activas dos días antes de la llegada. Las ampliaciones estudian decisiones los lunes a las 00:00 y, en la cola persistente, una ventana de llegadas de 28 días. Esta evidencia evalúa esos escenarios retrospectivos. la validación exacta del uso original requiere identificadores estables, fecha y hora auténticas de creación, historial de estados, snapshots de variables y registros de contactos.

La logística mejora frente a la regla sencilla en la primera simulación semanal, pero la priorización de la cola persistente alcanza una precisión próxima a la prevalencia. El reparto proporcional mejora la cobertura vacacional con una pérdida de detección global cuantificada. Estos resultados todavía no acreditan una mejora operativa suficiente para justificar un despliegue real. Las cancelaciones evitadas gracias a contactos siguen sin estar medidas.

## Reproducir la entrega completa

Desde la raíz con Python 3.14.5:

```powershell
python -m pip install -r requirements.txt
python -B -m src.machine_learning
python -B -m src.weekly_queue
python -B -m src.persistent_queue
python -B -m unittest discover -s tests -v
```

La revisión final ejecutó los tres experimentos y las once pruebas en el entorno existente de Python 3.14.5, pandas 3.0.3, NumPy 2.4.6 y scikit-learn 1.9.0. Se verificó la reproducción exacta de los informes. La instalación en un entorno nuevo queda pendiente de comprobar.

`data/` contiene el original y el Gold heredados, cuyas huellas se verifican al ejecutar. Los scripts ajustan exclusivamente con entrenamiento y regeneran sus informes. Los IDs representan filas técnicas del histórico. Los informes de preparación y el notebook de la Entrega 02 se conservan para dar continuidad al proyecto.

Los resultados adicionales reutilizan prueba y tienen carácter exploratorio. La reconstrucción del estado activo depende de fechas estimadas y del último estado fechado. la falta de snapshots e historial de estados limita su correspondencia con la operación real.
