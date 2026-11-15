# Cancelaciones hoteleras · Entrega 03

El proyecto continúa la [Entrega 02](docs/entregas/02_datos_y_preparacion.md). La [memoria de Machine Learning](docs/entregas/03_machine_learning.md) contiene los cinco apartados, la decisión para el MVP y los resultados.

## Reproducir

Desde la raíz con Python 3.14.5:

```powershell
python -m pip install -r requirements.txt
python -B -m src.machine_learning
python -B -m unittest discover -s tests -v
```

`data/` contiene el original y el Gold heredados. El experimento lee y verifica el Gold y escribe cuatro archivos `ml_*` en `reports/`. Los informes de preparación y el notebook de la Entrega 02 se incluyen para dar continuidad al proyecto. Los IDs de los informes nuevos identifican filas del histórico.

