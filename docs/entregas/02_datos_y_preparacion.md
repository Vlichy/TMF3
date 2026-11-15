# Entrega 02 · Datos y preparación

## 1. Qué datos necesita el producto

El producto ayudaría al equipo de reservas a priorizar el seguimiento de reservas con riesgo de cancelación. Cada fila representa una reserva registrada, incluida si terminó cancelada. Para preparar el prototipo necesitamos el hotel, la fecha prevista de llegada, la antelación, la duración de la estancia y el canal de venta. Para aprender de reservas históricas necesitamos además saber si se cancelaron. La habitación reservada, el régimen y la composición del grupo aportan contexto. Para un piloto sería deseable contar con un identificador interno, la fecha real de creación y una copia de los campos tal como estaban al reservar. Conviene disponer de al menos un ciclo anual completo para observar la estacionalidad.

## 2. Fuentes reales y viabilidad

La referencia es [Hotel booking demand datasets, de Antonio, de Almeida y Nunes](https://pmc.ncbi.nlm.nih.gov/articles/PMC6297060/), DOI 10.1016/j.dib.2018.11.126. Los suplementos del artículo contienen las extracciones de un hotel vacacional y uno urbano en Portugal, con llegadas previstas entre julio de 2015 y agosto de 2017. Es una publicación estática: sirve para un prototipo reproducible y no ofrece reservas recientes.

Para reproducir esta entrega se usa la copia consolidada ya disponible en [data/raw/hotel_bookings.csv](../../data/raw/hotel_bookings.csv), con 119.390 filas y 32 columnas. La huella esperada está en [el inventario de inspección](../../reports/inspeccion_dataset.json). Los dos ficheros del artículo son una referencia alternativa; si se usan para reconstruir el insumo, hay que armonizar nombres, añadir el tipo de hotel y verificar el esquema y los recuentos. No se presupone que generen un CSV idéntico. Para actualizar el producto haría falta una exportación autorizada del sistema de reservas del hotel.

## 3. Uso, privacidad y datos personales

El artículo se publica bajo [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) y se atribuye aquí a sus autores. Antes de redistribuir una copia de terceros, comprobaríamos sus condiciones concretas. Los autores retiraron los datos que identificaban a clientes y hoteles. Para un piloto no necesitamos nombres, contactos ni datos de pago. Si hace falta unir la versión inicial de una reserva con su resultado, basta un identificador interno seudonimizado, separado de las variables del modelo y con acceso restringido.

## 4. Del dato bruto al dato utilizable

| Capa | Qué contiene y cómo se obtiene |
| --- | --- |
| Bronze / Raw | CSV local de origen, conservado sin cambios. Se comprueba su SHA-256 antes de preparar los datos. |
| Silver | [Limpieza en memoria](../../src/cleaning.py): se normalizan tipos, se mantienen nulos de país y número de niños y se señalan anomalías. Se conservan las filas repetidas porque sin un ID de reserva no se puede distinguir una duplicación técnica de dos reservas iguales. No se necesitan cruces entre tablas. |
| Gold | [CSV preparado](../../data/processed/hotel_bookings_prepared.csv): 119.390 filas, 14 variables candidatas, la etiqueta `is_canceled` (1 = cancelada) y metadatos separados para auditoría. |

Las seis variables numéricas son `lead_time`, `total_nights`, `adults`, `children`, `babies` y `children_missing`. El total de noches suma las de semana y fin de semana; el indicador permite distinguir un número de niños desconocido de un cero. Las ocho categóricas son `hotel`, `arrival_date_month`, `meal`, `country`, `market_segment`, `distribution_channel`, `reserved_room_type` y `customer_type`.

Se excluyen de los predictores la etiqueta, los metadatos y los campos que revelan el resultado, como `reservation_status` y `reservation_status_date`. También se excluyen campos cuya versión inicial es incierta, como el número de cambios de la reserva. La imputación y la codificación se dejan para la siguiente fase y se ajustarán solo con entrenamiento.

Desde la raíz, con Python 3.14.5 y las [dependencias del proyecto](../../requirements.txt), ejecutar:

```powershell
python -m src.preparation
```

El [script](../../src/preparation.py) verifica la entrada y regenera el CSV preparado y sus informes locales. El [notebook de preparación](../../notebooks/preparacion.ipynb) explica el flujo paso a paso. Esta entrega aporta referencias a los datos existentes sin añadir otra copia de los CSV.

## 5. Riesgos y limitaciones

- Faltan la fecha real de creación, un ID estable y la versión de los campos en el momento de reservar; no se puede demostrar que todos los predictores estuvieran disponibles entonces.
- Dos hoteles portugueses y un histórico cerrado en 2017 limitan la representatividad para otros hoteles y para la operación actual.
- Si la copia principal dejara de estar disponible, recurriríamos a los suplementos originales y volveríamos a verificar cobertura, esquema y recuentos antes de usar el Gold.
