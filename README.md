# Motor de detección de comportamiento en puntos de venta

Código del motor descrito en la sección 3.3.2 del proyecto de grado *Reconocimiento de gestos de manos en puntos de venta mediante deep learning*.

Este codigo implenta la estructura planteada por el autor, usando las distintas estrategias de vision por computador establecidas en el documento. Para su uso adecuado se uso claude en apoyo de construcción, revision y debuggin de codigo.


## Arquitectura


| Archivo | Capa | Qué hace |
|---|---|---|
| `percepcion.py` | Percepción | Corre el estimador de pose multipersona (RF-DETR Keypoint, pesos preentrenados sin ajuste) y devuelve cada persona con su caja y 17 keypoints COCO. Un keypoint con confianza menor a 0,30 se considera no visible. `Rastreador` mantiene un id de pista por IoU. `AsignadorRol` decide si la persona es cajero o cliente según el lado de la línea de rol en que queda su torso. |
| `zonas.py` | Zonas | Convierte la posición de cada muñeca en **visitas** a las regiones. Una visita solo cuenta si la mano cruza 3 bandas concéntricas en orden al entrar y en orden inverso al salir. |
| `pipeline.py` | Traducción | Convierte las visitas en eventos con el rol del actor, los ordena por tiempo y se los entrega al motor. Con `--grabar` dibuja sobre el video la caja, el id, el rol, las muñecas y el rótulo de cada alerta. |
| `motor.py` | Decisión |Una máquina de estados agrupa los eventos en sesiones y sobre cada sesión se evalúan las reglas. |
| `configurar_roi.py` | Configuración | Herramienta interactiva para dibujar las regiones sobre un fotograma y guardarlas en `rois.json`. |

## Instalación

Requiere Python 3.14 (versión probada: 3.14.5).

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
