"""
Este archivo corresponde a la capa de percepción.

Toma un fotograma y retorna una lista de personas con pose, identidad y rol.
"""

from dataclasses import dataclass
import numpy as np
from ultralytics import YOLO
# ---------------------------------------------------------------------------
# Esqueleto COCO, 17 puntos
# ---------------------------------------------------------------------------

KEYPOINTS = [
    "nariz", "ojo_izq", "ojo_der", "oreja_izq", "oreja_der",
    "hombro_izq", "hombro_der", "codo_izq", "codo_der",
    "muneca_izq", "muneca_der", "cadera_izq", "cadera_der",
    "rodilla_izq", "rodilla_der", "tobillo_izq", "tobillo_der",
]
KP = {n: i for i, n in enumerate(KEYPOINTS)}

CONF_KP_MIN = 0.30     


@dataclass
class Persona:
    # Keypoints: [x, y, confianza].
    caja: np.ndarray
    confianza: float
    keypoints: np.ndarray
    id: int | None = None
    rol: str | None = None

    def punto(self, nombre: str) -> tuple[float, float, float]:
        x, y, c = self.keypoints[KP[nombre]]
        return float(x), float(y), float(c)

    def visible(self, nombre: str) -> bool:
        return self.keypoints[KP[nombre]][2] >= CONF_KP_MIN

    @property
    def centro(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.caja
        return float((x1 + x2) / 2), float((y1 + y2) / 2)


# ---------------------------------------------------------------------------
# Estimador de pose
# ---------------------------------------------------------------------------

class EstimadorPose:

    def __init__(self, pesos: str = "yolo26n-pose.pt", umbral: float = 0.4):
        self.umbral = umbral
        self.modelo = YOLO(pesos)

    def detectar(self, frame_bgr: np.ndarray) -> list[Persona]:
        r = self.modelo.predict(frame_bgr, conf=self.umbral, verbose=False)[0]
        if r.keypoints is None or len(r.boxes) == 0:
            return []
        cajas = r.boxes.xyxy.cpu().numpy().astype(float)
        confs = r.boxes.conf.cpu().numpy().astype(float)
        kps = r.keypoints.data.cpu().numpy().astype(float)  
        return [Persona(caja=cajas[i], confianza=float(confs[i]), keypoints=kps[i])
                for i in range(len(cajas))]


# ---------------------------------------------------------------------------
# Seguimiento por IoU con histéresis
# ---------------------------------------------------------------------------

def _iou(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return float(inter / (area_a + area_b - inter))


@dataclass
class _Pista:
    id: int
    caja: np.ndarray
    aciertos: int = 0
    perdidos: int = 0
    confirmada: bool = False


class Rastreador:

    def __init__(self, iou_min: float = 0.3, min_aciertos: int = 5,
                 max_perdidos: int = 15):
        self.iou_min = iou_min
        self.min_aciertos = min_aciertos
        self.max_perdidos = max_perdidos
        self.pistas: list[_Pista] = []
        self._siguiente_id = 1
        self.nuevas_confirmadas: set = set()
        self.perdidas: set = set()

    def actualizar(self, personas: list[Persona]) -> list[Persona]:
        self.nuevas_confirmadas = set()
        self.perdidas = set()
        det_usadas = set()
        pistas_usadas = set()

        
        pares = []

        for i, persona in enumerate(personas):
            for j, pista in enumerate(self.pistas):
                iou = _iou(persona.caja, pista.caja)
                pares.append((iou, i, j))
        pares.sort(key=lambda par: par[0], reverse=True)

        
        for s, i, j in pares:
            if s < self.iou_min or i in det_usadas or j in pistas_usadas:
                continue
            det_usadas.add(i)
            pistas_usadas.add(j)
            pista = self.pistas[j]
            pista.caja = personas[i].caja
            pista.aciertos += 1
            pista.perdidos = 0
            if not pista.confirmada and pista.aciertos >= self.min_aciertos:
                pista.confirmada = True
                self.nuevas_confirmadas.add(pista.id)
            personas[i].id = pista.id

        for i, p in enumerate(personas):
            if i not in det_usadas:
                pista = _Pista(id=self._siguiente_id, caja=p.caja, aciertos=1)
                self._siguiente_id += 1
                self.pistas.append(pista)
                p.id = pista.id

        for j, pista in enumerate(self.pistas):
            if j not in pistas_usadas:
                pista.perdidos += 1

        vivas = []
        for pista in self.pistas:
            if pista.perdidos > self.max_perdidos:
                if pista.confirmada:
                    self.perdidas.add(pista.id)
            else:
                vivas.append(pista)
        self.pistas = vivas

        return personas

    def confirmadas(self) -> set:
        return {p.id for p in self.pistas if p.confirmada and p.perdidos == 0}

    def vaciar(self) -> set:
        idas = {p.id for p in self.pistas if p.confirmada}
        self.pistas = []
        return idas


class AsignadorRol:

    def __init__(self, linea, memoria: int = 15):
        self.p0 = np.asarray(linea[0], dtype=float)
        self.p1 = np.asarray(linea[1], dtype=float)
        self.memoria = memoria
        self.votos: dict = {}      

    def _lado(self, punto) -> int:
        v = self.p1 - self.p0
        w = np.asarray(punto, dtype=float) - self.p0
        return 1 if v[0] * w[1] - v[1] * w[0] > 0 else -1

    def _ancla(self, p: Persona):
        for par in (("cadera_izq", "cadera_der"), ("hombro_izq", "hombro_der")):
            if p.visible(par[0]) and p.visible(par[1]):
                return (p.keypoints[KP[par[0]]][:2] + p.keypoints[KP[par[1]]][:2]) / 2.0
        return np.array(p.centro)

    def asignar(self, personas: list[Persona]) -> list[Persona]:
        for p in personas:
            if p.id is None:
                continue
            hist = self.votos.setdefault(p.id, [])
            hist.append(self._lado(self._ancla(p)))
            if len(hist) > self.memoria:
                hist.pop(0)
            p.rol = "cajero" if sum(hist) > 0 else "cliente"
        return personas

    def rol_de(self, id_pista: int) -> str | None:
        hist = self.votos.get(id_pista)
        if not hist:
            return None
        return "cajero" if sum(hist) > 0 else "cliente"

    def olvidar(self, id_pista: int) -> None:
        self.votos.pop(id_pista, None)


def munecas(p: Persona) -> dict:
    salida = {}
    for lado, nombre in (("izq", "muneca_izq"), ("der", "muneca_der")):
        if p.visible(nombre):
            x, y, _ = p.punto(nombre)
            salida[lado] = (x, y)
    return salida
