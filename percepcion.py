"""
Este archivo corresponde a la capa de percepción.

Toma un fotograma y retorna una lista de personas con pose, identidad y rol.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Esqueleto COCO, 17 puntos
# ---------------------------------------------------------------------------

COCO_KEYPOINTS = [
    "nariz", "ojo_izq", "ojo_der", "oreja_izq", "oreja_der",
    "hombro_izq", "hombro_der", "codo_izq", "codo_der",
    "muneca_izq", "muneca_der", "cadera_izq", "cadera_der",
    "rodilla_izq", "rodilla_der", "tobillo_izq", "tobillo_der",
]
KP = {n: i for i, n in enumerate(COCO_KEYPOINTS)}

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

class Detector(ABC):
    @abstractmethod
    def detectar(self, frame_bgr: np.ndarray) -> list[Persona]:
        """Recibe un fotograma BGR de OpenCV y devuelve las personas encontradas."""


class DetectorRFDETR(Detector):

    def __init__(self, umbral: float = 0.4):
        from rfdetr import RFDETRKeypointPreview

        self.umbral = umbral
        device = None
        try:
            import torch
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
        except Exception:                    
            pass
        try:
            self.modelo = RFDETRKeypointPreview(device=device) if device else RFDETRKeypointPreview()
        except TypeError:                          # versiones sin argumento device
            self.modelo = RFDETRKeypointPreview()

    def detectar(self, frame_bgr: np.ndarray) -> list[Persona]:
        rgb = frame_bgr[:, :, ::-1].copy()
        r = self.modelo.predict(rgb, threshold=self.umbral)

        if isinstance(r, (tuple, list)):
            obj_kp = next((x for x in r if hasattr(x, "xy")), None)
            obj_det = next((x for x in r if hasattr(x, "xyxy")), None)
        else:
            obj_kp = r if hasattr(r, "xy") else None
            obj_det = r if hasattr(r, "xyxy") else None
        if obj_kp is None:
            return []

        xy = np.asarray(obj_kp.xy, dtype=float)            # (N, K, 2)
        if xy.size == 0:
            return []

        kconf = getattr(obj_kp, "keypoint_confidence", None)
        if kconf is None:
            kconf = getattr(obj_kp, "confidence", None)
        kconf = (np.asarray(kconf, dtype=float) if kconf is not None
                 else np.ones(xy.shape[:2]))
        if kconf.ndim == 1:                                # (N,) -> (N, K)
            kconf = np.repeat(kconf[:, None], xy.shape[1], axis=1)

        cajas = np.asarray(obj_det.xyxy, dtype=float) if obj_det is not None else None

        personas = []
        for i in range(len(xy)):
            k = np.concatenate([xy[i], kconf[i][:, None]], axis=1)   # (K, 3)

            if cajas is not None and i < len(cajas):
                caja = cajas[i]
            else:
                vis = k[k[:, 2] >= CONF_KP_MIN][:, :2]
                if len(vis) < 2:
                    continue
                x1, y1 = vis.min(axis=0)
                x2, y2 = vis.max(axis=0)
                mx, my = (x2 - x1) * 0.15, (y2 - y1) * 0.15
                caja = np.array([x1 - mx, y1 - my, x2 + mx, y2 + my])

            visibles = k[k[:, 2] >= CONF_KP_MIN]
            conf = float(visibles[:, 2].mean()) if len(visibles) else 0.0
            personas.append(Persona(caja=caja, confianza=conf, keypoints=k))

        return personas


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

        # Emparejamiento voraz por IoU, de mayor a menor.
        pares = sorted(
            ((_iou(p.caja, pista.caja), i, j)
             for i, p in enumerate(personas)
             for j, pista in enumerate(self.pistas)),
            key=lambda x: -x[0],
        )
        det_usadas, pistas_usadas = set(), set()
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
        """Al terminar el video, todas las pistas confirmadas se dan por idas."""
        idas = {p.id for p in self.pistas if p.confirmada}
        self.pistas = []
        return idas


# ---------------------------------------------------------------------------
# Rol por geometría y muñecas
# ---------------------------------------------------------------------------

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
