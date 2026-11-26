"""
Configurador de las regiones de interés.

Se usa con el comando:
    python3 configurar_roi.py video.mp4
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ZONAS = [
    ("Z_CAJON",     "Cajón registrador (interior)",                    "poligono", (60, 200, 255)),
    ("Z_MOSTRADOR", "Superficie del mostrador (zona de intercambio)",  "poligono", (120, 255, 120)),
    ("Z_OCULTA",    "Zona OCULTA: bajo el mostrador, maletas, repisa", "poligono", (80, 80, 255)),
    ("LINEA_ROL",   "Línea del mostrador. IMPORTANTE: el CAJERO queda a la DERECHA de la flecha", "linea", (255, 140, 60)),
]

VENTANA = "Configurador de ROIs  ·  ENTER=siguiente  u=deshacer  g=guardar  ESC=salir"


class Configurador:
    def __init__(self, ruta_video: str):
        self.cap = cv2.VideoCapture(ruta_video)
        if not self.cap.isOpened():
            raise SystemExit(
                f"No pude abrir '{ruta_video}'.\n"
                "Si OpenCV no lo lee, conviértelo a H.264, por ejemplo en macOS:\n"
                f"  avconvert -s {ruta_video} -o salida.mp4 -p PresetHighestQuality"
            )
        self.total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        self.idx = 0
        self.frame = None
        self._leer(0)

        self.i_zona = 0
        self.puntos: list[tuple[int, int]] = []
        self.resultado: dict = {}

    def _leer(self, idx: int) -> None:
        idx = max(0, min(idx, self.total - 1))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, fr = self.cap.read()
        if ok:
            self.frame = fr
            self.idx = idx


    def _click(self, evento, x, y, flags, param):
        if evento == cv2.EVENT_LBUTTONDOWN and self.i_zona < len(ZONAS):
            tipo = ZONAS[self.i_zona][2]
            if tipo == "linea" and len(self.puntos) >= 2:
                return
            self.puntos.append((x, y))

    def _cerrar_zona(self) -> None:
        if self.i_zona >= len(ZONAS):
            return
        clave, _, tipo, _ = ZONAS[self.i_zona]
        minimo = 2 if tipo == "linea" else 3
        if len(self.puntos) >= minimo:
            self.resultado[clave] = {"tipo": tipo, "puntos": self.puntos.copy()}
        self.puntos = []
        self.i_zona += 1


    def _pintar(self) -> np.ndarray:
        img = self.frame.copy()
        capa = img.copy()

        # zonas ya definidas
        for k, dato in self.resultado.items():
            color = next(c for kk, _, _, c in ZONAS if kk == k)
            pts = np.array(dato["puntos"], np.int32)
            if dato["tipo"] == "poligono":
                cv2.fillPoly(capa, [pts], color)
                cv2.polylines(img, [pts], True, color, 2)
                cx, cy = pts.mean(axis=0).astype(int)
                cv2.putText(img, k, (cx - 45, cy), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (0, 0, 0), 3)
                cv2.putText(img, k, (cx - 45, cy), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, color, 1)
            else:
                import numpy as _np
                p0, p1 = tuple(dato["puntos"][0]), tuple(dato["puntos"][1])
                cv2.line(img, p0, p1, color, 3)
                cv2.arrowedLine(img, p0, p1, color, 2, tipLength=0.05)
                # Marca de qué lado queda el cajero: perpendicular a la
                # derecha de la flecha. Es el error más fácil de cometer y el
                # más difícil de notar después.
                v = _np.array(p1, float) - _np.array(p0, float)
                n = _np.linalg.norm(v)
                if n > 1:
                    perp = _np.array([-v[1], v[0]]) / n
                    m = (_np.array(p0, float) + _np.array(p1, float)) / 2
                    for signo, txt, col in ((1, "CAJERO", (80, 220, 80)),
                                            (-1, "CLIENTE", (80, 160, 255))):
                        q = (m + perp * 55 * signo).astype(int)
                        cv2.putText(img, txt, (q[0] - 34, q[1]),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4)
                        cv2.putText(img, txt, (q[0] - 34, q[1]),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)

        img = cv2.addWeighted(capa, 0.28, img, 0.72, 0)

        # zona en construcción
        if self.i_zona < len(ZONAS):
            color = ZONAS[self.i_zona][3]
            for i, p in enumerate(self.puntos):
                cv2.circle(img, p, 5, color, -1)
                if i:
                    cv2.line(img, self.puntos[i - 1], p, color, 2)
            if (len(self.puntos) == 2 and ZONAS[self.i_zona][2] == "linea"):
                v = np.array(self.puntos[1], float) - np.array(self.puntos[0], float)
                n = np.linalg.norm(v)
                if n > 1:
                    perp = np.array([-v[1], v[0]]) / n
                    m = (np.array(self.puntos[0], float) + np.array(self.puntos[1], float)) / 2
                    for signo, txt, col in ((1, "CAJERO", (80, 220, 80)),
                                            (-1, "CLIENTE", (80, 160, 255))):
                        q = (m + perp * 55 * signo).astype(int)
                        cv2.putText(img, txt, (q[0] - 34, q[1]),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4)
                        cv2.putText(img, txt, (q[0] - 34, q[1]),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
            if len(self.puntos) > 2 and ZONAS[self.i_zona][2] == "poligono":
                cv2.line(img, self.puntos[-1], self.puntos[0], color, 1, cv2.LINE_AA)

        self._panel(img)
        return img

    def _panel(self, img) -> None:
        h, w = img.shape[:2]
        cv2.rectangle(img, (0, 0), (w, 76), (24, 24, 24), -1)

        if self.i_zona < len(ZONAS):
            clave, etiqueta, tipo, color = ZONAS[self.i_zona]
            n = "2 puntos" if tipo == "linea" else "3+ puntos"
            txt = f"[{self.i_zona + 1}/{len(ZONAS)}]  {etiqueta}   ({n}, llevas {len(self.puntos)})"
        else:
            color = (120, 255, 120)
            txt = "Todas las zonas definidas.  Presiona 'g' para guardar."

        cv2.putText(img, txt, (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2)
        cv2.putText(img, f"frame {self.idx}/{self.total - 1}   ,/. mover   u deshacer   "
                         f"r reiniciar   TAB saltar   ENTER siguiente   g guardar",
                    (14, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (170, 170, 170), 1)


    def ejecutar(self) -> dict | None:
        cv2.namedWindow(VENTANA, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(VENTANA, self._click)

        while True:
            cv2.imshow(VENTANA, self._pintar())
            k = cv2.waitKey(20) & 0xFF

            if k == 27:                       # ESC
                cv2.destroyAllWindows()
                return None
            elif k in (13, 10):               # ENTER
                self._cerrar_zona()
            elif k == 9:                      # TAB
                self.puntos = []
                self.i_zona += 1
            elif k == ord("u") and self.puntos:
                self.puntos.pop()
            elif k == ord("r"):
                self.puntos = []
            elif k == ord(","):
                self._leer(self.idx - 30)
            elif k == ord("."):
                self._leer(self.idx + 30)
            elif k == ord("g"):
                if self.puntos:
                    self._cerrar_zona()
                cv2.destroyAllWindows()
                return self.resultado


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit("Uso: python3 configurar_roi.py <video.mp4>")

    ruta = sys.argv[1]
    cfg = Configurador(ruta)
    res = cfg.ejecutar()

    if res is None:
        print("Cancelado, no se guardó nada.")
        return

    h = int(cfg.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cfg.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    salida = {
        "video": Path(ruta).name,
        "resolucion": [w, h],
        "fps": round(cfg.cap.get(cv2.CAP_PROP_FPS), 2),
        "zonas": res,
    }
    Path("rois.json").write_text(json.dumps(salida, indent=2, ensure_ascii=False))

    print(f"\nGuardado en rois.json  ({w}x{h} @ {salida['fps']} fps)")
    for k, v in res.items():
        print(f"   {k:<12} {v['tipo']:<10} {len(v['puntos'])} puntos")
    faltan = [k for k, _, _, _ in ZONAS if k not in res]
    if faltan:
        print(f"   sin definir: {', '.join(faltan)}")


if __name__ == "__main__":
    main()
