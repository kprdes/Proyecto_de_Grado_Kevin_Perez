"""
Pipeline: video pregrabado -> percepción -> zonas -> motor -> alertas.

Los comandos que permiten usar este pipeline son:
    python3 pipeline.py video.mp4                        
    python3 pipeline.py video.mp4 --grabar salida.mp4 

Requiere rois.json, generado con configurar_roi.py para la posición de cámara.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from motor import Config, Evento, Rol, TipoEvento, Vigia, Zona
from percepcion import AsignadorRol, DetectorRFDETR, Rastreador, munecas
from zonas import GestorZonas

NOMBRE_A_ZONA = {
    "Z_CAJON": Zona.CAJON,
    "Z_MOSTRADOR": Zona.MOSTRADOR,
    "Z_OCULTA": Zona.OCULTA,
}

ROLES = {"cajero": Rol.CAJERO, "cliente": Rol.CLIENTE}


class Pipeline:
    def __init__(self, rois: dict, detector, cfg: Config | None = None):
        self.detector = detector
        self.vigia = Vigia(cfg)
        self.rastreador = Rastreador(min_aciertos=5, max_perdidos=15)

        zonas = rois.get("zonas", {})
        poligonos = {n: np.asarray(d["puntos"], dtype=np.int32)
                     for n, d in zonas.items()
                     if n in NOMBRE_A_ZONA and d.get("tipo") == "poligono"}
        if not poligonos:
            raise SystemExit("rois.json no tiene zonas. Ejecuta primero: "
                             "python3 configurar_roi.py video.mp4")
        if "LINEA_ROL" not in zonas:
            raise SystemExit("rois.json no tiene LINEA_ROL: sin ella no se distingue "
                             "al cajero del cliente. Vuelve a correr configurar_roi.py.")

        cfg = self.vigia.cfg
        self.gestor = GestorZonas(poligonos, n_bandas=cfg.n_bandas, ancho=cfg.ancho_banda)
        self.asignador = AsignadorRol(zonas["LINEA_ROL"]["puntos"], memoria=15)

        self.alertas: list = []
        self.eventos: list = []
        self.personas: list = []
        self._presentes: dict = {}     

    def _emitir(self, ev: Evento) -> list:
        self.eventos.append(ev)
        nuevas = self.vigia.procesar(ev)
        self.alertas.extend(nuevas)
        return nuevas

    def procesar_frame(self, frame, t: float) -> list:
        nuevas_alertas = []

        personas = self.detector.detectar(frame)
        personas = self.rastreador.actualizar(personas)
        personas = self.asignador.asignar(personas)
        self.personas = personas
        confirmadas = self.rastreador.confirmadas()
        por_id = {p.id: p for p in personas if p.id is not None}

        for id_pista in self.rastreador.nuevas_confirmadas:
            rol = ROLES.get(self.asignador.rol_de(id_pista))
            if rol is None:
                continue
            self._presentes[id_pista] = rol
            conf = float(por_id[id_pista].confianza) if id_pista in por_id else 1.0
            nuevas_alertas += self._emitir(
                Evento(tipo=TipoEvento.PERSONA_ENTRA, t=t, actor=rol, confianza=conf))

        for id_pista in self.rastreador.perdidas:
            rol = self._presentes.pop(id_pista, None)
            self.asignador.olvidar(id_pista)
            if rol is not None:
                nuevas_alertas += self._emitir(
                    Evento(tipo=TipoEvento.PERSONA_SALE, t=t, actor=rol))
        manos = {}
        for p in personas:
            if p.id in confirmadas:
                for lado, punto in munecas(p).items():
                    manos[(p.id, lado)] = punto

        pendientes = []
        for visita in self.gestor.actualizar(t, manos):
            zona = NOMBRE_A_ZONA.get(visita.zona)
            rol = ROLES.get(self.asignador.rol_de(int(visita.actor)))
            if zona is None or rol is None:
                continue
            pendientes.append(Evento(tipo=TipoEvento.MANO_ENTRA, t=visita.t_entrada,
                                     actor=rol, zona=zona, mano=visita.mano, confianza=0.9))
            pendientes.append(Evento(tipo=TipoEvento.MANO_SALE, t=visita.t_salida,
                                     actor=rol, zona=zona, mano=visita.mano, confianza=0.9))
        for ev in sorted(pendientes, key=lambda e: e.t):
            nuevas_alertas += self._emitir(ev)

        nuevas = self.vigia.tick(t)
        self.alertas.extend(nuevas)
        return nuevas_alertas + nuevas

    def finalizar(self, t: float) -> list:
        for id_pista in self.rastreador.vaciar():
            rol = self._presentes.pop(id_pista, None)
            if rol is not None:
                self._emitir(Evento(tipo=TipoEvento.PERSONA_SALE, t=t, actor=rol))
        nuevas = self.vigia.finalizar(t)
        self.alertas.extend(nuevas)
        return nuevas


def dibujar_marcas(frame, personas, alertas_recientes):
    for p in personas:
        if p.id is None:
            continue
        col = (80, 220, 80) if p.rol == "cajero" else (80, 160, 255)
        x1, y1, x2, y2 = p.caja.astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
        cv2.putText(frame, f"#{p.id} {p.rol or '?'}", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
        for x, y in munecas(p).values():
            cv2.circle(frame, (int(x), int(y)), 6, col, -1)
    for a in alertas_recientes:
        cv2.putText(frame, f"{a.codigo} {a.nombre}", (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 60, 255), 2)
    return frame


def imprimir(a) -> None:
    print(f"  [{a.nivel:5s}] {a.t:7.1f}s  {a.codigo}  {a.nombre}  score={a.score:.2f}")
    print(f"          {a.detalle}")


def main():
    ap = argparse.ArgumentParser(description="Motor de detección de comportamiento sobre video.")
    ap.add_argument("video")
    ap.add_argument("--rois", default="rois.json")
    ap.add_argument("--grabar", default=None, help="ruta del video anotado de salida")
    args = ap.parse_args()

    rois = json.loads(Path(args.rois).read_text(encoding="utf-8"))
    print("Cargando estimador de pose...")
    pipe = Pipeline(rois, DetectorRFDETR())

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"No pude abrir '{args.video}'.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    escritor, ultimas, idx = None, [], 0
    print("Procesando. Ctrl-C para cortar.\n")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            nuevas = pipe.procesar_frame(frame, idx / fps)
            if nuevas:
                ultimas = nuevas
                for a in nuevas:
                    imprimir(a)
            if args.grabar:
                if escritor is None:
                    h, w = frame.shape[:2]
                    escritor = cv2.VideoWriter(
                        args.grabar, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                escritor.write(dibujar_marcas(frame, pipe.personas, ultimas))
            idx += 1
    except KeyboardInterrupt:
        print("\nInterrumpido.")

    for a in pipe.finalizar(idx / fps):
        imprimir(a)
    cap.release()
    if escritor is not None:
        escritor.release()

    print(f"\n{idx} fotogramas · {len(pipe.eventos)} eventos · {len(pipe.alertas)} alertas")
    if pipe.eventos:
        Path("eventos_generados.json").write_text(json.dumps(
            [{"tipo": e.tipo.value, "t": round(e.t, 2),
              "actor": e.actor.value if e.actor else None,
              "zona": e.zona.value if e.zona else None,
              "mano": e.mano} for e in pipe.eventos],
            indent=2, ensure_ascii=False), encoding="utf-8")
        print("Eventos guardados en eventos_generados.json")


if __name__ == "__main__":
    main()
