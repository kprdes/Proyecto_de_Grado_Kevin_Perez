"""
Este archivo corresponde a la capa de zonas

La literatura usa líneas rectas paralelas al borde. Aquí se usa la distancia con
signo al polígono, de modo que funciona con regiones de cualquier forma:

    banda 0 = fuera        banda 1, 2, 3 = cada vez más adentro

    NIVEL 1   posición -> índice de banda
    NIVEL 2   secuencia de bandas -> ENTRADA / SALIDA aceptada
"""

from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np


class EstadoMano(str, Enum):
    FUERA = "FUERA"
    ENTRANDO = "ENTRANDO" 
    DENTRO = "DENTRO"
    SALIENDO = "SALIENDO"


@dataclass
class Visita:
    """Un ciclo entrada-salida completo de una mano en una zona."""
    actor: str
    zona: str
    t_entrada: float
    t_salida: float | None = None
    mano: str = ""                       # 'izq' o 'der'


class DetectorVisitas:

    def __init__(self, actor: str, zona: str, poligono, mano: str = "",
                 n_bandas: int = 3, ancho: float = 12.0):
        self.actor = actor
        self.zona = zona
        self.mano = mano
        self.poligono = np.asarray(poligono, dtype=np.int32)
        self.n_bandas = n_bandas
        self.ancho = ancho

        self.estado = EstadoMano.FUERA
        self.banda = 0
        self.banda_max = 0            
        self.visita_abierta: Visita | None = None


    def _banda_de(self, punto) -> int:
        """Índice de banda del punto; 0 = fuera del polígono."""
        d = cv2.pointPolygonTest(self.poligono, (float(punto[0]), float(punto[1])), True)
        if d <= 0:
            return 0
        return min(self.n_bandas, int(d / self.ancho) + 1)


    def actualizar(self, t: float, punto=None) -> Visita | None:

        if punto is None:
            return None

        b = self._banda_de(punto)
        anterior = self.banda
        self.banda = b

        if self.estado is EstadoMano.FUERA:
            if b > anterior:
                self.estado = EstadoMano.ENTRANDO
                self.banda_max = b
                self._t_primer_cruce = t
            return None

        if self.estado is EstadoMano.ENTRANDO:
            if b == 0:
                self.estado = EstadoMano.FUERA  
                self.banda_max = 0
            elif b >= self.banda_max:
                self.banda_max = b
                if b >= self.n_bandas:               
                    self.estado = EstadoMano.DENTRO
                    self.visita_abierta = Visita(
                        actor=self.actor, zona=self.zona, mano=self.mano,
                        t_entrada=self._t_primer_cruce,
                    )
            return None

        
        if self.estado is EstadoMano.DENTRO:
            if b < anterior:
                self.estado = EstadoMano.SALIENDO
                self.banda_max = b
            return None

        
        if self.estado is EstadoMano.SALIENDO:
            if b >= self.n_bandas:
                self.estado = EstadoMano.DENTRO      
                return None
            if b <= self.banda_max:
                self.banda_max = b
                if b == 0:                            
                    self.estado = EstadoMano.FUERA
                    v = self.visita_abierta
                    self.visita_abierta = None
                    if v is not None:
                        v.t_salida = t
                        return v
            return None

        return None


class GestorZonas:
    

    def __init__(self, zonas: dict, n_bandas: int = 3, ancho: float = 12.0):
        self.zonas = zonas              
        self.n_bandas = n_bandas
        self.ancho = ancho
        self.detectores: dict[tuple, DetectorVisitas] = {}

    def _detector(self, actor: str, mano: str, zona: str) -> DetectorVisitas:
        clave = (actor, mano, zona)
        if clave not in self.detectores:
            self.detectores[clave] = DetectorVisitas(
                actor=actor, zona=zona, poligono=self.zonas[zona], mano=mano,
                n_bandas=self.n_bandas, ancho=self.ancho,
            )
        return self.detectores[clave]

    def actualizar(self, t: float, manos: dict) -> list[Visita]:
        cerradas = []
        for (actor, mano), punto in manos.items():
            for zona in self.zonas:
                v = self._detector(actor, mano, zona).actualizar(t, punto)
                if v is not None:
                    cerradas.append(v)
        return cerradas
