"""
Este archivo corresponde a la aapa de decisión el motor consume eventos, agrupa los eventos en sesiones con una máquina de estados que
modela el curso normal de una transacción, y sobre los hechos acumulados en
cada sesión evalúa las reglas de comportamiento.


    NIVEL 1   posición de la muñeca -> banda de la región   (zonas.py)
    NIVEL 2   secuencia de bandas   -> visita aceptada      (zonas.py)
    NIVEL 3   visitas               -> transacción y alerta
"""

from dataclasses import dataclass, field
from enum import Enum


# ===========================================================================
# 1. VOCABULARIO
# ===========================================================================

class Zona(str, Enum):
    CAJON = "Z_CAJON"         
    MOSTRADOR = "Z_MOSTRADOR"  
    OCULTA = "Z_OCULTA" 


class Rol(str, Enum):
    CAJERO = "cajero"
    CLIENTE = "cliente"


class TipoEvento(str, Enum):
    PERSONA_ENTRA = "PERSONA_ENTRA"
    PERSONA_SALE = "PERSONA_SALE"
    MANO_ENTRA = "MANO_ENTRA"
    MANO_SALE = "MANO_SALE"


@dataclass
class Evento:
    tipo: TipoEvento
    t: float
    actor: Rol | None = None
    zona: Zona | None = None
    mano: str = ""
    confianza: float = 1.0


# ===========================================================================
# 2. CONFIGURACIÓN
# ===========================================================================

@dataclass
class Config:
    # Las tolerancia son los segundos que la mano del cajero
    # puede permanecer dentro de una zona antes de que la visita cuente como
    # anomalía:
    tol_visita_cajon: float = 10.0
    tol_visita_oculta: float = 1.5

    ven_encadenamiento: float = 8.0
    holgura_orden: float = 2.0
    ven_intercambio: float = 5.0
    ven_cierre: float = 10.0

    exp_sesion: float = 300.0

    n_bandas: int = 3
    ancho_banda: float = 12.0

    umbral_alerta: float = 0.45


# ===========================================================================
# 3. MÁQUINA DE ESTADOS DE LA TRANSACCIÓN
# ===========================================================================

class Estado(str, Enum):
    INACTIVO = "INACTIVO"
    CLIENTE_PRESENTE = "CLIENTE_PRESENTE"
    INTERCAMBIO = "INTERCAMBIO"          
    MANIPULACION = "MANIPULACION"        
    ACCESO_CAJON = "ACCESO_CAJON"
    CIERRE = "CIERRE"
    RECUPERACION = "RECUPERACION"       


@dataclass
class Paso:
    """Una visita dentro de la sesión: quién, dónde, cuándo."""
    actor: Rol
    zona: Zona
    t_entrada: float
    t_salida: float | None = None
    mano: str = ""


@dataclass
class Sesion:
    id: int
    t_inicio: float
    t_fin: float | None = None
    estado: Estado = Estado.INACTIVO

    cliente_presente: bool = False
    cliente_alguna_vez: bool = False
    t_salida_cliente: float | None = None
    hubo_intercambio: bool = False

    pasos: list[Paso] = field(default_factory=list)      
    abiertos: dict[tuple, Paso] = field(default_factory=dict)
    eventos: list[Evento] = field(default_factory=list)

    def n_accesos_cajon(self) -> int:
        return sum(1 for p in self.pasos
                   if p.zona is Zona.CAJON and p.actor is Rol.CAJERO)


class MotorFSM:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.sesion: Sesion | None = None
        self._n = 0

    def _abrir(self, t: float) -> Sesion:
        self._n += 1
        self.sesion = Sesion(id=self._n, t_inicio=t)
        return self.sesion

    def _cerrar(self, t: float, motivo: str) -> Sesion | None:
        if self.sesion is None:
            return None
        s = self.sesion
        s.t_fin = t
        s.estado = Estado.CIERRE if motivo == "normal" else Estado.RECUPERACION
        for p in s.abiertos.values():     
            if p.t_salida is None:
                p.t_salida = t
        s.abiertos.clear()
        self.sesion = None
        return s


    def procesar(self, ev: Evento) -> Sesion | None:
        """Devuelve la sesión que se haya cerrado al procesar el evento, si alguna."""
        cerrada = self._expirar(ev.t)

        if self.sesion is None:
            if ev.tipo not in (TipoEvento.PERSONA_ENTRA, TipoEvento.MANO_ENTRA):
                return cerrada
            self._abrir(ev.t)

        s = self.sesion
        s.eventos.append(ev)

        if ev.tipo is TipoEvento.PERSONA_ENTRA and ev.actor is Rol.CLIENTE:
            s.cliente_presente = True
            s.cliente_alguna_vez = True
            s.t_salida_cliente = None
            s.estado = Estado.CLIENTE_PRESENTE

        elif ev.tipo is TipoEvento.PERSONA_SALE and ev.actor is Rol.CLIENTE:
            s.cliente_presente = False
            s.t_salida_cliente = ev.t

        elif ev.tipo is TipoEvento.MANO_ENTRA:
            clave = (ev.actor, ev.mano, ev.zona)
            s.abiertos[clave] = Paso(actor=ev.actor, zona=ev.zona,
                                     t_entrada=ev.t, mano=ev.mano)
            self._interpretar_entrada(s, ev.t, ev.actor, ev.zona)

        elif ev.tipo is TipoEvento.MANO_SALE:
            paso = s.abiertos.pop((ev.actor, ev.mano, ev.zona), None)
            if paso is not None:
                paso.t_salida = ev.t
                s.pasos.append(paso)

        return cerrada

    def _interpretar_entrada(self, s: Sesion, t: float, actor: Rol, zona: Zona) -> None:
        if zona is Zona.CAJON and actor is Rol.CAJERO:
            s.estado = Estado.ACCESO_CAJON
            return

        if zona is Zona.MOSTRADOR:
            otro = Rol.CLIENTE if actor is Rol.CAJERO else Rol.CAJERO
            simultanea = any(k[0] is otro and k[2] is Zona.MOSTRADOR
                             for k in s.abiertos)
            reciente = any(
                p.actor is otro and p.zona is Zona.MOSTRADOR
                and p.t_salida is not None
                and t - p.t_salida <= self.cfg.ven_intercambio
                for p in s.pasos
            )
            if simultanea or reciente:
                s.hubo_intercambio = True
                s.estado = Estado.INTERCAMBIO
            else:
                s.estado = Estado.MANIPULACION


    def _expirar(self, t: float) -> Sesion | None:
        s = self.sesion
        if s is None:
            return None
        if t - s.t_inicio > self.cfg.exp_sesion:
            return self._cerrar(t, "expiracion")
        if (s.t_salida_cliente is not None
                and not s.abiertos
                and t - s.t_salida_cliente > self.cfg.ven_cierre):
            return self._cerrar(t, "normal")
        return None

    def avanzar(self, t: float) -> Sesion | None:
        return self._expirar(t)

    def finalizar(self, t: float) -> Sesion | None:
        return self._cerrar(t, "normal") if self.sesion is not None else None


# ===========================================================================
# 4. REGLAS (Cuadro 3.11)
# ===========================================================================

@dataclass
class Alerta:
    codigo: str
    nombre: str
    modalidad: str
    t: float
    sesion_id: int
    score: float
    severidad: float
    confianza: float
    detalle: str
    clave: str = ""

    @property
    def nivel(self) -> str:
        if self.score >= 0.75:
            return "ALTA"
        if self.score >= 0.45:
            return "MEDIA"
        return "BAJA"


SEVERIDAD = {
    "R01": 0.85,   # acceso al cajón sin cliente
    "R02": 1.00,   # intercambio sin acceso al cajón
    "R03": 1.00,   # trayecto cajón -> zona oculta
    "R04": 0.55,   # permanencia prolongada en el cajón
    "R06": 0.50,   # más de dos accesos al cajón
    "R10": 0.75,   # mano en zona oculta (0,90 durante una venta)
}

MODALIDAD = {
    "R01": "Sustracción de efectivo registrado",
    "R02": "Venta no registrada",
    "R03": "Venta no registrada / sustracción",
    "R04": "Sustracción / manipulación",
    "R06": "Sustracción / manipulación",
    "R10": "Venta no registrada / sustracción",
}


def _conf(eventos, tipos=()) -> float:
    rel = [e for e in eventos if not tipos or e.tipo in tipos]
    return 1.0 if not rel else sum(e.confianza for e in rel) / len(rel)


class Evaluador:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.alertas: list[Alerta] = []
        self._vistas: set[str] = set()

    def _emitir(self, a: Alerta) -> Alerta | None:
        if a.clave in self._vistas or a.score < self.cfg.umbral_alerta:
            return None
        self._vistas.add(a.clave)
        self.alertas.append(a)
        return a

    def _alerta(self, codigo: str, nombre: str, **kw) -> Alerta:
        return Alerta(codigo=codigo, nombre=nombre, modalidad=MODALIDAD[codigo], **kw)



    def continuas(self, s: Sesion | None, t: float) -> list[Alerta]:
        out = []
        if s is None:
            return out
        cfg = self.cfg

        for (actor, mano, zona), paso in list(s.abiertos.items()):
            transcurrido = t - paso.t_entrada

            # R10 — la mano del cajero está en la zona oculta.
            if zona is Zona.OCULTA and actor is Rol.CAJERO:
                if transcurrido >= cfg.tol_visita_oculta:
                    pers = min(1.3, 1.0 + (transcurrido - cfg.tol_visita_oculta) / 15.0)
                    sev = SEVERIDAD["R10"]
                    if s.cliente_presente or s.hubo_intercambio:
                        sev = min(1.0, sev + 0.15)     # durante una venta, peor
                    a = self._alerta(
                        "R10", "Mano en zona oculta",
                        t=paso.t_entrada + cfg.tol_visita_oculta, sesion_id=s.id,
                        score=round(sev * pers, 3), severidad=sev, confianza=1.0,
                        detalle=(f"La mano del cajero permaneció {transcurrido:.0f}s "
                                 f"en {zona.value}, fuera de la vista."),

                        clave=f"R10:{s.id}:{actor}",
                    )
                    if (r := self._emitir(a)):
                        out.append(r)

            # R04 — permanencia prolongada en el cajón.
            if zona is Zona.CAJON and transcurrido > cfg.tol_visita_cajon:
                pers = min(1.3, transcurrido / cfg.tol_visita_cajon)
                a = self._alerta(
                    "R04", "Permanencia en el cajón",
                    t=paso.t_entrada + cfg.tol_visita_cajon, sesion_id=s.id,
                    score=round(SEVERIDAD["R04"] * pers, 3),
                    severidad=SEVERIDAD["R04"], confianza=1.0,
                    detalle=(f"La mano permaneció {transcurrido:.0f}s dentro del cajón "
                             f"(límite {cfg.tol_visita_cajon:.0f}s)."),
                    clave=f"R04:{s.id}:{paso.t_entrada}",
                )
                if (r := self._emitir(a)):
                    out.append(r)

        return out

    # -- por evento --------------------------------------------------------

    def por_evento(self, s: Sesion | None, ev: Evento) -> list[Alerta]:
        out = []
        if s is None:
            return out

        # R03 — la mano sale del cajón y entra a la zona oculta poco después.
        if (ev.tipo is TipoEvento.MANO_ENTRA
                and ev.zona is Zona.OCULTA and ev.actor is Rol.CAJERO):
            for p in reversed(s.pasos):
                if (p.zona is Zona.CAJON and p.actor is Rol.CAJERO
                        and p.t_salida is not None
                        and -self.cfg.holgura_orden
                            <= ev.t - p.t_salida <= self.cfg.ven_encadenamiento):
                    salto = max(0.0, ev.t - p.t_salida)
                    a = self._alerta(
                        "R03", "Trayecto cajón → zona oculta",
                        t=ev.t, sesion_id=s.id,
                        score=round(ev.confianza * SEVERIDAD["R03"], 3),
                        severidad=SEVERIDAD["R03"], confianza=ev.confianza,
                        detalle=(f"La mano salió del cajón y entró a {ev.zona.value} "
                                 f"{salto:.1f}s después, sin pasar por el mostrador."),
                        clave=f"R03:{s.id}",
                    )
                    if (r := self._emitir(a)):
                        out.append(r)
                    break

        return out

    def al_cierre(self, s: Sesion) -> list[Alerta]:
        out = []
        t = s.t_fin or 0.0
        n_cajon = s.n_accesos_cajon()
        conf = _conf(s.eventos, (TipoEvento.MANO_ENTRA,))

        # R01 — acceso al cajón sin cliente presente en la sesión.
        if n_cajon > 0 and not s.cliente_alguna_vez:
            a = self._alerta(
                "R01", "Acceso al cajón sin cliente",
                t=s.t_inicio, sesion_id=s.id,
                score=round(conf * SEVERIDAD["R01"], 3),
                severidad=SEVERIDAD["R01"], confianza=round(conf, 3),
                detalle="Hubo acceso al cajón sin ningún cliente presente.",
                clave=f"R01:{s.id}",
            )
            if (r := self._emitir(a)):
                out.append(r)

        # R02 — intercambio con el cliente sin acceso al cajón.
        if s.hubo_intercambio and n_cajon == 0:
            a = self._alerta(
                "R02", "Intercambio sin acceso al cajón",
                t=t, sesion_id=s.id,
                score=round(conf * SEVERIDAD["R02"], 3),
                severidad=SEVERIDAD["R02"], confianza=round(conf, 3),
                detalle=("Hubo intercambio mano a mano con el cliente y el cajero "
                         "nunca accedió al cajón."),
                clave=f"R02:{s.id}",
            )
            if (r := self._emitir(a)):
                out.append(r)

        # R06 — más de dos accesos al cajón en una misma transacción.
        if s.cliente_alguna_vez and n_cajon > 2:
            a = self._alerta(
                "R06", "Accesos múltiples al cajón",
                t=t, sesion_id=s.id,
                score=round(SEVERIDAD["R06"] * min(1.3, n_cajon / 2.0), 3),
                severidad=SEVERIDAD["R06"], confianza=1.0,
                detalle=f"{n_cajon} accesos al cajón en una sola transacción.",
                clave=f"R06:{s.id}",
            )
            if (r := self._emitir(a)):
                out.append(r)

        return out


# ===========================================================================
# 5. ORQUESTADOR
# ===========================================================================

class Vigia:
    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or Config()
        self.fsm = MotorFSM(self.cfg)
        self.eval = Evaluador(self.cfg)

    def procesar(self, ev: Evento) -> list[Alerta]:
        a = self.eval.continuas(self.fsm.sesion, ev.t)
        cerrada = self.fsm.procesar(ev)
        if cerrada is not None:
            a += self.eval.al_cierre(cerrada)
        a += self.eval.por_evento(self.fsm.sesion, ev)
        a += self.eval.continuas(self.fsm.sesion, ev.t)
        return a

    def tick(self, t: float) -> list[Alerta]:
        a = self.eval.continuas(self.fsm.sesion, t)
        cerrada = self.fsm.avanzar(t)
        if cerrada is not None:
            a += self.eval.al_cierre(cerrada)
        return a

    def finalizar(self, t: float) -> list[Alerta]:
        a = self.eval.continuas(self.fsm.sesion, t)
        cerrada = self.fsm.finalizar(t)
        if cerrada is not None:
            a += self.eval.al_cierre(cerrada)
        return a
