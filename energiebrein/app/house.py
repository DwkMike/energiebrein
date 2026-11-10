"""Grey-box huismodel (2 toestanden: lucht/inrichting Ti en gebouwschil/massa Te).

    Ci dTi/dt = (Te-Ti)/Rie + (To-Ti)/Ria + Q + gs*PV + gi*Pbase
    Ce dTe/dt = (Ti-Te)/Rie + (To-Te)/Rea

Eenheden: kW, kWh/K, K/kW, uren. PV-vermogen is een proxy voor zonnewinst door ramen,
basislast (huishoudelijke elektriciteit) is een proxy voor interne warmtewinst.
Gefit met 'multiple shooting': vanaf elke startuur 24 uur vooruit simuleren en de
kamertemperatuur vergelijken met de meting. Zo leert het model het echte dynamische
gedrag (inclusief traagheid van radiatoren en muren) in plaats van alleen 1-uurs sprongen.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, asdict

import numpy as np
from scipy.linalg import expm
from scipy.optimize import least_squares

log = logging.getLogger("energiebrein.house")

# param: Ci, Ce, Rie, Rea, Ria, gs, gi
DEFAULT = np.array([2.0, 15.0, 1.5, 6.0, 12.0, 0.15, 0.6])
LOWER = np.array([0.3, 2.0, 0.2, 1.0, 2.0, 0.0, 0.0])
UPPER = np.array([15.0, 80.0, 10.0, 40.0, 80.0, 0.6, 1.0])


@dataclass
class HouseModel:
    Ci: float
    Ce: float
    Rie: float
    Rea: float
    Ria: float
    gs: float
    gi: float
    rmse_24h: float = float("nan")
    rmse_1h: float = float("nan")
    n_uren: int = 0

    @property
    def params(self) -> np.ndarray:
        return np.array([self.Ci, self.Ce, self.Rie, self.Rea, self.Ria, self.gs, self.gi])

    @property
    def ua_w_per_k(self) -> float:
        """Totale warmteverliescoefficient in W/K (stationair)."""
        return 1000.0 * (1.0 / self.Ria + 1.0 / (self.Rie + self.Rea))

    def to_dict(self) -> dict:
        d = asdict(self)
        d["ua_w_per_k"] = round(self.ua_w_per_k, 1)
        d["tijdconstante_uur"] = round((self.Ci + self.Ce) / (self.ua_w_per_k / 1000.0), 1)
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()}

    def discrete(self, dt: float):
        return discretize(self.params, dt)

    def te_reeks(self, df) -> np.ndarray:
        """Geschatte schiltemperatuur voor elke rij van een uur-tabel (kolommen Ti, To, q_space, pv, base)."""
        Ti = df["Ti"].to_numpy(dtype=float)
        To = df["To"].to_numpy(dtype=float)
        U = np.nan_to_num(np.column_stack([To, df["q_space"].to_numpy(dtype=float),
                                           df["pv"].to_numpy(dtype=float), df["base"].to_numpy(dtype=float)]))
        return te_waarnemer(self.params, Ti, To, U)

    def te_na(self, df) -> float:
        """Schatting van Te aan het EINDE van de laatste rij (start van het volgende uur)."""
        if df is None or len(df) == 0:
            return float("nan")
        d = df[["Ti", "To", "q_space", "pv", "base"]].astype(float).ffill()
        Te = self.te_reeks(d)
        Ad, Bd = discretize(self.params, 1.0)
        last = d.iloc[-1]
        u = np.nan_to_num(np.array([last["To"], last["q_space"], last["pv"], last["base"]]))
        if not np.isfinite(Te[-1]) or not np.isfinite(last["Ti"]):
            return float("nan")
        return float(Ad[1, 0] * last["Ti"] + Ad[1, 1] * Te[-1] + Bd[1] @ u)

    def warmteverlies_kw(self, Ti: float, To: float) -> float:
        return (Ti - To) * self.ua_w_per_k / 1000.0


def continuous(p: np.ndarray):
    Ci, Ce, Rie, Rea, Ria, gs, gi = p
    A = np.array([[-(1 / Rie + 1 / Ria) / Ci, 1 / (Rie * Ci)],
                  [1 / (Rie * Ce), -(1 / Rie + 1 / Rea) / Ce]])
    # inputs u = [To, Q, PV, Pbase]
    B = np.array([[1 / (Ria * Ci), 1 / Ci, gs / Ci, gi / Ci],
                  [1 / (Rea * Ce), 0.0, 0.0, 0.0]])
    return A, B


def discretize(p: np.ndarray, dt: float):
    """Exacte discretisatie (zero-order hold) via matrix-exponent."""
    A, B = continuous(p)
    n, m = A.shape[0], B.shape[1]
    M = np.zeros((n + m, n + m))
    M[:n, :n] = A
    M[:n, n:] = B
    E = expm(M * dt)
    return E[:n, :n], E[:n, n:]


def te_evenwicht(p, Ti, To):
    Ci, Ce, Rie, Rea, Ria, gs, gi = p
    return To + (Ti - To) * Rea / (Rie + Rea)


def _segments(mask: np.ndarray, H: int, step: int) -> np.ndarray:
    """Startindexen s waarvoor mask[s:s+H+1] volledig True is."""
    n = len(mask)
    good = np.convolve(mask.astype(int), np.ones(H + 1, dtype=int), mode="valid") == H + 1
    starts = np.nonzero(good)[0]
    return starts[::step] if len(starts) else starts


def te_waarnemer(p, Ti, To, U, dt: float = 1.0) -> np.ndarray:
    """Schat de (niet gemeten) schiltemperatuur Te over de hele reeks.

    Te reageert traag op het gemiddelde buitenklimaat; de evenwichtswaarde bij de huidige
    buitentemperatuur is daarom een slechte beginschatting (nacht/dag-verschil). We laten Te
    meelopen met de gemeten kamertemperatuur: Te[k+1] = Ad10*Ti[k] + Ad11*Te[k] + Bd1.u[k].
    Bij ontbrekende data: herstart op evenwicht met het 24-uurs gemiddelde buiten.
    """
    Ad, Bd = discretize(p, dt)
    a10, a11 = float(Ad[1, 0]), float(Ad[1, 1])
    b1 = Bd[1]
    n = len(Ti)
    Te = np.empty(n)
    To_s = pd_rolling_mean(To, int(round(24 / dt)))
    ok = np.isfinite(Ti) & np.isfinite(To)
    cur = np.nan
    ub = U @ b1
    for k in range(n):
        if not ok[k]:
            cur = np.nan
            Te[k] = np.nan
            continue
        if not np.isfinite(cur):
            cur = te_evenwicht(p, Ti[k], To_s[k])
        Te[k] = cur
        cur = a10 * Ti[k] + a11 * cur + ub[k]
    return Te


def pd_rolling_mean(x: np.ndarray, w: int) -> np.ndarray:
    import pandas as pd
    return pd.Series(x).rolling(max(w, 1), min_periods=1).mean().to_numpy()


def simulate_multi(p, Ti, To, U, starts, H, Te=None):
    """Simuleer vanaf alle starts tegelijk H stappen (dt=1h). Geeft (S,H) voorspelde Ti."""
    Ad, Bd = discretize(p, 1.0)
    if Te is None:
        Te = te_waarnemer(p, Ti, To, U)
    S = len(starts)
    x = np.zeros((S, 2))
    x[:, 0] = Ti[starts]
    x[:, 1] = Te[starts]
    out = np.zeros((S, H))
    for k in range(H):
        u = U[starts + k]          # (S,4)
        x = x @ Ad.T + u @ Bd.T
        out[:, k] = x[:, 0]
    return out


def fit(df, H: int = 24, min_uren: int = 96) -> HouseModel | None:
    cols = ["Ti", "To", "q_space", "pv", "base"]
    if not set(cols) <= set(df.columns):
        log.warning("huismodel: kolommen ontbreken")
        return None
    d = df[cols].astype(float)
    # alleen stookseizoen (buiten < 16 graden) - in de zomer zegt kamertemperatuur weinig over warmteverlies
    mask = d.notna().all(axis=1).to_numpy() & (d["To"].to_numpy() < 16.0)
    Ti = d["Ti"].to_numpy()
    To = d["To"].to_numpy()
    U = np.column_stack([To, d["q_space"].to_numpy(), d["pv"].to_numpy(), d["base"].to_numpy()])
    U = np.nan_to_num(U)
    starts = _segments(mask, H, step=2)
    U_obs = U.copy()
    if mask.sum() < min_uren or len(starts) < 10:
        log.warning("huismodel: te weinig bruikbare uren (%d)", int(mask.sum()))
        return None
    target = np.stack([Ti[s + 1:s + H + 1] for s in starts])
    w = 1.0 / np.sqrt(np.arange(1, H + 1))  # korte horizon iets zwaarder

    def resid(logp):
        p = np.exp(logp)
        p[5:] = logp[5:]  # gs, gi lineair
        sim = simulate_multi(p, Ti, To, U, starts, H)
        r = ((sim - target) * w).ravel()
        # zachte regularisatie richting fysisch plausibele startwaarden
        reg = 0.02 * (logp[:5] - np.log(DEFAULT[:5]))
        return np.concatenate([r, reg])

    x0 = np.concatenate([np.log(DEFAULT[:5]), DEFAULT[5:]])
    lb = np.concatenate([np.log(LOWER[:5]), LOWER[5:]])
    ub = np.concatenate([np.log(UPPER[:5]), UPPER[5:]])
    best = None
    for scale in (1.0, 0.5, 2.0):  # meerdere starts tegen lokale minima
        x_init = x0.copy()
        x_init[:5] += np.log(scale)
        x_init = np.clip(x_init, lb + 1e-6, ub - 1e-6)
        try:
            res = least_squares(resid, x_init, bounds=(lb, ub), loss="soft_l1", f_scale=0.3, max_nfev=300)
        except Exception as e:  # pragma: no cover
            log.warning("fit mislukt: %s", e)
            continue
        if best is None or res.cost < best.cost:
            best = res
    if best is None:
        return None
    p = np.exp(best.x)
    p[5:] = best.x[5:]
    sim = simulate_multi(p, Ti, To, U, starts, H)
    err = sim - target
    m = HouseModel(*p, rmse_24h=float(np.sqrt(np.mean(err[:, -1] ** 2))),
                   rmse_1h=float(np.sqrt(np.mean(err[:, 0] ** 2))), n_uren=int(mask.sum()))
    log.info("huismodel: %s", m.to_dict())
    return m


def default_model() -> HouseModel:
    return HouseModel(*DEFAULT)
