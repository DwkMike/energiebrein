"""Warmtepomp-rendement (COP) en radiatormodel, geleerd uit eigen meetdata.

COP volgt een Carnot-vorm met geleerde kwaliteitsfactor:
    COP = eta(To) * Tc / (Tc - Te),  Tc = Tflow + 3 K (condensor), Te = To - 6 K (verdamper), in Kelvin
    eta(To) = e0 + e1 * To   (vangt o.a. ontdooiverliezen rond 0-5 graden)
Deze vorm extrapoleert fysisch correct naar kouder weer of andere aanvoertemperaturen.

Radiatoren (EN 442): afgegeven vermogen ~ (Tgem - Ti)^n met n ~ 1.3, dus
    Tgem - Ti = k * Q^(1/n),  Tflow = Tgem + dT/2,  dT = Q / (debiet * 1.163)
Hiermee rekent de planner uit welke aanvoertemperatuur nodig is voor een gewenst vermogen.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, asdict

import numpy as np
from scipy.optimize import least_squares

log = logging.getLogger("energiebrein.cop")
N_RAD = 1.3
K0 = 273.15


@dataclass
class CopModel:
    e0: float = 0.45
    e1: float = 0.004
    k_rad: float = 9.0       # K per kW^(1/n)
    debiet: float = 1.0      # m3/h
    cop_dhw: float = 2.7
    min_aanvoer: float = 25.0
    max_aanvoer: float = 55.0
    rmse_cop: float = float("nan")
    n_uren: int = 0

    def to_dict(self):
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in asdict(self).items()}

    def cop(self, To, Tflow):
        To = np.asarray(To, dtype=float)
        Tflow = np.asarray(Tflow, dtype=float)
        Tc = Tflow + 3.0 + K0
        Te = To - 6.0 + K0
        eta = np.clip(self.e0 + self.e1 * To, 0.2, 0.75)
        c = eta * Tc / np.maximum(Tc - Te, 5.0)
        return np.clip(c, 1.0, 8.0)

    def aanvoer(self, q_kw, Ti):
        q = np.maximum(np.asarray(q_kw, dtype=float), 0.0)
        tgem = np.asarray(Ti, dtype=float) + self.k_rad * q ** (1.0 / N_RAD)
        tflow = tgem + q / (2 * 1.163 * max(self.debiet, 0.2))
        return np.clip(tflow, self.min_aanvoer, self.max_aanvoer)

    def cop_bij_vermogen(self, q_kw, Ti, To):
        return self.cop(To, self.aanvoer(q_kw, Ti))


def fit(df) -> CopModel:
    m = CopModel()
    need = {"q_hp", "p_hp", "To", "Tflow", "Tret", "dhw"}
    if not need <= set(df.columns):
        return m
    d = df.dropna(subset=["q_hp", "p_hp", "To", "Tflow"]).copy()
    if "flow" in d:
        f = d.loc[d["flow"] > 0.3, "flow"]
        if len(f) > 20:
            m.debiet = float(f.median())
    run = d[(d["p_hp"] > 0.3) & (d["q_hp"] > 0.5)]
    cop = run["q_hp"] / run["p_hp"]
    run = run[(cop > 1.0) & (cop < 8.0)]
    sh = run[~run["dhw"].astype(bool)]
    dhw = run[run["dhw"].astype(bool)]
    if len(dhw) >= 5:
        m.cop_dhw = float(np.clip((dhw["q_hp"] / dhw["p_hp"]).median(), 1.5, 4.5))
    if len(sh) < 24:
        log.warning("COP-model: te weinig uren (%d), standaardwaarden", len(sh))
        return m
    To = sh["To"].to_numpy()
    Tf = sh["Tflow"].to_numpy()
    y = (sh["q_hp"] / sh["p_hp"]).to_numpy()

    def r(x):
        mm = CopModel(e0=x[0], e1=x[1])
        return mm.cop(To, Tf) - y

    res = least_squares(r, [0.45, 0.004], bounds=([0.15, -0.02], [0.8, 0.03]), loss="soft_l1", f_scale=0.5)
    m.e0, m.e1 = map(float, res.x)
    m.rmse_cop = float(np.sqrt(np.mean(r(res.x) ** 2)))
    m.n_uren = int(len(sh))
    # radiatorconstante uit (Tgem - Ti) = k * q^(1/n)
    if "Ti" in sh:
        ok = sh.dropna(subset=["Ti", "Tret"])
        ok = ok[ok["q_space"] > 0.5]
        if len(ok) >= 24:
            tg = (ok["Tflow"] + ok["Tret"]) / 2 - ok["Ti"]
            x = ok["q_space"] ** (1.0 / N_RAD)
            k = float(np.median(tg / x))
            m.k_rad = float(np.clip(k, 2.0, 30.0))
    log.info("COP-model: %s", m.to_dict())
    return m
