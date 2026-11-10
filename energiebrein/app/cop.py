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
    n_dagen: int = 0
    dagfout_pct: float = float("nan")   # gemiddelde absolute afwijking voorspeld vs gemeten dagverbruik
    methode: str = "standaard"

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
    # 1) radiatorconstante uit (Tgem - Ti) = k * q^(1/n)
    if "Ti" in sh:
        ok = sh.dropna(subset=["Ti", "Tret"])
        ok = ok[ok["q_space"] > 0.5]
        if len(ok) >= 24:
            tg = (ok["Tflow"] + ok["Tret"]) / 2 - ok["Ti"]
            x = ok["q_space"] ** (1.0 / N_RAD)
            k = float(np.median(tg / x))
            m.k_rad = float(np.clip(k, 2.0, 30.0))
    # 2) COP fitten op DAGTOTALEN: voorspeld dagverbruik = som(q_ruimte / COP(q, Ti, To)) + som(q_tapwater / COP_tw)
    #    tegen het gemeten dagverbruik (kWh-meter). Uurwaarden zijn bij pendelen te ruizig (aan/uit binnen het uur);
    #    over een dag middelt dat uit. Dezelfde COP-functie als de planner gebruikt, dus consistent.
    dag = _dagtotalen_fit(d, m)
    if dag is None:
        # terugval: robuuste fit op uurwaarden
        To = sh["To"].to_numpy()
        Tf = sh["Tflow"].to_numpy()
        y = (sh["q_hp"] / sh["p_hp"]).to_numpy()

        def r(x):
            return CopModel(e0=x[0], e1=x[1]).cop(To, Tf) - y

        res = least_squares(r, [0.45, 0.004], bounds=([0.15, -0.005], [0.8, 0.03]), loss="soft_l1", f_scale=0.5)
        m.e0, m.e1 = map(float, res.x)
        m.rmse_cop = float(np.sqrt(np.mean(r(res.x) ** 2)))
        m.methode = "uur"
    m.n_uren = int(len(sh))
    log.info("COP-model: %s", m.to_dict())
    return m


def _dagtotalen_fit(d, m: CopModel):
    need = ["q_hp", "p_hp", "To", "dhw", "q_space"]
    if not set(need) <= set(d.columns):
        return None
    x = d.dropna(subset=["q_hp", "p_hp", "To"]).copy()
    x["Ti_ref"] = x["Ti"].fillna(20.0).clip(16, 24) if "Ti" in x else 20.0
    x["q_dhw"] = x["q_hp"].where(x["dhw"].astype(bool), 0.0)
    lok = x.index.tz_convert("Europe/Amsterdam")
    x["datum"] = lok.date
    n_per_dag = x.groupby("datum")["p_hp"].transform("count")
    x = x[n_per_dag >= 22]
    E = x.groupby("datum")["p_hp"].sum()
    Q = x.groupby("datum")["q_hp"].sum()
    goed = (E > 2.0) & (Q / E > 1.2) & (Q / E < 7.0)  # alleen echte stookdagen met plausibele dag-COP
    dagen = E.index[goed.to_numpy()]
    if len(dagen) < 5:
        return None
    x = x[x["datum"].isin(set(dagen))]
    E = E.loc[dagen].to_numpy()
    gidx = x["datum"].map({dd: i for i, dd in enumerate(dagen)}).to_numpy()
    qs = x["q_space"].fillna(0).to_numpy()
    qd = x["q_dhw"].to_numpy()
    To = x["To"].to_numpy()
    Ti = x["Ti_ref"].to_numpy()

    def voorspel(p):
        mm = CopModel(e0=p[0], e1=p[1], k_rad=m.k_rad, debiet=m.debiet)
        cop = mm.cop_bij_vermogen(qs, Ti, To)
        per_uur = qs / cop + qd / p[2]
        return np.bincount(gidx, weights=per_uur, minlength=len(dagen))

    def r(p):
        return (voorspel(p) - E) / np.maximum(E, 1.0)

    res = least_squares(r, [0.45, 0.004, m.cop_dhw], bounds=([0.15, -0.005, 1.5], [0.8, 0.03, 4.5]),
                        loss="soft_l1", f_scale=0.1)
    m.e0, m.e1, m.cop_dhw = map(float, res.x)
    fout = np.abs(r(res.x))
    m.dagfout_pct = float(100 * np.mean(fout))
    m.rmse_cop = float("nan")
    m.n_dagen = int(len(dagen))
    m.methode = "dagtotalen"
    return m
