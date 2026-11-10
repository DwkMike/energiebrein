"""Model-predictieve planner (MPC) voor accu, warmtepomp (ruimte + tapwater) en ELWA.

Lineair programma opgelost met HiGHS (scipy.optimize.linprog, method='highs').
Het warmtepomprendement hangt af van het vermogen (hogere aanvoertemperatuur), dat is
niet-lineair; daarom 'sequentieel lineair': oplossen, COP bijwerken bij het gevonden
vermogen, opnieuw oplossen (2-3 rondes convergeert in de praktijk).

Variabelen per tijdstap t (blok van V):
  0 imp   netafname kW          7 slo   comfort-ondershoot K
  1 exp   teruglevering kW      8 shi   comfort-overshoot K
  2 ch    accu laden kW         9 soc   accu-inhoud einde stap kWh
  3 dis   accu ontladen kW     10 Ti    kamertemp einde stap
  4 q     WP ruimte thermisch  11 Te    gebouwschil einde stap
  5 elwa  ELWA elektrisch kW   12 eb    boiler-energie einde stap kWh
  6 hdhw  WP tapwater therm.   13 curt  afgeregelde zon kW
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog
from scipy import sparse

log = logging.getLogger("energiebrein.opt")

V = 14
IMP, EXP, CH, DIS, Q, ELWA, HDHW, SLO, SHI, SOC, TI, TE, EB, CURT = range(V)


@dataclass
class Invoer:
    dt: float
    p_buy: np.ndarray
    p_sell: np.ndarray
    base: np.ndarray
    pv: np.ndarray
    To: np.ndarray
    Tlow: np.ndarray
    Thigh: np.ndarray
    dhw_vraag: np.ndarray      # kW thermisch tapwaterverbruik
    soc0: float                # kWh
    Ti0: float
    Te0: float
    eb0: float                 # kWh boven 'koud'
    soc_eind_min: float | None = None   # optioneel: accu minimaal zo vol aan het eind
    eindwaarde: bool = True             # waarde van opgeslagen energie aan het eind meetellen


def plan(inv: Invoer, house, cop, o: dict, accu_export_toegestaan: bool, iteraties: int = 3) -> dict:
    N = len(inv.p_buy)
    dt = inv.dt
    cap = o["accu_capaciteit_kwh"]
    pmax = o["accu_max_kw"]
    eta = float(np.sqrt(o["accu_rendement"]))  # rendement per richting
    soc_min = cap * o["accu_min_soc"] / 100.0
    soc_max = cap * o["accu_max_soc"] / 100.0
    qmax = o["wp_max_thermisch_kw"]
    cop_dhw = cop.cop_dhw if cop.cop_dhw else o["wp_cop_tapwater"]
    eb_max = o["boiler_max_kwh"]
    boiler_verlies = 0.01  # fractie per uur
    Ad, Bd = house.discrete(dt)
    grid_max = 17.0

    # startschatting ruimtevermogen = stationair warmteverlies naar ondergrens
    q_guess = np.clip(house.warmteverlies_kw(inv.Tlow, inv.To), 0.3, qmax)
    res_best = None
    for it in range(iteraties):
        cop_t = cop.cop_bij_vermogen(q_guess, inv.Tlow, inv.To)
        res = _los_op(inv, Ad, Bd, cop_t, cop_dhw, eta, soc_min, soc_max, pmax, qmax, eb_max,
                      boiler_verlies, grid_max, o, accu_export_toegestaan, house)
        if res is None:
            break
        res["cop"] = cop_t
        res_best = res
        q_new = np.maximum(res["q"], 0.2)
        if np.max(np.abs(q_new - q_guess)) < 0.1:
            break
        q_guess = 0.5 * q_guess + 0.5 * q_new
    if res_best is not None:
        res_best["aanvoer"] = cop.aanvoer(res_best["q"], res_best["Ti"])
    return res_best


def _los_op(inv, Ad, Bd, cop_t, cop_dhw, eta, soc_min, soc_max, pmax, qmax, eb_max,
            boiler_verlies, grid_max, o, accu_export, house):
    N = len(inv.p_buy)
    dt = inv.dt
    nv = N * V
    ix = lambda t, k: t * V + k  # noqa: E731

    c = np.zeros(nv)
    wear = o["accu_slijtage_per_kwh"]
    pen_lo, pen_hi = 3.0, 0.03  # EUR per K per uur
    for t in range(N):
        c[ix(t, IMP)] = dt * inv.p_buy[t]
        c[ix(t, EXP)] = -dt * inv.p_sell[t]
        c[ix(t, DIS)] = dt * wear
        c[ix(t, CH)] = dt * 0.001  # tie-breaker: niet zinloos laden
        c[ix(t, SLO)] = dt * pen_lo
        c[ix(t, SHI)] = dt * pen_hi
        c[ix(t, CURT)] = dt * 0.0005
    # eindwaarde van opgeslagen energie (anders 'leegt' de planner alles aan het eind)
    p_ref = float(np.median(inv.p_buy))
    cop_ref = float(np.median(cop_t))
    if inv.eindwaarde:
        c[ix(N - 1, SOC)] -= max(p_ref, 0.0) * eta * 0.9
        c[ix(N - 1, TI)] -= max(p_ref, 0.0) / cop_ref * house.Ci * 0.8
        c[ix(N - 1, TE)] -= max(p_ref, 0.0) / cop_ref * house.Ce * 0.8
        c[ix(N - 1, EB)] -= max(p_ref, 0.0) / cop_dhw * 0.9

    rows, cols, vals, beq = [], [], [], []
    r = 0

    def add(rr, k_list):
        for (cc, vv) in k_list:
            rows.append(rr); cols.append(cc); vals.append(vv)

    for t in range(N):
        # (1) vermogensbalans
        add(r, [(ix(t, IMP), 1), (ix(t, EXP), -1), (ix(t, CH), -1), (ix(t, DIS), 1),
                (ix(t, Q), -1.0 / cop_t[t]), (ix(t, ELWA), -1), (ix(t, HDHW), -1.0 / cop_dhw),
                (ix(t, CURT), -1)])
        beq.append(inv.base[t] - inv.pv[t]); r += 1
        # (2) accu
        add(r, [(ix(t, SOC), 1), (ix(t, CH), -dt * eta), (ix(t, DIS), dt / eta)])
        if t > 0:
            add(r, [(ix(t - 1, SOC), -1)]); beq.append(0.0)
        else:
            beq.append(inv.soc0)
        r += 1
        # (3,4) huis
        u_ext = np.array([inv.To[t], 0.0, inv.pv[t], inv.base[t]])
        for (state, rowA) in ((TI, 0), (TE, 1)):
            terms = [(ix(t, state), 1), (ix(t, Q), -Bd[rowA, 1])]
            rhs = float(Bd[rowA] @ u_ext)
            if t > 0:
                terms += [(ix(t - 1, TI), -Ad[rowA, 0]), (ix(t - 1, TE), -Ad[rowA, 1])]
            else:
                rhs += Ad[rowA, 0] * inv.Ti0 + Ad[rowA, 1] * inv.Te0
            add(r, terms); beq.append(rhs); r += 1
        # (5) boiler
        terms = [(ix(t, EB), 1), (ix(t, ELWA), -dt), (ix(t, HDHW), -dt)]
        rhs = -dt * inv.dhw_vraag[t]
        if t > 0:
            terms.append((ix(t - 1, EB), -(1 - boiler_verlies * dt)))
        else:
            rhs += (1 - boiler_verlies * dt) * inv.eb0
        add(r, terms); beq.append(rhs); r += 1
    A_eq = sparse.csr_matrix((vals, (rows, cols)), shape=(r, nv))

    rows, cols, vals, bub = [], [], [], []
    r = 0
    for t in range(N):
        add(r, [(ix(t, TI), -1), (ix(t, SLO), -1)]); bub.append(-inv.Tlow[t]); r += 1
        add(r, [(ix(t, TI), 1), (ix(t, SHI), -1)]); bub.append(inv.Thigh[t]); r += 1
        add(r, [(ix(t, Q), 1), (ix(t, HDHW), 1)]); bub.append(qmax); r += 1
        if not accu_export:
            # teruglevering alleen uit zon (accu levert niet aan het net)
            add(r, [(ix(t, EXP), 1)]); bub.append(max(inv.pv[t], 0.0)); r += 1
    A_ub = sparse.csr_matrix((vals, (rows, cols)), shape=(r, nv))

    bounds = []
    for t in range(N):
        b = [None] * V
        b[IMP] = (0, grid_max); b[EXP] = (0, grid_max)
        b[CH] = (0, pmax); b[DIS] = (0, pmax)
        b[Q] = (0, qmax); b[ELWA] = (0, o["elwa_max_kw"]); b[HDHW] = (0, qmax)
        b[SLO] = (0, None); b[SHI] = (0, None)
        b[SOC] = (soc_min, soc_max)
        if t == N - 1 and inv.soc_eind_min is not None:
            b[SOC] = (min(max(inv.soc_eind_min, soc_min), soc_max), soc_max)
        b[TI] = (-30, 60); b[TE] = (-30, 60)
        b[EB] = (0, eb_max)
        b[CURT] = (0, max(inv.pv[t], 0.0))
        bounds += b
    res = linprog(c, A_ub=A_ub, b_ub=np.array(bub), A_eq=A_eq, b_eq=np.array(beq),
                  bounds=bounds, method="highs")
    if res.status != 0:
        log.warning("planner: geen oplossing (%s)", res.message)
        return None
    x = res.x.reshape(N, V)
    kosten = float(np.sum(dt * (inv.p_buy * x[:, IMP] - inv.p_sell * x[:, EXP])))
    return {
        "imp": x[:, IMP], "exp": x[:, EXP], "ch": x[:, CH], "dis": x[:, DIS],
        "q": x[:, Q], "elwa": x[:, ELWA], "hdhw": x[:, HDHW], "slo": x[:, SLO], "shi": x[:, SHI],
        "soc": x[:, SOC], "Ti": x[:, TI], "Te": x[:, TE], "eb": x[:, EB], "curt": x[:, CURT],
        "p_hp": x[:, Q] / cop_t + x[:, HDHW] / cop_dhw,
        "energiekosten": kosten, "doel": float(res.fun),
    }
