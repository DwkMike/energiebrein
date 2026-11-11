"""Terugrekentest ALLEEN de accu (los van huis- en COP-model, dus op echte meetdata).

Per dag (uurdata, saldering):
  1. WERKELIJK: wat het huis echt kostte (prijs x netafname) + slijtage van de echte ontlaad-kWh.
  2. BREIN REALISTISCH: om 00:00 een 24-uursplan met
       - de echte prijzen (Zonneplan publiceert ze vooraf),
       - een VOORSPELD verbruik = gemiddelde van hetzelfde uur over de 7 dagen ervoor
         (inclusief warmtepomp/ELWA, zoals gemeten),
       - de echte zon (bovengrens voor de zonvoorspelling; in november klein effect).
     Dat plan wordt per uur vertaald naar wat het brein echt aan de AlphaESS kan opdragen:
       LADEN (Force Charging met X kW) / VASTHOUDEN (ontladen blokkeren) / EIGEN VERBRUIK (normaal).
     Daarna gesimuleerd met het ECHTE verbruik: verschillen vangt het net op.
  3. BREIN PERFECT: hetzelfde plan, maar met het echte verbruik vooraf bekend (bovengrens).
Eerlijkheid: start-SoC = werkelijke start-SoC, eind-SoC minimaal gelijk aan werkelijk eind.
Slijtage telt mee: accu_slijtage_per_kwh x ontladen kWh, voor alle varianten.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linprog

log = logging.getLogger("energiebrein.accutest")


def plan_accu(prijs, last, pv, soc0, soc_eind_min, o, export_toegestaan=False):
    """LP: minimaliseer kosten + slijtage. Variabelen per uur: imp, exp, ch, dis, soc."""
    N = len(prijs)
    V = 5
    IMP, EXP, CH, DIS, SOC = range(V)
    cap = o["accu_capaciteit_kwh"]
    pmax = o["accu_max_kw"]
    eta = float(np.sqrt(o["accu_rendement"]))
    smin = cap * o["accu_min_soc"] / 100
    smax = cap * o["accu_max_soc"] / 100
    wear = o["accu_slijtage_per_kwh"]
    ix = lambda t, k: t * V + k  # noqa: E731
    c = np.zeros(N * V)
    for t in range(N):
        c[ix(t, IMP)] = prijs[t]
        c[ix(t, EXP)] = -prijs[t]          # saldering
        c[ix(t, DIS)] = wear
        c[ix(t, CH)] = 0.0005
    rows, cols, vals, beq = [], [], [], []
    r = 0
    for t in range(N):
        for k, v in ((IMP, 1), (EXP, -1), (CH, -1), (DIS, 1)):
            rows.append(r); cols.append(ix(t, k)); vals.append(v)
        beq.append(last[t] - pv[t]); r += 1
        for k, v in ((SOC, 1), (CH, -eta), (DIS, 1 / eta)):
            rows.append(r); cols.append(ix(t, k)); vals.append(v)
        if t > 0:
            rows.append(r); cols.append(ix(t - 1, SOC)); vals.append(-1); beq.append(0.0)
        else:
            beq.append(soc0)
        r += 1
    A_eq = sparse.csr_matrix((vals, (rows, cols)), shape=(r, N * V))
    bounds = []
    for t in range(N):
        exp_max = 30.0 if export_toegestaan else max(pv[t], 0.0)
        lo_soc = smin
        if t == N - 1 and soc_eind_min is not None:
            lo_soc = min(max(soc_eind_min, smin), smax)
        bounds += [(0, 30), (0, exp_max), (0, pmax), (0, pmax), (lo_soc, smax)]
    res = linprog(c, A_eq=A_eq, b_eq=np.array(beq), bounds=bounds, method="highs")
    if res.status != 0:
        return None
    x = res.x.reshape(N, V)
    return {"ch": x[:, CH], "dis": x[:, DIS], "soc": x[:, SOC]}


def simuleer(plan, last, pv, soc0, o):
    """Voer het plan uit zoals de AlphaESS het kan: laden / vasthouden / eigen verbruik, met ECHT verbruik."""
    cap = o["accu_capaciteit_kwh"]
    pmax = o["accu_max_kw"]
    eta = float(np.sqrt(o["accu_rendement"]))
    smin = cap * o["accu_min_soc"] / 100
    smax = cap * o["accu_max_soc"] / 100
    soc = soc0
    netto = np.zeros(len(last))
    ontladen = 0.0
    modi = []
    for t in range(len(last)):
        overschot = pv[t] - last[t]          # >0: zon over, <0: tekort
        ch_plan, dis_plan = plan["ch"][t], plan["dis"][t]
        if ch_plan > 0.1 and dis_plan < 0.05:
            modus = "laden"
            ch = min(max(ch_plan, max(overschot, 0)), pmax, (smax - soc) / eta)
            dis = 0.0
        elif dis_plan < 0.05:
            modus = "vasthouden"             # niet ontladen; zonoverschot mag wel de accu in
            ch = min(max(overschot, 0), pmax, (smax - soc) / eta)
            dis = 0.0
        else:
            modus = "eigen verbruik"
            if overschot >= 0:
                ch = min(overschot, pmax, (smax - soc) / eta)
                dis = 0.0
            else:
                ch = 0.0
                dis = min(-overschot, pmax, max(soc - smin, 0) * eta)
        soc = soc + ch * eta - dis / eta
        netto[t] = last[t] - pv[t] + ch - dis
        ontladen += dis
        modi.append(modus)
    return netto, ontladen, soc, modi


def run(hist: pd.DataFrame, o: dict, dagen: int = 45) -> dict:
    need = ["price", "pv", "house", "grid", "soc", "batt"]
    if not set(need) <= set(hist.columns):
        return {"dagen": [], "melding": "kolommen ontbreken"}
    d = hist[need].copy()
    lok = d.index.tz_convert("Europe/Amsterdam")
    d["datum"] = lok.date
    d["uur"] = lok.hour
    cap = o["accu_capaciteit_kwh"]
    wear = o["accu_slijtage_per_kwh"]
    groepen = list(d.groupby("datum"))
    res = []
    for i, (datum, g) in enumerate(groepen):
        if i < 7 or i == len(groepen) - 1:
            continue
        g = g.dropna()
        if len(g) != 24:
            continue
        prijs = g["price"].to_numpy()
        last = g["house"].to_numpy()
        pv = g["pv"].to_numpy()
        # voorspeld verbruik: gemiddelde van hetzelfde uur, 7 voorgaande dagen
        vorig = pd.concat([x[1] for x in groepen[max(0, i - 7):i]])
        prof = vorig.groupby("uur")["house"].mean()
        last_fc = np.array([prof.get(u, np.nan) for u in g["uur"]])
        last_fc = np.where(np.isnan(last_fc), np.nanmean(last), last_fc)
        soc0 = float(g["soc"].iloc[0]) / 100 * cap
        soc_eind = float(g["soc"].iloc[-1]) / 100 * cap
        echt_kost = float(np.sum(prijs * g["grid"].to_numpy()))
        echt_ontl = float(np.clip(g["batt"].to_numpy(), 0, None).sum())
        p_real = plan_accu(prijs, last_fc, pv, soc0, soc_eind, o)
        p_perf = plan_accu(prijs, last, pv, soc0, soc_eind, o)
        if p_real is None or p_perf is None:
            continue
        net_r, ontl_r, eind_r, modi = simuleer(p_real, last, pv, soc0, o)
        net_p, ontl_p, eind_p, _ = simuleer(p_perf, last, pv, soc0, o)
        # verschil in eind-SoC eerlijk verrekenen tegen de gemiddelde laadprijs van die dag
        p_ref = float(np.percentile(prijs, 25))
        corr_r = (soc_eind - eind_r) * p_ref
        corr_p = (soc_eind - eind_p) * p_ref
        kost_r = float(np.sum(prijs * net_r)) + corr_r
        kost_p = float(np.sum(prijs * net_p)) + corr_p
        res.append({
            "datum": str(datum),
            "echt_eur": round(echt_kost + wear * echt_ontl, 2),
            "brein_realistisch_eur": round(kost_r + wear * ontl_r, 2),
            "brein_perfect_eur": round(kost_p + wear * ontl_p, 2),
            "besparing_realistisch_eur": round(echt_kost + wear * echt_ontl - kost_r - wear * ontl_r, 2),
            "besparing_perfect_eur": round(echt_kost + wear * echt_ontl - kost_p - wear * ontl_p, 2),
            "ontladen_kwh_echt": round(echt_ontl, 1),
            "ontladen_kwh_brein": round(ontl_r, 1),
            "uren_vasthouden": int(sum(m == "vasthouden" for m in modi)),
            "uren_laden": int(sum(m == "laden" for m in modi)),
            "prijs_min": round(float(prijs.min()), 3), "prijs_max": round(float(prijs.max()), 3),
        })
    if not res:
        return {"dagen": [], "melding": "te weinig volledige dagen met prijzen"}
    n = len(res)
    br = sum(x["besparing_realistisch_eur"] for x in res)
    bp = sum(x["besparing_perfect_eur"] for x in res)
    return {
        "dagen": res[-dagen:],
        "aantal_dagen": n,
        "besparing_realistisch_totaal_eur": round(br, 2),
        "besparing_realistisch_per_dag_eur": round(br / n, 2),
        "besparing_perfect_per_dag_eur": round(bp / n, 2),
        "dagen_brein_beter": int(sum(x["besparing_realistisch_eur"] > 0 for x in res)),
        "melding": "alleen accu; echte meetdata; realistisch = verbruik voorspeld met 7-dagenprofiel, "
                   "plan 1x per dag om 00:00 (het echte brein plant elk kwartier opnieuw); slijtage meegerekend",
    }
