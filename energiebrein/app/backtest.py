"""Terugrekentest: wat had het brein per dag gekost t.o.v. de werkelijkheid?

Per dag (uurdata) een 24-uursplan met de werkelijke prijzen, zon, basisverbruik en
buitentemperatuur ('perfecte voorspelling' = bovengrens van de winst).
Eerlijkheid:
  - niet kouder dan het in werkelijkheid was en niet kouder dan het stookschema
    (ondergrens = minimum van beide), eind-kamertemperatuur minimaal als werkelijk;
  - accu aan het eind minimaal even vol als werkelijk;
  - tapwater (WP-tapwater + ELWA) telt als vaste last, die wordt hier niet geoptimaliseerd;
  - kosten met dezelfde tariefregel als de werkelijkheid in die periode (saldering).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from optimizer import Invoer, plan
from schema import comfortgrenzen
from prices import terugleverprijs
from house import te_evenwicht

log = logging.getLogger("energiebrein.backtest")


def run(hist: pd.DataFrame, house, cop, o: dict, dagen: int = 30) -> dict:
    need = ["price", "pv", "base", "To", "Ti", "grid", "soc", "p_hp", "dhw"]
    if not set(need) <= set(hist.columns):
        return {"dagen": [], "melding": "kolommen ontbreken"}
    d = hist.copy()
    d["vast"] = d["base"] + d["p_hp"].fillna(0) * d["dhw"].astype(float) + d["elwa"].fillna(0)
    lok = d.index.tz_convert("Europe/Amsterdam")
    d["datum"] = lok.date
    resultaten = []
    Te_reeks = house.te_reeks(d[["Ti", "To", "q_space", "pv", "base"]].astype(float).ffill(limit=2))
    d["Te_est"] = Te_reeks
    for datum, g in list(d.groupby("datum"))[-dagen - 1:-1]:
        g = g.dropna(subset=["price", "pv", "base", "To", "Ti", "grid", "soc"])
        if len(g) < 24:
            continue
        g = g.iloc[:24]
        idx = g.index
        o2 = dict(o, tarief_modus="saldering")
        p_buy = g["price"].to_numpy()
        p_sell = terugleverprijs(idx, p_buy, o2)
        low_s, high_s, _ = comfortgrenzen(idx, o)
        Ti_echt = g["Ti"].to_numpy()
        Ti_next = np.append(Ti_echt[1:], Ti_echt[-1])
        low = np.minimum(low_s, Ti_next)
        low[-1] = Ti_echt[-1] - 0.1
        high = np.maximum(high_s, Ti_next + 0.2)
        cap = o["accu_capaciteit_kwh"]
        inv = Invoer(dt=1.0, p_buy=p_buy, p_sell=p_sell, base=g["vast"].to_numpy(), pv=g["pv"].to_numpy(),
                     To=g["To"].to_numpy(), Tlow=low, Thigh=high, dhw_vraag=np.zeros(24),
                     soc0=float(g["soc"].iloc[0]) / 100 * cap, Ti0=float(Ti_echt[0]),
                     Te0=float(g["Te_est"].iloc[0]) if np.isfinite(g["Te_est"].iloc[0])
                     else float(te_evenwicht(house.params, Ti_echt[0], g["To"].iloc[0])),
                     eb0=0.0, soc_eind_min=float(g["soc"].iloc[-1]) / 100 * cap, eindwaarde=False)
        o3 = dict(o, boiler_max_kwh=0.0, elwa_max_kw=0.0)
        r = plan(inv, house, cop, o3, accu_export_toegestaan=bool(o.get("accu_verkoop_toestaan_2026")))
        if r is None:
            continue
        echt = float(np.sum(g["price"] * g["grid"]))  # saldering: netto kWh * prijs
        brein = float(np.sum(p_buy * (r["imp"] - r["exp"])))
        resultaten.append({
            "datum": str(datum), "echt_eur": round(echt, 2), "brein_eur": round(brein, 2),
            "besparing_eur": round(echt - brein, 2),
            "wp_kwh_echt": round(float(g["p_hp"].fillna(0).sum()), 1),
            "wp_kwh_brein": round(float((r["q"] / r["cop"]).sum() + (g["p_hp"].fillna(0) * g["dhw"]).sum()), 1),
            "comfort_tekort_kh": round(float(r["slo"].sum()), 2),
        })
    if not resultaten:
        return {"dagen": [], "melding": "te weinig volledige dagen"}
    tot = sum(x["besparing_eur"] for x in resultaten)
    return {
        "dagen": resultaten,
        "aantal_dagen": len(resultaten),
        "besparing_totaal_eur": round(tot, 2),
        "besparing_per_dag_eur": round(tot / len(resultaten), 2),
        "echt_totaal_eur": round(sum(x["echt_eur"] for x in resultaten), 2),
        "melding": "perfecte voorspelling (bovengrens); realistische winst ligt lager",
    }
