"""Historische data ophalen en omzetten naar één uur-tabel met afgeleide grootheden."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

log = logging.getLogger("energiebrein.data")

# kolomnaam -> optie-sleutel met entity_id
KOLOMMEN = {
    "To": "ent_buiten_temp",
    "Ti": "ent_kamer_temp",
    "Tflow": "ent_aanvoer_temp",
    "Tret": "ent_retour_temp",
    "flow": "ent_debiet",          # m3/h
    "q_meet": "ent_wp_thermisch",  # kW thermisch volgens de warmtepomp zelf
    "p_hp_w": "ent_wp_elektrisch",  # W
    "pv_w": "ent_pv",              # W
    "house_w": "ent_huislast",     # W (totaal huis incl. WP en ELWA)
    "grid_w": "ent_net",           # W (+ = afname)
    "batt_w": "ent_accu_vermogen",  # W (+ = ontladen)
    "soc": "ent_soc",              # %
    "price": "ent_prijs",          # EUR/kWh incl. belasting
    "elwa_w": "ent_elwa",          # W
    "tank": "ent_boiler_temp",     # degC
}

DHW_AANVOER_GRENS = 56.0  # uur-gemiddelde aanvoer boven deze grens komt alleen bij tapwater voor


def _to_ts(v) -> pd.Timestamp:
    if isinstance(v, (int, float)):
        return pd.Timestamp(v, unit="ms", tz="UTC")
    return pd.Timestamp(v).tz_convert("UTC")


def stats_to_frame(stats: dict, opts: dict, kolommen=KOLOMMEN) -> pd.DataFrame:
    series = {}
    for kol, key in kolommen.items():
        eid = opts.get(key)
        rows = stats.get(eid) or []
        if not rows:
            log.info("geen statistiek voor %s (%s)", kol, eid)
            continue
        idx = [_to_ts(r["start"]) for r in rows]
        series[kol] = pd.Series([r.get("mean") for r in rows], index=idx, dtype="float64")
    if not series:
        return pd.DataFrame()
    df = pd.DataFrame(series).sort_index()
    df = df[~df.index.duplicated()]
    return df


def afleiden(df: pd.DataFrame) -> pd.DataFrame:
    """Voeg afgeleide kolommen toe (kW-eenheden)."""
    d = df.copy()
    for k in ["p_hp_w", "pv_w", "house_w", "grid_w", "batt_w", "elwa_w"]:
        if k not in d:
            d[k] = np.nan
    d["p_hp"] = d["p_hp_w"].clip(lower=0) / 1000.0
    d["pv"] = d["pv_w"].clip(lower=0) / 1000.0
    d["elwa"] = d["elwa_w"].fillna(0).clip(lower=0) / 1000.0
    d["grid"] = d["grid_w"] / 1000.0
    d["batt"] = d["batt_w"] / 1000.0
    d["house"] = d["house_w"].clip(lower=0) / 1000.0
    # basislast = alles behalve warmtepomp en ELWA
    d["base"] = (d["house"] - d["p_hp"].fillna(0) - d["elwa"]).clip(lower=0.05)
    # thermisch vermogen warmtepomp: debiet [m3/h] * 1.163 kWh/(m3 K) * dT
    # voorkeur: thermisch vermogen zoals de warmtepomp het zelf meet (uurgemiddelde klopt dan ook
    # bij pendelen); anders debiet x dT, wat bij aan/uit-bedrijf onderschat (gemiddelde x gemiddelde)
    if {"flow", "Tflow", "Tret"} <= set(d.columns):
        dT = (d["Tflow"] - d["Tret"]).clip(lower=0)
        q_calc = (d["flow"].clip(lower=0) * 1.163 * dT).clip(lower=0)
    else:
        q_calc = pd.Series(np.nan, index=d.index)
    if "q_meet" in d:
        d["q_hp"] = d["q_meet"].clip(lower=0).where(d["q_meet"].notna(), q_calc)
    else:
        d["q_hp"] = q_calc
    # tapwaterbedrijf: warmtepomp draait en de boiler warmt op (en niet door de ELWA),
    # of een aanvoer die alleen bij tapwater voorkomt
    tank = d["tank"] if "tank" in d else pd.Series(np.nan, index=d.index)
    tank_stijgt = tank.diff().fillna(0) > 0.8
    wp_aan = d["p_hp"].fillna(0) > 0.3
    elwa_uit = d["elwa"] < 0.3
    hoog = d.get("Tflow", pd.Series(np.nan, index=d.index)) > DHW_AANVOER_GRENS
    d["dhw"] = (wp_aan & elwa_uit & tank_stijgt) | hoog
    d["q_space"] = d["q_hp"].where(~d["dhw"], 0.0)
    d["import"] = d["grid"].clip(lower=0)
    d["export"] = (-d["grid"]).clip(lower=0)
    return d


def ophalen(ha, opts: dict, dagen: int = 400) -> pd.DataFrame:
    eids = [opts[k] for k in KOLOMMEN.values() if opts.get(k)]
    start = datetime.now(timezone.utc) - timedelta(days=dagen)
    stats = ha.statistics(eids, start, period="hour", types=["mean"])
    df = stats_to_frame(stats, opts)
    log.info("historie: %d uren, kolommen %s", len(df), list(df.columns))
    return afleiden(df) if len(df) else df
