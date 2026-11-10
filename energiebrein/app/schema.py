"""Comfortgrenzen en tapwatervraag per tijdstap."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _blokken(s: str):
    out = []
    for part in (s or "").split(","):
        part = part.strip()
        if "-" not in part:
            continue
        a, b = part.split("-")
        ha, ma = map(int, a.split(":"))
        hb, mb = map(int, b.split(":"))
        out.append((ha * 60 + ma, hb * 60 + mb))
    return out


def comfortgrenzen(idx: pd.DatetimeIndex, o: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ondergrens, bovengrens en 'comfortblok actief' per tijdstap (einde van de stap)."""
    lok = (idx + (idx[1] - idx[0] if len(idx) > 1 else pd.Timedelta(0))).tz_convert("Europe/Amsterdam")
    minuut = lok.hour * 60 + lok.minute
    blok = _blokken(o["comfort_blokken"])
    in_blok = np.zeros(len(idx), dtype=bool)
    for a, b in blok:
        in_blok |= (minuut >= a) & (minuut < b)
    eerste = min((a for a, _ in blok), default=6 * 60)
    laatste = max((b for _, b in blok), default=22 * 60 + 30)
    overdag = (minuut >= eerste) & (minuut < laatste)
    low = np.where(in_blok, o["comfort_temp"], np.where(overdag, o["eco_dag_temp"], o["eco_nacht_temp"]))
    high = np.full(len(idx), o["comfort_temp"] + 1.0)
    return low.astype(float), high, in_blok


def tapwatervraag(idx: pd.DatetimeIndex, o: dict) -> np.ndarray:
    """Thermische tapwatervraag (kW): 40% ochtend 06-09, 60% avond 18-22."""
    lok = idx.tz_convert("Europe/Amsterdam")
    h = lok.hour
    per_dag = o["tapwater_kwh_per_dag"]
    v = np.zeros(len(idx))
    v[(h >= 6) & (h < 9)] = 0.4 * per_dag / 3.0
    v[(h >= 18) & (h < 22)] = 0.6 * per_dag / 4.0
    return v


def boiler_energie(tank_temp: float | None, o: dict) -> float:
    """Ruwe inschatting van bruikbare boiler-energie uit de tanktemperatuur (40-60 graden = 0-max)."""
    if tank_temp is None or not np.isfinite(tank_temp):
        return o["boiler_max_kwh"] / 2
    return float(np.clip((tank_temp - 40.0) / 20.0, 0, 1) * o["boiler_max_kwh"])
