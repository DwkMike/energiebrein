"""Inkoop- en terugleverprijzen per tijdstap, inclusief de regels vanaf 2027.

Zonneplan all-in prijs (incl. btw)  p = (markt + opslag + EB) * (1 + btw)
=> markt ~= p / (1 + btw) - EB - opslag

Teruglevering:
  - t/m 31-12-2026 (saldering, netto-afnemer): teruglevering waard = inkoopprijs
  - vanaf 1-1-2027: markt + terugleververgoeding (+ zonnebonus% van markt als markt > 0)
  - optioneel scenario 'dubbele energiebelasting': EB over teruglevering eraf
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

GRENS_2027 = pd.Timestamp("2027-01-01T00:00:00", tz="Europe/Amsterdam").tz_convert("UTC")


def forecast_uit_state(state: dict | None) -> pd.Series:
    """Zonneplan forecast-attribuut -> Series (UTC index, EUR/kWh incl. belasting)."""
    if not state:
        return pd.Series(dtype=float)
    fc = (state.get("attributes") or {}).get("forecast") or []
    rows = []
    for f in fc:
        try:
            t = pd.Timestamp(f["start_date"]).tz_convert("UTC")
            amt = f["price_tax_included"]["amount"] if isinstance(f.get("price_tax_included"), dict) \
                else f.get("price_tax_included")
            rows.append((t, float(amt) / 1e7))
        except Exception:
            continue
    if not rows:
        return pd.Series(dtype=float)
    s = pd.Series(dict(rows)).sort_index()
    return s[~s.index.duplicated()]


def prijzen_voor(idx: pd.DatetimeIndex, fc: pd.Series, hist_prijs: pd.Series | None) -> tuple[np.ndarray, np.ndarray]:
    """Inkoopprijs per tijdstap + vlag of hij bekend (True) of geschat (False) is."""
    out = np.full(len(idx), np.nan)
    known = np.zeros(len(idx), dtype=bool)
    if len(fc):
        f = fc.reindex(idx, method="ffill", tolerance=pd.Timedelta("59min"))
        last = fc.index.max() + pd.Timedelta(minutes=15)
        m = f.notna().to_numpy() & (idx < last)
        out[m] = f.to_numpy()[m]
        known = m
    if np.isnan(out).any():
        lok = idx.tz_convert("Europe/Amsterdam")
        if hist_prijs is not None and len(hist_prijs.dropna()) > 48:
            h = hist_prijs.dropna()
            h = h[h.index >= h.index.max() - pd.Timedelta(days=14)]
            prof = h.groupby(h.index.tz_convert("Europe/Amsterdam").hour).median()
            fill = np.array([prof.get(x, h.median()) for x in lok.hour])
        else:
            fill = np.full(len(idx), np.nanmean(out) if (~np.isnan(out)).any() else 0.30)
        out = np.where(np.isnan(out), fill, out)
    return out, known


def marktprijs(p_buy: np.ndarray, o: dict) -> np.ndarray:
    return p_buy / (1.0 + o["btw"]) - o["energiebelasting"] - o["leverancier_opslag"]


def terugleverprijs(idx: pd.DatetimeIndex, p_buy: np.ndarray, o: dict) -> np.ndarray:
    markt = marktprijs(p_buy, o)
    na2027 = markt + o["terug_vergoeding"] + o["zonnebonus"] * np.maximum(markt, 0)
    if o.get("dubbele_eb"):
        na2027 = na2027 - o["eb_terug_per_kwh"]
    modus = o.get("tarief_modus", "auto")
    if modus == "saldering":
        return p_buy.copy()
    if modus == "2027":
        return na2027
    is_2027 = np.asarray(idx >= GRENS_2027)
    return np.where(is_2027, na2027, p_buy)


def is_saldering(t: datetime, o: dict) -> bool:
    modus = o.get("tarief_modus", "auto")
    if modus == "saldering":
        return True
    if modus == "2027":
        return False
    return pd.Timestamp(t).tz_convert("UTC") < GRENS_2027
