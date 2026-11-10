"""Voorspellers: weer (Open-Meteo), zonopbrengst en basisverbruik (LightGBM).

- Weer: Open-Meteo, inclusief 'global tilted irradiance' op het vlak van de panelen
  (Open-Meteo azimut: 0 = zuid, -90 = oost, 90 = west).
- Zon: LightGBM leert de echte opbrengst uit GTI + uur + seizoen + temperatuur
  (vangt omvormerlimiet, schaduw, vuil en paneelgedrag). Fallback: fysische formule.
- Basisverbruik (huis zonder warmtepomp/ELWA): LightGBM op uur, weekdag, buitentemp
  en het profiel van de afgelopen 7 dagen. Ook een 80%-kwantiel voor onzekerheid.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import requests

log = logging.getLogger("energiebrein.forecast")

try:
    import lightgbm as lgb
except Exception:  # pragma: no cover
    lgb = None

OM_URL = "https://api.open-meteo.com/v1/forecast"


class _Gbm:
    """Dunne wrapper rond de native LightGBM-API (geen scikit-learn nodig in de container)."""

    def __init__(self, n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=20,
                 objective="regression", alpha=None, monotone_constraints=None, verbose=-1):
        self.rounds = n_estimators
        self.params = {"objective": objective, "learning_rate": learning_rate, "num_leaves": num_leaves,
                       "min_data_in_leaf": min_child_samples, "verbose": verbose, "num_threads": 2}
        if alpha is not None:
            self.params["alpha"] = alpha
        if monotone_constraints is not None:
            self.params["monotone_constraints"] = monotone_constraints
        self.booster = None

    def fit(self, X, y):
        ds = lgb.Dataset(np.asarray(X, dtype=float), label=np.asarray(y, dtype=float), free_raw_data=True)
        self.booster = lgb.train(self.params, ds, num_boost_round=self.rounds)
        return self

    def predict(self, X):
        return self.booster.predict(np.asarray(X, dtype=float))


def weer_ophalen(lat: float, lon: float, helling: float, azimut_kompas: float,
                 past_days: int = 92, forecast_days: int = 3) -> pd.DataFrame:
    az = azimut_kompas - 180.0  # kompas (180 = zuid) -> Open-Meteo (0 = zuid)
    params = {
        "latitude": lat, "longitude": lon,
        "hourly": "temperature_2m,global_tilted_irradiance,cloud_cover,shortwave_radiation,wind_speed_10m",
        "tilt": helling, "azimuth": az,
        "past_days": past_days, "forecast_days": forecast_days,
        "timezone": "UTC",
    }
    laatste = None
    for poging in range(3):  # Open-Meteo is soms traag vanaf de Pi: korter wachten, vaker proberen
        try:
            r = requests.get(OM_URL, params=params, timeout=25)
            r.raise_for_status()
            break
        except Exception as e:
            laatste = e
            log.warning("Open-Meteo poging %d mislukt: %s", poging + 1, e)
    else:
        raise RuntimeError(f"Open-Meteo onbereikbaar: {laatste}")
    h = r.json()["hourly"]
    df = pd.DataFrame(h)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.set_index("time").rename(columns={
        "temperature_2m": "To_fc", "global_tilted_irradiance": "gti",
        "cloud_cover": "cloud", "shortwave_radiation": "ghi", "wind_speed_10m": "wind"})
    # Open-Meteo straling is gemiddelde over het VOORGAANDE uur; HA-statistiek 'start' is begin van het uur
    df[["gti", "ghi"]] = df[["gti", "ghi"]].shift(-1)
    return df


def _tijd_features(idx: pd.DatetimeIndex) -> pd.DataFrame:
    lok = idx.tz_convert("Europe/Amsterdam")
    doy = lok.dayofyear.to_numpy()
    return pd.DataFrame({
        "uur": lok.hour.to_numpy() + lok.minute.to_numpy() / 60.0,
        "dow": lok.dayofweek.to_numpy(),
        "weekend": (lok.dayofweek >= 5).astype(int),
        "doy_s": np.sin(2 * np.pi * doy / 365.25),
        "doy_c": np.cos(2 * np.pi * doy / 365.25),
    }, index=idx)


# ---------------------------------------------------------------- zon
class PvModel:
    def __init__(self, kwp: float, ac_max: float):
        self.kwp, self.ac_max = kwp, ac_max
        self.model = None
        self.mae = float("nan")

    def fysisch(self, gti):
        return np.clip(self.kwp * np.asarray(gti, dtype=float) / 1000.0 * 0.86, 0, self.ac_max)

    def _X(self, w: pd.DataFrame):
        X = _tijd_features(w.index)
        X["gti"] = w["gti"].to_numpy()
        X["ghi"] = w.get("ghi", w["gti"]).to_numpy()
        X["cloud"] = w.get("cloud", 0).to_numpy() if "cloud" in w else 0
        X["temp"] = w.get("To_fc", 10).to_numpy() if "To_fc" in w else 10
        X["fys"] = self.fysisch(w["gti"])
        return X[["uur", "doy_s", "doy_c", "gti", "ghi", "cloud", "temp", "fys"]]

    def fit(self, hist: pd.DataFrame, weer: pd.DataFrame):
        j = hist[["pv"]].join(weer, how="inner").dropna(subset=["pv", "gti"])
        if lgb is None or len(j) < 24 * 14:
            log.info("PV-model: fysische formule (%d uren)", len(j))
            return self
        X = self._X(j)
        y = j["pv"].to_numpy()
        n = len(j)
        split = int(n * 0.85)
        mono = [0, 0, 0, 1, 1, 0, 0, 1]
        self.model = _Gbm(n_estimators=400, learning_rate=0.03, num_leaves=15,
                          min_child_samples=10, monotone_constraints=mono)
        self.model.fit(X.iloc[:split], y[:split])
        pred = np.clip(self.model.predict(X.iloc[split:]), 0, self.ac_max)
        self.mae = float(np.mean(np.abs(pred - y[split:])))
        mae_fys = float(np.mean(np.abs(X["fys"].iloc[split:] - y[split:])))
        if mae_fys <= self.mae:
            log.info("PV-model: fysisch beter (%.3f vs %.3f kW), ML uit", mae_fys, self.mae)
            self.model, self.mae = None, mae_fys
        else:
            self.model.fit(X, y)  # hertrain op alles
            log.info("PV-model: LightGBM MAE %.3f kW (fysisch %.3f)", self.mae, mae_fys)
        return self

    def predict(self, weer: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            return self.fysisch(weer["gti"].fillna(0))
        X = self._X(weer.fillna({"gti": 0}))
        return np.clip(self.model.predict(X), 0, self.ac_max) * (weer["gti"].fillna(0).to_numpy() > 1)


# ---------------------------------------------------------------- basisverbruik
class LoadModel:
    def __init__(self):
        self.m50 = None
        self.m80 = None
        self.profiel = None
        self.mae = float("nan")
        self.mae_naief = float("nan")

    @staticmethod
    def _prof7(base: pd.Series) -> pd.Series:
        """Gemiddelde basislast op hetzelfde uur over de 7 voorgaande dagen."""
        s = base.copy()
        lok = s.index.tz_convert("Europe/Amsterdam")
        df = pd.DataFrame({"v": s.to_numpy(), "uur": lok.hour}, index=s.index)
        out = df.groupby("uur")["v"].transform(lambda x: x.shift(1).rolling(7, min_periods=2).mean())
        return out

    def _X(self, idx, To, prof):
        X = _tijd_features(idx)
        X["To"] = np.asarray(To, dtype=float)
        X["prof7"] = np.asarray(prof, dtype=float)
        return X[["uur", "dow", "weekend", "doy_s", "doy_c", "To", "prof7"]]

    def fit(self, hist: pd.DataFrame):
        d = hist[["base", "To"]].dropna()
        d = d[d["base"] < 15]
        if len(d) < 24 * 7:
            return self
        prof = self._prof7(d["base"])
        lok = d.index.tz_convert("Europe/Amsterdam")
        self.profiel = d["base"].groupby(lok.hour).median()
        ok = prof.notna()
        d, prof = d[ok], prof[ok]
        if lgb is None or len(d) < 24 * 14:
            return self
        X = self._X(d.index, d["To"], prof)
        y = d["base"].to_numpy()
        split = int(len(d) * 0.85)
        kw = dict(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=20)
        m = _Gbm(objective="quantile", alpha=0.5, **kw).fit(X.iloc[:split], y[:split])
        self.mae = float(np.mean(np.abs(m.predict(X.iloc[split:]) - y[split:])))
        self.mae_naief = float(np.mean(np.abs(prof.iloc[split:].to_numpy() - y[split:])))
        if self.mae >= self.mae_naief:
            log.info("Lastmodel: LightGBM (%.3f) niet beter dan 7-dagenprofiel (%.3f), profiel gebruikt",
                     self.mae, self.mae_naief)
            return self
        self.m50 = _Gbm(objective="quantile", alpha=0.5, **kw).fit(X, y)
        self.m80 = _Gbm(objective="quantile", alpha=0.8, **kw).fit(X, y)
        log.info("Lastmodel: MAE %.3f kW (naief 7-dagenprofiel %.3f)", self.mae, self.mae_naief)
        return self

    def predict(self, idx: pd.DatetimeIndex, To, hist_base: pd.Series):
        lok = idx.tz_convert("Europe/Amsterdam")
        recent = hist_base.dropna()
        recent = recent[recent.index >= recent.index.max() - pd.Timedelta(days=7)] if len(recent) else recent
        if len(recent):
            rl = recent.index.tz_convert("Europe/Amsterdam")
            p = recent.groupby(rl.hour).mean()
            prof = np.array([p.get(h, np.nan) for h in lok.hour])
        else:
            prof = np.full(len(idx), np.nan)
        if np.isnan(prof).any():
            fb = self.profiel if self.profiel is not None else pd.Series(0.4, index=range(24))
            prof = np.where(np.isnan(prof), [fb.get(h, 0.4) for h in lok.hour], prof)
        if self.m50 is None:
            return prof, prof * 1.3
        X = self._X(idx, To, prof)
        p50 = np.clip(self.m50.predict(X), 0.05, None)
        p80 = np.maximum(np.clip(self.m80.predict(X), 0.05, None), p50)
        return p50, p80


# ---------------------------------------------------------------- terugval: weer uit Home Assistant
def zonnestand(idx: pd.DatetimeIndex, lat: float, lon: float):
    """Zonnehoogte en -azimut (graden, azimut kompas) - eenvoudige NOAA-benadering, ruim nauwkeurig genoeg."""
    t = idx.tz_convert("UTC")
    doy = t.dayofyear.to_numpy()
    uur = t.hour.to_numpy() + t.minute.to_numpy() / 60.0
    g = 2 * np.pi / 365.0 * (doy - 1 + (uur - 12) / 24)
    decl = (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g) - 0.006758 * np.cos(2 * g)
            + 0.000907 * np.sin(2 * g) - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g))
    eqt = 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g) - 0.014615 * np.cos(2 * g)
                    - 0.040849 * np.sin(2 * g))
    tst = uur * 60 + eqt + 4 * lon
    ha = np.radians(tst / 4 - 180)
    la = np.radians(lat)
    cosz = np.sin(la) * np.sin(decl) + np.cos(la) * np.cos(decl) * np.cos(ha)
    elev = np.degrees(np.arcsin(np.clip(cosz, -1, 1)))
    az = np.degrees(np.arctan2(np.sin(ha), np.cos(ha) * np.sin(la) - np.tan(decl) * np.cos(la))) + 180
    return elev, az


def gti_schatting(idx, lat, lon, helling, azimut_kompas, bewolking_pct):
    """Instraling op het paneelvlak uit zonnestand + bewolking (Kasten-Czeplak)."""
    elev, az = zonnestand(idx, lat, lon)
    sin_e = np.clip(np.sin(np.radians(elev)), 0, None)
    ghi_hel = 1098 * sin_e * np.exp(-0.057 / np.maximum(sin_e, 0.05)) * (elev > 0)
    ghi = ghi_hel * (1 - 0.75 * (np.clip(bewolking_pct, 0, 100) / 100.0) ** 3.4)
    b, ap = np.radians(helling), np.radians(azimut_kompas)
    e, a = np.radians(np.maximum(elev, 0)), np.radians(az)
    cos_i = np.sin(e) * np.cos(b) + np.cos(e) * np.sin(b) * np.cos(a - ap)
    kd = 0.3 + 0.7 * (np.clip(bewolking_pct, 0, 100) / 100.0)  # diffuus aandeel
    straal = ghi * (1 - kd) * np.clip(cos_i, 0, None) / np.maximum(sin_e, 0.1)
    diffuus = ghi * kd * (1 + np.cos(b)) / 2
    return np.clip(straal + diffuus, 0, 1100)


def weer_uit_ha(ha, entity_id: str, lat, lon, helling, azimut_kompas) -> pd.DataFrame | None:
    """Uurverwachting van een HA-weerentiteit (temperatuur, bewolking) + geschatte GTI."""
    try:
        r = ha.s.post(f"{ha_rest()}/services/weather/get_forecasts?return_response",
                      json={"entity_id": entity_id, "type": "hourly"}, timeout=30)
        r.raise_for_status()
        fc = r.json()["service_response"][entity_id]["forecast"]
    except Exception as e:
        log.warning("weer uit HA mislukt: %s", e)
        return None
    df = pd.DataFrame(fc)
    df["time"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.set_index("time")
    out = pd.DataFrame(index=df.index)
    out["To_fc"] = df["temperature"].astype(float)
    out["cloud"] = df.get("cloud_coverage", pd.Series(50.0, index=df.index)).astype(float).fillna(50)
    mid = out.index + pd.Timedelta(minutes=30)
    out["gti"] = gti_schatting(mid, lat, lon, helling, azimut_kompas, out["cloud"].to_numpy())
    out["ghi"] = out["gti"]
    log.info("weer: terugval op %s (%d uur)", entity_id, len(out))
    return out


def ha_rest():
    from ha import REST
    return REST
