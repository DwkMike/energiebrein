"""Zelflerende dagvoorspelling voor zon en huisverbruik (v0.1.7).

Elke dag:
  * om ~21:00 legt het brein vast wat het voor MORGEN verwacht (zon, verbruik), samen met wat
    Forecast.Solar en de HA-dagbehoefte op dat moment zeggen ("snapshot");
  * bij de nachtelijke training vergelijkt het elke snapshot met de werkelijke meting en leert:
      - zon:      correctiefactor k_pv = som werkelijk / som voorspeld over de laatste 14 dagen
      - verbruik: het weermodel (graaddagen + weekend) wordt opnieuw gefit, plus een restcorrectie
  * de kwaliteit (gemiddelde afwijking in kWh/dag) van brein, Forecast.Solar en de HA-dagbehoefte
    wordt gepubliceerd, zodat je kunt zien welke voorspelling het beste is voordat je ermee stuurt.

Verbruik = huisverbruik ZONDER ELWA (zelfde definitie als automation.accu_leer_dagbehoefte).
Model:  kWh/dag = a + b * max(0, Tbasis - Tgem) + c * weekend   (gewogen, recente dagen tellen zwaarder)
"""
from __future__ import annotations

import json
import logging
import os
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

log = logging.getLogger("energiebrein.zelfleren")
TZ = "Europe/Amsterdam"
PAD = os.environ.get("EB_ZELFLEREN", "/data/zelflerend.json")
HALFWAARDE_DAGEN = 21.0
VENSTER_DAGEN = 60
EVAL_DAGEN = 14


def dagtotalen(hist: pd.DataFrame) -> pd.DataFrame:
    """Uurhistorie -> per lokale dag: zon kWh, verbruik excl. ELWA kWh, gemiddelde buitentemp, weekend."""
    need = {"pv", "house", "elwa", "To"}
    if hist is None or len(hist) == 0 or not need <= set(hist.columns):
        return pd.DataFrame()
    d = hist[list(need)].copy()
    d["datum"] = d.index.tz_convert(TZ).date
    g = d.groupby("datum")
    n = g["house"].count()
    out = pd.DataFrame({
        "zon": g["pv"].sum() * 24 / n.clip(lower=1),
        "verbruik": (g["house"].sum() - g["elwa"].sum()) * 24 / n.clip(lower=1),
        "To": g["To"].mean(),
        "n": n,
    })
    out = out[out["n"] >= 22].copy()   # alleen (bijna) volledige dagen; zomer/wintertijd = 23/25 uur
    out["weekend"] = [1 if pd.Timestamp(x).dayofweek >= 5 else 0 for x in out.index]
    return out


def _lokaal(nu=None) -> pd.Timestamp:
    """Altijd Nederlandse tijd (de App-container draait op UTC)."""
    if nu is None:
        return pd.Timestamp.now(tz=TZ)
    t = pd.Timestamp(nu)
    return t.tz_localize(TZ) if t.tzinfo is None else t.tz_convert(TZ)


def _gewichten(datums, ref: date) -> np.ndarray:
    leeftijd = np.array([(ref - x).days for x in datums], dtype=float)
    return 0.5 ** (leeftijd / HALFWAARDE_DAGEN)


class VerbruiksModel:
    def __init__(self):
        self.a, self.b, self.c, self.tbasis = 35.0, 0.0, 0.0, 16.0
        self.n = 0
        self.mae_model = float("nan")
        self.mae_ema = float("nan")
        self.methode = "standaard"

    def _fit(self, dd: pd.DataFrame, ref: date, tbasis: float):
        hdd = np.clip(tbasis - dd["To"].to_numpy(), 0, None)
        X = np.column_stack([np.ones(len(dd)), hdd, dd["weekend"].to_numpy()])
        y = dd["verbruik"].to_numpy()
        w = np.sqrt(_gewichten(dd.index, ref))
        coef, *_ = np.linalg.lstsq(X * w[:, None], y * w, rcond=None)
        coef[1] = max(coef[1], 0.0)     # kouder mag nooit minder verbruik geven
        res = y - X @ coef
        return coef, float(np.sqrt(np.average(res ** 2, weights=w ** 2)))

    def fit(self, dag: pd.DataFrame):
        dd = dag.dropna(subset=["verbruik", "To"]).tail(VENSTER_DAGEN)
        self.n = len(dd)
        if self.n < 10:
            if self.n:
                self.a = float(dd["verbruik"].mean())
            self.methode = "gemiddelde (te weinig dagen)"
            return self
        ref = dd.index[-1]
        best = None
        for tb in (14.0, 15.0, 16.0, 17.0, 18.0):
            coef, rmse = self._fit(dd, ref, tb)
            if best is None or rmse < best[2]:
                best = (tb, coef, rmse)
        self.tbasis, (self.a, self.b, self.c) = best[0], (float(x) for x in best[1])
        self.methode = "graaddagen+weekend"
        self._terugtest(dd)
        return self

    def _terugtest(self, dd: pd.DataFrame):
        """Een-dag-vooruit test: model (gefit op de dagen ervoor) vs de HA-methode (EMA 0,25)."""
        e_m, e_e = [], []
        ema = float(dd["verbruik"].iloc[:7].mean())
        for i in range(len(dd)):
            if i >= 14:
                tr = dd.iloc[:i]
                coef, _ = self._fit(tr, tr.index[-1], self.tbasis)
                x = dd.iloc[i]
                p = coef[0] + coef[1] * max(0.0, self.tbasis - x["To"]) + coef[2] * x["weekend"]
                e_m.append(abs(p - x["verbruik"]))
                e_e.append(abs(ema - x["verbruik"]))
            ema = 0.75 * ema + 0.25 * float(dd["verbruik"].iloc[i])
        if e_m:
            self.mae_model, self.mae_ema = float(np.mean(e_m)), float(np.mean(e_e))

    def predict(self, tgem: float, weekend: int) -> float:
        return max(0.0, self.a + self.b * max(0.0, self.tbasis - tgem) + self.c * weekend)

    def to_dict(self):
        r = lambda v: round(v, 3) if isinstance(v, float) and np.isfinite(v) else v  # noqa: E731
        return {"basis_kwh": r(self.a), "kwh_per_graaddag": r(self.b), "weekend_kwh": r(self.c),
                "graaddag_basis_c": self.tbasis, "dagen": self.n, "methode": self.methode,
                "terugtest_mae_model_kwh": r(self.mae_model), "terugtest_mae_ha_methode_kwh": r(self.mae_ema)}


class Zelflerend:
    def __init__(self, pad: str = PAD):
        self.pad = pad
        self.s = {"snapshots": {}, "k_pv": 1.0, "verbruik_bias": 0.0, "evaluatie": []}
        self.verbruik = VerbruiksModel()
        self.kwaliteit: dict = {}
        try:
            with open(pad) as f:
                self.s.update(json.load(f))
        except FileNotFoundError:
            pass
        except Exception as e:
            log.warning("zelflerend-bestand onleesbaar (%s), begin opnieuw", e)

    def _bewaar(self):
        try:
            os.makedirs(os.path.dirname(self.pad) or ".", exist_ok=True)
            tmp = self.pad + ".tmp"
            with open(tmp, "w") as f:
                json.dump(self.s, f)
            os.replace(tmp, self.pad)
        except Exception as e:
            log.warning("zelflerend opslaan mislukt: %s", e)

    # ---------------------------------------------------------------- leren (nachtelijk)
    def leer(self, hist: pd.DataFrame):
        dag = dagtotalen(hist)
        self.verbruik.fit(dag)
        ev = []
        for d_str, sn in sorted(self.s["snapshots"].items()):
            d = date.fromisoformat(d_str)
            if d not in dag.index:
                continue
            w = dag.loc[d]
            ev.append({"datum": d_str, "zon_echt": round(float(w["zon"]), 2),
                       "zon_brein_ruw": sn.get("zon_ruw"), "zon_brein": sn.get("zon"),
                       "zon_forecast_solar": sn.get("zon_fs"),
                       "verbruik_echt": round(float(w["verbruik"]), 2),
                       "verbruik_brein": sn.get("verbruik"), "verbruik_ha": sn.get("verbruik_ha")})
        ev = ev[-30:]
        self.s["evaluatie"] = ev
        recent = ev[-EVAL_DAGEN:]
        # zon: correctiefactor uit de som (stabiel, ook bij grijze dagen)
        zr = [(e["zon_echt"], e["zon_brein_ruw"]) for e in recent if e.get("zon_brein_ruw") is not None]
        if len(zr) >= 3 and sum(f for _, f in zr) > 3:
            self.s["k_pv"] = float(np.clip(sum(a for a, _ in zr) / sum(f for _, f in zr), 0.5, 2.0))
        # verbruik: restafwijking van het weermodel
        vr = [e["verbruik_echt"] - e["verbruik_brein"] for e in recent if e.get("verbruik_brein") is not None]
        if len(vr) >= 3:
            self.s["verbruik_bias"] = float(np.clip(np.mean(vr), -8, 8))
        self.kwaliteit = self._kwaliteit(recent)
        self._bewaar()
        log.info("zelflerend: k_pv=%.2f, verbruik-bias=%.1f kWh, model=%s, kwaliteit=%s",
                 self.s["k_pv"], self.s["verbruik_bias"], self.verbruik.to_dict(), self.kwaliteit)

    @staticmethod
    def _mae(paren):
        p = [(a, b) for a, b in paren if a is not None and b is not None]
        return (round(float(np.mean([abs(a - b) for a, b in p])), 2), len(p)) if p else (None, 0)

    def _kwaliteit(self, recent):
        zb, n = self._mae([(e["zon_echt"], e.get("zon_brein")) for e in recent])
        zf, _ = self._mae([(e["zon_echt"], e.get("zon_forecast_solar")) for e in recent])
        vb, _ = self._mae([(e["verbruik_echt"], e.get("verbruik_brein")) for e in recent])
        vh, _ = self._mae([(e["verbruik_echt"], e.get("verbruik_ha")) for e in recent])
        return {"dagen": n, "zon_mae_brein_kwh": zb, "zon_mae_forecast_solar_kwh": zf,
                "verbruik_mae_brein_kwh": vb, "verbruik_mae_ha_dagbehoefte_kwh": vh}

    # ---------------------------------------------------------------- voorspellen (elk kwartier)
    def voorspel(self, weer: pd.DataFrame | None, pv_model, nu: datetime | None = None) -> dict:
        nu = _lokaal(nu)
        vandaag = nu.date()
        morgen = vandaag + timedelta(days=1)
        out = {"vandaag": vandaag.isoformat(), "morgen": morgen.isoformat(),
               "k_pv": round(self.s["k_pv"], 3), "verbruik_bias": round(self.s["verbruik_bias"], 2)}
        if weer is None or len(weer) == 0:
            return out
        w = weer.copy()
        w["pv_kw"] = pv_model.predict(w)
        lok = w.index.tz_convert(TZ)
        w["datum"] = lok.date
        # uur-start >= begin van het huidige uur telt mee als 'rest van vandaag'
        rest = w[(w["datum"] == vandaag) & (lok >= nu.floor("h"))]
        for naam, d, sub in (("vandaag", vandaag, w[w["datum"] == vandaag]),
                             ("morgen", morgen, w[w["datum"] == morgen])):
            if len(sub) < 20:
                continue
            zon_ruw = float(sub["pv_kw"].sum())
            tgem = float(sub["To_fc"].mean()) if "To_fc" in sub else float("nan")
            weekend = 1 if pd.Timestamp(d).dayofweek >= 5 else 0
            verb = self.verbruik.predict(tgem, weekend) + self.s["verbruik_bias"] if np.isfinite(tgem) else None
            out[f"zon_{naam}_ruw"] = round(zon_ruw, 2)
            out[f"zon_{naam}"] = round(zon_ruw * self.s["k_pv"], 2)
            out[f"tgem_{naam}"] = round(tgem, 1) if np.isfinite(tgem) else None
            out[f"verbruik_{naam}"] = round(max(verb, 0.0), 1) if verb is not None else None
            out[f"weekend_{naam}"] = weekend
        out["zon_vandaag_rest"] = round(float(rest["pv_kw"].sum()) * self.s["k_pv"], 2)
        return out

    def snapshot(self, vp: dict, zon_fs: float | None, verbruik_ha: float | None, nu: datetime | None = None):
        """Eén keer per avond (na 21:00) vastleggen wat we voor morgen verwachten."""
        nu = _lokaal(nu)
        d = vp.get("morgen")
        if nu.hour < 21 or not d or d in self.s["snapshots"] or vp.get("zon_morgen") is None:
            return False
        self.s["snapshots"][d] = {"gemaakt": nu.isoformat(timespec="minutes"),
                                  "zon_ruw": vp.get("zon_morgen_ruw"), "zon": vp.get("zon_morgen"),
                                  "zon_fs": zon_fs, "verbruik": vp.get("verbruik_morgen"),
                                  "verbruik_ha": verbruik_ha, "tgem": vp.get("tgem_morgen")}
        oud = sorted(self.s["snapshots"])[:-60]
        for k in oud:
            self.s["snapshots"].pop(k, None)
        self._bewaar()
        log.info("zelflerend: snapshot voor %s vastgelegd: %s", d, self.s["snapshots"][d])
        return True
