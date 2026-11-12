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
    """Uurhistorie -> per lokale dag (kWh): zon, verbruik excl. ELWA, warmtepomp, airco, basis; gem. buitentemp."""
    need = {"pv", "house", "elwa", "To"}
    if hist is None or len(hist) == 0 or not need <= set(hist.columns):
        return pd.DataFrame()
    d = hist[[c for c in ["pv", "house", "elwa", "To", "p_hp", "airco"] if c in hist.columns]].copy()
    for c in ("p_hp", "airco"):
        if c not in d:
            d[c] = 0.0
    d["datum"] = d.index.tz_convert(TZ).date
    g = d.groupby("datum")
    n = g["house"].count()
    f = 24 / n.clip(lower=1)
    out = pd.DataFrame({
        "zon": g["pv"].sum() * f,
        "verbruik": (g["house"].sum() - g["elwa"].sum()) * f,
        "wp": g["p_hp"].sum() * f,
        "airco": g["airco"].sum() * f,
        "To": g["To"].mean(),
        "n": n,
    })
    out = out[out["n"] >= 22].copy()   # alleen (bijna) volledige dagen; zomer/wintertijd = 23/25 uur
    out["basis"] = (out["verbruik"] - out["wp"].fillna(0) - out["airco"].fillna(0)).clip(lower=0)
    out["weekend"] = [1 if pd.Timestamp(x).dayofweek >= 5 else 0 for x in out.index]
    # afwezig (vakantie): basis < 50 % van de mediaan -> niet meenemen in het leren
    med = out["basis"].median() if len(out) else 0
    out["afwezig"] = out["basis"] < 0.5 * med
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
    """Verbruik morgen = basis (recent, per weekdag/weekend) + airco (recent) + warmtepomp (graaddagen).

    De warmtepomp wordt alleen op stookdagen (WP > 5 kWh) gefit, zodat de zomer (alleen tapwater) en de
    start van het stookseizoen het model niet vertekenen. Vakantiedagen tellen niet mee.
    Een een-dag-vooruit terugtest vergelijkt dit met de HA-methode (EMA 0,25); de beste wint ('methode').
    """

    def __init__(self):
        self.basis_werk, self.basis_weekend, self.airco, self.wp_vast = 15.0, 15.0, 0.0, 0.0
        self.c_wp, self.tbasis, self.stookseizoen = 0.0, 16.0, False
        self.ema = 35.0
        self.n = 0
        self.mae_model = float("nan")
        self.mae_ema = float("nan")
        self.methode = "standaard"

    @staticmethod
    def _onderdelen(dd: pd.DataFrame, ref: date) -> dict:
        r = {}
        tw = dd[~dd["afwezig"]].tail(28)
        w = _gewichten(tw.index, ref) if len(tw) else np.array([])
        def gem(mask, kol, std):
            x = tw[mask]
            return float(np.average(x[kol], weights=w[mask.to_numpy()])) if len(x) else std
        alle = float(np.average(tw["basis"], weights=w)) if len(tw) else 15.0
        r["basis_werk"] = gem(tw["weekend"] == 0, "basis", alle)
        r["basis_weekend"] = gem(tw["weekend"] == 1, "basis", alle)
        r["airco"] = float(tw["airco"].tail(7).mean()) if len(tw) else 0.0
        stook = dd[dd["wp"] > 5].tail(VENSTER_DAGEN)
        r["stookseizoen"] = bool(len(dd) >= 3 and (dd["wp"].tail(5) > 5).sum() >= 3)
        r["wp_vast"] = float(dd["wp"].tail(7).mean()) if len(dd) else 0.0
        r["c_wp"], r["tbasis"] = 0.0, 16.0
        if len(stook) >= 4:
            ww = _gewichten(stook.index, ref)
            best = None
            for tb in (14.0, 15.0, 16.0, 17.0, 18.0):
                x = np.clip(tb - stook["To"].to_numpy(), 0, None)
                c = float(np.sum(ww * x * stook["wp"]) / max(np.sum(ww * x * x), 1e-6))
                res = float(np.sum(ww * (stook["wp"].to_numpy() - c * x) ** 2))
                if best is None or res < best[0]:
                    best = (res, tb, c)
            r["tbasis"], r["c_wp"] = best[1], max(best[2], 0.0)
        return r

    @staticmethod
    def _voorspel(r: dict, tgem: float, weekend: int) -> float:
        b = r["basis_weekend"] if weekend else r["basis_werk"]
        wp = r["c_wp"] * max(0.0, r["tbasis"] - tgem) if (r["stookseizoen"] and r["c_wp"] > 0) else r["wp_vast"]
        return max(0.0, b + r["airco"] + wp)

    def fit(self, dag: pd.DataFrame):
        dd = dag.dropna(subset=["verbruik", "To"]).tail(VENSTER_DAGEN + 30)
        self.n = len(dd)
        if self.n:
            ema = float(dd["verbruik"].iloc[:7].mean())
            for v in dd["verbruik"]:
                ema = 0.75 * ema + 0.25 * float(v)
            self.ema = ema
        if self.n < 10:
            self.methode = "HA-methode (te weinig dagen)"
            return self
        r = self._onderdelen(dd, dd.index[-1])
        for k, v in r.items():
            setattr(self, k, v)
        self._terugtest(dd)
        self.methode = "onderdelen (basis+airco+WP graaddagen)" if not (self.mae_model > self.mae_ema) \
            else "HA-methode (EMA) - onderdelenmodel scoorde slechter"
        return self

    def _terugtest(self, dd: pd.DataFrame, dagen: int = 21):
        e_m, e_e = [], []
        ema = float(dd["verbruik"].iloc[:7].mean())
        start = max(14, len(dd) - dagen)
        for i in range(len(dd)):
            if i >= start and not dd["afwezig"].iloc[i]:
                tr = dd.iloc[:i]
                r = self._onderdelen(tr, tr.index[-1])
                x = dd.iloc[i]
                e_m.append(abs(self._voorspel(r, x["To"], int(x["weekend"])) - x["verbruik"]))
                e_e.append(abs(ema - x["verbruik"]))
            ema = 0.75 * ema + 0.25 * float(dd["verbruik"].iloc[i])
        if e_m:
            self.mae_model, self.mae_ema = float(np.mean(e_m)), float(np.mean(e_e))

    def predict(self, tgem: float, weekend: int) -> float:
        if self.methode.startswith("HA-methode"):
            return self.ema
        return self._voorspel(self.__dict__, tgem, weekend)

    def to_dict(self):
        r = lambda v: round(v, 2) if isinstance(v, float) and np.isfinite(v) else v  # noqa: E731
        return {"methode": self.methode, "basis_werkdag_kwh": r(self.basis_werk),
                "basis_weekend_kwh": r(self.basis_weekend), "airco_kwh": r(self.airco),
                "wp_kwh_per_graaddag": r(self.c_wp), "graaddag_basis_c": self.tbasis,
                "stookseizoen": self.stookseizoen, "ha_methode_kwh": r(self.ema), "dagen": self.n,
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
