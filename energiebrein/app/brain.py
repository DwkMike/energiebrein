"""Het Energiebrein: leren, plannen en publiceren (v0.1 = schaduwmodus, stuurt niets aan)."""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import backtest
import cop as copmod
import data
import forecast
import house as housemod
from optimizer import Invoer, plan
from prices import forecast_uit_state, prijzen_voor, terugleverprijs, is_saldering
from schema import comfortgrenzen, tapwatervraag, boiler_energie

log = logging.getLogger("energiebrein")
VERSIE = "0.1.4"
UITVOER = os.environ.get("EB_UITVOER", "/homeassistant/energiebrein")


def _f(state: dict | None, default=None):
    try:
        return float(state["state"])
    except Exception:
        return default


class Brein:
    def __init__(self, ha, opts: dict):
        self.ha = ha
        self.o = opts
        self.hist = pd.DataFrame()
        self.house = housemod.default_model()
        self.cop = copmod.CopModel(cop_dhw=opts["wp_cop_tapwater"])
        self.pv = forecast.PvModel(opts["pv_kwp"], opts["pv_ac_max_kw"])
        self.load = forecast.LoadModel()
        self.weer = None
        self.bt = {}
        self.lat = self.lon = None
        self.laatste_training = None
        self.fouten: list[str] = []
        os.makedirs(UITVOER, exist_ok=True)

    # ------------------------------------------------------------------ leren
    def train(self):
        self.fouten = []
        cfg = self.ha.config()
        self.lat, self.lon = cfg.get("latitude"), cfg.get("longitude")
        self.hist = data.ophalen(self.ha, self.o)
        if len(self.hist) == 0:
            self.fouten.append("geen historische data")
            return
        m = housemod.fit(self.hist)
        if m is not None:
            self.house = m
        else:
            self.fouten.append("huismodel: standaardwaarden (te weinig stookdata)")
        self.cop = copmod.fit(self.hist)
        try:
            self.weer = forecast.weer_ophalen(self.lat, self.lon, self.o["pv_helling"], self.o["pv_azimut"])
            self.pv.fit(self.hist, self.weer)
        except Exception as e:
            self.fouten.append(f"weer/zon: {e}")
            log.exception("weer ophalen mislukt")
        self.load.fit(self.hist)
        try:
            self.bt = backtest.run(self.hist, self.house, self.cop, self.o)
        except Exception as e:
            self.fouten.append(f"terugrekentest: {e}")
            log.exception("backtest mislukt")
        self.laatste_training = datetime.now(timezone.utc)
        self._rapport()

    # ------------------------------------------------------------------ plannen
    def plan_nu(self) -> dict | None:
        o = self.o
        dt = 0.25
        N = int(o["horizon_uur"] / dt)
        t0 = pd.Timestamp.now(tz="UTC").floor("15min")
        idx = pd.date_range(t0, periods=N, freq="15min")
        try:
            self.weer = forecast.weer_ophalen(self.lat, self.lon, o["pv_helling"], o["pv_azimut"],
                                              past_days=1, forecast_days=3)
        except Exception as e:
            log.warning("weer verversen mislukt: %s", e)
        w = None
        if self.weer is not None:
            w = self.weer.reindex(self.weer.index.union(idx)).interpolate(method="time").reindex(idx)
        To = w["To_fc"].to_numpy() if w is not None else np.full(N, _f(self.ha.state(o["ent_buiten_temp"]), 5.0))
        To = np.nan_to_num(To, nan=float(np.nanmean(To)) if np.isfinite(np.nanmean(To)) else 5.0)
        pv = self.pv.predict(w) if w is not None else np.zeros(N)
        hist_base = self.hist["base"] if "base" in self.hist else pd.Series(dtype=float)
        base50, base80 = self.load.predict(idx, To, hist_base)
        fc = forecast_uit_state(self.ha.state(o["ent_prijs"]))
        hist_p = self.hist["price"] if "price" in self.hist else None
        p_buy, bekend = prijzen_voor(idx, fc, hist_p)
        p_sell = terugleverprijs(idx, p_buy, o)
        low, high, in_blok = comfortgrenzen(idx, o)
        soc = _f(self.ha.state(o["ent_soc"]), 50.0)
        Ti = _f(self.ha.state(o["ent_kamer_temp"]), float(low[0]))
        tank = _f(self.ha.state(o["ent_boiler_temp"]), None)
        inv = Invoer(dt=dt, p_buy=p_buy, p_sell=p_sell, base=base50, pv=pv, To=To, Tlow=low, Thigh=high,
                     dhw_vraag=tapwatervraag(idx, o), soc0=soc / 100 * o["accu_capaciteit_kwh"], Ti0=Ti,
                     Te0=self._te_nu(Ti, To[0]),
                     eb0=boiler_energie(tank, o))
        accu_export = (not is_saldering(t0, o)) or bool(o.get("accu_verkoop_toestaan_2026"))
        r = plan(inv, self.house, self.cop, o, accu_export_toegestaan=accu_export)
        if r is None:
            self.ha.set_state("sensor.energiebrein_status", "geen oplossing",
                              {"friendly_name": "Energiebrein status", "versie": VERSIE})
            return None
        self._publiceer(idx, r, inv, bekend, base80)
        return r

    def _te_nu(self, Ti: float, To0: float) -> float:
        try:
            tail = self.hist.iloc[-96:]
            te = self.house.te_na(tail)
            if np.isfinite(te):
                return te
        except Exception as e:
            log.debug("Te-schatting mislukt: %s", e)
        return float(housemod.te_evenwicht(self.house.params, Ti, To0))

    # ------------------------------------------------------------------ publiceren
    def _publiceer(self, idx, r, inv, bekend, base80):
        lok = idx.tz_convert("Europe/Amsterdam")
        accu = r["ch"] - r["dis"]
        stappen = [{
            "t": lok[i].strftime("%d-%m %H:%M"),
            "prijs": round(float(inv.p_buy[i]), 3),
            "accu_kw": round(float(accu[i]), 2),
            "soc": round(float(r["soc"][i] / self.o["accu_capaciteit_kwh"] * 100), 1),
            "wp_kw_th": round(float(r["q"][i]), 2),
            "aanvoer": round(float(r["aanvoer"][i]), 1),
            "kamer": round(float(r["Ti"][i]), 2),
            "elwa_kw": round(float(r["elwa"][i]), 2),
            "tapwater_kw_th": round(float(r["hdhw"][i]), 2),
            "net_kw": round(float(r["imp"][i] - r["exp"][i]), 2),
            "zon_kw": round(float(inv.pv[i]), 2),
        } for i in range(len(idx))]
        nu = stappen[0]
        gem = lambda a: round(float(np.mean(a)), 3)  # noqa: E731
        self.ha.set_state("sensor.energiebrein_status", "schaduw", {
            "friendly_name": "Energiebrein status", "icon": "mdi:brain", "versie": VERSIE,
            "laatste_plan": datetime.now(timezone.utc).isoformat(),
            "laatste_training": self.laatste_training.isoformat() if self.laatste_training else None,
            "prijzen_bekend_uur": round(float(bekend.sum()) * 0.25, 1),
            "meldingen": self.fouten,
        })
        self.ha.set_state("sensor.energiebrein_accu_advies", round(nu["accu_kw"] * 1000), {
            "friendly_name": "Energiebrein accu-advies", "unit_of_measurement": "W",
            "device_class": "power", "state_class": "measurement", "icon": "mdi:battery-sync",
            "toelichting": "+ = laden, - = ontladen", "plan": [{"t": s["t"], "kw": s["accu_kw"], "soc": s["soc"]}
                                                                 for s in stappen]})
        self.ha.set_state("sensor.energiebrein_aanvoer_advies", nu["aanvoer"] if nu["wp_kw_th"] > 0.2 else 0, {
            "friendly_name": "Energiebrein aanvoer-advies", "unit_of_measurement": "°C",
            "device_class": "temperature", "state_class": "measurement",
            "wp_thermisch_kw": nu["wp_kw_th"],
            "plan": [{"t": s["t"], "kw": s["wp_kw_th"], "aanvoer": s["aanvoer"], "kamer": s["kamer"]} for s in stappen]})
        self.ha.set_state("sensor.energiebrein_kamer_plan", nu["kamer"], {
            "friendly_name": "Energiebrein geplande kamertemperatuur", "unit_of_measurement": "°C",
            "device_class": "temperature", "state_class": "measurement"})
        self.ha.set_state("sensor.energiebrein_elwa_advies", "aan" if nu["elwa_kw"] > 0.2 else "uit", {
            "friendly_name": "Energiebrein ELWA-advies", "icon": "mdi:water-boiler",
            "vermogen_kw": nu["elwa_kw"],
            "plan": [{"t": s["t"], "kw": s["elwa_kw"], "wp_tapwater": s["tapwater_kw_th"]} for s in stappen]})
        self.ha.set_state("sensor.energiebrein_kosten_horizon", round(r["energiekosten"], 2), {
            "friendly_name": f"Energiebrein verwachte kosten {self.o['horizon_uur']} uur",
            "unit_of_measurement": "EUR", "icon": "mdi:cash",
            "netafname_kwh": round(float(np.sum(r["imp"]) * 0.25), 1),
            "teruglevering_kwh": round(float(np.sum(r["exp"]) * 0.25), 1),
            "wp_kwh": round(float(np.sum(r["p_hp"]) * 0.25), 1),
            "gem_cop": gem(r["cop"]),
            "comfort_tekort_kh": round(float(np.sum(r["slo"]) * 0.25), 2)})
        hm = self.house.to_dict()
        self.ha.set_state("sensor.energiebrein_huismodel", hm.get("ua_w_per_k"), {
            "friendly_name": "Energiebrein huismodel warmteverlies", "unit_of_measurement": "W/K",
            "icon": "mdi:home-thermometer", **hm})
        cm = self.cop.to_dict()
        self.ha.set_state("sensor.energiebrein_cop_model", round(float(self.cop.cop(7, 35)), 2), {
            "friendly_name": "Energiebrein COP bij 7/35", "icon": "mdi:heat-pump",
            "cop_bij_2_45": round(float(self.cop.cop(2, 45)), 2),
            "cop_bij_-5_50": round(float(self.cop.cop(-5, 50)), 2), **cm})
        if self.bt:
            self.ha.set_state("sensor.energiebrein_terugrekentest", self.bt.get("besparing_per_dag_eur", 0), {
                "friendly_name": "Energiebrein terugrekentest besparing per dag", "unit_of_measurement": "EUR",
                "icon": "mdi:chart-line", **{k: v for k, v in self.bt.items() if k != "dagen"},
                "dagen": self.bt.get("dagen", [])[-14:]})
        try:
            with open(os.path.join(UITVOER, "plan.json"), "w") as f:
                json.dump({"aangemaakt": datetime.now(timezone.utc).isoformat(), "stappen": stappen}, f)
        except Exception as e:
            log.warning("plan.json schrijven mislukt: %s", e)

    def _rapport(self):
        L = [f"# Energiebrein rapport (v{VERSIE})", "",
             f"Getraind: {datetime.now(timezone.utc).astimezone().strftime('%d-%m-%Y %H:%M')}",
             f"Historie: {len(self.hist)} uren", ""]
        L += ["## Huismodel", "```", json.dumps(self.house.to_dict(), indent=1), "```", ""]
        L += ["## Warmtepomp (COP)", "```", json.dumps(self.cop.to_dict(), indent=1), "```",
              f"COP 7/35: {self.cop.cop(7, 35):.2f} | 2/45: {self.cop.cop(2, 45):.2f} | -5/50: {self.cop.cop(-5, 50):.2f}",
              f"Methode: {self.cop.methode}, {self.cop.n_dagen} dagen, gemiddelde afwijking dagverbruik {self.cop.dagfout_pct:.1f}%", ""]
        L += ["## Voorspellers",
              f"- Zon: {'LightGBM' if self.pv.model is not None else 'fysisch'}; MAE {self.pv.mae:.3f} kW",
              f"- Basisverbruik: MAE {self.load.mae:.3f} kW (naief 7-dagenprofiel {self.load.mae_naief:.3f} kW)", ""]
        if self.bt:
            L += ["## Terugrekentest", self.bt.get("melding", ""), "",
                  f"Dagen: {self.bt.get('aantal_dagen', 0)}, werkelijk {self.bt.get('echt_totaal_eur')} EUR, "
                  f"besparing {self.bt.get('besparing_totaal_eur')} EUR ({self.bt.get('besparing_per_dag_eur')} EUR/dag)", "",
                  f"Modelafwijking t.o.v. werkelijke kosten: {self.bt.get('modelafwijking_pct')}%", "",
                  "| datum | werkelijk | model huidig | brein | besparing | WP kWh echt | WP kWh model | WP kWh brein | comfort-tekort Kh |",
                  "|---|---|---|---|---|---|---|---|---|"]
            for x in self.bt.get("dagen", []):
                L.append(f"| {x['datum']} | {x['echt_eur']} | {x['model_huidig_eur']} | {x['brein_eur']} | "
                         f"{x['besparing_eur']} | {x['wp_kwh_echt']} | {x['wp_kwh_model_huidig']} | "
                         f"{x['wp_kwh_brein']} | {x['comfort_tekort_kh']} |")
        if self.fouten:
            L += ["", "## Meldingen"] + [f"- {f}" for f in self.fouten]
        try:
            with open(os.path.join(UITVOER, "rapport.md"), "w") as f:
                f.write("\n".join(L))
        except Exception as e:
            log.warning("rapport schrijven mislukt: %s", e)
