"""End-to-end test met een gesimuleerd huis: kan het brein bekende parameters terugvinden en zinnig plannen?

Draaien:  python3 tests/test_synthetic.py
"""
import json
import os
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "energiebrein", "app"))
os.environ["EB_UITVOER"] = tempfile.mkdtemp()

import house as housemod  # noqa: E402
import cop as copmod  # noqa: E402
import forecast  # noqa: E402
import backtest  # noqa: E402
import brain  # noqa: E402
from optimizer import Invoer, plan  # noqa: E402

rng = np.random.default_rng(1)
OPTS = json.load(open(os.path.join(os.path.dirname(__file__), "opts.json")))

WAAR = housemod.HouseModel(Ci=1.6, Ce=22.0, Rie=1.2, Rea=7.0, Ria=9.0, gs=0.2, gi=0.7)
WAAR_COP = copmod.CopModel(e0=0.42, e1=0.005, k_rad=11.0)


def maak_historie(dagen=60):
    idx = pd.date_range("2026-09-12", periods=24 * dagen, freq="h", tz="UTC")
    n = len(idx)
    uur = idx.tz_convert("Europe/Amsterdam").hour.to_numpy()
    dag = np.arange(n) / 24
    To = 12 - dag * 0.12 + 4 * np.sin(2 * np.pi * (uur - 9) / 24) + rng.normal(0, 1, n)
    pv = np.clip(np.sin(np.pi * (uur - 7) / 10), 0, None) * rng.uniform(0.2, 4.5, n)
    pv[(uur < 7) | (uur > 17)] = 0
    base = 0.35 + 0.4 * ((uur >= 17) & (uur <= 21)) + 0.2 * ((uur >= 7) & (uur <= 9)) + rng.gamma(2, 0.08, n)
    Ad, Bd = WAAR.discrete(1.0)
    x = np.array([20.0, 15.0])
    Ti, q = np.zeros(n), np.zeros(n)
    for t in range(n):
        Ti[t] = x[0] + rng.normal(0, 0.05)
        sp = 20.5 if (6 <= uur[t] < 8 or 16 <= uur[t] < 23) else 19.0
        qt = np.clip(2.5 * (sp - x[0]) + WAAR.warmteverlies_kw(sp, To[t]) * 0.8, 0, 7)
        q[t] = qt
        x = Ad @ x + Bd @ np.array([To[t], qt, pv[t], base[t]])
    Tflow = WAAR_COP.aanvoer(q, Ti)
    Tret = Tflow - q / 1.163
    p_hp = np.where(q > 0, q / WAAR_COP.cop(To, Tflow) * rng.normal(1, 0.04, n), 0)
    price = 0.30 + 0.12 * ((uur >= 17) & (uur <= 21)) - 0.08 * ((uur >= 11) & (uur <= 15)) + rng.normal(0, 0.02, n)
    house_l = base + p_hp
    soc = np.full(n, 50.0)
    grid = house_l - pv
    df = pd.DataFrame({"To": To, "Ti": Ti, "Tflow": Tflow, "Tret": Tret, "flow": 1.0, "p_hp_w": p_hp * 1000,
                       "pv_w": pv * 1000, "house_w": house_l * 1000, "grid_w": grid * 1000, "batt_w": 0.0,
                       "soc": soc, "price": price, "elwa_w": 0.0, "tank": 50.0}, index=idx)
    import data
    return data.afleiden(df)


def test_huismodel(h):
    m = housemod.fit(h)
    assert m is not None, "huismodel niet gefit"
    print("UA echt %.0f W/K, gefit %.0f W/K | rmse1h %.3f rmse24h %.3f" % (
        WAAR.ua_w_per_k, m.ua_w_per_k, m.rmse_1h, m.rmse_24h))
    assert abs(m.ua_w_per_k - WAAR.ua_w_per_k) / WAAR.ua_w_per_k < 0.2
    assert m.rmse_24h < 0.6
    return m


def test_cop(h):
    c = copmod.fit(h)
    for To, Tf in [(7, 35), (2, 45), (-5, 50)]:
        print("COP %d/%d echt %.2f gefit %.2f" % (To, Tf, WAAR_COP.cop(To, Tf), c.cop(To, Tf)))
        assert abs(c.cop(To, Tf) - WAAR_COP.cop(To, Tf)) < 0.3
    print("k_rad echt 11 gefit %.1f" % c.k_rad)
    return c


def test_planner(m, c):
    N = 144
    idx = pd.date_range("2026-11-11", periods=N, freq="15min", tz="UTC")
    uur = idx.tz_convert("Europe/Amsterdam").hour.to_numpy()
    p = 0.30 + 0.20 * ((uur >= 17) & (uur <= 21)) - 0.12 * ((uur >= 2) & (uur <= 5))
    from schema import comfortgrenzen, tapwatervraag
    low, high, _ = comfortgrenzen(idx, OPTS)
    inv = Invoer(dt=0.25, p_buy=p, p_sell=p, base=np.full(N, 0.4), pv=np.zeros(N), To=np.full(N, 3.0),
                 Tlow=low, Thigh=high, dhw_vraag=tapwatervraag(idx, OPTS), soc0=10.0, Ti0=20.0, Te0=15.0, eb0=3.0)
    r = plan(inv, m, c, OPTS, accu_export_toegestaan=False)
    assert r is not None
    goedkoop = (uur >= 2) & (uur <= 5)
    duur = (uur >= 17) & (uur <= 21)
    print("laden in goedkope uren: %.1f kWh, ontladen in dure uren: %.1f kWh, comfort-tekort %.2f Kh" % (
        r["ch"][goedkoop].sum() * 0.25, r["dis"][duur].sum() * 0.25, r["slo"].sum() * 0.25))
    print("WP-vermogen goedkoop %.2f kW vs duur %.2f kW (gemiddeld)" % (r["q"][goedkoop].mean(), r["q"][duur].mean()))
    assert r["ch"][goedkoop].sum() > 1.0
    assert r["dis"][duur].sum() > 1.0
    assert r["slo"].sum() * 0.25 < 1.0
    return r


class NepHA:
    def __init__(self, h):
        self.h = h
        self.states = {}

    def config(self):
        return {"latitude": 51.9, "longitude": 5.3}

    def state(self, eid):
        if eid == OPTS["ent_prijs"]:
            idx = pd.date_range(pd.Timestamp.now(tz="UTC").floor("15min"), periods=96, freq="15min")
            return {"state": "0.3", "attributes": {"forecast": [
                {"start_date": t.isoformat(), "price_tax_included": {"amount": int((0.3 + 0.1 * (t.hour % 5 == 0)) * 1e7)}}
                for t in idx]}}
        return {"state": {"sensor.alphaess_soc_battery": "40", "sensor.tado_woonkamer_temperatuur": "20.1",
                          "sensor.wp_tank_temp": "50", "sensor.wp_buiten_temp": "4"}.get(eid, "0")}

    def set_state(self, eid, state, attrs=None):
        self.states[eid] = (state, attrs)


def test_brein(h):
    import data
    ha = NepHA(h)
    data.ophalen = lambda ha_, o, dagen=400: h
    forecast.weer_ophalen = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("geen netwerk in test"))
    b = brain.Brein(ha, OPTS)
    b.train()
    r = b.plan_nu()
    assert r is not None
    print("gepubliceerd:", sorted(ha.states))
    print("terugrekentest:", {k: v for k, v in b.bt.items() if k != "dagen"})
    rap = open(os.path.join(os.environ["EB_UITVOER"], "rapport.md")).read()
    assert "Terugrekentest" in rap


if __name__ == "__main__":
    h = maak_historie()
    m = test_huismodel(h)
    c = test_cop(h)
    test_planner(m, c)
    test_brein(h)
    print("ALLES OK")
