"""Toegang tot Home Assistant vanuit de App via de Supervisor-proxy.

REST:      http://supervisor/core/api/...
WebSocket: ws://supervisor/core/websocket
Auth:      Bearer $SUPERVISOR_TOKEN (vereist homeassistant_api: true in config.yaml)
"""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone

import requests
import websocket

log = logging.getLogger("energiebrein.ha")

REST = os.environ.get("EB_HA_REST", "http://supervisor/core/api")
WS = os.environ.get("EB_HA_WS", "ws://supervisor/core/websocket")


class HA:
    def __init__(self, token: str | None = None):
        self.token = token or os.environ.get("SUPERVISOR_TOKEN", "")
        self.s = requests.Session()
        self.s.headers.update({"Authorization": f"Bearer {self.token}",
                               "Content-Type": "application/json"})

    # ---------- REST ----------
    def config(self) -> dict:
        r = self.s.get(f"{REST}/config", timeout=30)
        r.raise_for_status()
        return r.json()

    def state(self, entity_id: str) -> dict | None:
        r = self.s.get(f"{REST}/states/{entity_id}", timeout=30)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.json()

    def set_state(self, entity_id: str, state, attributes: dict | None = None) -> None:
        body = {"state": str(state), "attributes": attributes or {}}
        try:
            r = self.s.post(f"{REST}/states/{entity_id}", data=json.dumps(body, default=str), timeout=30)
            r.raise_for_status()
        except Exception as e:  # publiceren mag de planner nooit laten crashen
            log.warning("set_state %s mislukt: %s", entity_id, e)

    # ---------- WebSocket ----------
    def _ws_call(self, payload: dict, timeout: int = 120) -> dict:
        ws = websocket.create_connection(WS, timeout=timeout)
        try:
            hello = json.loads(ws.recv())
            if hello.get("type") != "auth_required":
                raise RuntimeError(f"onverwacht websocket-antwoord: {hello}")
            ws.send(json.dumps({"type": "auth", "access_token": self.token}))
            auth = json.loads(ws.recv())
            if auth.get("type") != "auth_ok":
                raise RuntimeError(f"websocket-auth mislukt: {auth}")
            payload = dict(payload, id=1)
            ws.send(json.dumps(payload))
            while True:
                msg = json.loads(ws.recv())
                if msg.get("id") == 1:
                    if not msg.get("success", False):
                        raise RuntimeError(f"websocket-fout: {msg.get('error')}")
                    return msg.get("result")
        finally:
            ws.close()

    def statistics(self, ids: list[str], start: datetime, end: datetime | None = None,
                   period: str = "hour", types: list[str] | None = None) -> dict:
        """Langetermijnstatistieken. Geeft {statistic_id: [{start(ms), mean, ...}]}."""
        payload = {
            "type": "recorder/statistics_during_period",
            "start_time": start.astimezone(timezone.utc).isoformat(),
            "statistic_ids": ids,
            "period": period,
            "types": types or ["mean"],
        }
        if end is not None:
            payload["end_time"] = end.astimezone(timezone.utc).isoformat()
        return self._ws_call(payload) or {}


class TriggerLuisteraar:
    """Houdt één websocket open en wacht op een statuswijziging van een entiteit (event-driven).

    wacht(timeout) geeft True bij een wijziging en False bij time-out (vangnet).
    Bij een verbroken verbinding wordt automatisch opnieuw verbonden.
    """

    def __init__(self, ha: "HA", entity_id: str):
        self.ha = ha
        self.entity_id = entity_id
        self.ws = None

    def _verbind(self):
        ws = websocket.create_connection(WS, timeout=30)
        json.loads(ws.recv())
        ws.send(json.dumps({"type": "auth", "access_token": self.ha.token}))
        auth = json.loads(ws.recv())
        if auth.get("type") != "auth_ok":
            raise RuntimeError(f"websocket-auth mislukt: {auth}")
        ws.send(json.dumps({"id": 1, "type": "subscribe_trigger",
                            "trigger": {"platform": "state", "entity_id": self.entity_id}}))
        self.ws = ws
        log.info("event-driven: luistert naar %s", self.entity_id)

    def wacht(self, timeout: float) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                if self.ws is None:
                    self._verbind()
                self.ws.settimeout(max(1.0, deadline - time.time()))
                msg = json.loads(self.ws.recv())
                if msg.get("type") == "event":
                    return True
            except websocket.WebSocketTimeoutException:
                return False
            except Exception as e:
                log.warning("websocket verbroken (%s), opnieuw verbinden over 30 s", e)
                try:
                    if self.ws:
                        self.ws.close()
                except Exception:
                    pass
                self.ws = None
                time.sleep(30)
        return False
