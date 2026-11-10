"""Hoofdlus: bij start leren + terugrekentest, daarna elk kwartier plannen, elke nacht hertrainen."""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime

from brain import Brein, VERSIE
from ha import HA

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("energiebrein")

OPTIONS = os.environ.get("EB_OPTIONS", "/data/options.json")
HERTRAIN_UUR = 4  # lokale tijd, na middernacht als de nieuwe dag compleet is


def main():
    with open(OPTIONS) as f:
        opts = json.load(f)
    log.info("Energiebrein v%s start (modus: %s)", VERSIE, opts.get("modus"))
    ha = HA()
    brein = Brein(ha, opts)
    while True:
        try:
            brein.train()
            break
        except Exception:
            log.exception("training mislukt, opnieuw over 5 minuten")
            time.sleep(300)
    laatste_dag = datetime.now().date()
    while True:
        start = time.time()
        try:
            brein.plan_nu()
        except Exception:
            log.exception("plannen mislukt")
        nu = datetime.now()
        if nu.date() != laatste_dag and nu.hour >= HERTRAIN_UUR:
            try:
                brein.train()
                laatste_dag = nu.date()
            except Exception:
                log.exception("nachtelijke training mislukt")
        # wacht tot het volgende kwartier (+20 s zodat nieuwe prijzen/standen er zijn)
        rest = 900 - (time.time() % 900) + 20
        time.sleep(max(rest, 30) if time.time() - start < 900 else 30)


if __name__ == "__main__":
    main()
