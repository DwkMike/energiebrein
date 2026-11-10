# Energiebrein

Zelflerend energiebeheer voor Home Assistant: thuisaccu, warmtepomp (ruimte + tapwater) en
ELWA-boiler, gepland op dynamische stroomprijzen, zonvoorspelling en de warmtevraag van het huis.

## Hoe het werkt
| Onderdeel | Techniek |
|---|---|
| Huismodel | Grey-box 2R2C (lucht + gebouwschil), gefit op eigen meetdata met multiple shooting; schiltemperatuur via waarnemer |
| Warmtepomp | Carnot-COP-model met geleerde kwaliteitsfactor + radiatorvergelijking (EN 442, n=1,3) |
| Zon | Open-Meteo GTI op paneelvlak + LightGBM-correctie op eigen opbrengst |
| Basisverbruik | LightGBM (mediaan + 80%-kwantiel) op uur, weekdag, buitentemperatuur, 7-dagenprofiel |
| Planner | Model-predictieve regeling (MPC), lineair programma met HiGHS, 36 uur vooruit per kwartier |
| Tarieven | Saldering t/m 2026, vanaf 2027 marktprijs + vergoeding (+ zonnebonus), optie dubbele EB |

## Status
v0.1 = **schaduwmodus**: leert, plant en publiceert advies-sensoren (`sensor.energiebrein_*`)
plus een rapport in `/config/energiebrein/rapport.md`. Er wordt niets aangestuurd.

## Installatie
Instellingen → Apps → App store → ⋮ → Repositories → `https://github.com/DwkMike/energiebrein`,
daarna de App **Energiebrein** installeren en starten.

## Testen
`python3 tests/test_synthetic.py` — simuleert een huis met bekende eigenschappen en controleert of
het brein die terugvindt en zinnig plant.
