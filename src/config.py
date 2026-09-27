"""Zentrale Konfiguration des Projekts.

Hier werden die Produktkategorien, die Stichprobengroesse und die
Ausgabepfade festgelegt. Aenderungen fuer eigene Experimente nur hier
vornehmen.
"""
from pathlib import Path

# Drei Produktkategorien aus dem Amazon-Reviews-2023-Datensatz (McAuley Lab).
# Gueltige Namen entsprechen den HuggingFace-Konfigurationen
# "raw_review_<KATEGORIE>". Beispiele: All_Beauty, Automotive, Pet_Supplies,
# Video_Games, Office_Products, Toys_and_Games, ...
CATEGORIES = ["Automotive", "Pet_Supplies", "Video_Games"]

# Anzahl der Rezensionen, die pro Kategorie hoechstens verwendet werden.
# Kleiner waehlen, wenn Rechenzeit/Arbeitsspeicher begrenzt sind.
SAMPLES_PER_CATEGORY = 6000

# Anteil der Testdaten am Gesamtdatensatz.
TEST_SIZE = 0.2

# Fester Zufalls-Seed fuer reproduzierbare Ergebnisse.
RANDOM_STATE = 42

# Ausgabeordner fuer Tabellen und Abbildungen.
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
