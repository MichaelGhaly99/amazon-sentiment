# Sentimentanalyse von Amazon-Produktrezensionen

Textklassifikationssystem, das Amazon-Produktrezensionen anhand von Titel und
Beschreibung eine numerische Bewertung von 1 bis 5 Sternen zuweist.
Projektbericht zum Kurs DLBAIPNLP01 (Projekt: NLP), IU Internationale Hochschule.

## Aufgabe

- Daten: Amazon Reviews 2023 (McAuley Lab), mindestens drei Produktkategorien.
- Vorverarbeitung: Tokenisierung, Stoppwortentfernung, Stemming, TF-IDF-Codierung.
- Modell: Mehrklassen-Klassifikator (1-5 Sterne), hier Naive Bayes und
  Logistische Regression im Vergleich.
- Auswertung: Genauigkeit (Accuracy) und Konfusionsmatrix, zusaetzlich
  Accuracy je Kategorie und eine 3-Klassen-Polaritaetsanalyse.

## Projektstruktur

```
Projektbericht NLP/
├── requirements.txt        Python-Abhaengigkeiten
├── src/
│   ├── config.py           Kategorien, Stichprobengroesse, Pfade
│   ├── data.py             Datensammlung (Streaming von HuggingFace)
│   ├── preprocess.py       Textvorverarbeitung
│   ├── model.py            TF-IDF, Modelle, Evaluation, Plots
│   └── main.py             komplette Pipeline
└── results/                erzeugte Tabellen (CSV) und Abbildungen (PNG)
```

## Einrichtung und Ausfuehrung

1. Python 3.10 oder neuer installieren.
2. Virtuelle Umgebung anlegen und Abhaengigkeiten installieren:
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # macOS/Linux:
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. Pipeline starten:
   ```bash
   python src/main.py
   ```
   Beim ersten Lauf werden die NLTK-Ressourcen (stopwords, punkt) automatisch
   geladen und die Rezensionen per Streaming von HuggingFace geholt.

## Ergebnisse

Nach dem Lauf liegen im Ordner `results/`:

- `category_overview.csv` – Anzahl, Durchschnittsbewertung und mittlere
  Textlaenge je Kategorie
- `rating_distribution.csv` – Verteilung der Bewertungen je Kategorie
- `model_comparison.csv` – Accuracy der Modelle
- `accuracy_per_category.csv` – Accuracy je Produktkategorie
- `confusion_NaiveBayes.png`, `confusion_LogReg.png` – Konfusionsmatrizen
- `report_*.txt` – vollstaendige Klassifikationsberichte (Precision/Recall/F1)

## Konfiguration

Kategorien und Stichprobengroesse lassen sich in `src/config.py` anpassen.
Standard sind die Kategorien Automotive, Pet_Supplies und Video_Games mit
je bis zu 6000 Rezensionen.

## Datenquelle

McAuley Lab (2023): Amazon Reviews'23. https://amazon-reviews-2023.github.io/
