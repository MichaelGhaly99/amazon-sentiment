"""End-to-End-Pipeline der Sentimentanalyse.

Ausfuehren mit:  python src/main.py
Erzeugt im Ordner results/ alle Tabellen (CSV) und Abbildungen (PNG),
die fuer den Projektbericht benoetigt werden.
"""
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.naive_bayes import MultinomialNB

from config import RESULTS_DIR
from data import load_reviews
from model import (per_category_accuracy, split_data, to_polarity,
                   train_and_evaluate, vectorize)
from preprocess import preprocess_frame


def main():
    # 1) Daten sammeln
    df = load_reviews()

    # 2) Vorverarbeitung
    df = preprocess_frame(df)

    # 3) Datenbeschreibung (Tabellen fuer den Bericht)
    describe_data(df)

    # 4) Aufteilen in Trainings-/Testteil und in Ein-/Ausgabe
    Xtr, Xte, ytr, yte, cat_tr, cat_te = split_data(df)
    Xtr_v, Xte_v, _ = vectorize(Xtr, Xte)

    # 5) Zwei Modelle trainieren und bewerten (Mehrklassen, 1-5 Sterne)
    results = {}
    acc_nb, pred_nb = train_and_evaluate(
        Xtr_v, ytr, Xte_v, yte, "NaiveBayes", MultinomialNB())
    acc_lr, pred_lr = train_and_evaluate(
        Xtr_v, ytr, Xte_v, yte, "LogReg",
        LogisticRegression(max_iter=1000, C=1.0))
    results["NaiveBayes"] = acc_nb
    results["LogReg"] = acc_lr

    # 6) Accuracy je Kategorie (fuer das bessere Modell)
    best_pred = pred_lr if acc_lr >= acc_nb else pred_nb
    best_name = "LogReg" if acc_lr >= acc_nb else "NaiveBayes"
    per_cat = per_category_accuracy(yte, best_pred, cat_te)
    per_cat.to_csv(RESULTS_DIR / "accuracy_per_category.csv", index=False)
    print("\n[main] Accuracy je Kategorie (" + best_name + "):")
    print(per_cat.to_string(index=False))

    # 7) Optionale Zusatzanalyse: Sentimentpolaritaet (3 Klassen)
    ytr_p, yte_p = to_polarity(ytr), to_polarity(yte)
    pol_model = LogisticRegression(max_iter=1000)
    pol_model.fit(Xtr_v, ytr_p)
    acc_pol = accuracy_score(yte_p, pol_model.predict(Xte_v))
    print(f"\n[main] Polaritaet (negativ/neutral/positiv) Accuracy = {acc_pol:.4f}")

    # 8) Modellvergleich speichern
    pd.DataFrame(
        [{"Modell": k, "Accuracy": v} for k, v in results.items()]
        + [{"Modell": "LogReg (Polaritaet)", "Accuracy": acc_pol}]
    ).to_csv(RESULTS_DIR / "model_comparison.csv", index=False)

    print(f"\n[main] Fertig. Alle Ergebnisse liegen in: {RESULTS_DIR}")


def describe_data(df):
    """Erzeugt die im Bericht geforderten Datentabellen."""
    # Verteilung der Bewertungen je Kategorie
    dist = (df.groupby(["category", "rating"]).size()
            .unstack(fill_value=0))
    dist["gesamt"] = dist.sum(axis=1)
    dist.to_csv(RESULTS_DIR / "rating_distribution.csv")
    print("\n[data] Verteilung der Bewertungen je Kategorie:")
    print(dist.to_string())

    overview = (df.groupby("category")
                .agg(Anzahl=("rating", "size"),
                     Durchschnitt=("rating", "mean"),
                     mittl_Textlaenge=("clean_text", lambda s: s.str.split().map(len).mean()))
                .round(2))
    overview.to_csv(RESULTS_DIR / "category_overview.csv")
    print("\n[data] Kategorie-Ueberblick:")
    print(overview.to_string())


if __name__ == "__main__":
    main()
