"""Modelltraining und Evaluation.

Der Text wird per TF-IDF numerisch codiert. Als Mehrklassen-Klassifikatoren
(Bewertung 1 bis 5) kommen Multinomiale Naive Bayes und Logistische Regression
zum Einsatz. Ausgewertet werden Genauigkeit (Accuracy) und Konfusionsmatrix.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from config import RANDOM_STATE, RESULTS_DIR, TEST_SIZE


def split_data(df):
    """Teilt in Trainings-/Testteil (stratifiziert nach Bewertung).

    Die Spalten werden bewusst in einfache Python-Listen bzw. NumPy-Arrays
    umgewandelt. Das vermeidet Probleme mit Arrow-basierten pandas-Datentypen
    (neuere pandas-Versionen), mit denen scikit-learn beim Indizieren nicht
    zurechtkommt.
    """
    X = df["clean_text"].astype(str).tolist()
    y = df["rating"].astype(int).to_numpy()
    cat = df["category"].astype(str).tolist()
    return train_test_split(
        X, y, cat, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )


def vectorize(X_train, X_test):
    """Codiert den Text numerisch per TF-IDF (uni- und bigramme)."""
    vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=5)
    return vec.fit_transform(X_train), vec.transform(X_test), vec


def train_and_evaluate(Xtr, ytr, Xte, yte, name, model):
    """Trainiert ein Modell und berechnet Accuracy + Konfusionsmatrix."""
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    acc = accuracy_score(yte, pred)
    print(f"[model] {name}: Accuracy = {acc:.4f}")
    cm = confusion_matrix(yte, pred, labels=[1, 2, 3, 4, 5])
    _plot_confusion(cm, name)
    report = classification_report(yte, pred, digits=3, zero_division=0)
    (RESULTS_DIR / f"report_{name}.txt").write_text(report)
    return acc, pred


def _plot_confusion(cm, name):
    """Speichert die Konfusionsmatrix als Abbildung."""
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(5)); ax.set_xticklabels(range(1, 6))
    ax.set_yticks(range(5)); ax.set_yticklabels(range(1, 6))
    ax.set_xlabel("Vorhergesagte Bewertung"); ax.set_ylabel("Wahre Bewertung")
    ax.set_title(f"Konfusionsmatrix: {name}")
    for i in range(5):
        for j in range(5):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"confusion_{name}.png", dpi=150)
    plt.close(fig)


def per_category_accuracy(yte, pred, cat_te):
    """Berechnet die Accuracy getrennt nach Produktkategorie."""
    dfc = pd.DataFrame({"y": yte, "pred": pred, "cat": cat_te})
    rows = []
    for c, g in dfc.groupby("cat"):
        rows.append({"Kategorie": c, "Accuracy": accuracy_score(g["y"], g["pred"]),
                     "n_Test": len(g)})
    return pd.DataFrame(rows)


def to_polarity(y):
    """Bildet die 5 Klassen auf 3 Polaritaeten ab (negativ/neutral/positiv)."""
    mapping = {1: "negativ", 2: "negativ", 3: "neutral", 4: "positiv", 5: "positiv"}
    return np.array([mapping[int(v)] for v in y])
