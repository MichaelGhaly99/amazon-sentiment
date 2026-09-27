"""Textvorverarbeitung fuer die Sentimentanalyse.

Umgesetzt werden die im Aufgabenblatt geforderten Schritte:
Kleinschreibung, Entfernen von Nicht-Buchstaben, Tokenisierung,
Entfernen von Stoppwoertern und Stemming (Porter-Stemmer).
Die numerische Codierung des Textes erfolgt spaeter per TF-IDF im Modell.
"""
import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize


def _ensure_nltk() -> None:
    """Laedt die benoetigten NLTK-Ressourcen (einmalig)."""
    for resource, path in [
        ("stopwords", "corpora/stopwords"),
        ("punkt", "tokenizers/punkt"),
        ("punkt_tab", "tokenizers/punkt_tab"),
    ]:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(resource, quiet=True)


_ensure_nltk()
_STOPWORDS = set(stopwords.words("english"))
_STEMMER = PorterStemmer()
_NON_LETTER = re.compile(r"[^a-z\s]")


def clean_text(text: str) -> str:
    """Wendet die komplette Vorverarbeitungskette auf einen Text an."""
    text = text.lower()
    text = _NON_LETTER.sub(" ", text)          # Zahlen/Satzzeichen entfernen
    tokens = word_tokenize(text)
    tokens = [
        _STEMMER.stem(tok)
        for tok in tokens
        if tok not in _STOPWORDS and len(tok) > 2  # kurze Tokens verwerfen
    ]
    return " ".join(tokens)


def preprocess_frame(df):
    """Fuegt eine Spalte mit bereinigtem Text hinzu und entfernt leere Zeilen."""
    df = df.copy()
    df["clean_text"] = df["text"].astype(str).map(clean_text)
    df = df[df["clean_text"].str.len() > 0].reset_index(drop=True)
    return df
