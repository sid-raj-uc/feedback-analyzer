"""
VADER Sentiment Analysis Module
================================
Applies VADER (Valence Aware Dictionary and sEntiment Reasoner) 
sentiment analysis to customer reviews. Supports both document-level
and sentence-level analysis for capturing mixed sentiment.

VADER is particularly effective for:
  - Social media / informal text
  - Handling emphasis (CAPS, punctuation)
  - Negation and intensifier awareness
  - Speed: rule-based, no GPU required
"""

import logging
from pathlib import Path
from typing import Optional

import yaml
import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize
from tqdm import tqdm

# ── Load Config ────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT_DIR / "config.yaml"

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """
    VADER-based sentiment analysis engine.

    Attributes
    ----------
    analyzer : SentimentIntensityAnalyzer
        Core VADER analyzer instance.
    pos_threshold : float
        Compound score threshold for positive classification.
    neg_threshold : float
        Compound score threshold for negative classification.
    """

    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
        self.pos_threshold = config["sentiment"]["compound_threshold_pos"]
        self.neg_threshold = config["sentiment"]["compound_threshold_neg"]
        self.sentence_level = config["sentiment"]["sentence_level"]
        logger.info(
            f"VADER Analyzer initialized — "
            f"thresholds: pos ≥ {self.pos_threshold}, neg ≤ {self.neg_threshold}"
        )

    def analyze_text(self, text: str) -> dict:
        """
        Compute VADER sentiment scores for a single text.

        Parameters
        ----------
        text : str
            Input text to analyze.

        Returns
        -------
        dict
            Keys: compound, pos, neg, neu, label, confidence
        """
        if not isinstance(text, str) or len(text.strip()) == 0:
            return {
                "compound": 0.0,
                "pos": 0.0,
                "neg": 0.0,
                "neu": 1.0,
                "label": "neutral",
                "confidence": 0.0,
            }

        scores = self.analyzer.polarity_scores(text)
        compound = scores["compound"]

        # Classify
        if compound >= self.pos_threshold:
            label = "positive"
        elif compound <= self.neg_threshold:
            label = "negative"
        else:
            label = "neutral"

        # Confidence = absolute distance from zero, normalized to [0, 1]
        confidence = min(abs(compound), 1.0)

        return {
            "compound": round(compound, 4),
            "pos": round(scores["pos"], 4),
            "neg": round(scores["neg"], 4),
            "neu": round(scores["neu"], 4),
            "label": label,
            "confidence": round(confidence, 4),
        }

    def analyze_sentence_level(self, text: str) -> dict:
        """
        Perform sentence-level sentiment analysis for mixed-sentiment detection.

        Returns document-level aggregates plus per-sentence breakdown.

        Parameters
        ----------
        text : str
            Input text.

        Returns
        -------
        dict
            Document-level scores + sentence_scores list + mixed_sentiment flag
        """
        if not isinstance(text, str) or len(text.strip()) == 0:
            result = self.analyze_text("")
            result["sentence_scores"] = []
            result["mixed_sentiment"] = False
            result["n_sentences"] = 0
            return result

        sentences = sent_tokenize(text)
        sentence_scores = []
        labels_seen = set()

        for s in sentences:
            s_result = self.analyze_text(s)
            sentence_scores.append({
                "sentence": s,
                **s_result,
            })
            labels_seen.add(s_result["label"])

        # Document-level is still the full-text VADER score
        doc_result = self.analyze_text(text)
        doc_result["sentence_scores"] = sentence_scores
        doc_result["n_sentences"] = len(sentences)

        # Mixed sentiment = both positive and negative sentences present
        doc_result["mixed_sentiment"] = (
            "positive" in labels_seen and "negative" in labels_seen
        )

        return doc_result

    def analyze_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str = "clean_text",
        output_path: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Apply sentiment analysis to every review in a DataFrame.

        Adds columns:
          - sentiment_compound
          - sentiment_pos
          - sentiment_neg
          - sentiment_neu
          - sentiment_label
          - sentiment_confidence
          - mixed_sentiment
          - n_sentences

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with a text column.
        text_column : str
            Column name containing cleaned text.
        output_path : str, optional
            Save results to CSV.

        Returns
        -------
        pd.DataFrame
        """
        logger.info(f"Running VADER sentiment analysis on {len(df):,} reviews …")

        results = []
        for text in tqdm(df[text_column], desc="VADER sentiment", unit="rev"):
            if self.sentence_level:
                result = self.analyze_sentence_level(text)
            else:
                result = self.analyze_text(text)
            results.append(result)

        # Flatten into DataFrame columns
        df["sentiment_compound"] = [r["compound"] for r in results]
        df["sentiment_pos"] = [r["pos"] for r in results]
        df["sentiment_neg"] = [r["neg"] for r in results]
        df["sentiment_neu"] = [r["neu"] for r in results]
        df["sentiment_label"] = [r["label"] for r in results]
        df["sentiment_confidence"] = [r["confidence"] for r in results]

        if self.sentence_level:
            df["mixed_sentiment"] = [r.get("mixed_sentiment", False) for r in results]
            df["n_sentences"] = [r.get("n_sentences", 0) for r in results]

        # Log summary statistics
        label_counts = df["sentiment_label"].value_counts()
        logger.info("Sentiment distribution:")
        for label, count in label_counts.items():
            pct = count / len(df) * 100
            logger.info(f"  {label:<10s} → {count:>5,} ({pct:.1f}%)")

        logger.info(f"  Mean compound score: {df['sentiment_compound'].mean():.4f}")
        logger.info(f"  Std compound score:  {df['sentiment_compound'].std():.4f}")

        if self.sentence_level:
            mixed_count = df["mixed_sentiment"].sum()
            logger.info(f"  Mixed-sentiment reviews: {mixed_count:,} ({mixed_count/len(df)*100:.1f}%)")

        # Compute agreement with star ratings (if available)
        if "star_rating" in df.columns:
            star_sentiment = df["star_rating"].apply(
                lambda x: "positive" if x >= 4 else ("negative" if x <= 2 else "neutral")
            )
            agreement = (df["sentiment_label"] == star_sentiment).mean()
            logger.info(f"  VADER vs Star-rating agreement: {agreement:.1%}")

        # Save
        if output_path:
            out = ROOT_DIR / output_path
            out.parent.mkdir(parents=True, exist_ok=True)
            save_df = df.drop(columns=["sentence_scores"], errors="ignore")
            save_df.to_csv(out, index=False)
            logger.info(f"  ✓ Saved sentiment results → {out}")

        return df


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(
        level=config["logging"]["level"],
        format=config["logging"]["format"],
    )
    processed_path = ROOT_DIR / config["data"]["processed_path"]
    if not processed_path.exists():
        logger.error(f"Processed data not found: {processed_path}. Run preprocessor.py first.")
        raise FileNotFoundError(processed_path)

    df = pd.read_csv(processed_path)
    sa = SentimentAnalyzer()
    df = sa.analyze_dataframe(
        df,
        text_column="clean_text",
        output_path=config["sentiment"]["output_path"],
    )
    print(f"\nSentiment analysis complete for {len(df):,} reviews.")
    print(df[["review_id", "clean_text", "sentiment_compound", "sentiment_label"]].head(10))
