"""
Pain Point Extraction Module
==============================
Surfaces actionable pain points from customer feedback by combining
sentiment analysis results with topic clusters. Identifies the most
impactful negative themes and produces structured, prioritized output.

Key capabilities:
  - Cross-references negative sentiment with topic clusters
  - Ranks pain points by frequency, severity, and trend
  - Extracts representative quotes for each pain point
  - Generates structured JSON for downstream dashboards
"""

import json
import logging
from pathlib import Path
from collections import Counter, defaultdict
from typing import Optional

import yaml
import numpy as np
import pandas as pd

# ── Load Config ────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT_DIR / "config.yaml"

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

logger = logging.getLogger(__name__)


class PainPointExtractor:
    """
    Extracts and ranks actionable pain points from sentiment-annotated,
    topic-clustered review data.

    Methodology
    -----------
    1. Filter reviews to negative sentiment (compound <= threshold)
    2. Group by topic cluster
    3. For each cluster:
       - Compute severity (mean negative compound score)
       - Compute frequency (count of negative reviews)
       - Extract top keywords from the cluster
       - Select representative quotes
    4. Rank by composite score = frequency_weight * freq + severity_weight * severity
    """

    def __init__(
        self,
        freq_weight: float = 0.6,
        severity_weight: float = 0.4,
    ):
        self.neg_threshold = config["sentiment"]["compound_threshold_neg"]
        self.top_n = config["pain_points"]["top_n_pain_points"]
        self.freq_weight = freq_weight
        self.severity_weight = severity_weight
        logger.info("PainPointExtractor initialized.")

    def extract(
        self,
        df: pd.DataFrame,
        topic_summary: Optional[dict] = None,
        output_path: Optional[str] = None,
    ) -> list[dict]:
        """
        Extract pain points from the annotated DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain: sentiment_compound, sentiment_label, topic_id,
            clean_text, category, star_rating
        topic_summary : dict, optional
            Topic summary from BERTopic (topic_id → keywords).
        output_path : str, optional
            Save JSON output.

        Returns
        -------
        list[dict]
            Ranked list of pain points.
        """
        logger.info("Extracting pain points from negative feedback …")

        # 1. Filter to negative reviews (excluding outlier topics)
        neg_df = df[
            (df["sentiment_label"] == "negative")
            & (df["topic_id"] != -1)
        ].copy()

        logger.info(f"  Negative reviews with valid topics: {len(neg_df):,}")

        if len(neg_df) == 0:
            logger.warning("  No negative reviews found. Returning empty pain points.")
            return []

        # 2. Group by topic
        pain_points = []
        grouped = neg_df.groupby("topic_id")

        for topic_id, group in grouped:
            # Severity: mean absolute compound score (more negative = more severe)
            severity = abs(group["sentiment_compound"].mean())

            # Frequency
            frequency = len(group)

            # Category breakdown
            category_dist = group["category"].value_counts().to_dict()

            # Star rating stats
            avg_stars = group["star_rating"].mean()

            # Representative quotes (most negative, longest reviews)
            sorted_group = group.sort_values("sentiment_compound", ascending=True)
            representative_quotes = sorted_group["clean_text"].head(5).tolist()

            # Keywords from topic summary
            keywords = []
            if topic_summary and str(int(topic_id)) in topic_summary:
                kw_list = topic_summary[str(int(topic_id))].get("keywords", [])
                keywords = [kw["word"] for kw in kw_list[:8]]

            # Mixed sentiment count
            mixed_count = 0
            if "mixed_sentiment" in group.columns:
                mixed_count = int(group["mixed_sentiment"].sum())

            pain_points.append({
                "topic_id": int(topic_id),
                "keywords": keywords,
                "frequency": frequency,
                "severity": round(severity, 4),
                "avg_star_rating": round(avg_stars, 2),
                "category_distribution": category_dist,
                "mixed_sentiment_count": mixed_count,
                "representative_quotes": representative_quotes,
            })

        # 3. Normalize and compute composite score
        if pain_points:
            max_freq = max(pp["frequency"] for pp in pain_points)
            max_sev = max(pp["severity"] for pp in pain_points)

            for pp in pain_points:
                norm_freq = pp["frequency"] / max_freq if max_freq > 0 else 0
                norm_sev = pp["severity"] / max_sev if max_sev > 0 else 0
                pp["composite_score"] = round(
                    self.freq_weight * norm_freq + self.severity_weight * norm_sev,
                    4,
                )

            # 4. Sort by composite score (descending)
            pain_points.sort(key=lambda x: x["composite_score"], reverse=True)

            # Keep top N
            pain_points = pain_points[:self.top_n]

            # Add rank
            for i, pp in enumerate(pain_points, 1):
                pp["rank"] = i

        logger.info(f"  Top {len(pain_points)} pain points extracted.")
        for pp in pain_points:
            kw_str = ", ".join(pp["keywords"][:4]) if pp["keywords"] else "N/A"
            logger.info(
                f"    #{pp['rank']}  Score: {pp['composite_score']:.3f}  "
                f"Freq: {pp['frequency']:>4d}  Severity: {pp['severity']:.3f}  "
                f"Keywords: {kw_str}"
            )

        # ── Category-Level Pain Points ─────────────────────────────────────
        category_pain = self._extract_category_level(neg_df, topic_summary)

        # ── Build final output ─────────────────────────────────────────────
        output = {
            "summary": {
                "total_reviews": len(df),
                "negative_reviews": len(df[df["sentiment_label"] == "negative"]),
                "negative_with_topics": len(neg_df),
                "topics_with_negative": len(pain_points),
                "negative_rate": round(
                    len(df[df["sentiment_label"] == "negative"]) / len(df) * 100, 2
                ),
            },
            "pain_points": pain_points,
            "category_insights": category_pain,
        }

        # Save
        if output_path:
            out = ROOT_DIR / output_path
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "w") as f:
                json.dump(output, f, indent=2, default=str)
            logger.info(f"  ✓ Pain points saved → {out}")

        return output

    def _extract_category_level(
        self,
        neg_df: pd.DataFrame,
        topic_summary: Optional[dict],
    ) -> dict:
        """
        Extract per-category pain point summaries.
        """
        category_insights = {}

        for category, cat_group in neg_df.groupby("category"):
            top_topics = cat_group["topic_id"].value_counts().head(3)
            topic_details = []

            for tid, count in top_topics.items():
                keywords = []
                if topic_summary and str(int(tid)) in topic_summary:
                    kw_list = topic_summary[str(int(tid))].get("keywords", [])
                    keywords = [kw["word"] for kw in kw_list[:5]]

                topic_details.append({
                    "topic_id": int(tid),
                    "count": int(count),
                    "keywords": keywords,
                })

            category_insights[category] = {
                "negative_count": len(cat_group),
                "avg_severity": round(abs(cat_group["sentiment_compound"].mean()), 4),
                "avg_stars": round(cat_group["star_rating"].mean(), 2),
                "top_pain_topics": topic_details,
            }

        return category_insights


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(
        level=config["logging"]["level"],
        format=config["logging"]["format"],
    )
    topic_results = ROOT_DIR / config["topic_model"]["output_path"]
    topic_summary_path = ROOT_DIR / "outputs/topic_summary.json"

    if not topic_results.exists():
        logger.error(f"Topic results not found: {topic_results}. Run topic_model.py first.")
        raise FileNotFoundError(topic_results)

    df = pd.read_csv(topic_results)

    topic_summary = None
    if topic_summary_path.exists():
        with open(topic_summary_path) as f:
            topic_summary = json.load(f)

    extractor = PainPointExtractor()
    results = extractor.extract(
        df,
        topic_summary=topic_summary,
        output_path=config["pain_points"]["output_path"],
    )
    print(f"\nExtracted {len(results.get('pain_points', []))} pain points.")
