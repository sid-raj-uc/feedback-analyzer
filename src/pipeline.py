"""
End-to-End NLP Pipeline Orchestrator
======================================
Orchestrates the full feedback analysis pipeline:

  1. Data Generation / Ingestion
  2. Text Preprocessing
  3. VADER Sentiment Analysis
  4. BERTopic Topic Modeling
  5. Pain Point Extraction
  6. Visualization Generation

Usage:
  python src/pipeline.py              # full pipeline
  python src/pipeline.py --skip-data  # skip data generation (use existing CSV)
"""

import os
import sys
import time
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime

import yaml
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# ── Project Modules ────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from data_generator import generate_dataset
from preprocessor import preprocess_dataframe
from sentiment import SentimentAnalyzer
from topic_model import TopicModeler
from pain_points import PainPointExtractor

# ── Load Config ────────────────────────────────────────────────────────────────
CONFIG_PATH = ROOT_DIR / "config.yaml"

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

# ── Logging ────────────────────────────────────────────────────────────────────
log_path = ROOT_DIR / config["logging"]["file"]
log_path.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=config["logging"]["level"],
    format=config["logging"]["format"],
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_path),
    ],
)
logger = logging.getLogger("pipeline")

# ── Plot Styling ───────────────────────────────────────────────────────────────
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")
PLOTS_DIR = ROOT_DIR / config["dashboard"]["plots_dir"]
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def generate_visualizations(df: pd.DataFrame, topic_summary: dict):
    """
    Generate static plots for the analysis results.
    """
    logger.info("Generating visualizations …")

    # ── 1. Sentiment Distribution ──────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Compound score histogram
    axes[0].hist(
        df["sentiment_compound"].dropna(),
        bins=50,
        color="#4A90D9",
        edgecolor="white",
        alpha=0.85,
    )
    axes[0].axvline(x=0.05, color="#2ECC71", linestyle="--", linewidth=1.5, label="Pos threshold")
    axes[0].axvline(x=-0.05, color="#E74C3C", linestyle="--", linewidth=1.5, label="Neg threshold")
    axes[0].set_title("VADER Compound Score Distribution", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Compound Score")
    axes[0].set_ylabel("Count")
    axes[0].legend()

    # Sentiment label pie chart
    label_counts = df["sentiment_label"].value_counts()
    colors = {"positive": "#2ECC71", "negative": "#E74C3C", "neutral": "#F39C12"}
    pie_colors = [colors.get(l, "#BDC3C7") for l in label_counts.index]

    axes[1].pie(
        label_counts.values,
        labels=label_counts.index,
        colors=pie_colors,
        autopct="%1.1f%%",
        startangle=90,
        textprops={"fontsize": 11},
    )
    axes[1].set_title("Sentiment Label Distribution", fontsize=13, fontweight="bold")

    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "sentiment_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── 2. Sentiment by Category ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 6))
    cat_sentiment = df.groupby("category")["sentiment_compound"].mean().sort_values()
    colors_bar = ["#E74C3C" if v < 0 else "#2ECC71" for v in cat_sentiment.values]
    cat_sentiment.plot(kind="barh", ax=ax, color=colors_bar, edgecolor="white")
    ax.set_title("Average Sentiment by Product Category", fontsize=13, fontweight="bold")
    ax.set_xlabel("Mean Compound Score")
    ax.axvline(x=0, color="gray", linestyle="-", linewidth=0.8)
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "sentiment_by_category.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── 3. Star Rating vs Sentiment ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(
        data=df,
        x="star_rating",
        y="sentiment_compound",
        ax=ax,
        palette="coolwarm",
    )
    ax.set_title("Sentiment Score Distribution by Star Rating", fontsize=13, fontweight="bold")
    ax.set_xlabel("Star Rating")
    ax.set_ylabel("Compound Sentiment Score")
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "sentiment_vs_stars.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── 4. Topic Distribution ──────────────────────────────────────────────
    valid_topics = df[df["topic_id"] != -1]
    if len(valid_topics) > 0:
        fig, ax = plt.subplots(figsize=(14, 6))
        topic_counts = valid_topics["topic_id"].value_counts().sort_index()
        topic_labels = []
        for tid in topic_counts.index:
            if topic_summary and str(int(tid)) in topic_summary:
                kws = topic_summary[str(int(tid))].get("keywords", [])
                kw_str = ", ".join([k["word"] for k in kws[:3]])
                topic_labels.append(f"T{int(tid)}: {kw_str}")
            else:
                topic_labels.append(f"Topic {int(tid)}")

        bars = ax.bar(
            range(len(topic_counts)),
            topic_counts.values,
            color=sns.color_palette("husl", len(topic_counts)),
            edgecolor="white",
        )
        ax.set_xticks(range(len(topic_counts)))
        ax.set_xticklabels(topic_labels, rotation=45, ha="right", fontsize=9)
        ax.set_title("Document Count per Topic Cluster", fontsize=13, fontweight="bold")
        ax.set_ylabel("Number of Reviews")
        plt.tight_layout()
        fig.savefig(PLOTS_DIR / "topic_distribution.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ── 5. Sentiment Heatmap: Topic × Category ────────────────────────────
    if len(valid_topics) > 0:
        fig, ax = plt.subplots(figsize=(14, 8))
        heatmap_data = valid_topics.pivot_table(
            values="sentiment_compound",
            index="topic_id",
            columns="category",
            aggfunc="mean",
        )

        # Create topic labels for heatmap
        hm_labels = []
        for tid in heatmap_data.index:
            if topic_summary and str(int(tid)) in topic_summary:
                kws = topic_summary[str(int(tid))].get("keywords", [])
                kw_str = ", ".join([k["word"] for k in kws[:2]])
                hm_labels.append(f"T{int(tid)}: {kw_str}")
            else:
                hm_labels.append(f"Topic {int(tid)}")

        sns.heatmap(
            heatmap_data,
            annot=True,
            fmt=".2f",
            cmap="RdYlGn",
            center=0,
            ax=ax,
            yticklabels=hm_labels,
            linewidths=0.5,
        )
        ax.set_title("Sentiment Heatmap: Topic × Category", fontsize=13, fontweight="bold")
        ax.set_ylabel("Topic Cluster")
        ax.set_xlabel("Product Category")
        plt.tight_layout()
        fig.savefig(PLOTS_DIR / "sentiment_heatmap.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ── 6. Word Cloud for Negative Reviews ─────────────────────────────────
    neg_text = " ".join(
        df[df["sentiment_label"] == "negative"]["topic_text"].dropna().tolist()
    )
    if neg_text.strip():
        wc = WordCloud(
            width=1200,
            height=600,
            background_color="white",
            colormap="Reds",
            max_words=100,
            collocations=False,
        ).generate(neg_text)

        fig, ax = plt.subplots(figsize=(14, 7))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title("Word Cloud — Negative Reviews", fontsize=14, fontweight="bold")
        fig.savefig(PLOTS_DIR / "negative_wordcloud.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ── 7. Word Cloud for Positive Reviews ─────────────────────────────────
    pos_text = " ".join(
        df[df["sentiment_label"] == "positive"]["topic_text"].dropna().tolist()
    )
    if pos_text.strip():
        wc = WordCloud(
            width=1200,
            height=600,
            background_color="white",
            colormap="Greens",
            max_words=100,
            collocations=False,
        ).generate(pos_text)

        fig, ax = plt.subplots(figsize=(14, 7))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title("Word Cloud — Positive Reviews", fontsize=14, fontweight="bold")
        fig.savefig(PLOTS_DIR / "positive_wordcloud.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ── 8. Review Length vs Sentiment ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(
        df["word_count"],
        df["sentiment_compound"],
        alpha=0.15,
        s=8,
        c=df["star_rating"],
        cmap="RdYlGn",
        vmin=1,
        vmax=5,
    )
    cb = plt.colorbar(ax.collections[0], ax=ax, label="Star Rating")
    ax.set_title("Review Length vs Sentiment Score", fontsize=13, fontweight="bold")
    ax.set_xlabel("Word Count")
    ax.set_ylabel("Compound Sentiment Score")
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "length_vs_sentiment.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"✓ All visualization saved to {PLOTS_DIR}")


def run_pipeline(skip_data: bool = False, skip_topics: bool = False):
    """
    Execute the full NLP feedback analysis pipeline.

    Parameters
    ----------
    skip_data : bool
        If True, skip data generation and use existing CSV.
    skip_topics : bool
        If True, skip BERTopic (useful for quick sentiment-only runs).
    """
    t_start = time.time()
    logger.info("=" * 70)
    logger.info("  NLP Customer Feedback Analyzer — Pipeline Start")
    logger.info(f"  Timestamp: {datetime.now().isoformat()}")
    logger.info("=" * 70)

    # ── Stage 1: Data ──────────────────────────────────────────────────────
    raw_path = ROOT_DIR / config["data"]["raw_path"]

    if skip_data and raw_path.exists():
        logger.info("[Stage 1/5] Loading existing dataset …")
        df = pd.read_csv(raw_path)
    else:
        logger.info("[Stage 1/5] Generating synthetic dataset …")
        df = generate_dataset(
            n_reviews=config["data"]["n_reviews"],
            seed=config["data"]["random_seed"],
            output_path=config["data"]["raw_path"],
        )

    stage1_time = time.time() - t_start
    logger.info(f"  ✓ Stage 1 complete — {len(df):,} reviews ({stage1_time:.1f}s)")

    # ── Stage 2: Preprocessing ─────────────────────────────────────────────
    t2 = time.time()
    logger.info("[Stage 2/5] Preprocessing text …")
    df = preprocess_dataframe(
        df,
        text_column="review_text",
        output_path=config["data"]["processed_path"],
    )
    stage2_time = time.time() - t2
    logger.info(f"  ✓ Stage 2 complete ({stage2_time:.1f}s)")

    # ── Stage 3: Sentiment Analysis ────────────────────────────────────────
    t3 = time.time()
    logger.info("[Stage 3/5] Running VADER sentiment analysis …")
    sa = SentimentAnalyzer()
    df = sa.analyze_dataframe(
        df,
        text_column="clean_text",
        output_path=config["sentiment"]["output_path"],
    )
    stage3_time = time.time() - t3
    logger.info(f"  ✓ Stage 3 complete ({stage3_time:.1f}s)")

    # ── Stage 4: Topic Modeling ────────────────────────────────────────────
    topic_summary = {}
    if not skip_topics:
        t4 = time.time()
        logger.info("[Stage 4/5] Running BERTopic clustering …")
        tm = TopicModeler()
        df = tm.fit_transform(df, text_column="topic_text")
        tm.save_results(df, output_path=config["topic_model"]["output_path"])
        topic_summary = tm.export_topic_summary()

        try:
            tm.save_model()
        except Exception as e:
            logger.warning(f"  Model serialization skipped: {e}")

        stage4_time = time.time() - t4
        logger.info(f"  ✓ Stage 4 complete ({stage4_time:.1f}s)")
    else:
        logger.info("[Stage 4/5] SKIPPED (--skip-topics)")
        df["topic_id"] = -1
        df["topic_probability"] = 0.0

    # ── Stage 5: Pain Point Extraction ─────────────────────────────────────
    t5 = time.time()
    logger.info("[Stage 5/5] Extracting actionable pain points …")
    extractor = PainPointExtractor()
    pain_results = extractor.extract(
        df,
        topic_summary=topic_summary,
        output_path=config["pain_points"]["output_path"],
    )
    stage5_time = time.time() - t5
    logger.info(f"  ✓ Stage 5 complete ({stage5_time:.1f}s)")

    # ── Visualizations ─────────────────────────────────────────────────────
    t_viz = time.time()
    generate_visualizations(df, topic_summary)
    viz_time = time.time() - t_viz

    # ── Summary ────────────────────────────────────────────────────────────
    total_time = time.time() - t_start
    logger.info("")
    logger.info("=" * 70)
    logger.info("  Pipeline Complete")
    logger.info("=" * 70)
    logger.info(f"  Total reviews processed:  {len(df):,}")
    logger.info(f"  Sentiment distribution:")
    for label in ["positive", "negative", "neutral"]:
        count = (df["sentiment_label"] == label).sum()
        pct = count / len(df) * 100
        logger.info(f"    {label:<10s}  {count:>5,}  ({pct:.1f}%)")
    if not skip_topics:
        n_topics = len(df[df["topic_id"] != -1]["topic_id"].unique())
        outlier_pct = (df["topic_id"] == -1).mean() * 100
        logger.info(f"  Topics discovered:       {n_topics}")
        logger.info(f"  Outlier rate:            {outlier_pct:.1f}%")
    n_pain = len(pain_results.get("pain_points", []))
    logger.info(f"  Pain points extracted:   {n_pain}")
    logger.info(f"  Total runtime:           {total_time:.1f}s")
    logger.info("")
    logger.info("  Output files:")
    logger.info(f"    → {config['data']['processed_path']}")
    logger.info(f"    → {config['sentiment']['output_path']}")
    logger.info(f"    → {config['topic_model']['output_path']}")
    logger.info(f"    → {config['pain_points']['output_path']}")
    logger.info(f"    → {config['dashboard']['plots_dir']}/")
    logger.info("=" * 70)

    return df, pain_results


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="NLP Customer Feedback Analyzer — Pipeline",
    )
    parser.add_argument(
        "--skip-data",
        action="store_true",
        help="Skip data generation; use existing raw CSV.",
    )
    parser.add_argument(
        "--skip-topics",
        action="store_true",
        help="Skip BERTopic clustering (sentiment-only mode).",
    )
    args = parser.parse_args()
    run_pipeline(skip_data=args.skip_data, skip_topics=args.skip_topics)
