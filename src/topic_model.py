"""
BERTopic Clustering Module
===========================
Implements neural topic modeling using BERTopic with:
  - Sentence-transformer embeddings (all-MiniLM-L6-v2)
  - UMAP dimensionality reduction
  - HDBSCAN density-based clustering
  - c-TF-IDF class-based term frequency weighting

This module discovers latent topic clusters from customer feedback,
enabling the identification of recurring themes and pain points.
"""

import logging
import json
from pathlib import Path
from typing import Optional

import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from hdbscan import HDBSCAN
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

# ── Load Config ────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT_DIR / "config.yaml"

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

logger = logging.getLogger(__name__)


class TopicModeler:
    """
    BERTopic-based topic discovery engine.

    Pipeline:
        1. Embed documents using sentence-transformers
        2. Reduce dimensionality with UMAP
        3. Cluster with HDBSCAN
        4. Extract topic representations via c-TF-IDF

    Attributes
    ----------
    topic_model : BERTopic
        Fitted BERTopic model.
    topics : list[int]
        Topic assignments for each document.
    probabilities : np.ndarray
        Topic assignment probabilities.
    """

    def __init__(self):
        tc = config["topic_model"]

        # 1. Embedding model
        self.embedding_model_name = tc["embedding_model"]
        logger.info(f"Loading embedding model: {self.embedding_model_name}")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)

        # 2. UMAP
        umap_cfg = tc["umap"]
        self.umap_model = UMAP(
            n_neighbors=umap_cfg["n_neighbors"],
            n_components=umap_cfg["n_components"],
            min_dist=umap_cfg["min_dist"],
            metric=umap_cfg["metric"],
            random_state=umap_cfg["random_state"],
        )

        # 3. HDBSCAN
        hdb_cfg = tc["hdbscan"]
        self.hdbscan_model = HDBSCAN(
            min_cluster_size=hdb_cfg["min_cluster_size"],
            min_samples=hdb_cfg["min_samples"],
            metric=hdb_cfg["metric"],
            cluster_selection_method=hdb_cfg["cluster_selection_method"],
            prediction_data=True,
        )

        # 4. Vectorizer for c-TF-IDF
        vec_cfg = tc["vectorizer"]
        self.vectorizer = CountVectorizer(
            stop_words=vec_cfg["stop_words"],
            ngram_range=tuple(vec_cfg["n_gram_range"]),
            min_df=vec_cfg["min_df"],
            max_df=vec_cfg["max_df"],
        )

        # 5. BERTopic
        nr_topics = tc["nr_topics"]
        if nr_topics != "auto":
            nr_topics = int(nr_topics)

        self.topic_model = BERTopic(
            embedding_model=self.embedding_model,
            umap_model=self.umap_model,
            hdbscan_model=self.hdbscan_model,
            vectorizer_model=self.vectorizer,
            top_n_words=tc["top_n_words"],
            nr_topics=nr_topics,
            calculate_probabilities=tc["calculate_probabilities"],
            verbose=True,
        )

        self.topics = None
        self.probabilities = None
        logger.info("TopicModeler initialized.")

    def fit_transform(
        self,
        df: pd.DataFrame,
        text_column: str = "topic_text",
    ) -> pd.DataFrame:
        """
        Fit the BERTopic model on review text and assign topic labels.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with preprocessed text column.
        text_column : str
            Column name of the text to cluster.

        Returns
        -------
        pd.DataFrame
            DataFrame with added 'topic_id' and 'topic_probability' columns.
        """
        # Filter out empty documents
        docs = df[text_column].fillna("").tolist()
        valid_mask = [len(d.strip()) > 0 for d in docs]
        valid_docs = [d for d, m in zip(docs, valid_mask) if m]

        logger.info(f"Fitting BERTopic on {len(valid_docs):,} valid documents "
                     f"(filtered {len(docs) - len(valid_docs):,} empty) …")

        # Fit + transform
        topics, probs = self.topic_model.fit_transform(valid_docs)

        # Map back to full DataFrame
        topic_ids = []
        topic_probs = []
        valid_idx = 0
        for m in valid_mask:
            if m:
                topic_ids.append(topics[valid_idx])
                topic_probs.append(
                    float(probs[valid_idx].max()) if probs is not None and len(probs.shape) > 1
                    else float(probs[valid_idx]) if probs is not None
                    else 0.0
                )
                valid_idx += 1
            else:
                topic_ids.append(-1)
                topic_probs.append(0.0)

        df["topic_id"] = topic_ids
        df["topic_probability"] = topic_probs

        self.topics = topics
        self.probabilities = probs

        # Log topic summary
        topic_info = self.topic_model.get_topic_info()
        logger.info(f"\n{'='*60}")
        logger.info(f"Discovered {len(topic_info) - 1} topics (excluding outliers)")
        logger.info(f"{'='*60}")

        for _, row in topic_info.iterrows():
            tid = row["Topic"]
            count = row["Count"]
            name = row.get("Name", f"Topic_{tid}")
            if tid == -1:
                logger.info(f"  [Outliers]  Count: {count:>5,}")
            else:
                # Get top keywords
                topic_words = self.topic_model.get_topic(tid)
                keywords = ", ".join([w for w, _ in topic_words[:5]])
                logger.info(f"  Topic {tid:>3d}  Count: {count:>5,}  Keywords: {keywords}")

        # Outlier statistics
        outlier_count = sum(1 for t in topic_ids if t == -1)
        outlier_pct = outlier_count / len(topic_ids) * 100
        logger.info(f"\nOutlier rate: {outlier_count:,} / {len(topic_ids):,} ({outlier_pct:.1f}%)")

        return df

    def get_topic_info(self) -> pd.DataFrame:
        """Return the topic info DataFrame from the fitted model."""
        return self.topic_model.get_topic_info()

    def get_topic_keywords(self, topic_id: int, top_n: int = 10) -> list[tuple[str, float]]:
        """Get the top keywords for a specific topic."""
        return self.topic_model.get_topic(topic_id)[:top_n]

    def get_representative_docs(self, topic_id: int) -> list[str]:
        """Get representative documents for a specific topic."""
        return self.topic_model.get_representative_docs(topic_id)

    def save_model(self, path: Optional[str] = None):
        """Save the fitted BERTopic model to disk."""
        save_path = ROOT_DIR / (path or config["topic_model"]["model_path"])
        save_path.parent.mkdir(parents=True, exist_ok=True)
        self.topic_model.save(
            str(save_path),
            serialization="safetensors",
            save_ctfidf=True,
            save_embedding_model=self.embedding_model_name,
        )
        logger.info(f"✓ Model saved → {save_path}")

    def save_results(
        self,
        df: pd.DataFrame,
        output_path: Optional[str] = None,
    ):
        """Save topic assignment results to CSV."""
        out = ROOT_DIR / (output_path or config["topic_model"]["output_path"])
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        logger.info(f"✓ Topic results saved → {out}")

    def export_topic_summary(self, output_path: Optional[str] = None) -> dict:
        """
        Export a JSON summary of all discovered topics with their keywords
        and representative documents.
        """
        out_path = ROOT_DIR / (output_path or "outputs/topic_summary.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        topic_info = self.get_topic_info()
        summary = {}

        for _, row in topic_info.iterrows():
            tid = int(row["Topic"])
            if tid == -1:
                continue
            keywords = self.get_topic_keywords(tid)
            try:
                rep_docs = self.get_representative_docs(tid)
            except Exception:
                rep_docs = []

            summary[str(tid)] = {
                "topic_id": tid,
                "count": int(row["Count"]),
                "name": row.get("Name", f"Topic_{tid}"),
                "keywords": [{"word": w, "score": round(s, 4)} for w, s in keywords],
                "representative_docs": rep_docs[:3],
            }

        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"✓ Topic summary exported → {out_path}")
        return summary


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
    tm = TopicModeler()
    df = tm.fit_transform(df, text_column="topic_text")
    tm.save_results(df)
    tm.export_topic_summary()
    print(f"\nTopic modeling complete. {len(tm.get_topic_info()) - 1} topics discovered.")
