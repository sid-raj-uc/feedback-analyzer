"""
Text Preprocessing Module
=========================
Cleans and normalizes raw review text for downstream NLP tasks.
Handles HTML removal, URL stripping, special character cleaning,
tokenization, and stopword removal.
"""

import re
import logging
from pathlib import Path
from typing import Optional

import yaml
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tqdm import tqdm

# ── Load Config ────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT_DIR / "config.yaml"

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

logger = logging.getLogger(__name__)

# ── NLTK Downloads ─────────────────────────────────────────────────────────────
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

STOP_WORDS = set(stopwords.words(config["preprocessing"]["language"]))

# ── Regex Patterns ─────────────────────────────────────────────────────────────
HTML_TAG_RE = re.compile(r"<[^>]+>")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
SPECIAL_CHAR_RE = re.compile(r"[^a-zA-Z0-9\s.,!?'-]")
MULTI_SPACE_RE = re.compile(r"\s+")
EMAIL_RE = re.compile(r"\S+@\S+\.\S+")


def clean_text(
    text: str,
    lowercase: bool = True,
    remove_html: bool = True,
    remove_urls: bool = True,
    remove_special_chars: bool = True,
    remove_numbers: bool = False,
    min_token_length: int = 2,
) -> str:
    """
    Apply a comprehensive text cleaning pipeline to a single review.

    Parameters
    ----------
    text : str
        Raw review text.
    lowercase : bool
        Convert to lowercase.
    remove_html : bool
        Strip HTML tags.
    remove_urls : bool
        Remove URLs.
    remove_special_chars : bool
        Remove non-alphanumeric / non-punctuation characters.
    remove_numbers : bool
        Strip numeric characters.
    min_token_length : int
        Minimum token length to keep.

    Returns
    -------
    str
        Cleaned text.
    """
    if not isinstance(text, str) or len(text.strip()) == 0:
        return ""

    # 1. Remove HTML tags
    if remove_html:
        text = HTML_TAG_RE.sub(" ", text)

    # 2. Remove emails
    text = EMAIL_RE.sub(" ", text)

    # 3. Remove URLs
    if remove_urls:
        text = URL_RE.sub(" ", text)

    # 4. Lowercase
    if lowercase:
        text = text.lower()

    # 5. Remove special characters (keep basic punctuation for VADER)
    if remove_special_chars:
        text = SPECIAL_CHAR_RE.sub(" ", text)

    # 6. Remove numbers
    if remove_numbers:
        text = re.sub(r"\d+", " ", text)

    # 7. Normalize whitespace
    text = MULTI_SPACE_RE.sub(" ", text).strip()

    return text


def clean_text_for_topic_modeling(text: str) -> str:
    """
    More aggressive cleaning for topic modeling — removes stopwords
    and short tokens, keeping only meaningful content words.

    Parameters
    ----------
    text : str
        Pre-cleaned text (run clean_text first).

    Returns
    -------
    str
        Token-filtered text suitable for TF-IDF / BERTopic.
    """
    if not text:
        return ""

    tokens = word_tokenize(text)
    filtered = [
        t for t in tokens
        if t not in STOP_WORDS
        and len(t) >= config["preprocessing"]["min_token_length"]
        and t.isalpha()
    ]
    return " ".join(filtered)


def preprocess_dataframe(
    df: pd.DataFrame,
    text_column: str = "review_text",
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Apply the full preprocessing pipeline to a DataFrame of reviews.

    Adds columns:
      - clean_text : cleaned text (for VADER sentiment)
      - topic_text : aggressively cleaned text (for BERTopic)
      - word_count : number of words in clean_text
      - is_empty   : flag for empty/missing reviews

    Parameters
    ----------
    df : pd.DataFrame
        Raw reviews DataFrame.
    text_column : str
        Column containing raw text.
    output_path : str, optional
        If provided, save processed DataFrame to this CSV path.

    Returns
    -------
    pd.DataFrame
        Processed DataFrame with additional columns.
    """
    cfg = config["preprocessing"]
    logger.info(f"Preprocessing {len(df):,} reviews …")

    tqdm.pandas(desc="Cleaning text")

    # 1. Basic cleaning (preserves punctuation for VADER)
    df["clean_text"] = df[text_column].progress_apply(
        lambda x: clean_text(
            x,
            lowercase=cfg["lowercase"],
            remove_html=cfg["remove_html"],
            remove_urls=cfg["remove_urls"],
            remove_special_chars=cfg["remove_special_chars"],
            remove_numbers=cfg["remove_numbers"],
            min_token_length=cfg["min_token_length"],
        )
    )

    # 2. Aggressive cleaning for topic modeling
    tqdm.pandas(desc="Preparing topic text")
    df["topic_text"] = df["clean_text"].progress_apply(clean_text_for_topic_modeling)

    # 3. Metadata columns
    df["word_count"] = df["clean_text"].apply(lambda x: len(x.split()) if x else 0)
    df["is_empty"] = df["clean_text"].apply(lambda x: len(x.strip()) == 0)

    empty_count = df["is_empty"].sum()
    logger.info(f"  Empty/invalid reviews flagged: {empty_count:,} ({empty_count/len(df)*100:.1f}%)")
    logger.info(f"  Avg. word count: {df['word_count'].mean():.1f}")

    # 4. Optionally save
    if output_path:
        out = ROOT_DIR / output_path
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        logger.info(f"  ✓ Saved processed data → {out}")

    return df


# ── CLI Entry Point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(
        level=config["logging"]["level"],
        format=config["logging"]["format"],
    )
    raw_path = ROOT_DIR / config["data"]["raw_path"]
    if not raw_path.exists():
        logger.error(f"Raw data not found at {raw_path}. Run data_generator.py first.")
        raise FileNotFoundError(raw_path)

    df = pd.read_csv(raw_path)
    df = preprocess_dataframe(
        df,
        text_column="review_text",
        output_path=config["data"]["processed_path"],
    )
    print(f"\nPreprocessed {len(df):,} reviews.")
    print(df[["review_id", "clean_text", "word_count", "is_empty"]].head(10))
