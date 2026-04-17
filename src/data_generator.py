"""
Synthetic Customer Feedback Data Generator
==========================================
Generates 10,000+ realistic product reviews with varied sentiment,
product categories, and behavioral feedback patterns for the NLP pipeline.
"""

import os
import random
import csv
import yaml
import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# ── Load Config ────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT_DIR / "config.yaml"

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=config["logging"]["level"],
    format=config["logging"]["format"],
)
logger = logging.getLogger(__name__)

# ── Review Templates / Building Blocks ─────────────────────────────────────────

# Sentiment-aligned review fragments (realistic behavioral feedback)
POSITIVE_FRAGMENTS = [
    "Absolutely love this product! It exceeded all my expectations.",
    "Great quality for the price. Would definitely buy again.",
    "Fast shipping and the item was exactly as described.",
    "This is a game changer. My daily routine has improved significantly.",
    "Excellent customer service when I had a question about the product.",
    "Perfect gift idea. The recipient was thrilled.",
    "Premium build quality. You can tell this was made with care.",
    "I've been using this for months now and it still works flawlessly.",
    "The design is sleek and modern. Fits perfectly in my space.",
    "Best purchase I've made this year. Highly recommend to everyone.",
    "Outstanding value. Comparable to products that cost twice as much.",
    "Setup was incredibly easy. Had it running in under 5 minutes.",
    "The material feels luxurious and durable at the same time.",
    "I was skeptical at first but this product truly delivers on its promises.",
    "Impressed with the attention to detail in the packaging and product.",
    "My whole family loves this. We use it every single day.",
    "Works exactly as advertised. No complaints whatsoever.",
    "The color is vibrant and hasn't faded after multiple washes.",
    "Ergonomic design makes it comfortable for extended use.",
    "Energy efficient and quiet. Barely notice it running.",
]

NEGATIVE_FRAGMENTS = [
    "Product arrived damaged and customer service was unhelpful.",
    "Terrible quality. Broke after just two weeks of normal use.",
    "The sizing is completely off. Had to return immediately.",
    "Misleading product description. What I received looks nothing like the photos.",
    "Battery life is abysmal. Dies within an hour of moderate use.",
    "Overpriced for what you get. There are much better alternatives out there.",
    "The instructions were incomprehensible. Took hours to assemble.",
    "Started making a grinding noise after the first month. Very concerning.",
    "Tried contacting support three times with no response. Extremely frustrating.",
    "The material feels cheap and flimsy. Definitely not worth the premium price.",
    "Delivered two weeks late with no tracking updates or communication.",
    "The app that pairs with this device crashes constantly on both iOS and Android.",
    "Disappointed with the finish. There are visible scratches right out of the box.",
    "Does not work as advertised. The main feature is practically useless.",
    "Strong chemical smell that hasn't gone away even after airing it out for days.",
    "The zipper broke on the first use. Clearly a manufacturing defect.",
    "This product caused a skin rash. Had to stop using it immediately.",
    "Way too noisy. I can hear it from two rooms away.",
    "The remote control stopped working after a week. No replacement available.",
    "Refund process was a nightmare. Took over a month to get my money back.",
]

NEUTRAL_FRAGMENTS = [
    "It's okay. Does what it's supposed to do but nothing special.",
    "Average product. Met my basic needs but I wasn't blown away.",
    "Decent for the price point. You get what you pay for.",
    "It works fine but the design could be more modern.",
    "Arrived on time and in good condition. Product is adequate.",
    "Not bad but not great either. Middle of the road.",
    "Functional but lacks some features I was hoping for.",
    "It's a solid basic option if you don't need anything fancy.",
    "Quality is acceptable. Serves its purpose without any issues.",
    "Expected more based on the reviews but it's passable.",
]

MIXED_FRAGMENTS = [
    "The product itself is great but the shipping was terrible. Arrived damaged.",
    "Love the design but hate the battery life. Needs improvement.",
    "Excellent features but the build quality feels cheap for the price.",
    "Customer service was amazing but the product broke within a month.",
    "Great concept but poor execution. The software is buggy and slow.",
    "The taste is wonderful but the packaging is wasteful and hard to open.",
    "Comfortable to wear but the stitching started coming undone quickly.",
    "Easy to set up but the performance degrades over time noticeably.",
    "Looks beautiful on the shelf but doesn't function as well as I'd hoped.",
    "The core functionality works well but the companion app is terrible.",
]

# Pain-point specific fragments (for behavioral feedback surfacing)
PAIN_POINT_TEMPLATES = {
    "shipping_delay": [
        "Shipping took over {days} days. Completely unacceptable delivery time.",
        "My order was lost in transit and took {days} days to finally arrive.",
        "Delivery was delayed by {days} days with zero communication from the seller.",
        "Tracking showed delivered but package didn't arrive for another {days} days.",
    ],
    "product_defect": [
        "The {component} was broken right out of the box. Manufacturing defect.",
        "Found a crack in the {component} during unboxing. Quality control issue.",
        "The {component} stopped working after {days} days. Clearly defective.",
        "Multiple defects in the {component}. This is a quality control nightmare.",
    ],
    "customer_service": [
        "Called customer service {times} times and got transferred each time.",
        "Email support never responded after {times} follow-up messages.",
        "Live chat disconnected me {times} times before resolving my issue.",
        "Support agent was rude and dismissive after waiting {times} minutes on hold.",
    ],
    "price_value": [
        "Not worth the ${price} price tag. You can find better for half the cost.",
        "Feels like a ${price2} product being sold at a ${price} markup.",
        "The quality doesn't justify the ${price} price point at all.",
        "Severely overpriced at ${price}. Complete waste of money.",
    ],
    "usability": [
        "The interface is confusing and unintuitive. Takes forever to figure out.",
        "Setup process is ridiculously complicated. Needs better documentation.",
        "Controls are not responsive. I have to press buttons multiple times.",
        "The menu system is buried and illogical. Basic tasks require too many steps.",
    ],
    "durability": [
        "Broke after {days} days of normal use. Built to fail.",
        "The {component} wore out incredibly fast. Expected much better longevity.",
        "Paint started peeling after just {days} days. Terrible durability.",
        "Material degraded quickly. Within {days} days it looked years old.",
    ],
}

COMPONENTS = [
    "screen", "button", "hinge", "handle", "lid", "motor",
    "cord", "switch", "sensor", "filter", "seal", "latch",
    "bracket", "panel", "connector", "valve", "knob", "spring",
]

REVIEWER_NAMES = [
    "Alex M.", "Jordan T.", "Casey R.", "Taylor S.", "Morgan P.",
    "Riley K.", "Quinn B.", "Avery L.", "Dakota W.", "Cameron H.",
    "Skyler J.", "Drew N.", "Jamie F.", "Robin C.", "Sage D.",
    "Parker V.", "Blair G.", "Reese O.", "Finley A.", "Emerson Z.",
]


def _fill_template(template: str) -> str:
    """Fill in placeholders within a pain-point template."""
    return (
        template
        .replace("{days}", str(random.randint(3, 30)))
        .replace("{times}", str(random.randint(2, 8)))
        .replace("{price}", str(random.choice([30, 50, 75, 100, 150, 200, 300])))
        .replace("{price2}", str(random.choice([10, 15, 20, 25])))
        .replace("{component}", random.choice(COMPONENTS))
    )


def _generate_single_review(
    review_id: int,
    categories: list[str],
    seed_rng: random.Random,
) -> dict:
    """Generate a single synthetic review record."""
    # Weighted sentiment distribution: realistic skew (more positive than negative)
    sentiment_bucket = seed_rng.choices(
        ["positive", "negative", "neutral", "mixed"],
        weights=[0.45, 0.25, 0.15, 0.15],
        k=1,
    )[0]

    # Select fragments based on sentiment bucket
    if sentiment_bucket == "positive":
        n_sentences = seed_rng.randint(1, 3)
        fragments = seed_rng.sample(POSITIVE_FRAGMENTS, min(n_sentences, len(POSITIVE_FRAGMENTS)))
        star_rating = seed_rng.choices([4, 5], weights=[0.3, 0.7], k=1)[0]
    elif sentiment_bucket == "negative":
        n_sentences = seed_rng.randint(1, 4)  # negative reviews tend to be longer
        fragments = seed_rng.sample(NEGATIVE_FRAGMENTS, min(n_sentences, len(NEGATIVE_FRAGMENTS)))
        # Inject pain-point specific content ~60% of the time
        if seed_rng.random() < 0.6:
            pain_type = seed_rng.choice(list(PAIN_POINT_TEMPLATES.keys()))
            tmpl = seed_rng.choice(PAIN_POINT_TEMPLATES[pain_type])
            fragments.append(_fill_template(tmpl))
        star_rating = seed_rng.choices([1, 2], weights=[0.5, 0.5], k=1)[0]
    elif sentiment_bucket == "neutral":
        n_sentences = seed_rng.randint(1, 2)
        fragments = seed_rng.sample(NEUTRAL_FRAGMENTS, min(n_sentences, len(NEUTRAL_FRAGMENTS)))
        star_rating = 3
    else:  # mixed
        fragments = [seed_rng.choice(MIXED_FRAGMENTS)]
        star_rating = seed_rng.choice([2, 3, 4])

    review_text = " ".join(fragments)
    category = seed_rng.choice(categories)
    reviewer = seed_rng.choice(REVIEWER_NAMES)

    # Random date in the past 2 years
    days_ago = seed_rng.randint(0, 730)
    review_date = (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d")

    # Helpfulness votes
    helpful_votes = max(0, int(seed_rng.gauss(5, 10)))
    total_votes = helpful_votes + max(0, int(seed_rng.gauss(2, 5)))

    return {
        "review_id": review_id,
        "reviewer_name": reviewer,
        "category": category,
        "star_rating": star_rating,
        "review_text": review_text,
        "review_date": review_date,
        "helpful_votes": helpful_votes,
        "total_votes": total_votes,
    }


def generate_dataset(
    n_reviews: int = 10000,
    seed: int = 42,
    output_path: str = "data/raw_reviews.csv",
) -> pd.DataFrame:
    """
    Generate a synthetic customer review dataset.

    Parameters
    ----------
    n_reviews : int
        Number of reviews to generate (default: 10,000).
    seed : int
        Random seed for reproducibility.
    output_path : str
        CSV output path relative to project root.

    Returns
    -------
    pd.DataFrame
    """
    rng = random.Random(seed)
    np.random.seed(seed)
    categories = config["data"]["categories"]
    out_path = ROOT_DIR / output_path

    logger.info(f"Generating {n_reviews:,} synthetic reviews …")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    reviews = []
    for i in tqdm(range(1, n_reviews + 1), desc="Generating reviews", unit="rev"):
        reviews.append(_generate_single_review(i, categories, rng))

    df = pd.DataFrame(reviews)

    # Add some realistic noise: ~2% of reviews have empty text (simulates data quality issues)
    noise_indices = rng.sample(range(len(df)), k=int(0.02 * len(df)))
    df.loc[noise_indices, "review_text"] = ""

    df.to_csv(out_path, index=False, quoting=csv.QUOTE_ALL)
    logger.info(f"✓ Saved {len(df):,} reviews → {out_path}")

    # Print distribution summary
    logger.info("Star rating distribution:")
    for star in sorted(df["star_rating"].unique()):
        count = (df["star_rating"] == star).sum()
        pct = count / len(df) * 100
        logger.info(f"  ★{'★' * (star - 1)}{'☆' * (5 - star)}  {star}/5  →  {count:>5,} ({pct:.1f}%)")

    logger.info(f"Category distribution:")
    for cat, cnt in df["category"].value_counts().items():
        logger.info(f"  {cat:<20s} → {cnt:>5,}")

    return df


# ── CLI Entry Point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    n = config["data"]["n_reviews"]
    seed = config["data"]["random_seed"]
    raw_path = config["data"]["raw_path"]
    df = generate_dataset(n_reviews=n, seed=seed, output_path=raw_path)
    print(f"\n{'='*60}")
    print(f"Generated {len(df):,} reviews → {raw_path}")
    print(f"Columns: {list(df.columns)}")
    print(f"{'='*60}")
