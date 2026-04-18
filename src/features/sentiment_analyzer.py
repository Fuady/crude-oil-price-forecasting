"""
src/features/sentiment_analyzer.py
------------------------------------
NLP sentiment analysis for oil market news headlines using FinBERT.

FinBERT is a BERT model fine-tuned on financial text — far more accurate
than general-purpose sentiment tools (VADER, TextBlob) for energy headlines.

Pipeline:
  1. Load oil/energy news headlines (from NewsAPI or cached)
  2. Run FinBERT inference on each headline
  3. Aggregate to daily sentiment scores
  4. Create rolling sentiment features for the ML model

FinBERT paper: Araci, D. (2019). FinBERT: Financial Sentiment Analysis
with Pre-trained Language Models. arXiv:1908.10063

In production (ADNOC/Aramco trading desk):
  - Reuters Eikon headlines via API
  - Bloomberg Terminal news feed
  - OPEC press releases parsed on publish
  - Social media (carefully filtered)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
import warnings

warnings.filterwarnings("ignore")

RAW_DATA_DIR = Path("data/raw")

# Sample oil market headlines for demonstration
SAMPLE_HEADLINES = [
    ("OPEC+ agrees to extend production cuts through Q2, boosting crude prices", "bullish"),
    ("US crude inventories fall sharply, largest draw since January", "bullish"),
    ("Saudi Arabia maintains $80/bbl price target, signals further cuts if needed", "bullish"),
    ("China stimulus boosts oil demand outlook, refiners increase throughput", "bullish"),
    ("Brent crude climbs on Middle East supply disruption fears", "bullish"),
    ("IEA raises 2024 oil demand forecast amid stronger Asian consumption", "bullish"),
    ("Iraq, Kuwait pledge OPEC+ compliance after months of overproduction", "bullish"),
    ("US shale production growth slows as rig count falls for third week", "bullish"),
    ("Geopolitical tensions in Red Sea lift shipping costs, crude risk premium", "bullish"),
    ("Dollar weakens on Fed rate cut signals, boosting dollar-denominated oil", "bullish"),
    ("US crude inventories build unexpectedly, adding 4.2 million barrels", "bearish"),
    ("OPEC+ members consider increasing output at December meeting", "bearish"),
    ("China economic slowdown weighs on oil demand outlook", "bearish"),
    ("US shale producers report record output, WTI pressured below $75", "bearish"),
    ("IEA warns of oil supply surplus in 2024 as non-OPEC production surges", "bearish"),
    ("Global recession fears mount, crude oil slides to 4-month lows", "bearish"),
    ("Russia increases seaborne crude exports despite sanctions pressure", "bearish"),
    ("Brent crude drops on weak manufacturing data from US and Europe", "bearish"),
    ("Nigeria restores Forcados export terminal after brief disruption", "bearish"),
    ("Crude oil market weighed by rising interest rates, dollar strength", "bearish"),
    ("Oil edges higher on mixed inventory data, market awaits OPEC guidance", "neutral"),
    ("Crude prices range-bound ahead of EIA weekly petroleum report", "neutral"),
    ("Oil traders cautious as OPEC meeting date approaches", "neutral"),
    ("Brent holds steady near $82 as supply and demand factors balance", "neutral"),
    ("Energy markets mixed as geopolitical risks offset demand concerns", "neutral"),
]


class FinBERTSentimentAnalyzer:
    """
    Sentiment analyzer using FinBERT (ProsusAI/finbert).

    Falls back to lexicon-based scoring if transformers not available
    or model can't be downloaded (e.g., air-gapped environment).
    """

    def __init__(self, use_gpu: bool = False, batch_size: int = 32):
        self.use_gpu = use_gpu
        self.batch_size = batch_size
        self.model = None
        self.tokenizer = None
        self._use_finbert = False
        self._oil_lexicon = self._build_oil_lexicon()

    def _load_finbert(self) -> bool:
        """Attempt to load FinBERT model."""
        try:
            from transformers import pipeline
            device = 0 if self.use_gpu else -1
            self.model = pipeline(
                "text-classification",
                model="ProsusAI/finbert",
                device=device,
                truncation=True,
                max_length=512,
            )
            self._use_finbert = True
            print("  ✓ FinBERT model loaded")
            return True
        except Exception as e:
            print(f"  FinBERT unavailable ({e}). Using lexicon-based fallback.")
            return False

    def _build_oil_lexicon(self) -> Dict[str, float]:
        """
        Domain-specific oil market sentiment lexicon.
        Scores: +1 = strongly bullish, -1 = strongly bearish.
        """
        return {
            # Strongly bullish
            "cut": 0.6, "cuts": 0.6, "draw": 0.7, "drawdown": 0.7,
            "shortage": 0.8, "disruption": 0.7, "conflict": 0.5,
            "sanctions": 0.6, "geopolitical": 0.4, "tensions": 0.4,
            "bullish": 0.9, "surge": 0.7, "rally": 0.7, "jump": 0.6,
            "climb": 0.5, "rise": 0.5, "gain": 0.5, "boost": 0.6,
            "strong": 0.4, "demand": 0.3, "stimulus": 0.5, "recovery": 0.4,
            "opec": 0.2, "compliance": 0.5, "extend": 0.4, "tighten": 0.5,
            # Strongly bearish
            "build": -0.6, "surplus": -0.7, "glut": -0.8, "oversupply": -0.8,
            "bearish": -0.9, "drop": -0.6, "fall": -0.5, "decline": -0.5,
            "plunge": -0.8, "slide": -0.6, "crash": -0.9, "collapse": -0.9,
            "weak": -0.4, "slowdown": -0.5, "recession": -0.7, "fears": -0.3,
            "increase output": -0.6, "raise production": -0.6, "shale": -0.2,
            "record output": -0.5, "oversupply": -0.8, "inventory build": -0.7,
            # Modifiers
            "unexpectedly": 0.2, "sharply": 0.3, "record": 0.2,
        }

    def _lexicon_score(self, text: str) -> float:
        """Score a headline using the oil market lexicon."""
        text_lower = text.lower()
        score = 0.0
        n_matches = 0
        for term, weight in self._oil_lexicon.items():
            if term in text_lower:
                score += weight
                n_matches += 1
        if n_matches == 0:
            return 0.0
        return float(np.clip(score / max(n_matches, 1), -1, 1))

    def analyze_headlines(
        self, headlines: List[str], batch_size: int = None
    ) -> List[Dict]:
        """
        Analyze a list of headlines and return sentiment scores.

        Returns list of dicts with: score (-1 to 1), label, confidence
        """
        if not self._use_finbert:
            self._load_finbert()

        results = []

        if self._use_finbert and self.model is not None:
            try:
                bs = batch_size or self.batch_size
                for i in range(0, len(headlines), bs):
                    batch = headlines[i: i + bs]
                    outputs = self.model(batch)
                    for out in outputs:
                        label = out["label"].lower()
                        confidence = out["score"]
                        score = confidence if label == "positive" else (
                            -confidence if label == "negative" else 0.0
                        )
                        results.append({
                            "score": score,
                            "label": label,
                            "confidence": confidence,
                        })
                return results
            except Exception:
                pass   # Fall through to lexicon

        # Lexicon fallback
        for headline in headlines:
            score = self._lexicon_score(headline)
            label = "positive" if score > 0.1 else "negative" if score < -0.1 else "neutral"
            results.append({
                "score": score,
                "label": label,
                "confidence": abs(score),
            })
        return results


def load_or_create_sentiment(price_index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Load pre-computed sentiment from disk, or generate synthetic.
    Returns DataFrame aligned to price_index.
    """
    sentiment_path = RAW_DATA_DIR / "news_sentiment.parquet"

    if sentiment_path.exists():
        df = pd.read_parquet(sentiment_path)
        df.index = pd.to_datetime(df.index)
        df = df.reindex(price_index).ffill().bfill()
        return df

    # Fallback: generate synthetic
    return _generate_aligned_sentiment(price_index)


def _generate_aligned_sentiment(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Generate realistic synthetic sentiment aligned to a date index."""
    rng = np.random.default_rng(42)
    n = len(index)

    base = rng.normal(0.05, 0.20, n)
    sentiment = pd.Series(base).ewm(span=5).mean().values
    sentiment = np.clip(sentiment, -1, 1)

    pos_frac = np.clip((sentiment + 1) / 2 * 0.8 + 0.1, 0.1, 0.9)
    neg_frac = np.clip(0.9 - pos_frac, 0.05, 0.7)
    neu_frac = 1 - pos_frac - neg_frac

    df = pd.DataFrame({
        "sentiment_score":  sentiment,
        "sentiment_ma5":    pd.Series(sentiment).rolling(5,  min_periods=1).mean().values,
        "sentiment_ma20":   pd.Series(sentiment).rolling(20, min_periods=1).mean().values,
        "sentiment_std":    pd.Series(sentiment).rolling(10, min_periods=1).std().fillna(0).values,
        "n_headlines":      rng.integers(5, 35, n).astype(float),
        "pct_positive":     pos_frac,
        "pct_negative":     neg_frac,
        "pct_neutral":      neu_frac,
        "sentiment_momentum": pd.Series(sentiment).diff(5).fillna(0).values,
    }, index=index)

    return df


def add_sentiment_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add NLP sentiment features to the price DataFrame.

    Features:
      - sentiment_score: daily aggregate score (-1 bearish, +1 bullish)
      - sentiment_ma5/ma20: smoothed sentiment (reduces noise)
      - sentiment_regime: sustained positive or negative sentiment
      - sentiment_divergence: price direction vs sentiment direction
    """
    out = df.copy()

    sent = load_or_create_sentiment(df.index)

    for col in sent.columns:
        out[col] = sent[col].values

    # Sentiment regime: is sentiment consistently positive/negative?
    if "sentiment_ma5" in out.columns:
        out["sent_positive_regime"] = (out["sentiment_ma5"] > 0.1).astype(int)
        out["sent_negative_regime"] = (out["sentiment_ma5"] < -0.1).astype(int)

    # Sentiment divergence: price moving up but sentiment turning negative (or vice versa)
    if "roc_5d" in out.columns and "sentiment_score" in out.columns:
        price_up    = (out["roc_5d"] > 0).astype(int)
        sent_pos    = (out["sentiment_score"] > 0).astype(int)
        out["sentiment_price_divergence"] = (price_up != sent_pos).astype(int)

    # Extreme sentiment flags (contrarian signals)
    if "sentiment_score" in out.columns:
        out["extreme_bearish_sentiment"] = (out["sentiment_score"] < -0.5).astype(int)
        out["extreme_bullish_sentiment"] = (out["sentiment_score"] >  0.5).astype(int)

    return out
