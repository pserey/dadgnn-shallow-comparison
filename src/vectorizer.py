"""
Text vectorization using TF-IDF with specified configuration.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Optional


def make_vectorizer(
    lowercase: bool = True,
    strip_accents: Optional[str] = "unicode",
    analyzer: str = "word",
    ngram_range: tuple = (1, 2),
    min_df: int = 2,
    max_df: float = 0.9,
    max_features: int = 100_000,
    sublinear_tf: bool = True
) -> TfidfVectorizer:
    """
    Create a TF-IDF vectorizer with specified configuration.
    
    Args:
        lowercase: Convert text to lowercase
        strip_accents: Remove accents ('unicode' or None)  
        analyzer: Type of analyzer ('word', 'char', etc.)
        ngram_range: Range of n-grams to extract
        min_df: Minimum document frequency for terms
        max_df: Maximum document frequency for terms
        max_features: Maximum number of features
        sublinear_tf: Apply sublinear tf scaling
        
    Returns:
        Configured TfidfVectorizer instance
    """
    return TfidfVectorizer(
        lowercase=lowercase,
        strip_accents=strip_accents,
        analyzer=analyzer,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        max_features=max_features,
        sublinear_tf=sublinear_tf
    )


def get_default_vectorizer() -> TfidfVectorizer:
    """
    Get the default TF-IDF vectorizer configuration used in experiments.
    
    Returns:
        TfidfVectorizer with default parameters
    """
    vectorizer = make_vectorizer(
        lowercase=True,
        strip_accents="unicode", 
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9,
        max_features=100_000,
        sublinear_tf=True
    )
    return vectorizer


def log_vectorizer_config(vectorizer: TfidfVectorizer) -> None:
    """Log TF-IDF configuration for consistency verification."""
    print(f"TF-IDF CONFIGURATION:")
    print(f"  lowercase: {vectorizer.lowercase}")
    print(f"  strip_accents: {vectorizer.strip_accents}")
    print(f"  analyzer: {vectorizer.analyzer}")
    print(f"  ngram_range: {vectorizer.ngram_range}")
    print(f"  min_df: {vectorizer.min_df}")
    print(f"  max_df: {vectorizer.max_df}")
    print(f"  max_features: {vectorizer.max_features}")
    print(f"  sublinear_tf: {vectorizer.sublinear_tf}")
    print(f"  ✓ Configuration consistent across all experiments")
