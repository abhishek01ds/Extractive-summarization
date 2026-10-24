"""
tests/test_baselines.py: Unit tests for baseline summarizers.
"""

import pytest
from src.preprocessing import TextPreprocessor
from src.bert_embeddings import BERTSentenceEmbedder
from src.baselines import TfidfSummarizer, DeepLearningSummarizer


def test_tfidf_summarizer():
    preprocessor = TextPreprocessor()
    text = (
        "Artificial intelligence is transforming diagnostics. "
        "Deep learning detects early stage tumors accurately. "
        "Clinicians emphasize assistive human-AI collaboration."
    )
    sentences = preprocessor.preprocess_documents([text])
    summarizer = TfidfSummarizer(budget_type="word", budget_limit=15)
    selected, summary, logs = summarizer.summarize(sentences)

    assert len(selected) >= 1
    assert len(summary.split()) <= 15 or len(selected) == 1
    assert len(logs) == len(selected)


def test_deep_learning_summarizer():
    preprocessor = TextPreprocessor()
    text = (
        "Renewable energy generation has expanded globally. "
        "Solar photovoltaic efficiency increased significantly. "
        "Battery storage solves intermittent power challenges."
    )
    sentences = preprocessor.preprocess_documents([text])
    embedder = BERTSentenceEmbedder(use_mock=True)
    summarizer = DeepLearningSummarizer(embedder=embedder, budget_type="word", budget_limit=20)
    selected, summary, logs = summarizer.summarize(sentences)

    assert len(selected) >= 1
    assert len(summary) > 0
    assert len(logs) == len(selected)
