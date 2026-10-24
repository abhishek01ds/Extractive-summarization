"""
tests/test_summary_generator.py: Unit tests for greedy summary generation algorithm,
marginal gain maximization, budget constraint compliance, and order preservation.
"""

import pytest
from src.preprocessing import Sentence, TextPreprocessor
from src.bert_embeddings import BERTSentenceEmbedder
from src.graph_builder import SentenceGraphBuilder
from src.dsn_model import SGCSummModel
from src.summary_generator import GreedySummaryGenerator


@pytest.fixture
def dummy_cluster_sentences():
    # 5 sample sentences from 2 documents
    raw_texts = [
        ("doc_1", 0, "Cambodian leaders held an emergency summit in Phnom Penh yesterday."),
        ("doc_1", 1, "King Sihanouk appealed for national unity and constitutional peace."),
        ("doc_1", 2, "Negotiators discussed forming a joint government coalition."),
        ("doc_2", 0, "International monitors confirmed the democratic election was valid."),
        ("doc_2", 1, "Economic recovery remains the primary challenge facing the new cabinet.")
    ]
    preprocessor = TextPreprocessor()
    sentences = []
    for g_idx, (d_id, s_id, text) in enumerate(raw_texts):
        c_tok, s_tok = preprocessor.process_sentence_tokens(text)
        sentences.append(Sentence(
            raw_text=text,
            doc_id=d_id,
            sent_id=s_id,
            global_id=g_idx,
            clean_tokens=c_tok,
            stemmed_tokens=s_tok
        ))
    return sentences


def test_greedy_selection_and_budget_enforcement(dummy_cluster_sentences):
    embedder = BERTSentenceEmbedder(embedding_dim=32, use_mock=True)
    graph_builder = SentenceGraphBuilder(top_n=5, top_k=5)
    model = SGCSummModel(num_channels=31, num_vertices=5, num_neighbors=5, hidden_dim=8)

    # Test with strict byte budget: 150 bytes
    max_bytes = 150
    generator = GreedySummaryGenerator(
        model=model,
        embedder=embedder,
        graph_builder=graph_builder,
        budget_type="byte",
        budget_limit=max_bytes,
        restore_order=True
    )

    selected_sents, summary_text, logs = generator.generate_summary(dummy_cluster_sentences)

    assert len(selected_sents) > 0
    total_bytes = len(summary_text.encode("utf-8"))
    assert total_bytes <= max_bytes, f"Generated summary bytes ({total_bytes}) exceeded budget ({max_bytes})"
    assert len(logs) == len(selected_sents)


def test_order_preservation(dummy_cluster_sentences):
    embedder = BERTSentenceEmbedder(embedding_dim=32, use_mock=True)
    graph_builder = SentenceGraphBuilder(top_n=5, top_k=5)
    model = SGCSummModel(num_channels=31, num_vertices=5, num_neighbors=5, hidden_dim=8)

    generator = GreedySummaryGenerator(
        model=model,
        embedder=embedder,
        graph_builder=graph_builder,
        budget_type="word",
        budget_limit=200,
        restore_order=True
    )

    selected_sents, _, _ = generator.generate_summary(dummy_cluster_sentences)

    # Verify that in selected_sents, order is strictly sorted by doc_id then sent_id
    doc_sent_tuples = [(s.doc_id, s.sent_id) for s in selected_sents]
    assert doc_sent_tuples == sorted(doc_sent_tuples), "Selected sentences must restore original document order"


def test_empty_and_single_sentence_handling():
    embedder = BERTSentenceEmbedder(embedding_dim=16, use_mock=True)
    graph_builder = SentenceGraphBuilder(top_n=3, top_k=3)
    model = SGCSummModel(num_channels=15, num_vertices=3, num_neighbors=3, hidden_dim=4)

    generator = GreedySummaryGenerator(
        model=model,
        embedder=embedder,
        graph_builder=graph_builder,
        budget_limit=200
    )

    # Empty input
    sents, text, logs = generator.generate_summary([])
    assert sents == []
    assert text == ""

    # Single sentence
    s1 = Sentence("Single sentence cluster.", "doc_1", 0, 0, ["single"], ["singl"])
    sents, text, logs = generator.generate_summary([s1])
    assert len(sents) == 1
    assert text == "Single sentence cluster."
