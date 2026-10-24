"""
tests/test_preprocessing.py: Unit tests for text preprocessing, sentence segmentation,
stopword elimination, stemming, and metadata preservation.
"""

import pytest
from src.preprocessing import TextPreprocessor, Sentence


@pytest.fixture
def preprocessor():
    return TextPreprocessor()


def test_sentence_segmentation(preprocessor):
    text = "The president addressed the nation yesterday. He emphasized renewable energy and economic reforms! Will Congress pass the bill?"
    sentences = preprocessor.segment_text(text)
    
    assert len(sentences) == 3
    assert "The president addressed the nation yesterday." in sentences
    assert "He emphasized renewable energy and economic reforms!" in sentences
    assert "Will Congress pass the bill?" in sentences


def test_stopword_removal_and_stemming(preprocessor):
    sentence = "The political leaders are stabilizing their democratic economies."
    clean_tokens, stemmed_tokens = preprocessor.process_sentence_tokens(sentence)
    
    # 'the', 'are', 'their' should be filtered out as stopwords
    assert "the" not in clean_tokens
    assert "are" not in clean_tokens
    assert "their" not in clean_tokens
    
    # Meaningful words must be stemmed
    assert "polit" in stemmed_tokens or "leader" in stemmed_tokens
    assert "stabil" in stemmed_tokens or "economi" in stemmed_tokens


def test_metadata_preservation(preprocessor):
    docs = {
        "doc_A": "First sentence of doc A. Second sentence of doc A.",
        "doc_B": "Sentence of doc B."
    }
    sentences = preprocessor.preprocess_documents(docs)
    
    assert len(sentences) == 3
    # Check document IDs and sentence IDs
    assert sentences[0].doc_id == "doc_A"
    assert sentences[0].sent_id == 0
    assert sentences[0].global_id == 0
    assert sentences[0].raw_text == "First sentence of doc A."

    assert sentences[1].doc_id == "doc_A"
    assert sentences[1].sent_id == 1
    assert sentences[1].global_id == 1

    assert sentences[2].doc_id == "doc_B"
    assert sentences[2].sent_id == 0
    assert sentences[2].global_id == 2
    
    # Verify byte and word length calculations
    assert sentences[0].byte_length > 0
    assert sentences[0].word_count == 5


def test_empty_and_malformed_text_handling(preprocessor):
    # Empty documents, whitespace-only, punctuation-only
    empty_docs = {
        "doc_empty": "",
        "doc_spaces": "   \n\t  \r\n  ",
        "doc_punct": " ... ??? !!! "
    }
    sentences = preprocessor.preprocess_documents(empty_docs)
    assert len(sentences) == 0
