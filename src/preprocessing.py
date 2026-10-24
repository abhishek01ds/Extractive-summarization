"""
src/preprocessing.py: Text preprocessing for SGCSumm.
Implements:
1. Document loading & sentence segmentation (S1)
2. Stopword removal (S2)
3. Stemming (S3)
4. Preservation of original sentence text and document/sentence metadata.
5. Robust handling of edge cases (empty documents, excessive whitespace, punctuation-only lines).
"""

from dataclasses import dataclass, field
import re
from typing import List, Dict, Optional, Union
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


@dataclass
class Sentence:
    """
    Data structure representing a segmented sentence with metadata.
    Preserves original text for final summary generation while maintaining
    cleaned and stemmed representations for similarity and graph modeling.
    """
    raw_text: str
    doc_id: str
    sent_id: int
    global_id: int
    clean_tokens: List[str] = field(default_factory=list)
    stemmed_tokens: List[str] = field(default_factory=list)

    @property
    def byte_length(self) -> int:
        """Returns byte length of raw sentence in UTF-8."""
        return len(self.raw_text.encode("utf-8"))

    @property
    def word_count(self) -> int:
        """Returns word count of raw sentence."""
        return len(self.raw_text.split())

    def __repr__(self) -> str:
        return f"<Sentence doc={self.doc_id} sent={self.sent_id} words={self.word_count}>"


class TextPreprocessor:
    """
    Preprocessor implementing Algorithm 1 text preparation:
    - segmentSentences: S0 -> S1
    - removeStopWords: S1 -> S2
    - stemWords: S2 -> S3
    """
    def __init__(self, language: str = "english"):
        self.language = language
        self._ensure_nltk_resources()
        self.stop_words = set(stopwords.words(self.language))
        self.stemmer = PorterStemmer()

    @staticmethod
    def _ensure_nltk_resources():
        """Ensure necessary NLTK datasets are downloaded."""
        for resource in ["punkt", "punkt_tab", "stopwords"]:
            try:
                nltk.data.find(f"tokenizers/{resource}" if "punkt" in resource else f"corpora/{resource}")
            except (LookupError, OSError):
                try:
                    nltk.download(resource, quiet=True)
                except Exception:
                    pass

    def segment_text(self, text: str) -> List[str]:
        """
        Split a raw text string into a list of sentence strings.
        Handles irregular whitespaces and common abbreviations safely.
        """
        if not text or not text.strip():
            return []
            
        # Normalize carriage returns and excess whitespace
        clean_text = re.sub(r"\r\n|\r", "\n", text)
        clean_text = re.sub(r"\n{2,}", "\n", clean_text)
        
        try:
            raw_sentences = nltk.sent_tokenize(clean_text, language=self.language)
        except Exception:
            # Fallback regex splitter if NLTK tokenizer fails
            raw_sentences = re.split(r"(?<=[.!?])\s+", clean_text)

        # Filter out trivial sentences (empty or whitespace only)
        valid_sentences = []
        for s in raw_sentences:
            s_stripped = s.strip()
            # Must contain at least one alphanumeric character
            if any(char.isalnum() for char in s_stripped):
                valid_sentences.append(s_stripped)

        return valid_sentences

    def process_sentence_tokens(self, sentence_text: str) -> tuple[List[str], List[str]]:
        """
        Tokenize sentence, remove stopwords (S2), and stem words (S3).
        
        Returns:
            (clean_tokens, stemmed_tokens)
        """
        # Tokenize by alphanumeric words
        words = re.findall(r"\b\w+\b", sentence_text.lower())
        
        # Stopword removal
        clean_tokens = [w for w in words if w not in self.stop_words and not w.isnumeric()]
        
        # Stemming (Porter stemmer for verbs and nouns)
        stemmed_tokens = [self.stemmer.stem(w) for w in clean_tokens]
        
        return clean_tokens, stemmed_tokens

    def preprocess_documents(
        self,
        documents: Union[List[str], Dict[str, str]]
    ) -> List[Sentence]:
        """
        Preprocess a collection of documents belonging to a topic cluster.
        
        Args:
            documents: List of document text strings OR Dict of {doc_id: doc_text}.
            
        Returns:
            List of structured Sentence objects preserving document order and global IDs.
        """
        sentences: List[Sentence] = []
        global_idx = 0

        if isinstance(documents, list):
            doc_items = [(f"doc_{idx}", doc_text) for idx, doc_text in enumerate(documents)]
        elif isinstance(documents, dict):
            doc_items = list(documents.items())
        else:
            raise ValueError("Input documents must be a list of strings or dict of {doc_id: text}")

        for doc_id, doc_text in doc_items:
            if not doc_text or not doc_text.strip():
                continue

            raw_sents = self.segment_text(doc_text)
            for sent_idx, raw_sent in enumerate(raw_sents):
                clean_toks, stemmed_toks = self.process_sentence_tokens(raw_sent)
                
                # Note: Even if all words were stopwords (e.g. "To be or not to be"),
                # we keep the sentence if it has text, but provide fallback tokens
                if not stemmed_toks:
                    stemmed_toks = [w.lower() for w in re.findall(r"\b\w+\b", raw_sent)]
                    
                sentence_obj = Sentence(
                    raw_text=raw_sent,
                    doc_id=doc_id,
                    sent_id=sent_idx,
                    global_id=global_idx,
                    clean_tokens=clean_toks,
                    stemmed_tokens=stemmed_toks
                )
                sentences.append(sentence_obj)
                global_idx += 1

        return sentences
