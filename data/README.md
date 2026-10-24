# SGCSumm Dataset Preparation Guide

This folder contains datasets and dataset-loading documentation for **SGCSumm: Submodular Graph Convolutional Summarizer**.

---

## 1. Directory Structure

```text
data/
├── README.md               # Dataset documentation and preparation instructions
├── sample_duc2004/         # Built-in offline smoke-test dataset (simulates DUC 2004)
│   ├── d30001t.json        # Cambodian political crisis cluster (3 docs + gold reference)
│   └── d30002t.json        # Hurricane Mitch disaster cluster (3 docs + gold reference)
└── cache/                  # Disk cache for sentence BERT embeddings
```

---

## 2. Expected Data Format

SGCSumm loads document clusters formatted as JSON files. Each JSON file represents a topic cluster $D = \{d_1, \dots, d_p\}$ with human reference summaries:

```json
{
  "cluster_id": "d30001t",
  "topic": "Topic description / title",
  "documents": {
    "doc_1": "Full raw text of document 1...",
    "doc_2": "Full raw text of document 2...",
    "doc_3": "Full raw text of document 3..."
  },
  "reference_summaries": [
    "Abstractive or extractive reference summary written by human annotators..."
  ]
}
```

---

## 3. Obtaining and Preparing DUC 2004

### About DUC 2004 Task 2
- **Corpus:** Document Understanding Conference (DUC) 2004 Task 2 (Generic Multi-Document Summarization).
- **Structure:** 50 document clusters, each containing approximately 10 news articles on a single topic.
- **Reference:** 4 human-written abstractive summaries per cluster (capped at 665 bytes).
- **Access / Licensing:** DUC datasets are administered by the National Institute of Standards and Technology (NIST). Users must request access via the [NIST Past DUC Materials portal](https://duc.nist.gov/).

### Converting Raw DUC 2004 SGML to SGCSumm JSON
Once you obtain the official DUC 2004 Task 2 files, run the provided utility or preprocess them into the format above:
1. Extract `<TEXT>` contents from each document in the cluster folder.
2. Pair with the 4 reference summaries from `eval/models/2/`.
3. Save each cluster as `{cluster_id}.json` in `data/duc2004/`.

---

## 4. Obtaining and Preparing CNN / DailyMail

### About CNN / DailyMail
- **Corpus:** Single-document news summarization corpus consisting of approximately 100K articles with highlight bullet points.
- **Access:** Available via Hugging Face Datasets (`datasets.load_dataset("cnn_dailymail", "3.0.0")`).
- **Extractive Nature:** Since the highlights are already extractive bullet sentences, they do not require abstractive-to-extractive conversion.

---

## 5. Built-in Smoke-Test Dataset

To verify and train the pipeline immediately on any laptop without downloading multi-gigabyte corpora, SGCSumm automatically provides `sample_duc2004/`.
It contains real news text simulating the DUC 2004 multi-document clusters.
