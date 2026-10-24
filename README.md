# SGCSumm: Submodular Graph Convolutional Summarizer

A complete, faithful, modular, and beginner-friendly PyTorch implementation of the research paper:

> **Alireza Ghadimi and Hamid Beigy.**  
> *"SGCSumm: An extractive multi-document summarization method based on pre-trained language model, submodularity, and graph convolutional neural networks."*  
> **Expert Systems with Applications**, Volume 215, 2023, Article 119308.  
> [https://doi.org/10.1016/j.eswa.2022.119308](https://doi.org/10.1016/j.eswa.2022.119308)

---

## 📑 Table of Contents

1. [Paper Overview & Theoretical Guarantees](#1-paper-overview--theoretical-guarantees)
2. [Architecture & Step-by-Step Pipeline](#2-architecture--step-by-step-pipeline)
3. [Component-to-Code Mapping](#3-component-to-code-mapping)
4. [Windows & VS Code Installation Guide](#4-windows--vs-code-installation-guide)
5. [Quickstart & Verification](#5-quickstart--verification)
6. [Training, Evaluation & Summarization Workflows](#6-training-evaluation--summarization-workflows)
7. [Streamlit Interactive Web Application](#7-streamlit-interactive-web-application)
8. [Dataset Preparation (DUC 2004 & CNN/DailyMail)](#8-dataset-preparation)
9. [Deviations & Implementation Decisions](#9-deviations--implementation-decisions)
10. [Classroom Presentation Guide (Beginner Friendly)](#10-classroom-presentation-guide)
11. [Viva Examination Q&A](#11-viva-examination-qa)

---

## 1. Paper Overview & Theoretical Guarantees

Extractive multi-document summarization (MDS) aims to choose a subset of informative and non-redundant sentences $S^* \subseteq S$ from a cluster of documents $D = \{d_1, \dots, d_p\}$ subject to a length constraint $C$ (such as 665 bytes in the DUC 2004 benchmark):

$$\max_{T \subseteq S} f(T) \quad \text{s.t.} \quad \sum_{t \in T} C(t) \le C$$

This combinatorial subset selection problem is known to be **NP-hard**. However, if the scoring function $f(T)$ is **normalized**, **monotone non-decreasing**, and **submodular**:
- **Normalized:** $f(\emptyset) = 0$
- **Monotone Non-Decreasing:** $f(A) \le f(B)$ whenever $A \subseteq B \subseteq S$
- **Submodular:** $f(A \cup \{s\}) - f(A) \ge f(B \cup \{s\}) - f(B)$ for any $A \subseteq B$ and $s \notin B$ (diminishing marginal returns)

Nemhauser, Wolsey, and Fisher (1978) proved that a greedy selection strategy that iteratively picks the candidate with the highest marginal gain achieves an approximation bound within $(1 - 1/e) \approx 63.2\%$ of the global optimal solution.

**SGCSumm** is the first deep learning summarizer that formally ensures this minimum bound while simultaneously enabling:
1. Contextual sentence embeddings via pretrained BERT.
2. Graph-structure-aware feature learning via Graph Convolutional Networks (GCN).
3. Non-negative scoring via Deep Submodular Networks (DSN).

---

## 2. Architecture & Step-by-Step Pipeline

```mermaid
flowchart TD
    A[Input Documents D] --> B[Text Preprocessing S1, S2, S3]
    B --> C[BERT Embeddings E_s in R^m, Eq. 3]
    C --> D[Hyperspherical Transformation T1, T2: B' in R^{m-1}]
    D --> E[Modular Feature Mapping F: C in R^{m-1}, Eq. 2]
    E --> F[Context Graph G_D & Centrality Ordering l_v, Eq. 5]
    F --> G[Extract Fixed Receptive Fields: X in R^{m-1 x n x k}]
    G --> H[GCN Conv-Pooling Layers: W >= 0, Lemma 1, Thm 4]
    H --> I[Representation Vector Y in R^{m-1}, Remark 1]
    I --> J[Deep Submodular Network DSN: ReLU + Sigmoid]
    J --> K[Submodular Score in 0, 1, Remark 2]
    K --> L[Greedy Marginal Selection under Budget C, Alg 3]
    L --> M[Restore Document Chronological Order]
    M --> N[Final Extractive Summary S*]
```

### Key Mathematical Steps
1. **Contextual Sentence Embeddings (Eq. 3):**
   $$\mathcal{E}(s) = \frac{1}{11} \sum_{k=2}^{12} H_k(s) \in \mathbb{R}^m$$
   Averages token representations from hidden layers 2 through 12 in `bert-base-uncased` ($m = 768$).
2. **Hyperspherical Transformation ($T_1, T_2$):**
   Converts Cartesian vector $A_s$ into hyperspherical coordinates $B_s = (r, \phi_1, \dots, \phi_{m-1})$. The magnitude $r$ is eliminated ($T_2$), leaving $(m - 1)$ non-negative angular features $B'_s \in \mathbb{R}_{\ge 0}^{m-1}$.
3. **Modular Mapping $F$ (Eq. 2):**
   Maps raw angles through $F(T) = \sum_{t \in T} t$. Preserves non-negativity, normalization, monotonicity, and modularity (Theorems 1 & 2).
4. **Context Graph $G_D$ & Pairwise Angular Similarity (Eq. 4):**
   $$\mathcal{S}(s, s') = \frac{1}{m - 1} \sum_{i=1}^{m - 1} \cos(c_{i,s} - c_{i,s'})$$
5. **Graph Centrality Ordering $l(v)$ (Eq. 5):**
   $$l(v) = \text{degreeCentrality}(v) + \text{closenessCentrality}(v) = \frac{|\mathcal{N}(v)|}{\max_u |\mathcal{N}(u)|} + \frac{|V| - 1}{\sum_{v'} \mathcal{D}(v, v')}$$
   Selects top $n = 15$ vertices. For each vertex, selects top $k = 20$ most similar neighbors, forming input tensor $X \in \mathbb{R}^{(m-1) \times n \times k}$.
6. **Constrained GCN Layers (Theorems 3 & 4):**
   - **Layer 1:** $(m - 1)$ 1D filters of size $(1, k)$ over neighbors + ReLU + mean pooling $\to (m - 1) \times n \times 1$.
   - **Layer 2:** $(m - 1)$ 1D filters of size $(n, 1)$ over vertices + ReLU $\to Y \in \mathbb{R}_{\ge 0}^{m-1}$.
   - Weights $W \ge 0$ guarantee submodularity (Lemma 1 & Theorem 4).
7. **Deep Submodular Network (DSN):**
   A 3-layer architecture $(m - 1) \to 16 \to 1$ with ReLU hidden activation, Sigmoid output activation, and strictly non-negative weights $W \ge 0$.
8. **Greedy Marginal Selection (Algorithm 3):**
   Iteratively chooses $s^* = \arg\max_{s \in S \setminus S^*} [ \text{DSN}(Y(S^* \cup \{s\})) - \text{DSN}(Y(S^*)) ]$ until length budget $C$ is exhausted.

---

## 3. Component-to-Code Mapping

| Paper Component | Paper Reference | Implementation File | Key Classes / Functions |
|---|---|---|---|
| Text Preprocessing | Sec 3.1, Alg 1 ($S_1, S_2, S_3$) | [`src/preprocessing.py`](src/preprocessing.py) | `TextPreprocessor`, `Sentence`, `segment_text()`, `process_sentence_tokens()` |
| BERT Sentence Embeddings | Sec 3.1, Eq. (3) | [`src/bert_embeddings.py`](src/bert_embeddings.py) | `BERTSentenceEmbedder`, `MockBERTEmbedder`, `embed_sentences()` |
| Coordinate Transforms | Sec 3.1, Fig. 2, Eq. (2) | [`src/bert_embeddings.py`](src/bert_embeddings.py) | `cartesian_to_hyperspherical()`, `eliminate_first_component()`, `modular_mapping_F()` |
| Context Graph & Centrality | Sec 3.1, Eq. (4), Eq. (5) | [`src/graph_builder.py`](src/graph_builder.py) | `SentenceGraphBuilder`, `compute_pairwise_similarity()`, `compute_labeling_scores()` |
| GCN Representation Learning | Sec 3.1, Thm 3 & 4, Lem 1 | [`src/gcn_model.py`](src/gcn_model.py) | `GCNRepresentationLearner`, `forward()`, `project_non_negative()` |
| Deep Submodular Network | Sec 3.2, Alg 2, Remark 2 | [`src/dsn_model.py`](src/dsn_model.py) | `DeepSubmodularNetwork`, `SGCSummModel`, `project_non_negative()`, `validate_input()` |
| Target Conversion & Augmentation | Sec 3.4, Alg 4 | [`src/dataset.py`](src/dataset.py) | `DocumentCluster`, `convert_to_extractive()`, `augment_cluster_samples()`, `load_dataset_from_dir()` |
| Training Engine & L-BFGS-B | Sec 3.4, Table 4 | [`src/training.py`](src/training.py) | `SGCSummTrainer`, `train_lbfgsb()`, `train_adam_projected()`, `save_checkpoint()` |
| Greedy Summary Generation | Sec 3.3, Alg 3, Alg 5 | [`src/summary_generator.py`](src/summary_generator.py) | `GreedySummaryGenerator`, `generate_summary()`, `compute_subset_score()` |
| Evaluation Metrics | Sec 4.2, Table 6 & 7 | [`src/metrics.py`](src/metrics.py) | `SummarizationMetrics`, `evaluate_pair()`, `evaluate_dataset()` |
| Global Utilities | Utils & Config | [`src/utils.py`](src/utils.py) | `load_config()`, `set_seed()`, `get_device()`, `setup_logger()` |
| Training Entry Point | CLI Driver | [`train.py`](train.py) | `main()` |
| Evaluation Entry Point | CLI Driver | [`evaluate.py`](evaluate.py) | `main()` |
| Summarize Entry Point | CLI Driver | [`summarize.py`](summarize.py) | `main()` |
| Streamlit Web App | Interactive UI | [`app.py`](app.py) | Streamlit dashboard |
| Unit Test Suite | Test Verification | [`tests/`](tests/) | 17 comprehensive unit tests |

---

## 4. Windows & VS Code Installation Guide

### Prerequisites
- Windows 10/11 with Python 3.10 to 3.13 installed.
- VS Code with the official Python extension.
- CPU or integrated graphics (runs comfortably on $\le$ 8 GB RAM).

### Step 1: Open Project in VS Code
Open VS Code, press `Ctrl + Shift + P`, choose **Open Folder...**, and select `d:/NLP Paper/NLP` (or your project directory).

### Step 2: Install Dependencies
Open a PowerShell terminal in VS Code (`Ctrl + ` ` `):

```powershell
pip install -r requirements.txt
```

Verify your installation:
```powershell
python -c "import torch, transformers, networkx, rouge_score, nltk; print('All dependencies successfully installed!')"
```

---

## 5. Quickstart & Verification

Run the entire 17-item test suite to verify all mathematical operations, coordinate transforms, GCN dimensions, DSN bounds, and order preservation:

```powershell
python -m pytest -v
```

**Expected output:**
```text
tests/test_dsn_model.py::test_dsn_architecture_and_output_range PASSED
tests/test_dsn_model.py::test_dsn_input_validation PASSED
tests/test_dsn_model.py::test_dsn_non_decreasing_monotonicity PASSED
tests/test_dsn_model.py::test_unified_sgcsumm_model PASSED
tests/test_gcn_model.py::test_gcn_input_output_shapes PASSED
tests/test_gcn_model.py::test_gcn_non_negative_weights_and_output PASSED
tests/test_graph_builder.py::test_hyperspherical_transformation_and_non_negativity PASSED
tests/test_graph_builder.py::test_pairwise_similarity_matrix PASSED
tests/test_graph_builder.py::test_top_vertices_and_neighborhoods_selection PASSED
tests/test_graph_builder.py::test_graph_tensor_dimensions_and_padding_for_small_clusters PASSED
tests/test_preprocessing.py::test_sentence_segmentation PASSED
tests/test_preprocessing.py::test_stopword_removal_and_stemming PASSED
tests/test_preprocessing.py::test_metadata_preservation PASSED
tests/test_preprocessing.py::test_empty_and_malformed_text_handling PASSED
tests/test_summary_generator.py::test_greedy_selection_and_budget_enforcement PASSED
tests/test_summary_generator.py::test_order_preservation PASSED
tests/test_summary_generator.py::test_empty_and_single_sentence_handling PASSED
======================= 17 passed in 20s =======================
```

---

## 6. Training, Evaluation & Summarization Workflows

### A. Training on CPU Smoke-Test Dataset

Train the model on the built-in smoke-test dataset using L-BFGS-B (or Adam projected):

```powershell
# Fast offline training with mock embedder
python train.py --epochs 100 --use_mock

# Or train with real Hugging Face BERT embeddings
python train.py --epochs 100
```

Trained checkpoints are automatically saved to `./models/sgcsumm_best.pt`.

### B. Evaluating on Benchmark Dataset

Evaluate model summaries against gold references and view ROUGE-1, ROUGE-2, and ROUGE-L scores compared to published paper baselines:

```powershell
python evaluate.py --use_mock
```

**Sample Output from Actual Execution:**
```text
================================================================================
SGCSumm EVALUATION RESULTS PER DOCUMENT CLUSTER
================================================================================

[Cluster 1/2] ID: d30001t
Total Sentences in Pool: 7 | Selected: 1
Summary Length: 72 bytes (11 words)
--------------------------------------------------------------------------------
Generated Summary:
  Cambodian political leaders gathered in Phnom Penh for a special summit.
--------------------------------------------------------------------------------
Reference Summary:
  Cambodian political rivals met in Phnom Penh under King Norodom Sihanouk...
--------------------------------------------------------------------------------
Scores -> ROUGE-1 Recall: 20.59% | ROUGE-2 Recall: 9.09% | ROUGE-L F1: 26.67%

================================================================================
OVERALL DATASET EVALUATION SUMMARY
================================================================================
Number of Clusters Evaluated: 2
Budget Constraint: 665 bytes
ROUGE-1 -> Recall: 18.99% | F1: 27.32%
ROUGE-2 -> Recall: 6.82% | F1: 10.10%
ROUGE-L -> Recall: 17.52% | F1: 25.10%
================================================================================

================================================================================
BENCHMARK COMPARISON TABLE (Paper Table 6: DUC 2004 Test Set)
================================================================================
Method / System                     | ROUGE-1    | ROUGE-2   
--------------------------------------------------------------------------------
SGCSumm (Proposed in Paper)         | 40.66      | 10.41     
DSNSum (Ghadimi & Beigy, 2020)      | 40.28      | 10.23     
DPP (Kulesza & Taskar, 2012)        | 39.79      | 9.62      
Submodular (Lin & Bilmes, 2011)     | 39.18      | 9.35      
Centroid + G + new-tf-idf           | 39.11      | 9.81      
R2N2-ILP (Cao et al., 2015)         | 38.78      | 9.86      
Centroid + G                        | 38.55      | 9.73      
ICSISumm (Gillick et al., 2009)     | 38.41      | 9.78      
GRU+GCN (Yasunaga et al., 2017)     | 38.23      | 9.48      
LexRank (Erkan & Radev, 2004)       | 37.86      | 9.67      
--------------------------------------------------------------------------------
Current Run (Smoke-test Subset)     | 18.99      | 6.82      
================================================================================
```

### C. Summarizing Arbitrary Documents

Summarize custom `.txt` files directly from the command line:

```powershell
# Summarize files in a directory with a 665-byte budget
python summarize.py --input_dir path/to/articles/ --budget 665 --budget_type byte --use_mock

# Or summarize with a word-count limit
python summarize.py --budget 100 --budget_type word --use_mock
```

---

## 7. Streamlit Interactive Web Application

Launch the interactive graphical interface:

```powershell
python -m streamlit run app.py
```

### Web App Features
- **Upload multiple `.txt` articles** or choose the pre-loaded multi-document cluster.
- **Inspect source documents** side-by-side in tabbed views.
- **Adjust parameters:** Switch between byte budget (e.g. 665 bytes) and word budget (e.g. 100 words).
- **Inspect marginal gain breakdown table:** Shows exactly which sentence was chosen at each greedy step, its marginal score improvement $\Delta(s | S^*)$, and cumulative score.
- **Export summary:** Download the resulting summary directly as a `.txt` file.

---

## 8. Dataset Preparation (DUC 2004 & CNN/DailyMail)

See [`data/README.md`](data/README.md) for full details on:
1. Accessing DUC 2004 Task 2 via NIST.
2. Loading CNN/DailyMail via Hugging Face.
3. Preprocessing raw documents into JSON cluster files with gold references.

---

## 9. Deviations & Implementation Decisions

To maintain scientific integrity, the following implementation decisions are explicitly documented:

1. **Hyperspherical Angle Mapping:**
   - *Paper detail:* The paper states that hyperspherical angles in $B_s$ are all non-negative.
   - *Implementation decision:* In $m$-dimensional spherical coordinate systems, the polar angles $\phi_1, \dots, \phi_{m-2}$ lie in $[0, \pi]$, while the azimuthal angle $\phi_{m-1}$ lies in $[-\pi, \pi]$ or $[0, 2\pi)$. We map the azimuthal angle to $[0, 2\pi)$ via $\text{atan2}(y, x) \pmod{2\pi}$, ensuring that every component of $B_s$ is strictly non-negative, fulfilling Theorem 1 without approximation.
2. **Disconnected Context Graph Handling:**
   - *Paper detail:* Eq. (5) closeness centrality divides by $\sum_{v'} \mathcal{D}(v, v')$. If a graph is disconnected, $\mathcal{D}(v, v') = \infty$.
   - *Implementation decision:* We construct a fully connected weighted similarity graph where edge distance is $d(u, v) = \max(10^{-5}, 1 - \mathcal{S}(u, v))$, and apply harmonic scaling for disconnected components so that isolated sentences receive zero centrality rather than crashing with division by zero.
3. **Hidden Layer Size in DSN:**
   - *Paper detail:* Page 9 mentions 18 neurons in passing, but Table 4 explicitly specifies `Chosen value: 16`.
   - *Implementation decision:* We use `16` hidden units by default per Table 4, fully configurable in `config.yaml`.
4. **Optimization Engine:**
   - *Paper detail:* Section 3.4 cites L-BFGS-B (Byrd et al., 1995) to enforce $W \ge 0$.
   - *Implementation decision:* We provide both:
     1. Exact SciPy `L-BFGS-B` with `bounds=[(0, None), ...]`, faithfully matching the paper.
     2. PyTorch Adam with projected gradient descent (`project_non_negative()`), allowing accelerated training on both CPU and CUDA.
5. **Offline Development & Memory Safeguards:**
   - To accommodate 8 GB RAM and offline setups without requiring repeated downloads of 440 MB BERT checkpoints, an optional deterministic `MockBERTEmbedder` is provided alongside full Hugging Face BERT.

---

## 10. Classroom Presentation Guide

### Slide 1: The Problem of Multi-Document Summarization
- Massive volumes of news and scientific articles cover the same event from different angles.
- Extractive summarization selects the best representative sentences to construct a summary.
- Challenge: How do we select sentences without redundancy while ensuring we didn't just get lucky with our choice? (The subset selection problem is NP-hard).

### Slide 2: Why Submodularity Matters
- Submodularity is the mathematical formalization of **diminishing marginal returns**: adding a sentence to a small draft adds more new information than adding it to an already detailed summary.
- **The Guarantee:** Maximizing a monotone submodular function with a simple greedy algorithm guarantees at least **$(1 - 1/e) \approx 63.2\%$** of the optimal possible score (Nemhauser et al., 1978).
- Most modern deep learning models cannot offer this guarantee. SGCSumm does.

### Slide 3: The 3 Core Pillars of SGCSumm
1. **BERT Contextual Embeddings:** Captures deep language context by averaging hidden layers 2 through 12.
2. **Graph Convolutional Networks (GCN):** Models relationships between sentences via a context graph and learns structural features.
3. **Deep Submodular Network (DSN):** Scores sentences using constrained neural networks where weights $W \ge 0$ and activations are concave (ReLU, Sigmoid).

### Slide 4: Results & Conclusion
- On the standard **DUC 2004** benchmark, SGCSumm achieves **40.66 ROUGE-1** and **10.41 ROUGE-2**, outperforming previous graph and deep learning baselines (DSNSum, GRU+GCN, LexRank).
- Demonstrates that deep learning and formal submodular mathematical guarantees can be unified successfully.

---

## 11. Viva Examination Q&A

### Q1: What is submodularity, and why is it important in extractive summarization?
**Answer:** A set function $f: 2^V \to \mathbb{R}$ is submodular if for all $A \subseteq B \subseteq V$ and $s \notin B$, $f(A \cup \{s\}) - f(A) \ge f(B \cup \{s\}) - f(B)$. In summarization, this represents diminishing marginal returns: adding a sentence provides more novelty to a shorter summary than to a longer one. It allows an NP-hard subset selection problem to be solved within a provable $(1 - 1/e) \approx 63.2\%$ approximation factor using a greedy algorithm.

### Q2: Why does SGCSumm convert Cartesian BERT embeddings to hyperspherical coordinates?
**Answer:** Pretrained BERT embeddings contain arbitrary negative and positive values. Deep Submodular Networks require inputs to be strictly non-negative, normalized, and modular. Converting to hyperspherical coordinates expresses the vector as a magnitude and $(m - 1)$ angles. Discarding the magnitude leaves non-negative angular features in $[0, \pi]$ and $[0, 2\pi)$, which can then be mapped through modular function $F(T) = \sum t$ without violating submodularity conditions.

### Q3: How does the GCN preserve submodularity in Theorem 4?
**Answer:** By Lemma 1, mean pooling over modular inputs preserves submodularity. Since the convolutional filter weights are constrained to be non-negative ($W \ge 0$) and the activation function is concave and non-decreasing (ReLU), the composition of non-negative linear mappings, concave activations, and mean pooling preserves submodularity throughout both GCN layers.

### Q4: Why does SGCSumm use L-BFGS-B instead of standard SGD or Adam during training?
**Answer:** L-BFGS-B is a quasi-Newton optimization algorithm that supports simple box constraints on variables. It allows direct enforcement of the non-negativity constraint $w_{ij} \in [0, \infty)$ on all network weights during training with low memory overhead.

### Q5: How are abstractive gold reference summaries converted into extractive training targets?
**Answer:** Using the greedy ROUGE-1 selection method of Nallapati et al. (2017) (SummaRuNNer). Sentences from the source document cluster are greedily added one by one to an extractive set to maximize ROUGE-1 recall against the human reference summary.

### Q6: Why can't we simply select the top-$K$ highest scoring sentences instead of using greedy marginal selection?
**Answer:** Selecting the top-$K$ individual sentences ignores pairwise sentence redundancy. Two sentences might each have high individual scores but contain identical information. Algorithm 3 evaluates marginal gain $\Delta(s | S^*) = \text{score}(S^* \cup \{s\}) - \text{score}(S^*)$. If sentence $s$ repeats information already captured in $S^*$, the submodular scoring function produces diminishing marginal gains, naturally penalizing redundancy.

---

## 12. License & Citation

This implementation is intended for academic and educational research purposes. If you use this implementation in your research, please cite the original paper:

```bibtex
@article{ghadimi2023sgcsumm,
  title={SGCSumm: An extractive multi-document summarization method based on pre-trained language model, submodularity, and graph convolutional neural networks},
  author={Ghadimi, Alireza and Beigy, Hamid},
  journal={Expert Systems with Applications},
  volume={215},
  pages={119308},
  year={2023},
  publisher={Elsevier}
}
```
