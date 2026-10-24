"""
app.py: Streamlit Multi-Model Summarization Benchmarking Dashboard.
Enables side-by-side comparison of three distinct summarization paradigms:
1. Statistical Baseline (TF-IDF)
2. Deep Learning Baseline (BERT Contextual Semantic Centroid + MMR)
3. SGCSumm Model (Submodular Graph Convolutional Neural Network with (1 - 1/e) guarantee)
"""

import os
import time
import streamlit as st
import numpy as np
import torch
import pandas as pd

from src.utils import get_device
from src.preprocessing import TextPreprocessor
from src.bert_embeddings import BERTSentenceEmbedder
from src.graph_builder import SentenceGraphBuilder
from src.dsn_model import SGCSummModel
from src.summary_generator import GreedySummaryGenerator
from src.baselines import TfidfSummarizer, DeepLearningSummarizer

# ==========================================
# Page Configuration & Modern Styling
# ==========================================
st.set_page_config(
    page_title="Summarization Benchmark: TF-IDF vs Deep Learning vs SGCSumm",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern glassmorphism & presentation styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(120deg, #1E3A8A, #3B82F6, #10B981);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .model-card-tfidf {
        border-top: 4px solid #3B82F6;
        background-color: rgba(59, 130, 246, 0.04);
        padding: 1.2rem;
        border-radius: 0.6rem;
        min-height: 280px;
    }
    .model-card-dl {
        border-top: 4px solid #8B5CF6;
        background-color: rgba(139, 92, 246, 0.04);
        padding: 1.2rem;
        border-radius: 0.6rem;
        min-height: 280px;
    }
    .model-card-sgc {
        border-top: 4px solid #10B981;
        background-color: rgba(16, 185, 129, 0.04);
        padding: 1.2rem;
        border-radius: 0.6rem;
        min-height: 280px;
    }
    .badge {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        font-size: 0.75rem;
        font-weight: 600;
        border-radius: 9999px;
        margin-bottom: 0.8rem;
    }
    .badge-tfidf { background-color: #DBEAFE; color: #1E40AF; }
    .badge-dl { background-color: #EDE9FE; color: #5B21B6; }
    .badge-sgc { background-color: #D1FAE5; color: #065F46; }
    .metric-row {
        display: flex;
        gap: 12px;
        font-size: 0.85rem;
        color: #6B7280;
        margin-top: 0.8rem;
        border-top: 1px dashed rgba(0,0,0,0.1);
        padding-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# Sample Text Demos for Quick Presentation
# ==========================================
SAMPLE_ARTICLES = {
    "Artificial Intelligence in Medicine": (
        "Artificial intelligence is transforming modern healthcare through advanced diagnostic precision and predictive analytics. "
        "Deep learning models trained on millions of radiology scans can now detect early-stage malignant tumors faster than human specialists. "
        "Furthermore, hospital networks are implementing natural language processing systems to automate clinical documentation and patient triage. "
        "However, medical ethicists warn that algorithmic bias and black-box decision making remain substantial barriers to regulatory approval. "
        "Clinicians emphasize that artificial intelligence should act as an assistive co-pilot rather than an autonomous replacement for physicians. "
        "Recent clinical trials demonstrate that human-AI collaboration yields significantly higher diagnostic accuracy than either operating alone. "
        "Governments worldwide are establishing strict validation standards to guarantee patient safety, data privacy, and algorithm transparency."
    ),
    "Climate Change & Renewable Transition": (
        "The global transition toward renewable energy has accelerated rapidly in response to escalating climate disruptions. "
        "Solar photovoltaic efficiency has doubled over the past decade while production costs plummeted by more than eighty percent. "
        "Grid operators face significant integration challenges due to the intermittent nature of wind and solar power generation. "
        "Advanced grid-scale lithium-iron-phosphate and solid-state batteries are emerging as vital solutions for long-duration energy storage. "
        "Simultaneously, international energy agencies highlight that existing transmission infrastructure requires trillions in modernization investments. "
        "Decarbonizing industrial manufacturing and heavy aviation remains difficult without clean hydrogen and carbon capture technologies. "
        "Policymakers stress that comprehensive carbon pricing and cross-border cooperation are essential to achieve net-zero targets."
    ),
    "Deep Space Exploration & Astronomy": (
        "Astronomers analyzing infrared data from next-generation space telescopes have identified unexpectedly massive galaxies in the early universe. "
        "These cosmic structures formed merely hundreds of millions of years following the Big Bang, challenging prevailing cosmological theories. "
        "Spectroscopic observations reveal rich chemical compositions including carbon, oxygen, and silicate dust much earlier than previously thought. "
        "The scientific community is reconsidering initial stellar formation rates and the evolutionary mechanisms of supermassive black holes. "
        "Future orbital observatories scheduled for the next decade will probe exoplanetary atmospheres for distinct biosignatures. "
        "International space consortiums continue expanding robotic rover missions on Mars to search for subsurface microbial biosignatures."
    )
}

# ==========================================
# Sidebar Settings
# ==========================================
st.sidebar.title("⚙️ Benchmark Controls")

st.sidebar.subheader("🎯 Summary Length Strategy")
length_mode = st.sidebar.radio(
    "Length Constraint Mode",
    ["Percentage of Original Text (Rule)", "Fixed Word / Byte Budget"],
    index=0,
    help="Percentage rule ensures summary is proportionally sized to the input document (e.g. 30% of original words)."
)

if length_mode == "Percentage of Original Text (Rule)":
    target_pct = st.sidebar.slider(
        "Summary Ratio (X% of Original Words)",
        min_value=15,
        max_value=60,
        value=30,
        step=5,
        help="Summary will contain approximately X% of the original word count."
    )
    target_ratio = target_pct / 100.0
    budget_type = "word"
    budget_limit = 200
else:
    target_ratio = None
    target_pct = None
    budget_type = st.sidebar.selectbox("Budget Constraint Metric", ["word", "byte"], index=0)
    default_budget = 75 if budget_type == "word" else 500
    budget_limit = st.sidebar.slider(
        f"Budget Limit ({budget_type.capitalize()}s)",
        min_value=30 if budget_type == "word" else 200,
        max_value=250 if budget_type == "word" else 1500,
        value=default_budget,
        step=10 if budget_type == "word" else 50,
        help="Target maximum length for all three summarization models."
    )

min_sentences = st.sidebar.slider(
    "Minimum Sentences Guarantee",
    min_value=1,
    max_value=5,
    value=2,
    help="Guarantees that at least this many key sentences are selected, preventing premature 1-sentence termination."
)

st.sidebar.subheader("🧠 Model Engine")
embedding_mode = st.sidebar.radio(
    "Embedding Backend",
    ["Fast Mock (Instant / Offline)", "Pretrained BERT (bert-base-uncased)"],
    index=0,
    help="Fast Mock runs deterministic embeddings in milliseconds without network downloads. Pretrained BERT uses full 768-dim contextual representations."
)
use_mock = (embedding_mode == "Fast Mock (Instant / Offline)")

checkpoint_path = st.sidebar.text_input("SGCSumm Checkpoint", value="./models/sgcsumm_best.pt")

st.sidebar.markdown("---")
st.sidebar.markdown(
    r"""### 🔬 Model Comparison Overview
- **1. TF-IDF (Statistical):** Frequency-weighted word relevance. Fast & interpretable, but lacks word semantics.
- **2. Deep Learning (BERT + MMR):** Deep contextual embeddings with cosine centroid relevance and diversity penalty.
- **3. SGCSumm (Proposed):** GCN graph topology + Deep Submodular Network guaranteeing a $(1 - 1/e) \approx 63.2\%$ approximation bound.
"""
)

# ==========================================
# Main Header
# ==========================================
st.markdown('<div class="main-header">⚡ Multi-Model Summarization Comparison Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Input any text and compare <b>TF-IDF</b> (Statistical Baseline), <b>Deep Learning</b> (BERT Semantic Centroid), and <b>SGCSumm</b> (Submodular GCN) side-by-side.</div>', unsafe_allow_html=True)

# Quick sample loader buttons
st.write("**Quick Demo: Populate with sample text**")
preset_cols = st.columns([1, 1, 1, 0.6])
if "input_text_area" not in st.session_state:
    st.session_state["input_text_area"] = SAMPLE_ARTICLES["Artificial Intelligence in Medicine"]

with preset_cols[0]:
    if st.button("🏥 AI in Healthcare", width="stretch"):
        st.session_state["input_text_area"] = SAMPLE_ARTICLES["Artificial Intelligence in Medicine"]
        st.rerun()

with preset_cols[1]:
    if st.button("🌱 Climate & Energy", width="stretch"):
        st.session_state["input_text_area"] = SAMPLE_ARTICLES["Climate Change & Renewable Transition"]
        st.rerun()

with preset_cols[2]:
    if st.button("🚀 Deep Space Astronomy", width="stretch"):
        st.session_state["input_text_area"] = SAMPLE_ARTICLES["Deep Space Exploration & Astronomy"]
        st.rerun()

with preset_cols[3]:
    if st.button("🧹 Clear", width="stretch"):
        st.session_state["input_text_area"] = ""
        st.rerun()

# Dedicated Text Input Area
user_text = st.text_area(
    "📝 Write or paste your article / text below:",
    value=st.session_state.get("input_text_area", ""),
    height=200,
    placeholder="Type or paste the paragraphs you want summarized here..."
)

# Input text live statistics & rule indicator
if user_text.strip():
    words = len(user_text.split())
    chars = len(user_text)
    preprocessor = TextPreprocessor()
    segmented = preprocessor.segment_text(user_text)
    sent_count = len(segmented)
    
    if target_ratio is not None:
        target_w = max(20, int(words * target_ratio))
        rule_info = f" • 🎯 **Target Length Rule:** **{target_pct}% of original text** (≈ **{target_w} words**, min **{min_sentences} sentences**)"
    else:
        rule_info = f" • 🎯 **Target Length Rule:** Fixed Budget (max **{budget_limit} {budget_type}s**, min **{min_sentences} sentences**)"

    st.caption(f"📊 **Input Stats:** {words} words | {chars} characters | {sent_count} sentences{rule_info}")
else:
    st.caption("📊 **Input Stats:** 0 words | 0 characters | 0 sentences")

st.markdown("<br>", unsafe_allow_html=True)

# Execution Action Button
run_summarization = st.button("🚀 Generate & Compare All 3 Summaries", type="primary", width="stretch")

# ==========================================
# Cached Model Initialization
# ==========================================
@st.cache_resource(show_spinner=False)
def load_embedder(mock_mode: bool):
    return BERTSentenceEmbedder(
        model_name="bert-base-uncased",
        embedding_dim=768,
        use_cache=True,
        cache_dir="./data/cache",
        use_mock=mock_mode,
        device="cpu"
    )

@st.cache_resource(show_spinner=False)
def load_sgcsumm_model(ckpt_path: str):
    model = SGCSummModel(
        num_channels=767,
        num_vertices=15,
        num_neighbors=20,
        hidden_dim=16
    )
    if os.path.exists(ckpt_path):
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu")
            model.load_state_dict(ckpt["model_state_dict"])
            model.project_non_negative()
        except Exception:
            pass
    return model

# ==========================================
# Run Summarization Pipeline
# ==========================================
if run_summarization:
    if not user_text.strip():
        st.warning("⚠️ Please enter or paste some text before clicking Generate.")
        st.stop()

    with st.spinner("Preprocessing sentences, computing embeddings, and running all 3 summarizers..."):
        preprocessor = TextPreprocessor()
        sentences = preprocessor.preprocess_documents({"user_doc": user_text})

        if not sentences or len(sentences) < 1:
            st.error("No valid sentences could be extracted from your input. Please provide complete sentences with punctuation.")
            st.stop()

        # Shared embedder
        embedder = load_embedder(use_mock)

        # ------------------------------------
        # 1. TF-IDF Summarizer
        # ------------------------------------
        t0 = time.time()
        tfidf_model = TfidfSummarizer(budget_type=budget_type, budget_limit=int(budget_limit))
        tfidf_sents, tfidf_summary, tfidf_logs = tfidf_model.summarize(
            sentences, min_sentences=min_sentences, target_ratio=target_ratio
        )
        t_tfidf = (time.time() - t0) * 1000

        # ------------------------------------
        # 2. Deep Learning (BERT Centroid + MMR)
        # ------------------------------------
        t0 = time.time()
        dl_model = DeepLearningSummarizer(embedder=embedder, budget_type=budget_type, budget_limit=int(budget_limit))
        dl_sents, dl_summary, dl_logs = dl_model.summarize(
            sentences, min_sentences=min_sentences, target_ratio=target_ratio
        )
        t_dl = (time.time() - t0) * 1000

        # ------------------------------------
        # 3. SGCSumm (Submodular GCN Model)
        # ------------------------------------
        t0 = time.time()
        sgc_model = load_sgcsumm_model(checkpoint_path)
        graph_builder = SentenceGraphBuilder(top_n=15, top_k=20)
        generator = GreedySummaryGenerator(
            model=sgc_model,
            embedder=embedder,
            graph_builder=graph_builder,
            budget_type=budget_type,
            budget_limit=int(budget_limit),
            restore_order=True,
            device="cpu"
        )
        sgc_sents, sgc_summary, sgc_logs = generator.generate_summary(
            sentences, min_sentences=min_sentences, target_ratio=target_ratio
        )
        t_sgc = (time.time() - t0) * 1000

    # Store in session state for tabs / downloads
    st.session_state["results"] = {
        "sentences": sentences,
        "tfidf": {"summary": tfidf_summary, "sents": tfidf_sents, "time_ms": t_tfidf, "logs": tfidf_logs},
        "dl": {"summary": dl_summary, "sents": dl_sents, "time_ms": t_dl, "logs": dl_logs},
        "sgc": {"summary": sgc_summary, "sents": sgc_sents, "time_ms": t_sgc, "logs": sgc_logs},
    }

# Display results if available
if "results" in st.session_state and st.session_state["results"]:
    res = st.session_state["results"]
    orig_words = max(1, len(user_text.split()))

    st.markdown("---")
    st.subheader("🎯 Side-by-Side Model Summaries")

    col1, col2, col3 = st.columns(3)

    # ------------------ Column 1: TF-IDF ------------------
    with col1:
        st.markdown('<span class="badge badge-tfidf">1. Statistical Baseline</span>', unsafe_allow_html=True)
        st.markdown("### 📊 TF-IDF Summarizer")
        
        summ1 = res["tfidf"]["summary"]
        w1 = len(summ1.split())
        s1 = len(res["tfidf"]["sents"])
        t1 = res["tfidf"]["time_ms"]
        ratio1 = (w1 / orig_words) * 100

        st.markdown(f'<div class="model-card-tfidf">{summ1 if summ1 else "<i>No summary generated</i>"}</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="metric-row">'
            f'<span>⏱️ <b>{t1:.1f} ms</b></span>'
            f'<span>📝 <b>{w1} words</b> ({s1} sents)</span>'
            f'<span>📉 <b>{ratio1:.1f}% size</b></span>'
            f'</div>',
            unsafe_allow_html=True
        )

    # ------------------ Column 2: Deep Learning ------------------
    with col2:
        st.markdown('<span class="badge badge-dl">2. Deep Contextual Neural</span>', unsafe_allow_html=True)
        st.markdown("### 🧠 Deep Learning (BERT)")
        
        summ2 = res["dl"]["summary"]
        w2 = len(summ2.split())
        s2 = len(res["dl"]["sents"])
        t2 = res["dl"]["time_ms"]
        ratio2 = (w2 / orig_words) * 100

        st.markdown(f'<div class="model-card-dl">{summ2 if summ2 else "<i>No summary generated</i>"}</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="metric-row">'
            f'<span>⏱️ <b>{t2:.1f} ms</b></span>'
            f'<span>📝 <b>{w2} words</b> ({s2} sents)</span>'
            f'<span>📉 <b>{ratio2:.1f}% size</b></span>'
            f'</div>',
            unsafe_allow_html=True
        )

    # ------------------ Column 3: SGCSumm ------------------
    with col3:
        st.markdown('<span class="badge badge-sgc">3. Paper Implementation</span>', unsafe_allow_html=True)
        st.markdown("### 🏆 SGCSumm (Submodular GCN)")
        
        summ3 = res["sgc"]["summary"]
        w3 = len(summ3.split())
        s3 = len(res["sgc"]["sents"])
        t3 = res["sgc"]["time_ms"]
        ratio3 = (w3 / orig_words) * 100

        st.markdown(f'<div class="model-card-sgc">{summ3 if summ3 else "<i>No summary generated</i>"}</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="metric-row">'
            f'<span>⏱️ <b>{t3:.1f} ms</b></span>'
            f'<span>📝 <b>{w3} words</b> ({s3} sents)</span>'
            f'<span>📉 <b>{ratio3:.1f}% size</b></span>'
            f'</div>',
            unsafe_allow_html=True
        )

    # ==========================================
    # Audience Comparison & Analysis
    # ==========================================
    st.markdown("---")
    st.subheader("📊 Comparative Benchmark & Audience Analytics")

    tab_summary, tab_consensus, tab_details = st.tabs([
        "📈 Metric Comparison Table",
        "🤝 Sentence Consensus & Overlap",
        "🔍 Sentence Selection Breakdown"
    ])

    with tab_summary:
        comparison_df = pd.DataFrame([
            {
                "Model Paradigm": "TF-IDF (Statistical Baseline)",
                "Approach": "Frequency Weights (Bag of Words)",
                "Theoretical Guarantee": "None (Heuristic Ranking)",
                "Sentences Selected": len(res["tfidf"]["sents"]),
                "Summary Words": len(res["tfidf"]["summary"].split()),
                "Inference Latency": f"{res['tfidf']['time_ms']:.1f} ms",
                "Compression Ratio": f"{(len(res['tfidf']['summary'].split()) / orig_words)*100:.1f}%"
            },
            {
                "Model Paradigm": "Deep Learning (BERT Semantic Centroid)",
                "Approach": "Contextual Embedding Centroid + MMR",
                "Theoretical Guarantee": "Heuristic MMR Diversity",
                "Sentences Selected": len(res["dl"]["sents"]),
                "Summary Words": len(res["dl"]["summary"].split()),
                "Inference Latency": f"{res['dl']['time_ms']:.1f} ms",
                "Compression Ratio": f"{(len(res['dl']['summary'].split()) / orig_words)*100:.1f}%"
            },
            {
                "Model Paradigm": "SGCSumm (Proposed Method)",
                "Approach": "GCN Graph Topology + Deep Submodular Network",
                "Theoretical Guarantee": "Greedy (1 - 1/e) ~ 63.2% Optimal",
                "Sentences Selected": len(res["sgc"]["sents"]),
                "Summary Words": len(res["sgc"]["summary"].split()),
                "Inference Latency": f"{res['sgc']['time_ms']:.1f} ms",
                "Compression Ratio": f"{(len(res['sgc']['summary'].split()) / orig_words)*100:.1f}%"
            }
        ])
        st.table(comparison_df)

    with tab_consensus:
        st.markdown("**Which sentences were chosen by each model?**")
        all_sents = res["sentences"]
        tfidf_texts = set(s.raw_text for s in res["tfidf"]["sents"])
        dl_texts = set(s.raw_text for s in res["dl"]["sents"])
        sgc_texts = set(s.raw_text for s in res["sgc"]["sents"])

        consensus_records = []
        for i, s in enumerate(all_sents):
            in_tfidf = s.raw_text in tfidf_texts
            in_dl = s.raw_text in dl_texts
            in_sgc = s.raw_text in sgc_texts
            count = sum([in_tfidf, in_dl, in_sgc])

            if count > 0:
                agreement_badge = "🌟 Unanimous (3/3)" if count == 3 else ("⚖️ Agreed (2/3)" if count == 2 else "🎯 Model-Specific (1/3)")
                consensus_records.append({
                    "Sentence #": f"S{i+1}",
                    "Sentence Text": s.raw_text,
                    "TF-IDF": "✅" if in_tfidf else "—",
                    "Deep Learning": "✅" if in_dl else "—",
                    "SGCSumm": "✅" if in_sgc else "—",
                    "Consensus": agreement_badge
                })

        if consensus_records:
            st.dataframe(pd.DataFrame(consensus_records), width="stretch")
        else:
            st.info("No sentences selected.")

    with tab_details:
        subtab_tf, subtab_dl, subtab_sgc = st.tabs(["TF-IDF Logs", "Deep Learning Logs", "SGCSumm Marginal Gains"])
        
        with subtab_tf:
            if res["tfidf"]["logs"]:
                st.dataframe(pd.DataFrame(res["tfidf"]["logs"]), width="stretch")
            else:
                st.write("No logs available.")

        with subtab_dl:
            if res["dl"]["logs"]:
                st.dataframe(pd.DataFrame(res["dl"]["logs"]), width="stretch")
            else:
                st.write("No logs available.")

        with subtab_sgc:
            if res["sgc"]["logs"]:
                sgc_rows = []
                for log, sent in zip(res["sgc"]["logs"], res["sgc"]["sents"]):
                    sgc_rows.append({
                        "Sentence #": sent.sent_id,
                        "Marginal Gain Δ": f"{log.get('marginal_gain', 0):.5f}",
                        "Cumulative DSN Score": f"{log.get('cumulative_score', 0):.5f}",
                        "Cost": f"{log.get('cost', 0)} {budget_type}s",
                        "Sentence Text": sent.raw_text
                    })
                st.dataframe(pd.DataFrame(sgc_rows), width="stretch")
            else:
                st.write("No logs available.")

    # ==========================================
    # Export Options
    # ==========================================
    st.markdown("---")
    st.write("### 💾 Export Summaries")
    dl_col1, dl_col2, dl_col3, dl_col4 = st.columns(4)

    with dl_col1:
        st.download_button(
            "📥 TF-IDF Summary (.txt)",
            data=res["tfidf"]["summary"],
            file_name="summary_tfidf.txt",
            mime="text/plain",
            width="stretch"
        )
    with dl_col2:
        st.download_button(
            "📥 Deep Learning Summary (.txt)",
            data=res["dl"]["summary"],
            file_name="summary_deep_learning.txt",
            mime="text/plain",
            width="stretch"
        )
    with dl_col3:
        st.download_button(
            "📥 SGCSumm Summary (.txt)",
            data=res["sgc"]["summary"],
            file_name="summary_sgcsumm.txt",
            mime="text/plain",
            width="stretch"
        )
    with dl_col4:
        full_report = (
            f"# Multi-Model Summarization Benchmark Report\n\n"
            f"## Original Input Text ({orig_words} words)\n{user_text}\n\n"
            f"---\n\n"
            f"## 1. TF-IDF Summary ({res['tfidf']['time_ms']:.1f} ms)\n{res['tfidf']['summary']}\n\n"
            f"---\n\n"
            f"## 2. Deep Learning (BERT) Summary ({res['dl']['time_ms']:.1f} ms)\n{res['dl']['summary']}\n\n"
            f"---\n\n"
            f"## 3. SGCSumm Summary ({res['sgc']['time_ms']:.1f} ms)\n{res['sgc']['summary']}\n\n"
        )
        st.download_button(
            "📊 Full Benchmark Report (.md)",
            data=full_report,
            file_name="summarization_benchmark_report.md",
            mime="text/markdown",
            width="stretch"
        )
