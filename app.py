"""
app.py

Streamlit application for the Corporate Narrative Consistency Engine.
Provides a UI for contradiction detection using NLI and semantic search
using a FAISS index.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import streamlit as st

# Ensure src/ is on the path so we can import internal modules standalone
_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

try:
    from models.nli import NLIModel
    from retrieval.faiss_index import FAISSIndex
except ImportError as e:
    st.error(f"Cannot import internal modules. Ensure the app is run from the project root. Detail: {e}")
    st.stop()


# ---------------------------------------------------------------------------
# Page configuration & Sidebar
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Corporate Narrative Consistency Engine",
    page_icon="🔍",
    layout="wide",
)

with st.sidebar:
    st.title("About the Project")
    st.write(
        "The Corporate Narrative Consistency Engine identifies conflicting statements "
        "across SEC filings and internal communications. It ensures your corporate "
        "messaging remains perfectly aligned."
    )
    st.markdown("[View on GitHub](https://github.com/Jainishpatel1019/corporate-narrative-engine)")
    
    st.header("Models Used")
    st.markdown(
        "- **Semantic Search:** `all-MiniLM-L6-v2` + FAISS\n"
        "- **Consistency NLI:** `cross-encoder/nli-deberta-v3-base`"
    )


# ---------------------------------------------------------------------------
# Caching Model Loading
# ---------------------------------------------------------------------------

@st.cache_resource
def get_nli_model() -> NLIModel | None:
    """Load the NLI model instance once and reuse across sessions."""
    try:
        return NLIModel()
    except Exception as e:
        st.error(f"Failed to load NLI model: {e}")
        return None


@st.cache_resource
def get_faiss_index() -> tuple[FAISSIndex | None, bool]:
    """Load the FAISS index instance once and reuse across sessions."""
    try:
        index = FAISSIndex()
        index_path = "data/faiss.index"
        
        # If the file exists, attempt to load it
        if os.path.exists(index_path):
            index.load(index_path)
            return index, True
            
        return index, False
    except Exception as e:
        st.error(f"Failed to load FAISS index: {e}")
        return None, False


nli_model = get_nli_model()
faiss_index, index_loaded = get_faiss_index()


# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------

st.title("Corporate Narrative Consistency Engine")

tab1, tab2 = st.tabs(["Contradiction Detector", "Filing Search"])

# --------------------------
# Tab 1: Contradiction
# --------------------------

with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        text_a = st.text_area("Filing A (older)", height=250, placeholder="Enter the older premise here...")
    
    with col2:
        text_b = st.text_area("Filing B (newer)", height=250, placeholder="Enter the newer hypothesis here...")
        
    if st.button("Detect Contradiction", type="primary"):
        if not text_a.strip() or not text_b.strip():
            st.warning("Please provide both Filing A and Filing B to detect contradictions.")
        elif nli_model is None:
            st.error("NLI Model failed to load.")
        else:
            try:
                # Measure latency
                t0 = time.perf_counter()
                result = nli_model.predict(text_a, text_b)
                latency_ms = (time.perf_counter() - t0) * 1000.0
                
                label = result["label"]
                score = result["score"]
                
                # Determine metric styling
                if label == "contradiction":
                    color = "red"
                elif label == "entailment":
                    color = "green"
                else:  # neutral
                    color = "gray"
                
                st.markdown(f"### Result: <span style='color:{color}; text-transform:uppercase;'>{label}</span>", unsafe_allow_html=True)
                
                # Render score as progress bar
                st.progress(score, text=f"Confidence Score: {score:.2f}")
                st.caption(f"Inference latency: {latency_ms:.1f} ms")
                
            except Exception as e:
                st.error(f"An error occurred during inference: {e}")


# --------------------------
# Tab 2: Filing Search
# --------------------------

with tab2:
    if not index_loaded:
        st.warning(
            "⚠️ FAISS index not found at `data/faiss.index`. "
            "Please build the index first using the data pipeline script."
        )
        
    query = st.text_input("Search Query", placeholder="e.g. Revenue growth in Q3")
    
    col3, col4 = st.columns([1, 4])
    with col3:
        top_k = st.slider("Result Count (top_k)", min_value=1, max_value=10, value=5)
    with col4:
        # vertical alignment trick
        st.write("")
        st.write("")
        search_pressed = st.button("Search", type="primary")
        
    if search_pressed:
        if not query.strip():
            st.warning("Please enter a search query.")
        elif not index_loaded or faiss_index is None:
            st.error("Cannot perform search because the FAISS index is not loaded.")
        else:
            try:
                # Measure latency
                t0 = time.perf_counter()
                results = faiss_index.search(query, top_k=top_k)
                latency_ms = (time.perf_counter() - t0) * 1000.0
                
                st.success(f"Retrieved {len(results)} chunks in {latency_ms:.1f} ms")
                
                if not results:
                    st.info("No matching chunks found.")
                else:
                    for r in results:
                        score = r.get("score", 0.0)
                        chunk_idx = r.get("chunk_index", "N/A")
                        text_val = r.get("text", "")
                        
                        # Show as expandable cards
                        with st.expander(f"Chunk Index: {chunk_idx} | Semantic Similarity: {score:.3f}"):
                            st.write(text_val)
                            
            except Exception as e:
                st.error(f"An error occurred during semantic search: {e}")
