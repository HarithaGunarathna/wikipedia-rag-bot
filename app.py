import streamlit as st
from rag_pipeline import WikipediaRAG, load_embeddings

st.set_page_config(page_title="Wikipedia RAG Bot", page_icon="📚", layout="centered")

st.title("📚 Wikipedia RAG Bot")
st.caption("Ask anything — I'll check the cache first, then search Wikipedia if needed.")

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    model_name = st.selectbox(
        "Ollama model",
        ["llama3", "llama3.2", "mistral", "phi3", "gemma3:1b", "llama2"],
        help="The model must be pulled first: ollama pull <model>",
    )

    max_pages = st.slider(
        "Wikipedia pages to fetch",
        min_value=1, max_value=5, value=3,
        help="Only used on a cache miss. More pages = richer context but slower.",
    )

    st.divider()
    st.subheader("Cache (ChromaDB)")

    if "rag" in st.session_state:
        cache_count = st.session_state.rag.cache_size()
        st.metric("Cached chunks", cache_count)
    else:
        st.caption("Cache loading…")

    if st.button("Clear cache", type="secondary"):
        if "rag" in st.session_state:
            st.session_state.rag.clear_cache()
            st.success("Cache cleared.")
            st.rerun()

    st.divider()
    st.markdown("**Setup checklist**")
    st.markdown("1. Install [Ollama](https://ollama.com)")
    st.markdown("2. Run `ollama serve`")
    st.markdown(f"3. Run `ollama pull {model_name}`")
    st.markdown("4. `pip install -r requirements.txt`")

# ── Session state ─────────────────────────────────────────────────────────────
if "embeddings" not in st.session_state:
    with st.spinner("Loading embedding model (first run only)…"):
        st.session_state.embeddings = load_embeddings()

if "rag" not in st.session_state or st.session_state.get("active_model") != model_name:
    st.session_state.rag = WikipediaRAG(
        model_name=model_name,
        embeddings=st.session_state.embeddings,
    )
    st.session_state.active_model = model_name

if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Chat history ──────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            badge = "🟢 From cache" if msg.get("from_cache") else "🌐 From Wikipedia"
            st.caption(badge)
        st.write(msg["content"])
        if msg.get("sources"):
            with st.expander("Wikipedia sources"):
                for src in msg["sources"]:
                    st.markdown(f"- [{src['title']}]({src['url']})")

# ── Input ─────────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask a question…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Checking cache and thinking…"):
            try:
                answer, sources, from_cache = st.session_state.rag.answer(prompt, max_pages)
            except Exception as e:
                answer = f"Error: {e}\n\nMake sure Ollama is running (`ollama serve`) and `{model_name}` is pulled."
                sources, from_cache = [], False

        badge = "🟢 From cache" if from_cache else "🌐 From Wikipedia"
        st.caption(badge)
        st.write(answer)
        if sources:
            with st.expander("Wikipedia sources"):
                for src in sources:
                    st.markdown(f"- [{src['title']}]({src['url']})")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
        "from_cache": from_cache,
    })
