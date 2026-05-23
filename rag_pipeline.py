import hashlib
from pathlib import Path
from typing import Optional

import wikipedia
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

CHROMA_PATH = str(Path(__file__).parent / "chroma_db")
COLLECTION_NAME = "wikipedia_cache"
# Cosine similarity threshold — chunks scoring above this are considered a cache hit
CACHE_THRESHOLD = 0.45

PROMPT = PromptTemplate.from_template(
    """Use the following Wikipedia content to answer the question accurately and concisely.
If the answer is not found in the context, say "I couldn't find information about that in Wikipedia."

Context:
{context}

Question: {question}

Answer:"""
)


def load_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


class WikipediaRAG:
    def __init__(self, model_name: str = "llama3", embeddings: Optional[HuggingFaceEmbeddings] = None):
        self.llm = OllamaLLM(model=model_name)
        self.embeddings = embeddings or load_embeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        self._chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
        self.vectorstore = Chroma(
            client=self._chroma_client,
            collection_name=COLLECTION_NAME,
            embedding_function=self.embeddings,
        )

    def cache_size(self) -> int:
        return self.vectorstore._collection.count()

    def clear_cache(self):
        self._chroma_client.delete_collection(COLLECTION_NAME)
        self.vectorstore = Chroma(
            client=self._chroma_client,
            collection_name=COLLECTION_NAME,
            embedding_function=self.embeddings,
        )

    def _fetch_pages(self, query: str, max_pages: int):
        docs, sources = [], []
        try:
            titles = wikipedia.search(query, results=max_pages)
        except Exception:
            return docs, sources

        for title in titles:
            try:
                page = wikipedia.page(title, auto_suggest=False)
                docs.append(page.content)
                sources.append({"title": page.title, "url": page.url})
            except wikipedia.DisambiguationError as e:
                try:
                    page = wikipedia.page(e.options[0], auto_suggest=False)
                    docs.append(page.content)
                    sources.append({"title": page.title, "url": page.url})
                except Exception:
                    continue
            except Exception:
                continue

        return docs, sources

    def _add_to_cache(self, docs: list, sources: list):
        texts, metadatas, ids = [], [], []
        seen: set = set()
        for doc, src in zip(docs, sources):
            for chunk in self.text_splitter.split_text(doc):
                chunk_id = hashlib.md5(chunk.encode()).hexdigest()
                if chunk_id in seen:
                    continue
                seen.add(chunk_id)
                texts.append(chunk)
                metadatas.append(src)
                ids.append(chunk_id)
        if texts:
            self.vectorstore.add_texts(texts=texts, metadatas=metadatas, ids=ids)

    def answer(self, query: str, max_pages: int = 3):
        """Returns (answer_text, sources, from_cache)."""
        cached = self.vectorstore.similarity_search_with_relevance_scores(query, k=4)
        top_score = cached[0][1] if cached else 0.0

        if top_score >= CACHE_THRESHOLD:
            context = "\n\n".join(doc.page_content for doc, _ in cached)
            sources = []
            seen: set = set()
            for doc, _ in cached:
                title = doc.metadata.get("title", "")
                url = doc.metadata.get("url", "")
                if title and title not in seen:
                    sources.append({"title": title, "url": url})
                    seen.add(title)
            from_cache = True
        else:
            docs, sources = self._fetch_pages(query, max_pages)
            if not docs:
                return "I couldn't find any relevant Wikipedia articles for your query.", [], False
            self._add_to_cache(docs, sources)
            fresh = self.vectorstore.similarity_search_with_relevance_scores(query, k=4)
            context = "\n\n".join(doc.page_content for doc, _ in fresh)
            from_cache = False

        chain = (
            {"context": lambda _: context, "question": RunnablePassthrough()}
            | PROMPT
            | self.llm
            | StrOutputParser()
        )
        return chain.invoke(query).strip(), sources, from_cache
