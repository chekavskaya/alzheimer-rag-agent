import streamlit as st
from rag.retrieval import Retriever
from rag.generator import Generator
from rag.pipeline import RAGPipeline



DATA_PATH = "data/pubmed_chunks.csv"  
import os
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")



@st.cache_resource
def load_retriever():
    return Retriever(
        data_path=DATA_PATH,
        text_column="text",
        metadata_columns=["pmid", "title"]
    )


@st.cache_resource
def load_generator():
    return Generator(api_key=OPENROUTER_API_KEY)


@st.cache_resource
def load_pipeline():
    retriever = load_retriever()
    generator = load_generator()
    return RAGPipeline(retriever, generator)


rag = load_pipeline()



st.title("RAG Assistant for Alzheimer's Research")
st.markdown(
    "Задайте вопрос о потенциальных терапевтических мишенях для болезни Альцгеймера. "
    "Система покажет сгенерированный ответ с источниками из научных статей."
)

query = st.text_area("Введите вопрос:", height=100)


if st.button("Получить ответ"):
    if not query.strip():
        st.warning("Введите текст вопроса!")
    else:
        with st.spinner("Ищем релевантные статьи и генерируем ответ..."):
            result = rag.answer(query)


        
        st.subheader("Ответ RAG:")
        st.write(result["answer"])


        
        st.subheader("Использованные источники:")
        for i, src in enumerate(result["sources"], 1):
            pmid = src.get("pmid", "N/A")
            title = src.get("title", "N/A")
            st.markdown(f"[{i}] PMID {pmid} — {title}")


