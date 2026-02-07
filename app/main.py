import streamlit as st
import os
from rag.retrieval import Retriever
from rag.generator import Generator
from rag.pipeline import RAGPipeline

st.set_page_config(page_title="AD Target Explorer", layout="wide")


DATA_PATH = "data/pubmed_chunks.csv"
API_KEY = os.getenv("OPENROUTER_API_KEY")

@st.cache_resource
def init_rag():
    retriever = Retriever(DATA_PATH)
    generator = Generator(API_KEY)
    return RAGPipeline(retriever, generator)

rag = init_rag()

st.title("Alzheimer's Research RAG-Agent")
st.sidebar.header("What is it?")
st.sidebar.info("The agent analyzes the latest publications from PubMed and helps find therapeutic targets.")


st.subheader("Examples:")
example_queries = [
    "What are potential targets for Alzheimer's disease treatment?",
    "Are the targets druggable with small molecules or biologics?"
]
selected_example = st.selectbox("Select an example or enter your own:", [""] + example_queries)

query = st.text_area("Your request:", value=selected_example if selected_example else "", height=100)

if st.button("Analyze"):
    if not query.strip():
        st.error("Please enter your request.")
    else:
        with st.spinner("Analyzing..."):
            result = rag.answer(query)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("Answer")
                st.markdown(result["answer"])
                st.success(f"Faithfulness Score: {result['metrics']['faithfulness']}")

            with col2:
                st.subheader("")
                for src in result["sources"]:
                    with st.expander(f"[{src['id']}] {src['title']}"):
                        st.write(f"**PMID:** {src['pmid']}")
                        st.write(f"**Relevance Score:** {src['score']:.2f}")
                        st.markdown(f"[PubMed link](https://pubmed.ncbi.nlm.nih.gov/{src['pmid']}/)")