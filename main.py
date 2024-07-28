import time
import streamlit as st
from langchain_community.llms import CTransformers
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor

def load_llm():
    model_name = 'aryarishit/phi3-unsloth-resumebot-GGUF'

    llm = CTransformers(
            model = model_name,
            max_new_tokens = 128,
            temperature = 0.5
        )
    return llm

def get_index():
    Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-mpnet-base-v2")
    docs = SimpleDirectoryReader('Info_docs').load_data()
    Settings.llm = None
    Settings.chunk_size = 84
    Settings.chunk_overlap = 25
    index = VectorStoreIndex.from_documents(docs)

    return index

def get_context(index, query,top_k = 2):
    top_k = top_k

    # configure retriever
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=top_k,
    )

    # assemble query engine
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.2)],
    )
    # query documents
    query = query
    response = query_engine.query(query)

    # reformat response
    context = ""
    for i in range(len(response.source_nodes)):
        context = context + response.source_nodes[i].text + "\n\n"

    return context

def get_alpaca_prompt(context, query):

    instruction_string = '''[INST]Consider you are assistant to Rishit Arya, and answers on behalf of him, Given the following context and a question, generate an answer based on the given context only. If the answer to the question is not found in the context, strictly state "I don't know." only, don't try to make up an answer.Answer pricesly to what is asked it as if you are answering to Rishit's potential client. \nContext:{}
Question:{}[\INST] \nAnswer:'''

    prompt = instruction_string.format(
            context,
            query # input
        )
    return prompt

st.title("Ask-Rishit")

if "llm" not in st.session_state:
    st.session_state['llm'] = None
if "embeddings" not in st.session_state:
    st.session_state['embeddings'] = None

if st.session_state['llm'] is None:
    with st.spinner('Loading the model'):
        llm = load_llm()
    st.session_state['llm'] = llm

if st.session_state['embeddings'] is None:
    index = get_index()
    st.session_state['embeddings'] = index

query = st.text_input('Enter your Question')

if st.button('Generate') and st.session_state['llm'] is not None and st.session_state['embeddings'] is not None:
    with st.spinner('Generating.......'):
        llm = st.session_state['llm']
        index = st.session_state['embeddings']
        context = get_context(index, query)
        st.write("Context: " + context)
        prompt = get_alpaca_prompt(context, query)

        start_time = time.time()
        response = llm.invoke(prompt)
        end_time = time.time()
        time_taken = round(end_time-start_time, 2)
    st.write(response)
    st.caption('Time taken:' + str(time_taken))
