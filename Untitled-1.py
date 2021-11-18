
import streamlit as st
import pandas as pd
import gdown
# import argparse
# import code
# import prettytable
# import logging
from drqa import retriever
import pandas as pd
# import numpy as np

from sentence_transformers import SentenceTransformer, util

@st.cache(allow_output_mutation=True)
def load_model():

    url = 'https://drive.google.com/uc?id=1ddcuoAy0z9aDUVuktZZb_Nu_4Di40bRv'
    # output = '20150428_collected_images.tgz'
    gdown.download(url, quiet=False)

    ranker = retriever.get_class('tfidf')(tfidf_path='text-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz')

    passage_encoder = SentenceTransformer('facebook-dpr-ctx_encoder-single-nq-base')
    query_encoder = SentenceTransformer('facebook-dpr-question_encoder-single-nq-base')

    # questions = pd.read_csv("Gita_QA - Sheet1.csv").Question.unique()
    text = pd.read_csv("df_gita(asitis).csv").text.to_list()
    title = pd.read_csv("df_gita(asitis).csv").id.to_list()

    passages = []
    for a,b in zip(title,text):
        passages.append(a+" [SEP] "+b)


    passage_embeddings = passage_encoder.encode(passages)

    return text,passage_embeddings,query_encoder,ranker

    
st.title("GITA QUESTION AND ANSWER")
    
st.header("GITA WEBAPP")

text,passage_embeddings, query_encoder, ranker = load_model()

def display(text="Button clicked"):
    st.text(text)

st.text_input("Enter your question ","Type and submit",key="query")

st.button("Submit",key = "submitted")

if (st.session_state.submitted):
    display(st.session_state.query)

    query = st.session_state.query
    # query = questions[0]
    doc_names, doc_scores = ranker.closest_docs(query, k=5)


    query_embedding = query_encoder.encode(query)

    scores = util.dot_score(query_embedding, passage_embeddings)

    indices = sorted(range(len(scores[0])), key=lambda i: scores[0][i], reverse=True)[:5]

    st.text(text[indices[0]]+"\n"+text[indices[1]]+"\n"+text[indices[2]]+"\n"+text[indices[3]]+
    "\n"+text[indices[4]])

    st.text(doc_names)

    # st.text(scores[0][indices[0]]+" "+scores[0][indices[1]]+" "+scores[0][indices[2]]+" "+scores[0][indices[3]])