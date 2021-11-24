
import streamlit as st
import pandas as pd
import gdown
import os
# import argparse
# import code
# import prettytable
# import logging
from drqa import retriever
import pandas as pd
# import numpy as np

from sentence_transformers import SentenceTransformer, util
from utils import get_unique_N,sort_list

# if os.path.isdir('pygaggle') == False:
#         os.system("git clone --recursive https://github.com/castorini/pygaggle.git")

# import pygaggle
# os.system("pip install pygaggle")
# os.system("pip install -U transformers")
# from pygaggle import pygaggle

# os.system("cd pygaggle")

from base import Query, Text
from Re_ranking import MonoBERT

@st.cache(allow_output_mutation=True)
def load_model():
    if os.path.exists('text-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz') == False:
        url = 'https://drive.google.com/uc?id=1ddcuoAy0z9aDUVuktZZb_Nu_4Di40bRv'
        # otput = '20150428_collected_images.tgz'
        gdown.download(url, quiet=False)
    
    ranker = retriever.get_class('tfidf')(tfidf_path='text-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz')

    passage_encoder = SentenceTransformer('facebook-dpr-ctx_encoder-single-nq-base')
    query_encoder = SentenceTransformer('facebook-dpr-question_encoder-single-nq-base')

    # questions = pd.read_csv("Gita_QA - Sheet1.csv").Question.unique()

    df_verses = pd.read_csv('df_gita(asitis).csv')
    text = df_verses.text.to_list()
    title = df_verses.id.to_list()

    passages = []
    for a,b in zip(title,text):
        passages.append(a+" [SEP] "+b)


    passage_embeddings = passage_encoder.encode(passages)

    reranker =  MonoBERT()

    return df_verses,text,passage_embeddings,query_encoder,ranker,reranker

    
st.title("GITA QUESTION AND ANSWER")
    
st.header("GITA WEBAPP")

df_verses, text, passage_embeddings, query_encoder, ranker, reranker = load_model()

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

    # st.markdown(text[indices[0]])
    # st.markdown("\n"+text[indices[1]])
    # st.markdown("\n"+text[indices[2]])
    # st.markdown("\n"+text[indices[3]])
    # st.markdown("\n"+text[indices[4]])

    # st.markdown(doc_names)

    # for idx, verse in enumerate(doc_names):
    # print(col)
    
    doc_names = pd.Series(doc_names).map(df_verses.set_index('id')['text'])
    verses = doc_names.to_list()
    
    for i in range(5):
        verses.append(text[indices[i]])

    q = Query(query)

    texts = [ Text(v, {'docid': i}, 0) for i,v in enumerate(verses)] # Note, pyserini scores don't matter since T5 will ignore them.

    reranked = reranker.rerank(q, texts)
    reranked_result = list(get_unique_N(sort_list([t.text for t in reranked],[t.score for t in reranked]),5))

    for verse in reranked_result:
        st.markdown("\n"+verse)
    # st.text(scores[0][indices[0]]+" "+scores[0][indices[1]]+" "+scores[0][indices[2]]+" "+scores[0][indices[3]])
