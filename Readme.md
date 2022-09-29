
# Gita Webapp

Contains code for streamlit webapp "Gita Recommender"

Retrieves relevant verses from Bhagavad Gita text for the question entered by the user.
App can be deployed locally or via colab notebooks.


- Uses DrQA and Dense Passage retriever to retrieve relevant verses independently
- Uses MonoBERT re-ranker to shuffle the order of the verses
