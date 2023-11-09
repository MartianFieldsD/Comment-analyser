import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
Model = SentenceTransformer('bert-base-uncased')
with open('Untitled.tsv', 'r') as file:
    lines = file.readlines()
data = []
for line in lines:
    parts = line.strip().split('|')
    date = parts[0].strip()
    comment = parts[1].strip()
    categorized = parts[2].strip()
    data.append([date, comment, categorized])
df = pd.DataFrame(data, columns=['Date', 'Comment', 'Categorized'])
import pickle 
df['Categorized'] = df['Categorized'].replace({'Yes a': 'Yes'})
labelencoder = LabelEncoder()
df['Categorized'] = labelencoder.fit_transform(df['Categorized'])
pickled_model = pickle.load(open('model.pkl', 'rb'))
st.title("Comment Categorization App")
user_input = st.text_input("Enter a comment:")
if user_input:
    user_bert = Model.encode([user_input])
    prediction = pickled_model.predict(user_bert)
    original_category = labelencoder.inverse_transform(prediction)
    st.write(f"Predicted Category: {original_category[0]}")
