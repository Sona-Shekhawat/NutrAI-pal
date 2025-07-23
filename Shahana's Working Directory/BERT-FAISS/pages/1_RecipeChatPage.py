import streamlit as st
import pickle 
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain_ollama import ChatOllama

# load the model
model = ChatOllama(model="llama3.1")

st.title("NUTRIAI-PAL")

# Load FAISS index
index = faiss.read_index(r"D:\Academics -\project-nutiai\NutrAI-pal\Shahana's Working Directory\BERT-FAISS\faiss_index.bin")

# Load scaler
with open(r"D:\Academics -\project-nutiai\NutrAI-pal\Shahana's Working Directory\BERT-FAISS\scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Load embeddings (only needed if you plan to modify or inspect them)
combined_embeddings = np.load(r"D:\Academics -\project-nutiai\NutrAI-pal\Shahana's Working Directory\BERT-FAISS\combined_embeddings.npy")

# Load BERT model
bert_model = SentenceTransformer('all-MiniLM-L6-v2')

df1=pd.read_csv(r"D:\Academics -\project-nutiai\NutrAI-pal\Shahana's Working Directory\BERT-FAISS\Dataset_combined.csv")

def recommendation(user_input,nutritions_goals,df1,bert_model,index):
    query_text =user_input
    query_embed = bert_model.encode([query_text])
    # example nutrient goals
    query_nutri_df = pd.DataFrame.from_dict(nutritions_goals)
    query_nutrient = scaler.transform(query_nutri_df)
    # combine
    query_vector = np.hstack((query_embed, query_nutrient))
    D, I = index.search(query_vector.reshape(1, -1), k=3)
    recommendations = [
        (df1.iloc[idx], D[0][i])
        for i, idx in enumerate(I[0])
    ]
    return recommendations


if "nutritions_goals" not in st.session_state and "user_input" not in st.session_state:
    st.warning("⚠️ Please submit your nutrition goals and input first")
else:
    nutritions_goals = st.session_state["nutritions_goals"]
    user_input=st.session_state["user_input"]



recommendations =recommendation(user_input,nutritions_goals,df1,bert_model,index)

recipes_text = "\n".join([
f"{row['name']} (Tags: {row['tags']})"
for row, _ in recommendations
])
# collect all the messages
messages = [
    SystemMessage(
        content=f"""
You are a nutritional cooking assistant. The user has the following pantry items: {user_input}.
Their nutritional goals are: {nutritions_goals}.
Here are a few suggested recipes based on their ingredients and dietary targets:
{recipes_text}
"""
    ),
    HumanMessage(content=user_input)
]
# get the answer
result = model.invoke(messages)

st.write(result.content)

query=st.chat_input("Any Query....?")

if query:
    response=model.invoke(query)
    st.write(response.content)