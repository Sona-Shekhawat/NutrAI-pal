import streamlit as st
import pickle
import pandas as pd
import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer
from langchain_ollama import ChatOllama
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

# === SETUP ===
device = 'cuda' if torch.cuda.is_available() else 'cpu'
bert_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

model = ChatOllama(model="llama3.1")

st.title("🥗 NUTRIAI-PAL")

# === LOAD DATA ===
index = faiss.read_index(r"D:\Academics -\project-nutiai\NutrAI-pal\Shahana's Working Directory\BERT-FAISS\faiss_index.bin")
with open(r"D:\Academics -\project-nutiai\NutrAI-pal\Shahana's Working Directory\BERT-FAISS\scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

combined_embeddings = np.load(r"D:\Academics -\project-nutiai\NutrAI-pal\Shahana's Working Directory\BERT-FAISS\combined_embeddings.npy")
df1 = pd.read_csv(r"D:\Academics -\project-nutiai\NutrAI-pal\Shahana's Working Directory\BERT-FAISS\PreProcessedData.csv")

# === FAISS RECOMMENDATION ===
def recommendation(user_input, nutritions_goals, df1, bert_model, index):
    query_embed = bert_model.encode([user_input])
    query_nutri_df = pd.DataFrame.from_dict(nutritions_goals)
    query_nutrient = scaler.transform(query_nutri_df)

    ingredient_weight = 2.5
    nutrition_weight = 0.5

    query_embed_scaled = query_embed * ingredient_weight
    query_nutrient_scaled = query_nutrient * nutrition_weight
    query_vector = np.hstack((query_embed_scaled, query_nutrient_scaled))

    D, I = index.search(query_vector.reshape(1, -1), k=3)
    recommendations = [
        (df1.iloc[idx], D[0][i])
        for i, idx in enumerate(I[0])
    ]
    return recommendations

# === SESSION STATE ===
if "nutritions_goals" not in st.session_state or "user_input" not in st.session_state:
    st.warning("Please submit your nutrition goals and input first")
    st.stop()
else:
    nutritions_goals = st.session_state["nutritions_goals"]
    user_input = st.session_state["user_input"]

# === GET TOP RECIPES ===
recommendations = recommendation(user_input, nutritions_goals, df1, bert_model, index)

recipes_text = "\n".join([
    f"{row['name']} (Tags: {row['tags']}) (Ingredients:{row['ingredients']}) (Instructions:{row['instructions']})"
    for row, _ in recommendations
])

# === INITIAL PROMPT (NO MEMORY) ===
initial_prompt_template = PromptTemplate(
    input_variables=["pantry", "goals", "recipes"],
    template="""
You are a smart and health-focused nutritional cooking assistant.

The user has:
Pantry: {pantry}
Nutritional goals: {goals}

Here are some candidate recipes:
{recipes}

Choose the top 2 recipes that match:
- Pantry items (most important)
- Nutritional goals
- Practicality

Return this structure:
Title:  
Ingredients:  
Instructions:  
Cooking Time:  
Serving Size:  
Nutritional Info:  
---
"""
)

initial_chain = LLMChain(
    llm=model,
    prompt=initial_prompt_template,
    verbose=True
)

# === RUN INITIAL RESPONSE ===
initial_response = initial_chain.invoke({
    "pantry": user_input,
    "goals": nutritions_goals,
    "recipes": recipes_text
})

st.markdown("### 🍽 Top Recipes")
st.markdown(initial_response["text"])

# === CHAT PROMPT + MEMORY (1 input only!) ===
chat_prompt = PromptTemplate(
    input_variables=["query"],
    template="""
You're NutriBot — a smart, friendly nutrition assistant.

You've already shown the user some recipe suggestions.
Now, continue the conversation.

User: {query}
NutriBot:"""
)

chat_memory = ConversationBufferMemory(input_key="query")

chat_chain = LLMChain(
    llm=model,
    prompt=chat_prompt,
    memory=chat_memory,
    verbose=True
)

# === CHAT HANDLER ===
query = st.chat_input("Ask follow-up questions or request more ideas!")

if query:
    response = chat_chain.run(query)
    st.markdown(f"**NutriBot:** {response}")


