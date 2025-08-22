import streamlit as st
import pickle
import pandas as pd
import numpy as np
import faiss
import torch
import pathlib
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
import os

# =====================
# PATHS
# =====================

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent  # Go up from /pages
DATA_DIR = BASE_DIR / "Data"

index_path = DATA_DIR / "faiss_index.bin"
scaler_path = DATA_DIR / "scaler.pkl"
embeddings_path = DATA_DIR / "combined_embeddings.npy"
csv_path = DATA_DIR / "PreProcessedData.csv"

# =====================
# SETTINGS
# =====================
device = 'cuda' if torch.cuda.is_available() else 'cpu'
bert_model = SentenceTransformer('all-mpnet-base-v2', device=device)

# Google Gemini setup (replace with your API key)
from dotenv import load_dotenv
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

model = ChatGoogleGenerativeAI(model="gemini-2.5-pro")  # Correct Gemini model

# =====================
# INDEX BUILDER (if missing)
# =====================
def build_faiss_index():
    st.warning("⚠️ FAISS index not found. Building new one...")
    df = pd.read_csv(csv_path)
    nutrient_columns = ["calories", "protein", "fat", "carbs"]  # adjust as per your CSV

    # Create embeddings
    text_embeddings = bert_model.encode(df["ingredients"].astype(str).tolist())

    # Scale nutrients
    scaler = StandardScaler()
    nutrient_data = scaler.fit_transform(df[nutrient_columns])

    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    # Combine
    ingredient_weight = 2.5
    nutrition_weight = 0.5
    text_scaled = text_embeddings * ingredient_weight
    nutri_scaled = nutrient_data * nutrition_weight
    combined_embeddings = np.hstack((text_scaled, nutri_scaled))
    np.save(embeddings_path, combined_embeddings)

    # Build FAISS
    index = faiss.IndexFlatL2(combined_embeddings.shape[1])
    index.add(combined_embeddings)
    faiss.write_index(index, str(index_path))
    st.success("✅ FAISS index built successfully!")
    return df, index, scaler, combined_embeddings

# =====================
# LOAD DATA
# =====================
if not index_path.exists() or not scaler_path.exists() or not embeddings_path.exists():
    df1, index, scaler, combined_embeddings = build_faiss_index()
else:
    df1 = pd.read_csv(csv_path)
    index = faiss.read_index(str(index_path))
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    combined_embeddings = np.load(embeddings_path)

# =====================
# STREAMLIT UI
# =====================
st.title("🥗 NUTRIAI-PAL")

if "nutritions_goals" not in st.session_state or "user_input" not in st.session_state:
    st.warning("Please submit your nutrition goals and input first")
    st.stop()
else:
    nutritions_goals = st.session_state["nutritions_goals"]
    user_input = st.session_state["user_input"]

# =====================
# RECOMMENDATION FUNCTION
# =====================
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
    recommendations = [(df1.iloc[idx], D[0][i]) for i, idx in enumerate(I[0])]
    return recommendations

# =====================
# GET RECOMMENDATIONS
# =====================
recommendations = recommendation(user_input, nutritions_goals, df1, bert_model, index)
recipes_text = "\n".join([
    f"{row['name']} (Tags: {row['tags']}) (Ingredients:{row['ingredients']}) (Instructions:{row['instructions']})"
    for row, _ in recommendations
])

# =====================
# INITIAL PROMPT
# =====================
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

Return this structure in a good language:
Title:  
Ingredients:  
Instructions:  
Cooking Time:  
Serving Size:  
Nutritional Info:  
---
"""
)
initial_chain = LLMChain(llm=model, prompt=initial_prompt_template, verbose=True)
initial_response = initial_chain.invoke({
    "pantry": user_input,
    "goals": nutritions_goals,
    "recipes": recipes_text
})

st.markdown("### 🍽 Top Recipes")
st.markdown(initial_response["text"])

# =====================
# CHAT PROMPT + MEMORY
# =====================
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
chat_chain = LLMChain(llm=model, prompt=chat_prompt, memory=chat_memory, verbose=True)

# =====================
# CHAT INPUT HANDLER
# =====================
query = st.chat_input("Ask follow-up questions or request more ideas!")
if query:
    response = chat_chain.run(query)
    st.markdown(f"**NutriBot:** {response}")
