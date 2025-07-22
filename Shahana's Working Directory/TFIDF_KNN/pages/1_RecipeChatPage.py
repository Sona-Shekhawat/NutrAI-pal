import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import streamlit as st
import pickle
import re
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain_ollama import ChatOllama


# load the model
model = ChatOllama(model="llama3.1")

st.title("NUTRIAI-PAL")

# Load trained models
with open(r"D:\Academics -\project-nutiai\NutrAI-pal\Shahana's Working Directory\TFIDF_KNN\tfidf.pkl", "rb") as f:
    tfidf = pickle.load(f)

with open(r"D:\Academics -\project-nutiai\NutrAI-pal\Shahana's Working Directory\TFIDF_KNN\scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open(r"D:\Academics -\project-nutiai\NutrAI-pal\Shahana's Working Directory\TFIDF_KNN\knn_model.pkl", "rb") as f:
    knn_model = pickle.load(f)

df1 = pd.read_csv(r"D:\Academics -\project-nutiai\NutrAI-pal\Shahana's Working Directory\TFIDF_KNN\Dataset_combined.csv")


def get_recipe_recommendations(user_input,nutritions,df1,tfidf, knn, top_n=5):
    df_test=pd.DataFrame.from_dict(nutritions)
    nutritions_goals=df_test.values
    values= scaler.fit_transform(nutritions_goals)
    processed_input = re.sub(r'[^a-z0-9\s]', '', user_input.lower())
    processed_input = processed_input.replace(',', ' ')
    input_tfidf = tfidf.transform([processed_input]).toarray()
    input_combined=np.concatenate([input_tfidf,values],axis=1)
    distances, indices = knn.kneighbors(input_combined, n_neighbors=top_n)
    recommendations = [
        (df1.iloc[idx], distances[0][i])
        for i, idx in enumerate(indices[0])
    ]
    return recommendations

if "nutritions_goals" not in st.session_state and "user_input" not in st.session_state:
    st.warning("⚠️ Please submit your nutrition goals and input first")
else:
    nutritions_goals = st.session_state["nutritions_goals"]
    user_input=st.session_state["user_input"]

    recommendations = get_recipe_recommendations(user_input,nutritions_goals,df1, tfidf, knn_model, top_n=10)

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



