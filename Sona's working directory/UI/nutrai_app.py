# creating a streamlit application 
import streamlit as st
import pandas as pd
import pickle
import numpy as np
# from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.chains import RetrievalQA
# from langchain.vectorstores import FAISS  # or your KNN-based retriever
# from langchain_google_genai import GoogleGenerativeAIEmbeddings

import os 

st.title("Nutra-AI")
st.header("AI powered nutrition-based recipe recommendation system")

# loading the model and vectorizors, scalers
with open("knn.pkl","rb") as f:
    knn = pickle.load(f)

# have to give our custom function 
from my_custom_tokenizer import custom_tokenizer
with open("ingredient_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)
with open("nutrients_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# loading our dataset
recipes = pd.read_csv("Dataset_combined2.csv")

# building the UI
st.subheader("Enter your information")

age = st.number_input("Age", value=25)
gender = st.radio("Gender",["Male","Female"],1)
height = st.number_input("Height (cm)", value= 155)
weight = st.number_input("Weight (kg)", value = 55)
fiber = st.number_input("Fiber (g)",value=8)
sodium = st.number_input("sodium (mg)",value = 500)

def calculate_nutrition(age, weight_kg, height_cm, gender):
    """
    Calculate daily nutrition requirements based on age, weight, height, and gender.
    
    Returns: dict with calories, protein, carbs, fat, fiber, sodium
    """
    
    # Step 1: Calculate Basal Metabolic Rate (BMR)
    if gender== 0:
        bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age + 5
    else:
        bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age - 161

    # Assume sedentary activity level (can be adjusted later)
    calories = bmr * 1.2

    # Macronutrient distribution (based on total calories)
    protein_grams = weight_kg * 0.8  # grams per kg of body weight
    fat_grams = (0.25 * calories) / 9  # 25% of calories from fat
    carbs_grams = (0.50 * calories) / 4  # 50% from carbs


    return {
        "calories": round(calories/3.5),   # divinding the final by 3 as these estimates are for whole three meals, multiplying by 4 as the serving size of recipes are 4
        "protein (g)": round(protein_grams/3.5, 1),
        "carbohydrates (g)": round(carbs_grams/3.5, 1),
        "fat (g)": round(fat_grams/3.5, 1)
    }

ingredients = st.text_area("Enter your pantry items",value="rice coconut dal vegetable")

if st.button("Get recipes"):
    st.session_state.selected_recipe_index = None

    #scaling nutrients
    calories, protein, carb, fat = calculate_nutrition(age,weight,height,gender).values()
    nutrients = np.array([calories, protein, carb, fat, fiber, sodium], ndmin=2)
    st.write("Based on your BMR, your nutritional needs are: ")
    st.write(pd.DataFrame(nutrients,columns=["calories","protein (g)", "carbohydrates (g)", "fat (g)", "fiber (g)", "sodium (mg)"]))
    scaled_nutrients = scaler.transform(nutrients*0.5)
    #vectorizing ingredients
    vectorizer_ingredients = vectorizer.transform([ingredients])
    vectorizer_ingredients = vectorizer_ingredients.toarray()*2.5  

    input_features = np.hstack((scaled_nutrients,vectorizer_ingredients))

    distances, indices = knn.kneighbors(input_features)

    st.write("The top five recommended recipes based on your input: ")

    matched_recipes = recipes.iloc[indices[0]]
    resultant_names = matched_recipes['name']

    #preserve the needed session variables:
    st.session_state.user_input = {
        'matched_recipes': matched_recipes, 'resultant_names':resultant_names, 'pantry': ingredients
    }
    st.session_state.user_nutrition = {'calories':calories, 'protien':protein, 'carbohydrates':carb, 'fat': fat, 'sodium':sodium, 'fiber':fiber }

# Initialize session state
if "selected_recipe_index" not in st.session_state:
    st.session_state.selected_recipe_index = None
# In Streamlit, your script reruns top-to-bottom every time a user interacts with the page. So we need a way to persist state — i.e., to remember if the user has clicked a recipe or not.

#Page 1
# Recipe Titles 
if "user_input" in st.session_state and st.session_state.selected_recipe_index is None:
    st.title("Recommended Recipes")
    st.write("Click on a recipe to view its details:")
    
    #fetching resultant names from the previous session state
    resultant_names = st.session_state.user_input['resultant_names']
    for i,row in resultant_names.items():
        if st.button(row, key=f"recipe_{i}"):
            st.session_state.selected_recipe_index = i
            st.rerun()   # forces that rerun immediately, allowing the change to take effect right away

# page 2
elif "user_input" in st.session_state:
    matched_recipes = st.session_state.user_input['matched_recipes']
    idx = st.session_state.selected_recipe_index
    recipe = matched_recipes.loc[idx]
    #name
    st.title(recipe["name"])

    #ingredients
    st.subheader("Ingredients")
    # cleaning unicode data
    from clean_unicode_help import clean_unicode
    import ast
    # cleaning unicode
    

    ingredients = ast.literal_eval(recipe["ingredients"])
    for item in ingredients:
        cleaned = clean_unicode(item)
        st.markdown(f"\t\t\t- {cleaned}")

    #nutrition
    st.subheader("Nutrition")
    nutrition = ast.literal_eval(recipe["nutrition"])
    for key,value in nutrition.items():
        st.markdown(f"\t\t\t- {key}: {value}")

    #instructions
    st.subheader("Instructions")
    instructions = recipe["instructions"]
    st.markdown(instructions)

    # for passing to llm
    result_formatted = recipe[['name','ingredients','nutrition','instructions']]

    # user query
    st.subheader("Need a variation? Ask AI")
    user_needs = st.text_input("Enter your request",key="user_query", value="Make it vegetarian")

    #context prompt
    template = '''You are a nutrition assistant
                here are some recipies name with their ingredients and nutrition:
                {result_formatted}
                Here is the user entered nutrition needs and pantry items: 
                {user_input}
                Answer the following user query: {user_needs}, keeping in mind the required nutrition needs.
                
'''

    if user_needs:
        # Load your Gemini API Key
        os.environ["GOOGLE_API_KEY"] = "AIzaSyAonrrlhxfzhR5MyaPG_sVrPpDsUjyjpgk"  # use environment variable or Render secret

        # Set up LLM
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.3)

        prompt_template = ChatPromptTemplate.from_template(template)
        prompt = prompt_template.format(result_formatted=result_formatted,user_input=st.session_state.user_nutrition,
                                        user_needs=user_needs)

        try:
            response = llm.invoke(prompt)
            st.markdown("### AI Response")
            st.write(response.content)
        except Exception as e:
            st.error(f"Error from LLaMA/Ollama: {e}")
    
    # Navigation button
    if st.button("<- Back to recipe list"):
        st.session_state.selected_recipe_index = None
        st.rerun()
