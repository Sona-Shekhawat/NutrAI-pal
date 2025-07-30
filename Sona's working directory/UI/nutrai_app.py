# creating a streamlit application 
import streamlit as st
import pandas as pd
import pickle
import numpy as np
from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate

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
recipes = pd.read_csv("Dataset_combined.csv")

# building the UI
st.subheader("Enter your information")

age = st.number_input("Age")
gender = st.radio("Gender",["Male","Female"],1)
height = st.number_input("Height (cm)")
weight = st.number_input("Weight (kg)")

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

    # Fiber recommendation
    if gender.lower() == 'male':
        fiber_grams = 30 if age >= 18 else 25
    else:
        fiber_grams = 25 if age >= 18 else 20

    # Sodium (upper limit, in mg)
    sodium_mg = 2300  # standard for healthy adults

    return {
        "calories": round(calories),
        "protein (g)": round(protein_grams, 1),
        "carbohydrates (g)": round(carbs_grams, 1),
        "fat (g)": round(fat_grams, 1),
        "fiber (g)": fiber_grams,
        "sodium (mg)": sodium_mg
    }

ingredients = st.text_area("Enter your pantry items",placeholder="paneer capsicum milk cheeze")

if st.button("Get recipes"):
    st.session_state.selected_recipe_index = None

    #scaling nutrients
    nutrients = np.array(list(calculate_nutrition(age,weight,height,gender).values()), ndmin=2)
    scaled_nutrients = scaler.transform(nutrients)
    #vectorizing ingredients
    vectorizer_ingredients = vectorizer.transform([ingredients])
    vectorizer_ingredients = vectorizer_ingredients.toarray()

    input_features = np.hstack((scaled_nutrients,vectorizer_ingredients))

    distances, indices = knn.kneighbors(input_features)

    st.write("The top five recommended recipes based on your input: ")

    matched_recipes = recipes.iloc[indices[0]]
    resultant_names = matched_recipes['name']

    #preserve the needed session variables:
    st.session_state.user_input = {
        'matched_recipes': matched_recipes, 'resultant_names':resultant_names, 'pantry': ingredients
    }
    st.session_state.user_nutrition = calculate_nutrition(age,weight,height,gender)

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
    ingredients = eval(recipe["ingredients"])
    for item in ingredients:
        st.markdown(f"\t\t\t- {item}")

    #nutrition
    st.subheader("Nutrition")
    nutrition = eval(recipe["nutrition"])
    for key,value in nutrition.items():
        st.markdown(f"\t\t\t- {key}: {value}")

    #instructions
    st.subheader("Instructions")
    instructions = eval(recipe["instructions"])
    for item in instructions:
        st.markdown(f"\t\t\t- {item}")

    # for passing to llm
    result_formatted = recipe[['name','ingredients','nutrition','instructions']]

    # user query
    st.subheader("Need a variation? Ask AI")
    user_needs = st.text_input("Enter your request",key="user_query" ,value="Replace paneer with tofu, make it gluten-free...")

    #context prompt
    template = '''You are a nutrition assistant
                here are some recipies name with their ingredients and nutrition:
                {result_formatted}
                Here is the user entered nutrition needs and pantry items: 
                {user_input}
                Answer the following user query: {user_needs}, keeping in mind the required nutrition needs.
                You can make ingredient substitions only from the provided recipies, do not generate a new recipe.
                If you are unable to make substitions just say I don't know.
'''

    if user_needs:
        model = ChatOllama(model="llama3.1")
        prompt_template = ChatPromptTemplate.from_template(template)
        prompt = prompt_template.format(result_formatted=result_formatted,user_input=st.session_state.user_nutrition,
                                        user_needs=user_needs)

        try:
            response = model.invoke(prompt)
            st.markdown("### AI Response")
            st.write(response.content)
        except Exception as e:
            st.error(f"Error from LLaMA/Ollama: {e}")
    
    # Navigation button
    if st.button("<- Back to recipe list"):
        st.session_state.selected_recipe_index = None
        st.rerun()
