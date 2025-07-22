import streamlit as st

user_input = st.chat_input("👨‍🍳 What's in your pantry today?")

with st.form("nutrition_form"):
        st.markdown("### 🎯 Enter Your Nutrition Goals")
        calories = st.number_input("Calories (kcal)", min_value=0.0, step=10.0, value=200.0, format="%.2f")
        proteins = st.number_input("Proteins (g)", min_value=0.0, step=5.0, value=45.0, format="%.2f")
        carbohydrates = st.number_input("Carbohydrates (g)", min_value=0.0, step=5.0, value=78.0, format="%.2f")
        fiber = st.number_input("Fiber (g)", min_value=0.0, step=1.0, value=15.0, format="%.2f")
        fat = st.number_input("Fat (g)", min_value=0.0, step=5.0, value=20.0, format="%.2f")
        sodium = st.number_input("Sodium (g)", min_value=0.0, step=1.0, value=4.0, format="%.2f")

        submitted = st.form_submit_button("Submit")

        if submitted:
            nutritions_goals={
                "calories": [calories],
                "proteins": [proteins],
                "carbohydrates": [carbohydrates],
                "fiber": [fiber],
                "fat": [fat],
                "sodium": [sodium]
            }
            st.session_state["nutritions_goals"] = nutritions_goals
            st.success("✅ Nutrition goals submitted.")



if user_input:
    st.session_state["user_input"] = user_input

    # 3. If both data present, show redirect button
    if "nutritions_goals" in st.session_state:
        st.success("✅ Pantry input received.")
        st.markdown("### ✅ All Set! Click below to get your recipes:")
        st.page_link("pages/1_RecipeChatPage.py", label="🍽 Go to Recipe Page", icon="➡️")
    else:
        st.warning("⚠️ Please submit nutrition goals before proceeding.")


