import pandas as pd
import json

# Load JSON data from file
with open("recipes.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv("recipes_final.csv", index=False, encoding='utf-8')
