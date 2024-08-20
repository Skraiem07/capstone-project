import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
# Load data
file_path = '../data/cosmetics.csv'
df = pd.read_csv(file_path)
df['features'] = df['Ingredients']

def recommend_cosmetics(skin_type, label_filter, rank_filter, brand_filter, price_range, ingredient_input=None, num_recommendations=5):
    # Apply initial filters
    filtered_products = df[df[skin_type] == 1]
    
    if label_filter != 'All':
        filtered_products = filtered_products[filtered_products['Label'] == label_filter]
    
    filtered_products = filtered_products[
        (filtered_products['Rank'] >= rank_filter[0]) & 
        (filtered_products['Rank'] <= rank_filter[1])
    ]
    
    if brand_filter != 'All':
        filtered_products = filtered_products[filtered_products['Brand'] == brand_filter]
    
    filtered_products = filtered_products[
        (filtered_products['Price'] >= price_range[0]) & 
        (filtered_products['Price'] <= price_range[1])
    ]

    # Apply ingredient filter on the filtered DataFrame
    if ingredient_input:
        ingredients = [ingredient.strip().lower() for ingredient in ingredient_input.split(',')]
        filtered_products = filtered_products[
            filtered_products['Ingredients'].apply(lambda x: all(ingredient in x.lower() for ingredient in ingredients))
        ]
        
        if not filtered_products.empty:
            # Apply cosine similarity to the filtered DataFrame based on ingredients
            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix_filtered = vectorizer.fit_transform(filtered_products['Ingredients'])
            input_vec = vectorizer.transform([ingredient_input])
            cosine_similarities = cosine_similarity(input_vec, tfidf_matrix_filtered).flatten()
            recommended_indices = cosine_similarities.argsort()[-num_recommendations:][::-1]
            filtered_products = filtered_products.iloc[recommended_indices]

    return filtered_products.sort_values(by=['Rank']).head(num_recommendations)
