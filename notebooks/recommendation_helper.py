import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast

# Load data
file_path = '../data/clean_products.csv'
df = pd.read_csv(file_path)

def combine_ingredients_highlights(row):
    """Convert the ingredients column from string representation to a list and combine it into a single string."""
    ingredients_str = row['ingredients']
    if pd.isna(ingredients_str):
        ingredients_str = '[]'  # Use an empty list as a fallback

    # Convert string representation of list to actual list
    try:
        ingredients = ast.literal_eval(ingredients_str)
    except (ValueError, SyntaxError):
        ingredients = []  # Default to an empty list if literal_eval fails

    # Combine ingredients into a single string
    combined_ingredients = ', '.join(ingredients)
    return combined_ingredients

def filter_and_recommend_products(filtered_products, ingredient_input, num_recommendations):
    """Filter products based on ingredient input and apply cosine similarity for recommendations."""
    # Ensure 'Combined' column exists
    filtered_products['Combined'] = filtered_products.apply(combine_ingredients_highlights, axis=1)
    
    if ingredient_input:
        ingredients = [ingredient.strip().lower() for ingredient in ingredient_input.split(',')]
        filtered_products = filtered_products[
            filtered_products['Combined'].apply(lambda x: all(ingredient in x.lower() for ingredient in ingredients))
        ]
        
        if not filtered_products.empty:
            # Apply TF-IDF vectorization and cosine similarity
            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix_filtered = vectorizer.fit_transform(filtered_products['Combined'])
            input_vec = vectorizer.transform([ingredient_input])
            cosine_similarities = cosine_similarity(input_vec, tfidf_matrix_filtered).flatten()
            recommended_indices = cosine_similarities.argsort()[-num_recommendations:][::-1]
            filtered_products = filtered_products.iloc[recommended_indices]

    return filtered_products.sort_values(by=['rating'], ascending=False).head(num_recommendations)

def recommend_cosmetics(product_category, brand_filter, price_range, rating_range, skin_type=None, ingredient_input=None, num_recommendations=5):
    """Recommend cosmetics based on various filters and ingredient input."""
    # Apply initial filters
    filtered_products = df.copy()  # Use a copy of the DataFrame to avoid modifying the original
    
    if product_category != 'All':
        filtered_products = filtered_products[filtered_products['secondary_category'] == product_category]
    
    if skin_type != 'all':
        filtered_products = filtered_products[filtered_products[skin_type] == 1]
    
    filtered_products = filtered_products[
        (filtered_products['rating'] >= rating_range[0]) & 
        (filtered_products['rating'] <= rating_range[1])
    ]
    
    filtered_products = filtered_products[
        (filtered_products['price_usd'] >= price_range[0]) & 
        (filtered_products['price_usd'] <= price_range[1])
    ]
    
    if brand_filter != 'All':
        filtered_products = filtered_products[filtered_products['brand_name'] == brand_filter]
    
    # Filter and recommend products based on ingredients
    filtered_products = filter_and_recommend_products(filtered_products, ingredient_input, num_recommendations)
    
    return filtered_products
