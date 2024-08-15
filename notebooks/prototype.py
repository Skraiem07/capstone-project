import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Load data
file_path = '../data/cosmetics.csv'
df = pd.read_csv(file_path)
df['features'] = df['Ingredients']

# TF-IDF Vectorizer
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['features'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
indices = pd.Series(df.index, index=df['Name']).drop_duplicates()

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
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix_filtered = vectorizer.fit_transform(filtered_products['Ingredients'])
        input_vec = vectorizer.transform([ingredient_input])
        cosine_similarities = cosine_similarity(input_vec, tfidf_matrix_filtered).flatten()
        recommended_indices = cosine_similarities.argsort()[-num_recommendations:][::-1]
        filtered_products = filtered_products.iloc[recommended_indices]
    
    return filtered_products.sort_values(by=['Rank']).head(num_recommendations)

def main():
    st.title('Skincare Products Recommendation System')

    col1, col2, col3 = st.columns(3)

    with col1:
        skin_type = st.selectbox('Select your skin type:', ('Combination', 'Dry', 'Normal', 'Oily', 'Sensitive'))

    unique_labels = df['Label'].unique().tolist()
    unique_labels.insert(0, 'All')

    with col2:
        label_filter = st.selectbox('Filter by label (optional):', unique_labels)

    with col1:
        rank_filter = st.slider('Select rank range:', min_value=int(df['Rank'].min()), max_value=int(df['Rank'].max()), value=(int(df['Rank'].min()), int(df['Rank'].max())))

    unique_brands = df['Brand'].unique().tolist()
    unique_brands.insert(0, 'All')

    with col2:
        brand_filter = st.selectbox('Filter by brand (optional):', unique_brands)

    with col3:
        price_range = st.slider('Select price range:', min_value=float(df['Price'].min()), max_value=float(df['Price'].max()), value=(float(df['Price'].min()), float(df['Price'].max())))

    st.write("Or enter ingredients to get product recommendations (optional):") 
    ingredient_input = st.text_area("Ingredients (comma-separated)", "")

    if st.button('Find similar products!'):
        top_recommended_products = recommend_cosmetics(skin_type, label_filter, rank_filter, brand_filter, price_range, ingredient_input)
        
        st.subheader('Recommended Products')
        
        # Create a clickable table with expandable details
        for index, row in top_recommended_products.iterrows():
            with st.expander(f"{row['Name']}"):
                st.write(f"**Product Name**: {row['Name']}")
                st.write(f"**Label**: {row['Label']}")
                st.write(f"**Brand**: {row['Brand']}")
                st.write(f"**Ingredients**: {row['Ingredients']}")
                st.write(f"**Rank**: {row['Rank']}")
                st.write(f"**Price**: {row['Price']}")

if __name__ == "__main__":
    main()
