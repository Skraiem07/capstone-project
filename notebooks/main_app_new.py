import streamlit as st
import pandas as pd
from recommendation_helper import recommend_cosmetics
from sentiment_analysis import analyze_product_reviews, df_products
import time
from preprocess_documents import initialize_rag, get_recommendation,embeddings_dir
from rag_pdf import initialize_rag_2, get_recommendation_2,faiss_index_path
import base64

# Streamlit configuration
st.set_page_config(
    page_title="Porefectionist",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)

# Function to load and encode the image
def get_base64_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode()
    return encoded

# Specify the path to your local image
image_path = "serum.jpg"  # Replace with your actual image file name
base64_image = get_base64_image(image_path)

# Custom CSS for sidebar, buttons, background image, and expander
custom_style = f"""
<style>
    /* Sidebar styling */
    [data-testid="stSidebar"] {{
        background-color: #FFF3EE; /* Sidebar background color */
        color: #333; /* Sidebar text color */
    }}

    /* Sidebar title */
    [data-testid="stSidebar"] .sidebar .sidebar-title {{
        color: #D20103; /* Title color */
    }}

    /* Main content background */
    section.main {{
        background-image: url(data:image/jpg;base64,{base64_image});
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
/* Set a solid white background for the stExpander component */
[data-testid="stExpander"] {{
    background-color: white; /* Set background to white */
    border-radius: 8px; /* Optional: Add some border radius for rounded corners */
    padding: 10px; /* Optional: Add padding to make content look nicer */
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1); /* Optional: Add a subtle shadow */
}}

[data-testid="stExpander"] summary {{
    background-color: #white; /* Ensure the summary section also has a white background */
    border-radius: 8px; /* Match the border radius with the container */
    padding: 10px; /* Padding for the summary */
}}

[data-testid="stExpander"] summary:hover {{
    background-color: #white; /* Optional: Add a hover effect */
}}


    [data-testid="stSidebarCollapseButton"] {{
        display: none;
    }}

    /* Footer style */
     .footer {{
        position: fixed;
        bottom: 0;
        right: 0;
        padding: 10px;
        color: #555;
        font-size: 0.9rem;
        background-color: rgba(255, 255, 255, 0.9); /* Slightly opaque background for contrast */
        /* Remove border radius */
        border-radius: 0; 
        /* Remove box shadow if desired */
        box-shadow: none; 
    }}

</style>
<div class="footer">Powered by Rakib</div>
"""

st.markdown(custom_style, unsafe_allow_html=True)

def format_list_as_string(lst):
    """
    Convert a list into a formatted string with each item on a new line.
    """
    if isinstance(lst, list):
        # Join the list items into a single string separated by new lines
        formatted_string = '\n'.join(f" - {item.strip()}" for item in lst if item)
        return formatted_string
    return "Not available"

def main():

    # Initialize an empty DataFrame for top recommended products
    chain = initialize_rag_2(faiss_index_path)
    top_recommended_products = pd.DataFrame()
    skin_type = None

    with st.sidebar:
        st.title('Porefectionist')
        st.header('Personalized Skin Care Recommender')

        # Load unique product categories
        unique_labels = df_products['secondary_category'].unique().tolist()
        product_category = st.selectbox('Product Category:', unique_labels)

        
        skin_type = st.selectbox('Skin type:', ('All','Combination', 'Dry', 'Normal', 'Oily')).lower()

        # Filter options for brand, rating, and price
        brand_name = df_products[df_products['secondary_category'] == product_category]["brand_name"].unique().tolist()
        brand_name.insert(0, "All")
        brand_filter = st.selectbox('Brand:', brand_name)

        rating_filter = st.slider(
            'Rating range:',
            min_value=int(df_products['rating'].min()),
            max_value=int(df_products['rating'].max()),
            value=(int(df_products['rating'].min()), int(df_products['rating'].max()))
        )

        price_range = st.slider(
            'Price range:',
            min_value=df_products['price_usd'].min(),
            max_value=df_products['price_usd'].max(),
            value=(df_products['price_usd'].min(), float(df_products['price_usd'].max()))
        )

        # Ingredient input for filtering
        st.write("Enter ingredients(optional):")
        ingredient_input = st.text_area("Ingredients (comma-separated)", "")

        # Button to trigger product recommendation
        if st.button('Get products!', key='find_products'):
            st.session_state.button_clicked = True
            top_recommended_products = recommend_cosmetics(
                product_category, brand_filter, price_range,
                rating_filter, skin_type, ingredient_input
            )
        
        # Display greeting message if button hasn't been clicked yet
    if not top_recommended_products.empty:
        # If products are found, display them
        if not top_recommended_products.empty:
            st.subheader('Recommended Products')
            for _, row in top_recommended_products.iterrows():
                with st.expander(f"{row['product_name']} - {row['brand_name']}"):
                    st.write(f"**Product Name**: {row['product_name']}")
                    st.write(f"**Brand Name**: {row['brand_name']}")
                    st.write(f"**Price**: ${row['price_usd']:.2f}")
                    st.write(f"**Rating**: {row['rating']}")
                    st.write(f"**Ingredients**: {row['ingredients']}")
                    st.write(f"**Highlights**: {row['highlights']}")
                    st.write(f"**Product Category**: {row['secondary_category']}")
                    personal = get_recommendation_2(skin_type, row['product_name'], row['ingredients'],chain)
                    if personal !="": 
                        st.write(f"**Generated Recommendation**: {personal}")
                    else:
                        st.write("There is not enough information about the product available to give a recommendation.")

                 

                 
                    result = analyze_product_reviews(row['product_id'])
                    # Check if the result is a dictionary or a string
                    if isinstance(result, dict):
                        # If it's a dictionary, print the calculated metrics in a web-friendly format
                        st.write("**Product Review Summary**:")
                        for key, value in result.items():
                            st.write(f"{key}: {value}")
                    else:
                        st.write(f"{result}")
                                        
                    

        else:
            st.write("No recommended products available.")


if __name__ == "__main__":
    main()
