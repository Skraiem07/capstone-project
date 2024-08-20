import streamlit as st
import pandas as pd
from recommendation_helper import recommend_cosmetics,df
from streamlit_navigation_bar import st_navbar
import time
from rag_skin import get_recommendation
st.set_page_config(
    page_title="Porefectionist",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",

    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    },

    )

# Custom CSS for the sidebar
sidebar_style = """
<style>
    [data-testid="stSidebar"] button {
  background-color: #c2fbd7;
  border-radius: 100px;
  box-shadow: rgba(44, 187, 99, .2) 0 -25px 18px -14px inset,rgba(44, 187, 99, .15) 0 1px 2px,rgba(44, 187, 99, .15) 0 2px 4px,rgba(44, 187, 99, .15) 0 4px 8px,rgba(44, 187, 99, .15) 0 8px 16px,rgba(44, 187, 99, .15) 0 16px 32px;
  color: green;
  cursor: pointer;
  display: inline-block;
  font-family: CerebriSans-Regular,-apple-system,system-ui,Roboto,sans-serif;
  padding: 7px 20px;
  text-align: center;
  text-decoration: none;
  transition: all 250ms;
  border: 0;
  font-size: 16px;
  user-select: none;
  -webkit-user-select: none;
  touch-action: manipulation;
  
}
    [data-testid="stSidebarCollapseButton"] {
        display: none;
    }
button:hover {
  box-shadow: rgba(44,187,99,.35) 0 -25px 18px -14px inset,rgba(44,187,99,.25) 0 1px 2px,rgba(44,187,99,.25) 0 2px 4px,rgba(44,187,99,.25) 0 4px 8px,rgba(44,187,99,.25) 0 8px 16px,rgba(44,187,99,.25) 0 16px 32px;
  transform: scale(1.05) rotate(-1deg);
}
</style>
"""



# Custom CSS for sidebar
sidebar_color = "#73cfc9"  # Replace this with your desired color
sidebar_text_color = "##73cfc9"  # Replace this with your desired text color

st.markdown(
    f"""
    <style>
    .css-1d391kg {{
        background-color: {sidebar_color};
        color: {sidebar_text_color};
    }}
    .css-1d391kg a {{
        color: {sidebar_text_color};
    }}
    .css-1d391kg .stMarkdown {{
        color: {sidebar_text_color};
    }}
    </style>
    """,
    unsafe_allow_html=True
)

def main():
    top_recommended_products=pd.DataFrame()
    page = st_navbar(["Home", "Review analysis"])
   # Input features
# Inject the CSS into the Streamlit app
    st.markdown(sidebar_style, unsafe_allow_html=True)


    if page=="Home":
        with st.sidebar:
            st.title('Skincare Products Recommendation System')
             # Select skin type (first line)
            skin_type = st.selectbox('Select your skin type:', ('Combination', 'Dry', 'Normal', 'Oily', 'Sensitive'))
            # Load data from the helper script (you can load unique labels and brands directly)
            unique_labels = df['Label'].unique().tolist()
            unique_labels.insert(0, 'All')

            # Select label filter (second line)
            label_filter = st.selectbox('Filter by label (optional):', unique_labels)
            # Select rank filter (third line)
            rank_filter = st.slider('Select rank range:', 
                                    min_value=int(df['Rank'].min()), 
                                    max_value=int(df['Rank'].max()), 
                                    value=(int(df['Rank'].min()), int(df['Rank'].max())))

            unique_brands = df['Brand'].unique().tolist()
            unique_brands.insert(0, 'All')

            # Select brand filter (fourth line)
            brand_filter = st.selectbox('Filter by brand (optional):', unique_brands)

            # Select price range (fifth line)
            price_range = st.slider('Select price range:', 
                                    min_value=float(df['Price'].min()), 
                                    max_value=float(df['Price'].max()), 
                                    value=(float(df['Price'].min()), float(df['Price'].max())))

            # Enter ingredients (sixth line)
            st.write("Or enter ingredients to get product recommendations (optional):") 
            ingredient_input = st.text_area("Ingredients (comma-separated)", "")

            # Button to find similar products (seventh line)
            if st.button('Find similar products!'):
                top_recommended_products = recommend_cosmetics(skin_type, label_filter, rank_filter, brand_filter, price_range, ingredient_input)

    # Check if top_recommended_products is empty
        if top_recommended_products.empty:
                st.write("No recommended products available.")
        else:

            with st.spinner("Just a moment while we create your personalized recommendation 🧴✨"):
                st.empty()
               
          
        # Create a clickable table with expandable details
                st.subheader('Recommended Products')
                print(top_recommended_products)
                for index, row in top_recommended_products.iterrows():
                    personal= get_recommendation(skin_type, label_filter, row['Ingredients'])
                    with st.expander(f"{row['Name']}"):
                        st.write(f"**Product Name**: {row['Name']}")
                        st.write(f"**Label**: {row['Label']}")
                        st.write(f"**Brand**: {row['Brand']}")
                        st.write(f"**Ingredients**: {row['Ingredients']}")
                        st.write(f"**Rank**: {row['Rank']}")
                        st.write(f"**Price**: {row['Price']}") 

                        #if st.button(f"Generate Recommendation for {row['Name']}", key=index):
                        #personal = get_recommendation(skin_type, label_filter, row['Ingredients'])

                        if personal !="": 
                            st.write(f"**Generated Recommendation**: {personal}")
                        else:
                            st.write("There is not enough information about the product available to give a recommendation.")

                       


    elif page=="Review analysis":
        st.write("review analysis")
        with st.sidebar:
            with st.echo():
                st.write("This code will be printed to the sidebar.")


        
   

    


if __name__ == "__main__":
    main()
