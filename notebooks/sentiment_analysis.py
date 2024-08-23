import pandas as pd

# Load data
file_path = '../data/clean_products.csv'
df_products = pd.read_csv(file_path,low_memory=False)
df_sentiment=pd.read_csv("../data/final_reviews.csv", low_memory=False)


"""def review_analysis(primary_category , brand_name, product_name):
    filtered_products = df_products[
    (df_products['primary_category'] == primary_category) &
    (df_products['brand_name'] == brand_name) &
    (df_products['product_name'] == product_name)]       
    if not filtered_products.empty:
       return filtered_products
    return "null"

"""


def analyze_product_reviews(product_id):
    # Filter reviews for the specific product ID
    product_reviews = df_sentiment[df_sentiment['product_id'] == product_id]
    
    # Check if there are any reviews for the product
    if product_reviews.empty:
        return f"No reviews found for product ID {product_id}"
    
    # Calculate the mean review score
    mean_review_score = product_reviews['rating'].mean()
    
    # Calculate the number of positive reviews
    num_positive_reviews = len(product_reviews[product_reviews['sentiment'] == 'Positive'])
    
    # Calculate the number of negative reviews
    num_negative_reviews = len(product_reviews[product_reviews['sentiment'] == 'Negative'])
    
    # Calculate the total number of reviews
    total_reviews = len(product_reviews)
    
    # Calculate the recommendation rate
    recommendation_rate = product_reviews['is_recommended'].mean() * 100  # Convert to percentage
    
    # Calculate the average sentiment score
    avg_sentiment_score = round(product_reviews['sentiment_score'].mean(),2)
    
    # Return a summary dictionary with formatted keys
    return {
        "Mean Review Score": mean_review_score,
        "Number of Positive Reviews": num_positive_reviews,
        "Number of Negative Reviews": num_negative_reviews,
        "Total Reviews": total_reviews,
        "Recommendation Rate (%)": recommendation_rate,
        "Average Sentiment Score": avg_sentiment_score
    }
