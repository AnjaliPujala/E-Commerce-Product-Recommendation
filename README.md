# E-Commerce-Product-Recommendation

This project is an E-commerce Product Recommendation System built using Google Colab. The dataset used for this project is from Kaggle, containing data about various grocery and gourmet food items. The recommendation system is content-based, using product names to find similar items.

## Project Description

The primary goal of this project is to recommend similar products based on their names. The process involves using TF-IDF Vectorizer and Cosine Similarity to measure the similarity between product names. Here’s a step-by-step breakdown:

1. **Data Import and Initial Inspection:**
   - Import necessary libraries: NumPy, Pandas, Matplotlib, Seaborn.
   - Load the dataset and display the first few rows.

2. **TF-IDF Vectorization:**
   - Initialize a `TfidfVectorizer` object.
   - Fit the vectorizer to the 'name' column of the DataFrame and transform it into a matrix of TF-IDF features.

3. **Cosine Similarity Calculation:**
   - Calculate the cosine similarity between all pairs of items based on their TF-IDF vectors.
   - Display the cosine similarity matrix.

4. **Recommendation Function:**
   - Define a function `Recommend` that takes an item name as input and returns the top 10 similar items based on cosine similarity.

## How to Use
 **Clone the Repository:**
   ```sh
   git clone https://github.com/AnjaliPujala/e-commerce-product-recommendation.git
   cd e-commerce-product-recommendation

   # Install Dependencies:
   pip install numpy pandas matplotlib seaborn scikit-learn

   # Run the Jupyter Notebook:
   # Open the Jupyter Notebook file in Google Colab or your local Jupyter environment and run the cells step-by-step to load the data, perform TF-IDF vectorization, calculate cosine similarity, and generate recommendations.
   
   # Generate Recommendations:
   # Use the `Recommend` function to get product recommendations. For example:
   Recommend('Nutraj 100% Natural Dried Premium California Walnut Kernels, 500g (2 X 250g) | Pure Without Shell Walnut Kernels | Akhrot ...')


