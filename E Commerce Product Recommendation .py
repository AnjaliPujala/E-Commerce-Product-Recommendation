import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data= pd.read_csv('C:\\Users\\HELLO\\OneDrive\\Desktop\\MyProjects\\All Grocery and Gourmet Foods.csv')
data.head()

#import requirements
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize a TfidfVectorizer object
vectorizer= TfidfVectorizer()
# Fit the vectorizer to the 'name' column of the DataFrame and transform it into a matrix of TF-IDF features
item_vectors= vectorizer.fit_transform(data['name'])

# Calculate the cosine similarity between all pairs of items based on their TF-IDF vectors
cosine_item_vectors = cosine_similarity(item_vectors)
# Display the cosine similarity matrix
cosine_item_vectors

data['name'][6]

item='Nutraj 100% Natural Dried Premium California Walnut Kernels, 500g (2 X 250g) | Pure Without Shell Walnut Kernels | Akhrot ...'
data[data['name']==item].index[0]

item='Nutraj 100% Natural Dried Premium California Walnut Kernels, 500g (2 X 250g) | Pure Without Shell Walnut Kernels | Akhrot ...'
# Keep the original item string for later use
original_item = item
item_vector = vectorizer.transform([item])  # Transform the item string

cosine_item = cosine_similarity(item_vector, item_vectors)

list_cosine_item = list(enumerate(cosine_item[0]))

sorted_item = sorted(list_cosine_item, key=lambda x:x[1], reverse=True)
sorted_item

sorted_item= sorted_item[1:11]
sorted_item

indices=[index for index,val in sorted_item]


data['name'][indices]

def Recommend(item):
  # Keep the original item string for later use
  original_item = item

  #Transform the item to float
  item_vector = vectorizer.transform([item])

  #cosine similarity
  cosine_item = cosine_similarity(item_vector, item_vectors)

  #Enumerate to get indices of similar items
  list_cosine_item = list(enumerate(cosine_item[0]))

  #Sort the list in descending order of similarity score
  sorted_item = sorted(list_cosine_item, key=lambda x:x[1], reverse=True)

  #Get the indices of the top 10 similar items
  sorted_item= sorted_item[1:11]

  #Get the names of the top 10 similar items
  indices=[index for index,val in sorted_item]
  recommended_items=data['name'][indices]

  #return item names
  return recommended_items

#TestCase 1
print("Test Case 1 : ",'\n',Recommend('Nutraj 100% Natural Dried Premium California Walnut Kernels, 500g (2 X 250g) | Pure Without Shell Walnut Kernels | Akhrot ...'))

#TestCase 2
print("Test Case 2 : ",'\n',Recommend('Dabur Vedic Tea - 500gm | Handpicked from Assam, Nilgiri & Darjeeling | Soulful Aroma & Rich Taste | 30+ Ayurvedic Herbs |...'))