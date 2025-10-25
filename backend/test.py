from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pandas as pd

# Load the SentenceTransformer model
model = SentenceTransformer('sentence_transformer_model')

# Load the FAISS index and embeddings
index = faiss.read_index('faiss_index_batting_stats')
embeddings = np.load('embeddings.npy')

# load data
file = 'C:/Users/Sai kumar/OneDrive/Desktop/RAG/batting_stats.csv'
df = pd.read_csv(file)

# Drop unnecessary columns
df_cleaned = df.drop(columns=['Unnamed: 16', 'Unnamed: 17', 'Unnamed: 18',])

# Drop rows with null values in critical columns
df_cleaned = df_cleaned.dropna(subset=['Player', 'Span', 'Matches', 'Innings', 'Runs'])
# df_cleaned.to_csv('./cleaned_df.csv')
# Make a copy of the original DataFrame to avoid modifications
df_cleaned = df.copy()
query = "How many matches did Virat Kohli play in T20?"
query_embedding = model.encode([query])[0].reshape(1, -1)
k = 5  # Number of nearest neighbors to retrieve
distances, indices = index.search(query_embedding, k)
closest_row = df_cleaned.iloc[indices[0][0]]
closest_row_df = closest_row.to_frame().T

from transformers import pipeline

# Assuming closest_row_df is the DataFrame containing the structured data
closest_row_dict = closest_row_df.to_dict(orient='records')[0]  # Convert to a dictionary

# Create a dictionary in the expected format
qa_input = {
    'question': query,
    'context': str(closest_row_dict)  # The context needs to be a string
}

# Load the pre-trained question-answering model
qa_pipeline = pipeline('question-answering', model='bert-large-uncased-whole-word-masking-finetuned-squad')

# Generate a response from the LLM
response = qa_pipeline(qa_input)
answer = response['answer']
print(answer)
