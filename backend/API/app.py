from fastapi import FastAPI, HTTPException 
from fastapi.middleware.cors import CORSMiddleware 
from pydantic import BaseModel 
from sentence_transformers import SentenceTransformer 
import faiss 
import numpy as np  
import pandas as pd 
from transformers import pipeline 


class Inputdata(BaseModel):
  text:str
  role:str

app= FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
def predict(data: Inputdata):
  try:
    model=SentenceTransformer('C:/Users/user/Desktop/final project/backend/sentence_transformer_model')
    index=faiss.read_index('C:/Users/user/Desktop/final project/backend/faiss_index_batting_stats')
    embeddings=np.load('C:/Users/user/Desktop/final project/backend/embeddings.npy')
    file='C:/Users/user/Desktop/final project/backend/batting_stats_cleaned.csv'
    df=pd.read_csv(file)
    
    query=data.text
    query_embedding = model.encode([query])[0].reshape(1,-1) 
    k=5 
    distances,indices =index.search(query_embedding,k)
    closest_row= df.iloc[indices[0][0]]
    closest_row_df= closest_row.to_frame().T
    closest_row_dict=closest_row_df.to_dict(orient='records')[0] 
    qa_input = {
      'question' : query,
      'context': str(closest_row_dict)
    }

    qa_pipeline = pipeline('question-answering', model='bert-large-uncased-whole-word-masking-finetuned-squad')

    response=qa_pipeline(qa_input)
    answer=response['answer']
    return {'text':answer,'role':'bot'}
  
  except Exception as e:
    raise HTTPException(status_code=400,detail=str(e))
  
  # Run the app with: uvicorn app:app --reload
  
  