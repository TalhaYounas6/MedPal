from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from query_engine import query_mediquery

app = FastAPI()

# CORS setup to allow React frontend to talk to FastAPI backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_methods=["POST", "OPTIONS"],  
    allow_headers=["Content-Type"], 
    allow_credentials=True,
)

class Question(BaseModel):
    question: str

@app.get("/")
async def root():
    return {"message": "MediQuery API is running"}    

@app.post("/ask")
async def ask_question(data: Question):
    try:
        response = query_mediquery(data.question)
        return {"answer": response}
    except Exception as e:
        print(f"Error in backend: {e}")
        
