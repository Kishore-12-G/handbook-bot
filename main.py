from fastapi import FastAPI
from pydantic import BaseModel
from app.chain import chain
from app.memory import conversation_memory

app = FastAPI()

class QueryRequest(BaseModel):
    question: str
    reset_memory: bool = False

class QueryResponse(BaseModel):
    answer: str

@app.post("/ask", response_model=QueryResponse)
async def ask_question(query: QueryRequest):
    try:
        if query.reset_memory:
            conversation_memory.clear()
        
        result = chain.invoke(query.question)
        
        if hasattr(result, 'content'):
            answer = result.content
        else:
            answer = str(result)

        print(conversation_memory.get_memory())
        
        return {"answer": answer}
    except Exception as e:
        return {"answer": f"Error: {str(e)}"}


    