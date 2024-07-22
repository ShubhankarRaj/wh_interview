from model_service.query_model import QueryRequest, QueryResponse
from model_backend.model_response import get_model_response
from fastapi import FastAPI, HTTPException

app = FastAPI()

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    
    try:
        response_text = get_model_response(question_text=request.query)
        
        return QueryResponse(results=response_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
