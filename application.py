from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import uvicorn
from agent import agent_executor
load_dotenv()

# Initialize FastAPI application
application = FastAPI()

# Add CORS middleware
application.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatCompletionRequest(BaseModel):
    message: str

class ChatCompletionResponse(BaseModel):
    content: str

@application.post("/chat_completion/", response_model=ChatCompletionResponse)
async def chat_completion(request: ChatCompletionRequest):
    try:
        # Run the agent with the user's message
        result = agent_executor.invoke({"input": request.message})
        return ChatCompletionResponse(response=result["output"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
if __name__ == "__main__":
    uvicorn.run(application, host="0.0.0.0", port=8000, reload=True)