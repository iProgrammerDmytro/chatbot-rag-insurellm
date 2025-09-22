from fastapi import APIRouter, HTTPException, status

from app.core.config import get_settings

from .schemas import AskRequest, AskResponse
from .services import ChatbotService

settings = get_settings()

router = APIRouter(prefix="/chatbot", tags=["chatbot"])


@router.post("/ask")
async def ask(payload: AskRequest) -> AskResponse:
    chatbot = ChatbotService()
    print(payload)

    try:
        return await chatbot.ask(payload)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chatbot failure: {type(e).__name__}: {e}",
        )
