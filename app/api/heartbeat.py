from fastapi import APIRouter

heartbeat_router = APIRouter()


# /healthzはcloud runで予約されているため、/heartbeatに変更
@heartbeat_router.get("/heartbeat", status_code=200)
def healthz() -> dict:
    """
    Healthz
    """
    return {"status": "OK"}
