from fastapi import APIRouter
from pydantic import BaseModel

from app.search import process_query
from app.cache import cache


router = APIRouter()


class QueryRequest(BaseModel):
    query: str


@router.post("/query")
def query(req: QueryRequest):

    return process_query(req.query)


@router.get("/cache/stats")
def cache_stats():

    return cache.stats()


@router.delete("/cache")
def clear_cache():

    cache.clear()

    return {"message": "cache cleared"}