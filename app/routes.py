from fastapi import APIRouter
from pydantic import BaseModel

from app.search import process_query
from app.cache import cache
from app.config import config


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


@router.get("/health")
def health():

    total_cache_entries = sum(
        len(cluster["entries"])
        for cluster in cache.cache.values()
    )

    return {
        "status": "ok",
        "documents_loaded": len(config.documents),
        "clusters": config.cluster_model.n_components,
        "cache_entries": total_cache_entries
    }