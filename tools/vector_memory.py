"""
tools/vector_memory.py
-----------------------
Tool: store_user_profile / retrieve_user_profile
Long-term memory using ChromaDB to store and retrieve candidate profiles
across sessions. Enables longitudinal feedback tracking.
"""

from typing import Any, Optional
import json
from observability import log_tool_call

try:
    import chromadb
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

_client = None
_collection = None

def _get_collection():
    global _client, _collection
    if _collection is None:
        if not CHROMA_AVAILABLE:
            return None
        _client = chromadb.Client()  # In-memory for dev; swap for persistent in prod
        _collection = _client.get_or_create_collection("user_profiles")
    return _collection


def store_user_profile(
    user_id: str,
    job_role: str,
    evaluated_skills: dict[str, str],
    session_summary: str,
    tool_context=None,
) -> dict[str, Any]:
    """
    Store or update a candidate's performance profile in vector memory.

    Args:
        user_id: Unique identifier for the candidate.
        job_role: The role they practiced for.
        evaluated_skills: Skill ratings from this session.
        session_summary: Brief text summary to embed.

    Returns:
        Status dict.
    """
    collection = _get_collection()
    if collection is None:
        result = {"status": "error", "message": "ChromaDB not available."}
        if tool_context:
            log_tool_call(tool_context.state, "verifier_critic", "store_user_profile", {"user_id": user_id, "job_role": job_role}, result)
        return result

    metadata = {
        "user_id": user_id,
        "job_role": job_role,
        "skills_json": json.dumps(evaluated_skills),
    }

    collection.upsert(
        ids=[user_id],
        documents=[session_summary],
        metadatas=[metadata],
    )

    result = {"status": "success", "user_id": user_id}
    if tool_context:
        log_tool_call(tool_context.state, "verifier_critic", "store_user_profile", {"user_id": user_id, "job_role": job_role}, result)
    return result


def retrieve_user_profile(user_id: str, tool_context=None) -> dict[str, Any]:
    """
    Retrieve a candidate's stored profile from vector memory.

    Args:
        user_id: Unique identifier for the candidate.

    Returns:
        Dict with past performance data, or empty if not found.
    """
    collection = _get_collection()
    if collection is None:
        result = {"status": "error", "message": "ChromaDB not available."}
        if tool_context:
            log_tool_call(tool_context.state, "context_optimizer", "retrieve_user_profile", {"user_id": user_id}, result)
        return result

    results = collection.get(ids=[user_id])
    if not results["documents"]:
        result = {"status": "not_found", "user_id": user_id}
        if tool_context:
            log_tool_call(tool_context.state, "context_optimizer", "retrieve_user_profile", {"user_id": user_id}, result)
        return result

    metadata = results["metadatas"][0]
    result = {
        "status": "found",
        "user_id": user_id,
        "job_role": metadata.get("job_role", ""),
        "past_skills": json.loads(metadata.get("skills_json", "{}")),
        "past_summary": results["documents"][0],
    }
    if tool_context:
        log_tool_call(tool_context.state, "context_optimizer", "retrieve_user_profile", {"user_id": user_id}, {"status": "found", "job_role": result["job_role"]})
    return result
