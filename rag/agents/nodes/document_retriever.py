import logfire
from rag.agents.conversation_state import AgentState
from rag.services.retrieval.vector_store import search_rag_knowledge
from rag.services.retrieval.reranker import rerank_documents

def retrieve_node(state: AgentState):
    """
    Performs vector search and semantic reranking for technical queries.
    """
    query = state["current_query"]
    
    
    # Standard Retrieval Logic
    with logfire.span("🔍 Knowledge Retrieval"):
        logfire.info(f"Searching Qdrant for: {query}")
        top_k = state.get("top_k", 4)
        raw_results = search_rag_knowledge(query, limit=max(15, top_k), scope=state.get("scope", "reference"))
        logfire.info(f"Retrieved {len(raw_results)} candidates from Vector DB")
        
        doc_contents = [doc['content'] for doc in raw_results]
        
        with logfire.span("⚖️ Semantic Reranking"):
            reranked_contents = rerank_documents(query, doc_contents, top_n=top_k)
            logfire.info(f"Reranking complete. Kept top {top_k} most relevant chunks.")
            
        formatted_docs = [f"CONTENT: {doc}" for doc in reranked_contents]
    
    return {
        "documents": formatted_docs,
        "status": f"Found technical context.",
        "plan": state["plan"] + ["Context Retrieved"]
    }
