import logging
from fastapi import FastAPI
import inngest
import inngest.fast_api
from inngest.experimental import ai
from dotenv import load_dotenv
import uuid
import os
import datetime


from data_loader import load_and_chunk_pdf,embed_texts
from vector_db import QdrantStorage
from custom_types import RAGSearchResult, RAGUpsertResult, RAGChunkAndSrc, RAGQueryResult

load_dotenv()
inngest_client = inngest.Inngest(
    app_id = "intellify_task_app",
    logger = logging.getLogger("uvicorn"),
    is_production = False,
    serializer = inngest.PydanticSerializer()
)


@inngest_client.create_function(
    fn_id = "RAG: Ingest PDF",
    trigger = inngest.TriggerEvent(
        event = "rag/ingest_pdf"
    )
)
async def rag_ingest_pdf(ctx : inngest.Context):#This function is to keep track that the PDF inputted by client should first retrieved by the inngest and format it to the API
    def _load(ctx: inngest.Context) -> RAGChunkAndSrc:
        pdf_path = ctx.event.data["pdf_path"]
        source_id = ctx.event.data.get("source_id", pdf_path)
        chunks = load_and_chunk_pdf(pdf_path)
        return RAGChunkAndSrc(chunks = chunks, source_id = source_id)

    def _upsert(chunks_and_src: RAGChunkAndSrc) -> RAGUpsertResult:
        chunks = chunks_and_src.chunks
        source_id = chunks_and_src.source_id
        vecs = embed_texts(chunks)
        ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}:{i}")) for i in range(len(chunks))]
        payloads = [{"source":source_id, "text":chunks[i]} for i in range(len(chunks))]
        QdrantStorage(dim=384).upsert(ids,vecs,payloads)
        return RAGUpsertResult(ingested = len(chunks))

    chunks_and_src = await ctx.step.run(
        "load-and-chunk", #step_id
        lambda: _load(ctx),
        output_type=RAGChunkAndSrc
    )

    ingested = await ctx.step.run(
        "embed-and-upsert", #step_id
        lambda: _upsert(chunks_and_src),
        output_type=RAGUpsertResult
    )

    return ingested.model_dump()


@inngest_client.create_function(
    fn_id = "RAG: Query PDF",
    trigger = inngest.TriggerEvent(event = "rag/query_pdf_ai")
)
async def rag_query_pdf_ai(ctx: inngest.Context):
    def _search(question: str, top_k: int = 5) -> RAGSearchResult:
        query_vec = embed_texts([question])[0]
        store = QdrantStorage(dim=384)
        found = store.search(query_vec, top_k)
        return RAGSearchResult(contexts=found["contexts"], sources = found["sources"])
    question = ctx.event.data["question"]

    top_k = int(ctx.event.data.get("top_k", 5))

    found = await ctx.step.run("embed-and-search", lambda: _search(question, top_k), output_type=RAGSearchResult)

    context_block = "\n\n".join(f"- {c}" for c in found.contexts)
    user_content = (
        "Use the following context to answer the question.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Questions:{question}\n"
        "Answer conscisely using the context above."
    )
    adapter = ai.openai.Adapter(
    auth_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    model="gemini-3.1-flash-lite-preview",
)
    res = await ctx.step.ai.infer(
        "llm-answer",
        adapter = adapter,
        body = {
            "max_tokens": 1024,
            "temperature":0.2,
            "messages":[
                {"role": "system", "content": "You only answer questions using only the provided context."},
                {"role":"user", "content": user_content}
            ]
        }

    )

    answer = res["choices"][0]["message"]["content"].strip()
    return {"answer":answer,"sources": found.sources, "num_contexts": len(found.contexts)}







app = FastAPI()

inngest.fast_api.serve(app, inngest_client, functions = [rag_ingest_pdf, rag_query_pdf_ai])

