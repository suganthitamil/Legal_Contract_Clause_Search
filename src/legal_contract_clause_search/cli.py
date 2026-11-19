"""Command-line interface:  uv run lccs --help

  lccs ingest            build the index (load -> chunk -> embed -> store)
  lccs search "..."      search from the terminal
  lccs answer "..."      search + Claude answer with citations
  lccs serve             start the API (Swagger UI at /docs)
  lccs eval              run the evaluation experiments
  lccs stats             list built indexes
  lccs download-models   fetch optional models (needs huggingface.co access)
"""

import logging
from datetime import date
from typing import Optional

import typer

from legal_contract_clause_search.config import Settings

app = typer.Typer(add_completion=False, no_args_is_help=True, help="Legal Contract Clause Search")


def _settings(**overrides) -> Settings:
    s = Settings()
    return s.model_copy(update={k: v for k, v in overrides.items() if v is not None})


@app.callback()
def main(verbose: bool = typer.Option(False, "--verbose", "-v")):
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    for noisy in ("httpx", "httpcore", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


@app.command()
def ingest(chunker: Optional[str] = typer.Option(None, help="clause|fixed|sliding|recursive|semantic"),
           embedder: Optional[str] = typer.Option(None, help="bge-small|bge-base|voyage-law-2"),
           store: Optional[str] = typer.Option(None, help="qdrant|chroma"),
           limit: Optional[int] = typer.Option(None, help="Only N contracts (quick test)"),
           headers: Optional[bool] = typer.Option(None, "--headers/--no-headers", help="Contextual chunk headers")):
    """Build (or rebuild) the search index."""
    from legal_contract_clause_search.ingest import ingest as run_ingest

    s = _settings(vector_store=store, contextual_headers=headers)
    stats, _ = run_ingest(s, chunker, embedder, limit=limit, recreate=True)
    typer.echo(f"\nIndexed {stats.chunks} chunks from {stats.contracts} contracts into "
               f"{stats.collection!r} in {stats.seconds:.0f}s (avg {stats.avg_tokens} tokens/chunk)")


def _request(query, types, after, before, top_k, mode, rerank, per_contract):
    from legal_contract_clause_search.model import SearchRequest
    return SearchRequest(query=query, contract_types=types or None, signed_after=after, signed_before=before,
                         top_k=top_k, mode=mode, rerank=rerank, max_per_contract=per_contract)


_TYPE = typer.Option(None, "--type", "-t", help="Contract type filter (repeatable)")
_AFTER = typer.Option(None, "--after", help="Signed on/after YYYY-MM-DD")
_BEFORE = typer.Option(None, "--before", help="Signed on/before YYYY-MM-DD")


def _date(value: Optional[str]) -> Optional[date]:
    return date.fromisoformat(value) if value else None


@app.command()
def search(query: str,
           type_: Optional[list[str]] = _TYPE, after: Optional[str] = _AFTER, before: Optional[str] = _BEFORE,
           top_k: int = typer.Option(5, "--top-k", "-k"),
           mode: Optional[str] = typer.Option(None, help="dense|sparse|hybrid"),
           rerank: Optional[bool] = typer.Option(None, "--rerank/--no-rerank"),
           per_contract: Optional[int] = typer.Option(None, "--per-contract", help="Max hits per contract"),
           chunker: Optional[str] = None, embedder: Optional[str] = None, store: Optional[str] = None):
    """Search clauses from the terminal."""
    from legal_contract_clause_search.retrieval import Retriever

    s = _settings(vector_store=store)
    r = Retriever(s, chunker=chunker, embedder=embedder)
    hits = r.search(_request(query, type_, _date(after), _date(before), top_k, mode, rerank, per_contract))
    typer.echo(f"{len(hits)} hits from {r.collection} (mode={r.resolve_mode(mode)})\n")
    for i, h in enumerate(hits, 1):
        typer.echo(f"[{i}] {h.score:.3f}  {h.contract_type} | {h.sign_date or 'no date'} | {h.title[:70]}")
        typer.echo(f"     {h.section_path or '-'}")
        typer.echo("     " + " ".join(h.text.split())[:300] + "\n")
    r.close()


@app.command()
def answer(question: str,
           type_: Optional[list[str]] = _TYPE, after: Optional[str] = _AFTER, before: Optional[str] = _BEFORE,
           top_k: int = typer.Option(5, "--top-k", "-k"),
           mode: Optional[str] = typer.Option(None, help="dense|sparse|hybrid")):
    """Search, then ask Claude to answer from the retrieved clauses (with citations)."""
    from legal_contract_clause_search.generation import GenerationError
    from legal_contract_clause_search.generation import answer as generate
    from legal_contract_clause_search.retrieval import Retriever

    s = _settings()
    r = Retriever(s)
    hits = r.search(_request(question, type_, _date(after), _date(before), top_k, mode, None, None))
    r.close()
    try:
        res = generate(question, hits, s)
    except GenerationError as e:
        typer.echo(f"Generation failed: {e}", err=True)
        raise typer.Exit(1)
    typer.echo(res.answer + "\n\nSources:")
    for i, h in enumerate(res.hits, 1):
        typer.echo(f"  [{i}] {h.title[:70]} | {h.section_path or '-'}")


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000, reload: bool = False):
    """Start the API. Swagger UI: http://localhost:8000/docs"""
    import uvicorn
    uvicorn.run("legal_contract_clause_search.api:app", host=host, port=port, reload=reload)


@app.command("eval")
def evaluate(limit: int = typer.Option(50, help="Contracts in the evaluation sample"),
             stage: str = typer.Option("all", help="chunkers|search|embedders|stores|all"),
             k: int = typer.Option(10, help="Retrieve this many per query")):
    """Run the evaluation experiments -> results/experiments.csv"""
    from legal_contract_clause_search.eval.runner import run

    run(_settings(), limit=limit, stage=stage, k=k)


@app.command()
def stats():
    """List built indexes and their settings."""
    from legal_contract_clause_search.index_meta import all_meta

    for key, m in all_meta(Settings().index_meta_path).items():
        typer.echo(f"{key}\n   {m.get('contracts')} contracts, {m.get('chunks')} chunks, "
                   f"avg {m.get('avg_tokens')} tokens, built {m.get('created')}")


@app.command("download-models")
def download_models(bge_base: bool = typer.Option(True, "--bge-base/--no-bge-base"),
                    reranker: bool = typer.Option(True, "--reranker/--no-reranker")):
    """Download models into models/ (run once on a network that allows huggingface.co)."""
    from fastembed import TextEmbedding
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    s = Settings()
    wanted = [("dense", "BAAI/bge-small-en-v1.5")]
    if bge_base:
        wanted.append(("dense", "BAAI/bge-base-en-v1.5"))
    if reranker:
        wanted.append(("rerank", s.reranker_model_name))
    for kind, name in wanted:
        try:
            (TextEmbedding if kind == "dense" else TextCrossEncoder)(name, cache_dir=str(s.models_dir))
            typer.echo(f"OK      {name}")
        except Exception as e:  # noqa: BLE001 - report and continue with the next model
            typer.echo(f"FAILED  {name}: {str(e)[:150]}")


if __name__ == "__main__":
    app()
