"""Answer generation with Claude — the "G" in RAG.

The retrieved clauses are sent as DOCUMENTS with citations enabled. Claude's
answer then comes back split into text blocks, and each claim carries the exact
clause text it relied on (`cited_text`) and which document it came from. That
makes every statement checkable against the original contract wording.

Rules in the system prompt: answer ONLY from the clauses, cite them, and say so
when the clauses don't answer the question (no guessing from outside knowledge).

Needs credentials: ANTHROPIC_API_KEY in .env (or `ant auth login`).
"""

from contextlib import contextmanager

import anthropic

from legal_contract_clause_search.config import Settings
from legal_contract_clause_search.model import AnswerResponse, Citation, SearchHit

SYSTEM_PROMPT = """You are a contract analysis assistant. You answer questions about commercial contracts using ONLY the contract clauses provided as documents.

- Support every statement with the clauses it comes from; mention which contract (and section, when known) each point refers to.
- When clauses from different contracts differ, say how they differ instead of merging them into one answer.
- If the provided clauses do not answer the question, say that the retrieved clauses don't cover it. Do not use outside knowledge or guess.
- Be concise. This is informational analysis, not legal advice."""


class GenerationError(RuntimeError):
    pass


def _document(i: int, h: SearchHit) -> dict:
    title = f"[{i}] {h.title} | {h.contract_type} | signed {h.sign_date or 'unknown'}"
    if h.section_path:
        title += f" | {h.section_path}"
    return {
        "type": "document",
        "source": {"type": "text", "media_type": "text/plain", "data": h.text},
        "title": title[:500],
        "citations": {"enabled": True},
    }


def _client(settings: Settings) -> anthropic.Anthropic:
    # No key in .env -> the SDK looks for ANTHROPIC_API_KEY in the environment itself.
    return anthropic.Anthropic(api_key=settings.anthropic_api_key) if settings.anthropic_api_key \
        else anthropic.Anthropic()


@contextmanager
def _api_errors():
    """Turn SDK exceptions into one GenerationError with a readable message."""
    try:
        yield
    except anthropic.AuthenticationError as e:
        raise GenerationError("Claude credentials missing/invalid: set ANTHROPIC_API_KEY in .env") from e
    except anthropic.RateLimitError as e:
        raise GenerationError("Claude API rate limit reached, try again shortly") from e
    except anthropic.APIStatusError as e:
        raise GenerationError(f"Claude API error {e.status_code}: {e.message}") from e
    except anthropic.APIConnectionError as e:
        raise GenerationError("Could not reach the Claude API (network)") from e
    except TypeError as e:   # raised by the SDK when no credentials can be resolved at all
        raise GenerationError(f"Claude client not configured: {e}") from e


def _parse_answer(response, hits: list[SearchHit]) -> tuple[str, list[Citation]]:
    """Join the text blocks and turn Claude's citations into [n] markers that point at `hits`."""
    parts: list[str] = []
    citations: list[Citation] = []
    for block in response.content:
        if block.type != "text":
            continue                       # thinking blocks etc. are not part of the answer
        parts.append(block.text)
        markers = []
        for c in getattr(block, "citations", None) or []:
            idx = getattr(c, "document_index", None)
            if idx is None or idx >= len(hits):
                continue
            h = hits[idx]
            citations.append(Citation(hit_index=idx + 1, cited_text=getattr(c, "cited_text", ""),
                                      doc_id=h.doc_id, title=h.title, section_path=h.section_path))
            markers.append(f"[{idx + 1}]")
        if markers:
            parts.append(" " + "".join(dict.fromkeys(markers)))
    return "".join(parts).strip(), citations


def answer(question: str, hits: list[SearchHit], settings: Settings) -> AnswerResponse:
    """Ask Claude to answer `question` using only the retrieved clauses in `hits`."""
    if not hits:
        return AnswerResponse(answer="No clauses matched the query and filters, so there is nothing to answer from.",
                              citations=[], hits=[], model=settings.claude_model)

    content = [_document(i, h) for i, h in enumerate(hits, 1)] + [{"type": "text", "text": question}]
    extra = {"betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"} if settings.claude_fallback else {}

    with _api_errors():
        # Streaming avoids HTTP timeouts on long answers; we just wait for the final message.
        with _client(settings).beta.messages.stream(
            model=settings.claude_model,
            max_tokens=16000,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": content}],
            output_config={"effort": settings.claude_effort},
            **extra,
        ) as stream:
            response = stream.get_final_message()

    if response.stop_reason == "refusal":
        return AnswerResponse(answer="The model declined to answer this request.", citations=[],
                              hits=hits, model=response.model)
    text, citations = _parse_answer(response, hits)
    return AnswerResponse(answer=text, citations=citations, hits=hits, model=response.model)

