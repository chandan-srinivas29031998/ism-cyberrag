import re

from src.config import GROQ_API_KEY, LLM_MODEL_NAME, LLM_PROVIDER, OLLAMA_BASE_URL
from src.guardrail import OOS_REFUSAL

# System prompt for the ISM CyberRAG assistant
SYSTEM_PROMPT = f"""You are an assistant that answers questions about the Australian Information Security Manual (ISM). You must answer using ONLY the context chunks provided below. Do not use any outside knowledge.

Rules:
1. Every factual claim in your answer MUST be directly supported by one of the provided context chunks. If a chunk contains the information, cite its ISM control ID in parentheses, for example: (ISM-1234). Do not state anything that is not in the provided context.
2. If the context chunks do not contain enough information to answer the question, say exactly: "{OOS_REFUSAL}" Do not guess or speculate.
3. Keep your answer concise and direct. Answer only what the question asks. Do not add tangential controls, implementation details, or background unless the question explicitly asks for them.
4. When multiple ISM controls are relevant, synthesize them into a coherent answer and cite each control ID. Group related controls together rather than listing them one by one without explanation.
5. Do not fabricate control IDs. Only cite IDs that appear in the provided context chunks.
6. Never cite "(ISM-None)", "(None)", or "(N/A)". If a relevant chunk has no control ID, describe the guideline or definition without a parenthetical control citation.
7. For definition questions, prefer the exact definition wording from the context. Check all chunks, including Cyber security terminology chunks, before refusing.
8. For multi-part questions, answer under the same parts asked by the question and keep each part limited to the strongest directly relevant controls."""


def _valid_control_ids(context_chunks: list[dict]) -> set[str]:
    """Return all ISM control IDs explicitly present in the supplied context."""
    valid = set()
    for chunk in context_chunks:
        control_id = chunk.get("control_id")
        if isinstance(control_id, str) and re.fullmatch(r"ISM-\d{4}", control_id):
            valid.add(control_id)
        valid.update(re.findall(r"\bISM-\d{4}\b", chunk.get("content", "")))
    return valid


def _sanitize_citations(answer: str, context_chunks: list[dict]) -> str:
    """
    Remove malformed or unsupported parenthetical ISM citations.

    The LLM can still emit strings such as (ISM-None) or
    (ISM-Cyber security terminology 10). We keep only real ISM control IDs
    that appear in the retrieved context.
    """
    valid_controls = _valid_control_ids(context_chunks)

    def replace_ism_parenthetical(match: re.Match) -> str:
        citation_text = match.group(1)
        cited_controls = re.findall(r"\bISM-\d{4}\b", citation_text)
        if not cited_controls:
            return ""

        kept = []
        for control in cited_controls:
            if control in valid_controls and control not in kept:
                kept.append(control)

        if not kept:
            return ""
        if len(kept) == len(cited_controls):
            return match.group(0)
        return f"({', '.join(kept)})"

    cleaned = re.sub(r"\(([^)]*\bISM-[^)]*)\)", replace_ism_parenthetical, answer)
    cleaned = re.sub(r"\s*\((?:ISM-)?(?:None|N/A|null)\)", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\s+([.,;:])", r"\1", cleaned)
    return cleaned.strip()


def generate_answer(question: str, context_chunks: list[dict]) -> str:
    """
    Generates an answer using the configured LLM provider (Groq or Ollama) with source context.

    Args:
        question:        The user's question.
        context_chunks:  List of chunk dicts (must have 'content' key,
                         optionally 'control_id', 'category', 'similarity').

    Returns:
        The generated answer string.
    """
    # Build context block from retrieved chunks
    context_parts = []
    for i, chunk in enumerate(context_chunks, 1):
        control = chunk.get("control_id") or "N/A"
        category = chunk.get("category", "")
        sim = chunk.get("similarity", "")
        header = f"[Chunk {i}] Control: {control}"
        if category:
            header += f" | Category: {category}"
        if sim:
            header += f" | Similarity: {sim:.4f}" if isinstance(sim, float) else f" | Similarity: {sim}"
        context_parts.append(f"{header}\n{chunk['content']}")

    context_text = "\n\n".join(context_parts)

    user_message = f"""Context:
{context_text}

Question: {question}"""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    if LLM_PROVIDER == "ollama":
        from openai import OpenAI
        # Ollama provides an OpenAI compatible API
        client = OpenAI(base_url=f"{OLLAMA_BASE_URL}/v1", api_key="ollama")
        
        response = client.chat.completions.create(
            model=LLM_MODEL_NAME, # using model from config
            messages=messages,
            temperature=0.1,
        )
        return _sanitize_citations(response.choices[0].message.content, context_chunks)
        
    else: # Default to Groq
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY must be set when LLM_PROVIDER='groq'. Check your .env file.")
        
        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY)
        
        response = client.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=messages,
            temperature=0.1,
            max_tokens=1024,
        )
        return _sanitize_citations(response.choices[0].message.content, context_chunks)
