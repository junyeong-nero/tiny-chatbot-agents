import re

SECTION_REFERENCE_PATTERNS = [
    r"제?\s*(\d+)\s*조\s*(?:제?\s*(\d+)\s*항)?",
    r"(\d+)\s*조\s*(\d+)\s*항",
]


def extract_section_reference(query: str) -> str | None:
    for pattern in SECTION_REFERENCE_PATTERNS:
        match = re.search(pattern, query)
        if not match:
            continue

        article = match.group(1)
        paragraph = match.group(2) if len(match.groups()) > 1 else None
        if paragraph:
            return f"제{article}조 제{paragraph}항"
        return f"제{article}조"

    return None


def extract_citations(answer: str) -> list[str]:
    return re.findall(r"\[참조:\s*([^\]]+)\]", answer)


def extract_usage_metadata(response: object) -> dict[str, int | str]:
    usage = getattr(response, "usage", {}) or {}
    return {
        "llm_model": getattr(response, "model", ""),
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
        "tokens_used": usage.get("total_tokens", 0),
    }
