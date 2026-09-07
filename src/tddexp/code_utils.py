from __future__ import annotations

import re


def strip_markdown_fences(text: str) -> str:
    text = strip_thinking_blocks(text)
    fence_idx = text.find("```")
    if fence_idx == -1:
        return text

    candidates = code_segments_from_fences(text)
    non_placeholder = [candidate for candidate in candidates if not looks_like_placeholder(candidate)]
    if non_placeholder:
        return max(non_placeholder, key=code_segment_rank).strip() + "\n"
    if candidates:
        return max(candidates, key=code_segment_rank).strip() + "\n"
    prefix = text[:fence_idx].strip()
    if prefix:
        return prefix + "\n"
    return text


def strip_thinking_blocks(text: str) -> str:
    """Remove completed reasoning blocks while preserving the final answer."""
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)


def code_segments_from_fences(text: str) -> list[str]:
    segments = text.split("```")
    candidates: list[str] = []
    prefix = segments[0].strip()
    if prefix and looks_like_code(prefix):
        candidates.append(prefix)

    for segment in segments[1:]:
        content = strip_fence_language(segment).strip()
        if content and looks_like_code(content):
            candidates.append(content)
    return candidates


def strip_fence_language(segment: str) -> str:
    stripped = segment.lstrip()
    for language in ("python", "py"):
        if stripped.lower().startswith(language):
            rest = stripped[len(language):]
            if not rest or rest[0].isspace():
                return rest.lstrip("\r\n")
    return segment


def looks_like_code(text: str) -> bool:
    return code_score(text) > 0


def code_segment_rank(text: str) -> tuple[int, int]:
    return code_score(text), len(text)


def code_score(text: str) -> int:
    score = 0
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(
            (
                "def ",
                "class ",
                "import ",
                "from ",
                "if ",
                "elif ",
                "else:",
                "for ",
                "while ",
                "try:",
                "except ",
                "with ",
                "return ",
                "print(",
            )
        ):
            score += 1
        elif re.match(r"^[A-Za-z_][A-Za-z0-9_,\s\[\]]*\s*=", stripped):
            score += 1
    return score


def looks_like_placeholder(text: str) -> bool:
    lowered = text.lower()
    return (
        "your implementation here" in lowered
        or "todo" in lowered
        or "pass" == lowered.strip()
    )


def assemble_candidate(
    prompt: str,
    completion: str,
    entry_point: str | None = None,
    prompt_is_code_prefix: bool = True,
) -> str:
    completion = strip_markdown_fences(completion)
    if not prompt_is_code_prefix:
        # A complete solution may define helpers after the entry point or use
        # a main block for stdin tasks. Keep that executable program intact.
        return completion.rstrip() + "\n"
    elif completion.startswith(prompt):
        code = completion
    else:
        code = prompt + completion
    code = strip_markdown_fences(code)
    if entry_point:
        code = keep_target_function(code, entry_point)
    return truncate_completion(code).rstrip() + "\n"


def keep_target_function(code: str, entry_point: str) -> str:
    lines = code.splitlines()
    def_pattern = re.compile(rf"^def\s+{re.escape(entry_point)}\s*\(")

    start = None
    for idx, line in enumerate(lines):
        if def_pattern.match(line):
            start = idx
            break
    if start is None:
        return code

    end = len(lines)
    for idx in range(start + 1, len(lines)):
        line = lines[idx]
        stripped = line.strip()
        if not stripped or line.startswith((" ", "\t")):
            continue
        if stripped.startswith(("def ", "class ", "# Test", "# test", "if __name__", "print(")):
            end = idx
            break

    kept = lines[:end]
    return "\n".join(kept) + "\n"


def truncate_completion(text: str) -> str:
    stops = [
        "\n\nif __name__",
        "\n\nprint(",
        "\n\n```",
        "\n```",
        "\nThis solution",
        "\nThe solution",
        "\nExplanation:",
        "\n###",
    ]
    end = len(text)
    for stop in stops:
        idx = text.find(stop)
        if idx != -1:
            end = min(end, idx)
    return text[:end]
