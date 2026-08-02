"""Single source of truth for chat-template rendering and think-block
handling, shared by the rollout orchestrator, the SFT trainer, and the
reward models.

Newer chat templates (Qwen3/Qwen3.5) manage the <think> block themselves:
with enable_thinking=True the generation prompt ends with an OPENED
`<think>\n`, so the model's raw output contains the reasoning and the
closing `</think>` but not the opening tag. All think-stripping must
handle both that continuation form and self-contained <think>...</think>
outputs.
"""

import re

THINK_RE = re.compile(r'<think>.*?</think>', re.DOTALL)


def visible_text(raw: str) -> str:
    """Text outside any think block. Handles outputs whose opening
    <think> tag lives in the prompt (template pre-opened)."""
    s = THINK_RE.sub('', raw or "")
    if '</think>' in s:
        s = s.split('</think>', 1)[1]
    return s


def render_generation_prompt(tokenizer, messages) -> str:
    """Chat-template prompt for generation, thinking mode enabled when
    the template supports it."""
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=True,
        )
    except TypeError:  # template without the kwarg
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )


def render_sft_texts(tokenizer, messages):
    """(prompt_text, full_text) for SFT on a conversation whose last
    message is the assistant response. prompt_text is an EXACT prefix of
    full_text, so loss masking by prompt length stays aligned.

    If the template pre-opens <think> in the prompt, the duplicate
    opening tag is stripped from the stored response body.
    """
    prompt = render_generation_prompt(tokenizer, messages[:-1])
    body = messages[-1]["content"]
    if prompt.rstrip().endswith("<think>") and body.lstrip().startswith("<think>"):
        body = body.lstrip()[len("<think>"):].lstrip("\n")
    return prompt, prompt + body + "<|im_end|>"
