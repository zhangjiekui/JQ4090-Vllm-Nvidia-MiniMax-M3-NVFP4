# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.reasoning.basic_parsers import BaseThinkingReasoningParser

if TYPE_CHECKING:
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest


class MiniMaxM3ReasoningParser(BaseThinkingReasoningParser):
    """Reasoning parser for MiniMax M3 explicit thinking blocks.

    MiniMax M3 emits reasoning as:

        <mm:think>reasoning text</mm:think>assistant content

    The M3 tokenizer exposes both markers as complete vocabulary entries, but
    generated marker text may be tokenized into smaller pieces. The streaming
    parser therefore uses text markers for extraction instead of relying on the
    single vocabulary IDs. The chat template may also prefill the start marker
    when ``thinking_mode="enabled"``, so generated text can begin directly
    inside a reasoning block without emitting ``<mm:think>`` again.
    """

    @property
    def start_token(self) -> str:
        return "<mm:think>"

    @property
    def end_token(self) -> str:
        return "</mm:think>"

    def __init__(self, tokenizer, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)
        self._start_token_ids = self._encode_marker(self.start_token)
        self._end_token_ids = self._encode_marker(self.end_token)
        chat_kwargs = kwargs.get("chat_template_kwargs", {}) or {}
        self._initial_in_reasoning = chat_kwargs.get("thinking_mode") == "enabled"
        self._reasoning_ended_streaming = False

    def _encode_marker(self, marker: str) -> tuple[int, ...]:
        try:
            token_ids = self.model_tokenizer.encode(marker, add_special_tokens=False)
        except TypeError:
            token_ids = self.model_tokenizer.encode(marker)
        return tuple(token_ids)

    @staticmethod
    def _contains_token_sequence(
        token_ids: Sequence[int], marker_ids: Sequence[int]
    ) -> bool:
        if not marker_ids or len(marker_ids) > len(token_ids):
            return False
        marker_len = len(marker_ids)
        return any(
            tuple(token_ids[i : i + marker_len]) == tuple(marker_ids)
            for i in range(len(token_ids) - marker_len + 1)
        )

    @staticmethod
    def _rfind_token_sequence(
        token_ids: Sequence[int], marker_ids: Sequence[int]
    ) -> int:
        if not marker_ids or len(marker_ids) > len(token_ids):
            return -1
        marker_len = len(marker_ids)
        for i in range(len(token_ids) - marker_len, -1, -1):
            if tuple(token_ids[i : i + marker_len]) == tuple(marker_ids):
                return i
        return -1

    @staticmethod
    def _ends_with_token_sequence_prefix(
        token_ids: Sequence[int], marker_ids: Sequence[int]
    ) -> bool:
        if not marker_ids:
            return False
        max_len = min(len(token_ids), len(marker_ids) - 1)
        for prefix_len in range(max_len, 0, -1):
            if tuple(token_ids[-prefix_len:]) == tuple(marker_ids[:prefix_len]):
                return True
        return False

    @staticmethod
    def _strip_partial_marker_suffix(text: str, marker: str) -> str:
        max_len = min(len(text), len(marker) - 1)
        for suffix_len in range(max_len, 0, -1):
            if marker.startswith(text[-suffix_len:]):
                return text[:-suffix_len]
        return text

    @staticmethod
    def _visible_delta(previous: str | None, current: str | None) -> str | None:
        if not current:
            return None
        if not previous:
            return current
        if current.startswith(previous):
            delta = current[len(previous) :]
            return delta or None
        return current

    def _visible_segments(self, text: str) -> tuple[str | None, str | None]:
        """Split text into (reasoning_content, content).

        Returns a tuple of (reasoning, content) where:
        - reasoning is the text between <mm:think> and </mm:think> tags
        - content is the text after the end tag (or before the start tag if no reasoning was started)

        Strategy:
        1. If text contains only the end marker or is shorter than it, return (None, None)
        2. If we're NOT initially in reasoning (no start tag emitted):
           - If text starts with end marker, skip it
           - If text doesn't contain start marker, everything is content
        3. If we ARE initially in reasoning:
           - If text doesn't contain start marker, everything before end marker is reasoning
           - If text contains start marker, everything after it is reasoning, everything before is content
        4. If text contains start marker:
           - Everything after start marker is reasoning
           - Everything before start marker is content
           - If end marker is found after start marker, everything after end marker is content too
        """
        if not text:
            return None, None

        # If text only contains partially visible markers, return (None, None)
        if len(text) < len(self.start_token) and self.start_token.startswith(text):
            return None, None
        if len(text) < len(self.end_token) and self.end_token.startswith(text):
            return None, None

        # If NOT initially in reasoning
        if not self._initial_in_reasoning:
            # Skip a stray end marker at the beginning (non-starting tag)
            if text.startswith(self.end_token):
                text = text[len(self.end_token):]
                if not text:
                    return None, None

            # If start marker not found, all content
            if self.start_token not in text:
                content_after = self._strip_partial_marker_suffix(
                    text, self.start_token
                )
                return None, content_after or None

            # Get content before <mm:think>
            content_before, _, m_content_between = text.partition(self.start_token)

            # Split between <mm:think> and </mm:think>
            reasoning_between, end_found, m_content_after = m_content_between.partition(
                self.end_token
            )

            if end_found:
                return reasoning_between or None, (content_before + m_content_after) or None

            # End not found yet - reasoning is ongoing
            reasoning_stripped = self._strip_partial_marker_suffix(
                reasoning_between, self.end_token
            )
            return reasoning_stripped or None, content_before or None

        # Initially IN reasoning (start tag already emitted by chat template)
        if self.start_token not in text:
            reason_between, end_found, rest_after = text.partition(self.end_token)
            if end_found:
                return reason_between or None, rest_after or None

            # End not found yet - still in reasoning
            reason_between_stripped = self._strip_partial_marker_suffix(
                reason_between, self.end_token
            )
            return reason_between_stripped or None, None

        # Have both start and potentially end markers
        content_before, _, m_content_between = text.partition(self.start_token)
        reason_between, end_found, m_after_end = m_content_between.partition(
            self.end_token
        )

        if end_found:
            return reason_between or None, (content_before + m_after_end) or None

        # End not found yet - reasoning is ongoing
        reason_between_stripped = self._strip_partial_marker_suffix(
            reason_between, self.end_token
        )
        return reason_between_stripped or None, content_before or None

    def extract_reasoning(
        self,
        model_output: str,
        request: "ChatCompletionRequest | ResponsesRequest",
    ) -> tuple[str | None, str | None]:
        # MiniMax M3 can start a response with a stray closer. Drop that first
        # token only; later unmatched closers stay visible as content.
        if not self._initial_in_reasoning and model_output.startswith(self.end_token):
            content_remaining = model_output[len(self.end_token) :]
            return None, content_remaining or None

        if self._initial_in_reasoning and self.start_token not in model_output:
            reasoning_between, end_found, content_remaining = model_output.partition(
                self.end_token
            )
            if not end_found:
                return model_output, None
            return reasoning_between, content_remaining or None

        if self.start_token not in model_output:
            return None, model_output

        content_before, _, after_start = model_output.partition(self.start_token)
        reasoning_between, end_found, content_after = after_start.partition(
            self.end_token
        )
        if not end_found:
            return reasoning_between, content_before or None

        return reasoning_between, (content_before + content_after) or None

    def is_reasoning_end_streaming(
        self, input_ids: Sequence[int], delta_ids: Iterable[int]
    ) -> bool:
        if self._reasoning_ended_streaming:
            return True

        delta_ids = tuple(delta_ids)
        if self._contains_token_sequence(delta_ids, self._end_token_ids):
            return True
        if self._contains_token_sequence(input_ids, self._end_token_ids):
            return True
        if self._initial_in_reasoning:
            return False
        if self._ends_with_token_sequence_prefix(input_ids, self._start_token_ids):
            return False
        if self._ends_with_token_sequence_prefix(input_ids, self._end_token_ids):
            return False
        if not self._contains_token_sequence(input_ids, self._start_token_ids):
            return bool(input_ids)
        return False

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        end_index = self._rfind_token_sequence(input_ids, self._end_token_ids)
        if end_index >= 0:
            return input_ids[end_index + len(self._end_token_ids) :]

        has_start = self._contains_token_sequence(input_ids, self._start_token_ids)
        if self._initial_in_reasoning and not has_start:
            return []

        if not has_start:
            return input_ids
        return []

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        if not delta_text:
            return None

        if not previous_text:
            self._reasoning_ended_streaming = False

        previous_reasoning, previous_content = self._visible_segments(previous_text)
        current_reasoning, current_content = self._visible_segments(current_text)

        if current_content is not None or self.end_token in current_text:
            self._reasoning_ended_streaming = True

        reasoning = self._visible_delta(previous_reasoning, current_reasoning)
        content_delta = self._visible_delta(previous_content, current_content)

        if reasoning is None and content_delta is None:
            return None

        return DeltaMessage(reasoning=reasoning, content=content_delta)

    def count_reasoning_tokens(self, token_ids: Sequence[int]) -> int:
        count = 0
        depth = 1 if self._initial_in_reasoning else 0
        i = 0
        while i < len(token_ids):
            start_len = len(self._start_token_ids)
            end_len = len(self._end_token_ids)
            if start_len > 0 and tuple(token_ids[i : i + start_len]) == (
                self._start_token_ids
            ):
                depth += 1
                i += start_len
                continue
            if end_len > 0 and tuple(token_ids[i : i + end_len]) == (
                self._end_token_ids
            ):
                if depth > 0:
                    depth -= 1
                i += end_len
                continue
            if depth > 0:
                count += 1
            i += 1
        return count

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        """Check if reasoning has ended in the given token sequence.

        Only returns True when the end marker is at the very end of the
        sequence (e.g. thinking_mode="disabled" where the chat template
        appends </think> as the prompt prefix).  End markers buried in
        the middle of the prompt (system-message instructions mentioning
        the tags) are correctly ignored.
        """
        end_index = self._rfind_token_sequence(input_ids, self._end_token_ids)
        if end_index < 0:
            return False

        # Anything after the end marker?
        after_end = input_ids[end_index + len(self._end_token_ids):]
        if not after_end:
            # End marker is the last token → reasoning ended (disabled prefix)
            return True

        # If a start marker appears after the end marker, a new block started
        if self._contains_token_sequence(after_end, self._start_token_ids):
            return False

        # End marker is in the middle (e.g. system-message instructions).
        # NOT a reasoning end — let the streaming logic decide.
        return False