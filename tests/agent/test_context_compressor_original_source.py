"""Original-source compression never recursively summarizes an earlier checkpoint."""

from unittest.mock import patch

from agent.context_compressor import (
    COMPRESSED_SUMMARY_METADATA_KEY,
    ContextCompressor,
    SUMMARY_PREFIX,
    _FROZEN_SUMMARY_END,
    _FROZEN_SUMMARY_START,
    _SUMMARY_END_MARKER,
)


class _HistoryStore:
    def __init__(self, messages):
        self.messages = messages

    def get_messages(self, session_id, *, include_compacted=False):
        assert session_id == "session-1"
        assert include_compacted is True
        return self.messages


def _summary(content: str, row_id: int) -> dict:
    return {
        "id": row_id,
        "role": "assistant",
        "content": f"{SUMMARY_PREFIX}\n{content}\n\n{_SUMMARY_END_MARKER}",
        COMPRESSED_SUMMARY_METADATA_KEY: True,
        "active": 0,
        "compacted": 1,
    }


def _compressor(history) -> ContextCompressor:
    with patch("agent.context_compressor.get_model_context_length", return_value=65_536):
        compressor = ContextCompressor(
            model="test/model",
            threshold_percent=0.75,
            protect_first_n=1,
            protect_last_n=1,
            tail_mode="legacy",
            summary_source="original",
            summary_source_windows=2,
            quiet_mode=True,
        )
    compressor._session_db = _HistoryStore(history)
    compressor._session_id = "session-1"
    return compressor


def _history():
    return [
        {"id": 1, "role": "user", "content": "old original turn", "active": 0, "compacted": 1},
        _summary("FIRST SUMMARY frozen fact", 2),
        {"id": 3, "role": "user", "content": "recent original turn", "active": 0, "compacted": 1},
        _summary("ROLLING SUMMARY must never be input", 4),
    ]


def test_original_source_uses_persisted_turns_not_previous_summary():
    compressor = _compressor(_history())
    compressor._previous_summary = "ROLLING SUMMARY must never be input"

    turns, frozen = compressor._original_summary_source([
        {"role": "user", "content": "current original turn"},
    ])
    serialized = compressor._serialize_for_summary(turns)
    prompt = compressor._build_summary_prompt(
        serialized,
        summary_budget=2_000,
        focus_topic=None,
        memory_context="",
        has_user_turn=True,
        include_previous_summary=False,
    )

    assert "recent original turn" in prompt
    assert "current original turn" in prompt
    assert "ROLLING SUMMARY must never be input" not in prompt
    assert "FIRST SUMMARY frozen fact" not in prompt
    assert _SUMMARY_END_MARKER not in prompt
    assert frozen == "FIRST SUMMARY frozen fact"


def test_frozen_summary_is_byte_identical_across_compactions():
    compressor = _compressor(_history())
    _, first_frozen = compressor._original_summary_source([
        {"role": "user", "content": "first current window"},
    ])
    first = compressor._append_frozen_summary("fresh recent checkpoint one", first_frozen)

    compressor._previous_summary = first
    compressor._session_db.messages.extend([
        {"id": 5, "role": "user", "content": "new original turn", "active": 0, "compacted": 1},
        _summary(first, 6),
    ])
    _, second_frozen = compressor._original_summary_source([
        {"role": "user", "content": "second current window"},
    ])
    second = compressor._append_frozen_summary("fresh recent checkpoint two", second_frozen)

    assert first_frozen == "FIRST SUMMARY frozen fact"
    assert second_frozen == first_frozen
    assert compressor._extract_frozen_summary(second) == first_frozen
    assert second.count(_FROZEN_SUMMARY_START) == 1
    assert second.count(_FROZEN_SUMMARY_END) == 1


def test_main_model_retry_does_not_expand_original_source_twice():
    compressor = _compressor(_history())
    compressor.summary_model = "test/aux-model"
    compressor._summary_has_user_turn = True
    summary = """## User Requests
- Continue the current task.

## Work Performed
- Preserved the current original turn.

    ## Current State
- Ready to continue.
"""

    def fall_back_to_main(*_args, **_kwargs):
        compressor.summary_model = compressor.model
        compressor._summary_model_fallen_back = True

    with (
        patch.object(
            compressor,
            "_original_summary_source",
            wraps=compressor._original_summary_source,
        ) as prepare_source,
        patch.object(
            compressor,
            "_call_summary_llm",
            side_effect=[RuntimeError("temporary auxiliary failure"), summary],
        ) as call_summary,
        patch.object(
            compressor,
            "_fallback_to_main_for_compression",
            side_effect=fall_back_to_main,
        ),
    ):
        compressor._generate_summary([
            {"role": "user", "content": "current original turn"},
        ])

    assert call_summary.call_count == 2
    assert prepare_source.call_count == 1
