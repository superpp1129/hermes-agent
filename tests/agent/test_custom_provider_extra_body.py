from types import SimpleNamespace

from agent.agent_init import _compression_output_reservation, _merge_custom_provider_extra_body




def test_custom_provider_extra_body_preserves_caller_override():
    agent = SimpleNamespace(
        provider="custom",
        model="google/gemma-4-31b-it",
        base_url="https://example.test/v1",
        request_overrides={
            "extra_body": {
                "reasoning_effort": "low",
                "caller_only": True,
            }
        },
    )

    _merge_custom_provider_extra_body(
        agent,
        [
            {
                "name": "gemma",
                "base_url": "https://example.test/v1",
                "model": "google/gemma-4-31b-it",
                "extra_body": {
                    "enable_thinking": True,
                    "reasoning_effort": "high",
                },
            }
        ],
    )

    assert agent.request_overrides["extra_body"] == {
        "enable_thinking": True,
        "reasoning_effort": "low",
        "caller_only": True,
    }




def test_named_custom_provider_extra_body_matches_provider_key():
    agent = SimpleNamespace(
        provider="custom:zai-coding-plan",
        model="glm-5.2",
        base_url="https://api.z.ai/api/coding/paas/v4",
        request_overrides={},
    )

    _merge_custom_provider_extra_body(
        agent,
        [
            {
                "provider_key": "other-provider",
                "name": "Other Provider",
                "base_url": "https://api.z.ai/api/coding/paas/v4",
                "model": "glm-5.2",
                "extra_body": {"enable_thinking": True},
            },
            {
                "provider_key": "zai-coding-plan",
                "name": "Z.AI Coding Plan",
                "base_url": "https://api.z.ai/api/coding/paas/v4/",
                "model": "glm-5.2",
                "extra_body": {"enable_thinking": False},
            },
        ],
    )

    assert agent.request_overrides == {"extra_body": {"enable_thinking": False}}


def test_provider_output_cap_reserves_room_for_automatic_compression():
    from agent.context_compressor import ContextCompressor

    agent = SimpleNamespace(
        provider="custom:local", model="local-model", base_url="http://localhost:1234/v1",
        max_tokens=None, request_overrides={},
    )
    _merge_custom_provider_extra_body(agent, [{
        "provider_key": "local", "model": "local-model", "base_url": agent.base_url,
        "extra_body": {"max_tokens": 8192},
    }])
    compressor = ContextCompressor(
        model=agent.model, config_context_length=65536, quiet_mode=True,
        max_tokens=_compression_output_reservation(agent),
    )
    assert compressor.max_tokens == 8192
    assert compressor.threshold_tokens <= (65536 - 8192) * 0.85
    assert compressor.threshold_tokens + 8192 < 65536
    assert agent.max_tokens is None  # accounting must not mutate wire settings
    assert agent.request_overrides["extra_body"] == {"max_tokens": 8192}


def test_output_reservation_honours_wire_override_and_ignores_invalid_values():
    agent = SimpleNamespace(max_tokens=4096, request_overrides={})
    assert _compression_output_reservation(agent) == 4096
    agent.request_overrides = {"max_tokens": 1024, "extra_body": {"max_tokens": 8192}}
    assert _compression_output_reservation(agent) == 8192
    agent.request_overrides = {"extra_body": {"max_completion_tokens": 2048}}
    assert _compression_output_reservation(agent) == 2048
    for invalid in (True, -1, 0, None, "8192"):
        agent.request_overrides = {"extra_body": {"max_tokens": invalid}}
        assert _compression_output_reservation(agent) == 4096


def test_real_agent_uses_same_output_cap_for_wire_and_auto_compression(tmp_path, monkeypatch):
    import yaml
    from run_agent import AIAgent

    config = {
        "model": {"context_length": 65536},
        "compression": {"enabled": True, "threshold": 0.5},
        "custom_providers": [{
            "provider_key": "local", "name": "Local", "model": "local-model",
            "base_url": "http://127.0.0.1:1/v1", "extra_body": {"max_tokens": 8192},
        }],
    }
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config))
    agent = AIAgent(
        model="local-model", provider="custom:local", base_url="http://127.0.0.1:1/v1",
        api_key="test-only", enabled_toolsets=[], quiet_mode=True,
        skip_memory=True, skip_context_files=True,
    )
    try:
        wire = agent._build_api_kwargs([{"role": "user", "content": "hello"}], tools_for_api=[])
        body = dict(wire)
        body.update(wire.get("extra_body") or {})
        assert body["max_tokens"] == 8192
        compressor = agent.context_compressor
        assert compressor.max_tokens == body["max_tokens"]
        assert compressor.threshold_tokens == 48742
        assert compressor.should_compress(48000) is False
        assert compressor.should_compress(50000) is True
        assert compressor.should_compress(63960) is True
    finally:
        agent.client.close()
