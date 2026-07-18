from sqlquality.llm import AnthropicProvider, LLMProvider, resolve_provider


class _Block:
    def __init__(self, text: str) -> None:
        self.type = "text"
        self.text = text


class _Message:
    def __init__(self, *texts: str) -> None:
        self.content = [_Block(t) for t in texts]


class _FakeMessages:
    def __init__(self, message: _Message) -> None:
        self._message = message
        self.calls: list[dict] = []

    def create(self, **kwargs) -> _Message:
        self.calls.append(kwargs)
        return self._message


class _FakeClient:
    def __init__(self, message: _Message) -> None:
        self.messages = _FakeMessages(message)


def test_anthropic_provider_suggest_concatenates_text_blocks():
    client = _FakeClient(_Message("consider ", "an index on customer_id"))
    provider = AnthropicProvider(client=client)
    assert isinstance(provider, LLMProvider)
    out = provider.suggest("prompt text")
    assert out == "consider an index on customer_id"
    # verify the API was called with the expected shape
    call = client.messages.calls[0]
    assert call["model"] == "claude-opus-4-8"
    assert call["messages"] == [{"role": "user", "content": "prompt text"}]
    assert call["max_tokens"] >= 256


def test_anthropic_provider_respects_model_override():
    client = _FakeClient(_Message("x"))
    AnthropicProvider(model="claude-haiku-4-5", client=client).suggest("p")
    assert client.messages.calls[0]["model"] == "claude-haiku-4-5"


def test_resolve_provider_none_by_default(monkeypatch):
    monkeypatch.delenv("SQLQUALITY_LLM", raising=False)
    assert resolve_provider() is None


def test_resolve_provider_none_when_disabled(monkeypatch):
    monkeypatch.setenv("SQLQUALITY_LLM", "off")
    assert resolve_provider() is None
