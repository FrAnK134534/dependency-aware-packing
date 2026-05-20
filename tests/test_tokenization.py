from dapacking.tokenization import active_tokenizer_name, configure_tokenizer, count_tokens, tokenize, truncate_to_tokens


def test_simple_tokenizer_remains_default() -> None:
    configure_tokenizer("simple")

    assert active_tokenizer_name() == "simple"
    assert tokenize("Hello, world!") == ["hello", ",", "world", "!"]
    assert count_tokens("Hello, world!") == 4


def test_simple_truncation_uses_lightweight_tokens() -> None:
    configure_tokenizer("simple")

    text, overflow = truncate_to_tokens("alpha beta gamma", 2)

    assert text == "alpha beta"
    assert overflow == 1
