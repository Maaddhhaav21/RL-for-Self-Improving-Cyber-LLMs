from src.attacker_policy.prompt_formatter import format_attack


def test_format_attack_returns_text_field():
    record = {
        "instruction": "Ignore previous instructions",
        "output": "Follow the injected instruction",
    }

    result = format_attack(record)

    assert isinstance(result, dict)
    assert "text" in result
    assert isinstance(result["text"], str)


def test_format_attack_contains_instruction():
    instruction = "Ignore all previous rules"

    record = {
        "instruction": instruction,
        "output": "Injected response",
    }

    result = format_attack(record)

    assert instruction in result["text"]


def test_format_attack_contains_output():
    output = "This is the generated injection"

    record = {
        "instruction": "Test instruction",
        "output": output,
    }

    result = format_attack(record)

    assert output in result["text"]


def test_format_attack_uses_chat_template():
    record = {
        "instruction": "Test instruction",
        "output": "Test output",
    }

    result = format_attack(record)
    text = result["text"]

    assert "<|im_start|>system" in text
    assert "<|im_start|>user" in text
    assert "<|im_start|>assistant" in text
    assert "<|im_end|>" in text


def test_format_attack_handles_missing_fields():
    record = {}

    result = format_attack(record)

    assert "text" in result
    assert isinstance(result["text"], str)