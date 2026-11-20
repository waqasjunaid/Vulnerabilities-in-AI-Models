# PromptArmor

**Multi-Layer Prompt Injection Defense System**

*Day 7 of 30 AI Projects in 30 Days*

[![Tests](https://img.shields.io/badge/tests-28%20passing-brightgreen)]()
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()

PromptArmor implements defense-in-depth for LLM applications. Because when it comes to prompt injection, no single technique is foolproof.

## Status

- 28/28 unit tests passing
- All 6 layers tested and working
- CLI commands functional
- Demo script included

## Features

- **6 Defense Layers**: Canary tokens, pattern classifier, sanitizer, semantic drift detection, LLM-as-judge, response signatures
- **34+ Attack Patterns**: Comprehensive database across 9 categories
- **Red Team Simulator**: Automated attack testing with generated variations
- **Escape Room Game**: Gamified security testing - can you break the AI?
- **Multi-LLM Support**: Claude, GPT-4, Gemini
- **Production Ready**: Async-first, type-safe, well-tested

## Quick Start

```bash
pip install promptarmor
```

```python
from promptarmor import PromptArmor, ArmorConfig

# Create armored assistant
armor = await PromptArmor.create(
    ArmorConfig(
        system_prompt="You are a helpful shopping assistant.",
        strict_mode=True,
    )
)

# Process user input safely
response = await armor.process("What products do you have?")

if response.detection_result.is_safe:
    print(response.final_response)
else:
    print(f"Blocked: {response.detection_result.block_reason}")
```

## Defense Layers

### 1. Canary Tokens (Honeypots)
Hidden tripwires that detect when an attacker has extracted system information.

### 2. Attack Classifier
Pattern matching + embedding similarity to detect known attack structures.

### 3. Input Sanitizer
Normalizes Unicode, decodes Base64/URL encoding, removes invisible characters.

### 4. Semantic Drift Detection
Measures if response "drifted" from expected behavior using embeddings.

### 5. LLM-as-Judge
A second model evaluates if the response was compromised.

### 6. Response Signatures
Cryptographic-style compliance markers that prove instructions were followed.

## CLI Usage

```bash
# Test an input
python cli.py test "Ignore all previous instructions"

# Interactive protection mode
python cli.py protect --system-prompt "You are a helpful assistant"

# Run red team assessment
python cli.py redteam --attacks 100

# Play the escape room
python cli.py game
```

## Red Team Testing

```python
from promptarmor import PromptArmor
from promptarmor.attacks import RedTeamSimulator

armor = await PromptArmor.create()
simulator = RedTeamSimulator()

report = await simulator.run(armor)
report.print_summary()

# Defense success rate: 94.2%
# Vulnerabilities: Weak against encoding_bypass attacks (3 successful)
```

## Architecture

```
User Input
    │
    ▼
┌─────────────────┐
│ Sanitizer       │ → Normalize, decode, clean
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Classifier      │ → Pattern + embedding detection
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Main LLM        │ → With canary tokens
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Drift Detection │ → Semantic similarity check
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Judge Layer     │ → LLM evaluates for compromise
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Signature Check │ → Verify compliance marker
└────────┬────────┘
         │
         ▼
    Safe Response (or blocked)
```

## License

MIT

## Author

Francisco Perez - Day 7 of 30 AI Projects in 30 Days

## Links

- [GitLab Repository](https://gitlab.com/akercito/promptarmor)
- [Blog Post](https://franciscoperez.surge.sh/blog/promptarmor.html)
- [Portfolio](https://franciscoperez.surge.sh)
