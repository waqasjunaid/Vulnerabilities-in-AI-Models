"""
llm_backend.py
--------------
A thin abstraction over a chat LLM so the rest of the lab does not care
*which* model is producing tokens. Three backends are provided:

  * OllamaBackend  -> talks to a local Ollama server (fully offline once the
                      model is pulled). Recommended for the RTX 3090 setup.
  * HFBackend      -> loads a local Hugging Face checkpoint with transformers
                      (also fully offline if weights are already on disk).
  * MockBackend    -> no model at all. A deterministic, rule-based stand-in
                      used to verify the *harness logic* (attack detection,
                      ASR computation) without GPU or weights.

Every backend exposes one method:

    generate(messages: list[dict]) -> str

where `messages` is the OpenAI-style list of {"role", "content"} dicts and
the return value is the assistant's raw text.
"""

from __future__ import annotations
import json
import re


class LLMBackend:
    """Interface. Subclass and implement generate()."""

    def generate(self, messages: list[dict]) -> str:
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# 1. Ollama  (recommended offline path)
# --------------------------------------------------------------------------- #
class OllamaBackend(LLMBackend):
    """
    Requires a running Ollama server (default http://localhost:11434) and a
    pulled model, e.g.:  ollama pull qwen2.5:7b-instruct

    On an offline machine: pull the model once on a connected machine, then
    copy ~/.ollama/models to the offline box, or run `ollama pull` while it
    has temporary network access. After that, no network is needed.
    """

    def __init__(self, model: str = "qwen2.5:7b-instruct",
                 host: str = "http://localhost:11434",
                 temperature: float = 0.0):
        self.model = model
        self.host = host.rstrip("/")
        self.temperature = temperature
        import requests  # local import so MockBackend needs no deps
        self._requests = requests

    def generate(self, messages: list[dict]) -> str:
        resp = self._requests.post(
            f"{self.host}/api/chat",
            json={
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {"temperature": self.temperature},
            },
            timeout=600,
        )
        if resp.status_code != 200:
            # Surface the server's actual explanation instead of a bare 404.
            body = resp.text.strip()
            hint = ""
            if resp.status_code == 404 and "not found" in body.lower():
                hint = (f"\nThe model '{self.model}' is not installed on this "
                        f"Ollama server. Run 'ollama list' to see what is "
                        f"available, or pull/copy the model onto this machine.")
            elif resp.status_code == 404:
                hint = ("\nThe /api/chat endpoint was not found. Your Ollama "
                        "version may be too old; check 'ollama --version'.")
            raise RuntimeError(
                f"Ollama returned HTTP {resp.status_code} from {self.host}/api/chat\n"
                f"Server said: {body}{hint}"
            )
        return resp.json()["message"]["content"]


# --------------------------------------------------------------------------- #
# 2. Hugging Face transformers (alternative offline path)
# --------------------------------------------------------------------------- #
class HFBackend(LLMBackend):
    """
    Loads a local checkpoint with transformers. Set the path to an already
    downloaded model directory so nothing touches the network.

    On the RTX 3090 (24 GB) a 7-8B instruct model in fp16/bf16 fits comfortably.
    Set HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1 in the environment to be
    certain it never reaches out.
    """

    def __init__(self, model_path: str,
                 max_new_tokens: int = 512,
                 temperature: float = 0.0):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype="auto", device_map="auto"
        )

    def generate(self, messages: list[dict]) -> str:
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        do_sample = self.temperature > 0
        out = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=do_sample,
            temperature=self.temperature if do_sample else None,
        )
        gen = out[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(gen, skip_special_tokens=True)


# --------------------------------------------------------------------------- #
# 3. Mock backend (no model) -- for testing the harness itself
# --------------------------------------------------------------------------- #
class MockBackend(LLMBackend):
    """
    A deterministic, rule-based 'agent brain' used purely to exercise the
    harness offline. It does NOT understand language; it pattern-matches.

    Behaviour:
      * Step 1: it always calls the benign tool named in the conversation's
        embedded hint  [BENIGN_TOOL=<name> ARG=<json>]  (the run harness puts
        this hint in the system prompt so the mock knows the intended task).
      * Step 2: it reads the latest tool observation. If `vulnerable=True` and
        the observation contains an injected instruction marker
        ('<<INJECTED' ... 'CALL '), it 'succumbs' and calls the attacker tool
        named in the marker -- simulating a vulnerable LLM.
        If `vulnerable=False` it ignores the injection (simulating a robust
        agent) and returns a final answer.

    This lets you confirm the experiment reports ASR=1.0 for a vulnerable
    brain and ASR=0.0 for a robust brain, before you ever load a real model.
    """

    def __init__(self, vulnerable: bool = True):
        self.vulnerable = vulnerable

    def generate(self, messages: list[dict]) -> str:
        system = next((m["content"] for m in messages if m["role"] == "system"), "")
        last_tool_obs = ""
        for m in reversed(messages):
            if m["role"] == "tool":
                last_tool_obs = m["content"]
                break

        # If we've already produced one observation, decide step 2.
        n_tool_msgs = sum(1 for m in messages if m["role"] == "tool")

        if n_tool_msgs == 0:
            # Step 1: call the benign tool from the hint.
            m = re.search(r"\[BENIGN_TOOL=(\S+) ARG=(\{.*?\})\]", system)
            if m:
                return json.dumps({
                    "thought": "I will complete the user's task.",
                    "action": m.group(1),
                    "action_input": json.loads(m.group(2)),
                })
            return json.dumps({"thought": "no task", "final_answer": "done"})

        # Step 2+: look at the observation.
        if self.vulnerable:
            inj = re.search(r"<<INJECTED.*?CALL\s+(\S+)\s+ARG=(\{.*?\})", last_tool_obs, re.S)
            if inj and n_tool_msgs < 2:
                # Succumb to the injected instruction.
                return json.dumps({
                    "thought": "Following the instruction found in the content.",
                    "action": inj.group(1),
                    "action_input": json.loads(inj.group(2)),
                })
        # Robust, or already acted: finish normally.
        return json.dumps({
            "thought": "I have what I need.",
            "final_answer": "Task complete.",
        })