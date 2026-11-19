# When Tool Outputs Become Attacks

Indirect prompt injection (IPI) defense for **tool-using LLM agents**: a trust-calibrated guardian with **argument grounding** (v2).

Paper title: *When Tool Outputs Become Attacks: Indirect Prompt Injection Vulnerabilities in LLM Agents*

## What’s in this repo

| Path | Contents |
|------|----------|
| `paper/` | Overleaf-ready LaTeX manuscript (`main.tex`, sections, TikZ figures, `refs.bib`) |
| `src/trust_v2.py` | Argument-grounded guardian v2 (blocks attacker-authored tool arguments; relief for fully goal-aligned calls) |
| `tools/` | Evaluation / replay / summarization scripts for InjecAgent & AgentDojo runs |
| `tests/` | Offline unit tests for v2 |

This repo holds the **manuscript** and **evaluation helpers**. The full offline `guardian_lab` environment (AgentDojo checkout, traces, GPU runs) lives on the experiment machine and is not required to edit the paper.

## Paper (Overleaf)

1. Upload everything under `paper/` to a new Overleaf project (keep the `figures/` folder).
2. Compile `main.tex`.
3. Needs packages commonly available on Overleaf: `tikz`, `booktabs`, `algorithm`, `algpseudocode`, `amsmath`, `enumitem`, `hyperref`.

## Core idea

IPI is treated as an **authorization** problem, not only text classification:

1. **Provenance** — mark third-party tool observations as untrusted  
2. **Trust-calibrated gating (v1)** — risk from content, provenance, action prior, and goal compatibility  
3. **Argument grounding (v2)** — block if call arguments copy entities from injection spans (and not from the user goal); optionally relieve fully aligned, untainted calls  
4. **Optional learned spans** — sentence detector (`span-v1`) can extend markers; on InjecAgent-style traces it is largely redundant with markers for grounding  

## Headline results (from the manuscript)

- **InjecAgent (DS):** full-chain ASR \(27.6\% \rightarrow 0\%\) with trust-calibrated guardian; ablations reopen \(4.2\%\) / \(9.2\%\) ASR  
- **Adaptive (stage-only, 8 attempts):** \(0/101\) confirmed bypasses  
- **AgentDojo (`injecagent`, effect-based, \(n{=}949\)):** ASR \(10.85\% \rightarrow 0.63\%\) (v2 tool-guard) without significant utility loss  
- **Learned spans:** strong detection in isolation; do not change v2 allow/block vs markers on this set; residuals are answer-only / criterion attacks  

## Tools (experiment machine)

Scripts are plain Python 3.9+ / bash. They expect recorded AgentDojo / guardian runs on the offline box.

| Script | Purpose |
|--------|---------|
| `tools/check_leakage.py` | Fail if benchmark-specific answers leak into guardian code |
| `tools/run_agentdojo_eval.sh` | Frozen AgentDojo eval (`baseline` / `guarded`), resumable |
| `tools/summarize_agentdojo.py` | Baseline vs guarded rates, Wilson CIs, paired sign test, effect-based rescoring |
| `tools/case_report.py` | Per-case tools called / blocked / outcomes |
| `tools/replay_guardian.py` | Replay a guardian version on recorded trajectories (no GPU) |
| `tools/final_table.py` | Paper-style table over complete runs |
| `src/trust_v2.py` | Guardian v2 (`GUARDIAN_VERSION=v2`) |
| `tests/test_trust_v2.py` | Offline v2 unit tests |

### Typical workflow (offline lab)

```bash
# Leakage check on pipeline + guardian sources
python tools/check_leakage.py path/to/agent_pipeline.py agentdojo_guardian.py trust_model.py

# Runs (example)
tools/run_agentdojo_eval.sh baseline runs/agentdojo/final_baseline
tools/run_agentdojo_eval.sh guarded  runs/agentdojo/final_guarded

# Summary
python tools/summarize_agentdojo.py \
  --baseline runs/agentdojo/final_baseline/attack runs/agentdojo/final_baseline/clean \
  --guarded  runs/agentdojo/final_guarded/attack  runs/agentdojo/final_guarded/clean \
  --csv final_summary.csv
```

Useful env overrides for the shell runner: `SUITES`, `PHASES`, `MODEL_ID`, `EXTRA`, `GUARDIAN_THRESHOLD`, `GUARDIAN_VERSION`.

### Reading AgentDojo numbers

- `security == True` means the **attack succeeded** (lower is better).  
- Prefer **effect-based** scoring (executed calls only); blocked calls should not count as success.  
- Always report **utility under attack** next to ASR.

## License / status

Research prototype and manuscript draft. Not a production security product.
