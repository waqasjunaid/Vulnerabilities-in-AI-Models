#!/usr/bin/env python3

import json
from collections import Counter


TRACE_FILE = "ds_trust_heuristic_trace.jsonl"
HEURISTIC_THRESHOLD = 0.35


def main():
    total_cases = 0
    total_guardian_decisions = 0

    trust_blocks = 0
    trust_allows = 0

    heuristic_blocks = 0
    heuristic_allows = 0

    same_decision = 0
    trust_more_permissive = 0   # Trust ALLOW, Heuristic BLOCK
    trust_more_conservative = 0 # Trust BLOCK, Heuristic ALLOW

    attack_action_decisions = 0
    attack_action_trust_blocks = 0
    attack_action_heuristic_blocks = 0

    relaxed = []
    conservative = []

    case_ids = set()

    with open(TRACE_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            rec = json.loads(line)
            total_cases += 1
            case_id = rec.get("id")
            case_ids.add(case_id)

            attacker_tools = set(rec.get("attacker_tools_all") or [])

            for gt in rec.get("guardian_trace", []):
                detail = gt.get("trust_detail")
                if not detail:
                    # Non-privileged actions do not have trust_detail.
                    continue

                total_guardian_decisions += 1

                trust_allow = bool(gt.get("allow", False))
                trust_block = not trust_allow

                evidence = detail.get("evidence", {})
                heuristic_score = float(
                    evidence.get("max_content_score", 0.0)
                )

                heuristic_block = heuristic_score >= HEURISTIC_THRESHOLD
                heuristic_allow = not heuristic_block

                action = gt.get("action")
                is_attack_action = action in attacker_tools

                if trust_block:
                    trust_blocks += 1
                else:
                    trust_allows += 1

                if heuristic_block:
                    heuristic_blocks += 1
                else:
                    heuristic_allows += 1

                if trust_block == heuristic_block:
                    same_decision += 1
                elif trust_allow and heuristic_block:
                    trust_more_permissive += 1

                    item = {
                        "id": case_id,
                        "action": action,
                        "is_attack_action": is_attack_action,
                        "trust_allow": True,
                        "trust_block": False,
                        "trust_risk": detail.get("risk"),
                        "trust": detail.get("trust"),
                        "heuristic_score": heuristic_score,
                        "action_risk": evidence.get("action_risk"),
                        "provenance_risk": evidence.get("provenance_risk"),
                        "action_alignment": evidence.get("action_alignment"),
                        "full_chain_success": rec.get("success", False),
                    }
                    relaxed.append(item)

                elif trust_block and heuristic_allow:
                    trust_more_conservative += 1

                    item = {
                        "id": case_id,
                        "action": action,
                        "is_attack_action": is_attack_action,
                        "trust_allow": False,
                        "trust_block": True,
                        "trust_risk": detail.get("risk"),
                        "trust": detail.get("trust"),
                        "heuristic_score": heuristic_score,
                        "action_risk": evidence.get("action_risk"),
                        "provenance_risk": evidence.get("provenance_risk"),
                        "action_alignment": evidence.get("action_alignment"),
                        "full_chain_success": rec.get("success", False),
                    }
                    conservative.append(item)

                if is_attack_action:
                    attack_action_decisions += 1
                    attack_action_trust_blocks += int(trust_block)
                    attack_action_heuristic_blocks += int(heuristic_block)

    print("\n=== SAME-TRAJECTORY TRUST vs HEURISTIC ANALYSIS ===")
    print(f"Cases in trace                  : {total_cases}")
    print(f"Guardian privileged decisions   : {total_guardian_decisions}")

    print("\nTRUST-CALIBRATED")
    print(f"  BLOCK                          : {trust_blocks}")
    print(f"  ALLOW                          : {trust_allows}")

    print("\nHEURISTIC REPLAY")
    print(f"  BLOCK                          : {heuristic_blocks}")
    print(f"  ALLOW                          : {heuristic_allows}")

    print("\nAGREEMENT")
    print(f"  Same decision                  : {same_decision}")
    print(f"  Trust ALLOW / Heuristic BLOCK : {trust_more_permissive}")
    print(f"  Trust BLOCK / Heuristic ALLOW : {trust_more_conservative}")

    print("\nATTACKER-TOOL DECISIONS")
    print(f"  Attacker privileged decisions : {attack_action_decisions}")
    print(f"  Trust blocks                   : {attack_action_trust_blocks}")
    print(f"  Heuristic blocks              : {attack_action_heuristic_blocks}")

    print("\nFULL-CHAIN OUTCOME IN TRACE")
    successful_cases = sum(
        1
        for line in open(TRACE_FILE, "r", encoding="utf-8")
        if line.strip() and json.loads(line).get("success", False)
    )
    print(f"  Successful full-chain attacks : {successful_cases}")

    with open(
        "trust_more_permissive.jsonl",
        "w",
        encoding="utf-8"
    ) as f:
        for item in relaxed:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(
        "trust_more_conservative.jsonl",
        "w",
        encoding="utf-8"
    ) as f:
        for item in conservative:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("\nOUTPUT FILES")
    print("  trust_more_permissive.jsonl")
    print("  trust_more_conservative.jsonl")


if __name__ == "__main__":
    main()