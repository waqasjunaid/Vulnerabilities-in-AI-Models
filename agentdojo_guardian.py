"""Opt-in bridge between the project guardian and AgentDojo ToolsExecutor.

This first integration targets AgentDojo's Workspace suite. The contextual
trust model remains an inference-time prototype, not a trained calibrator.
"""
from __future__ import annotations

import os
import re
import sys
from types import SimpleNamespace

import yaml

PROJECT_ROOT = os.environ.get(
    "GUARDIAN_PROJECT_ROOT", "/mnt/cisk/junaid/junaid/guardian_lab"
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# State-changing or externally visible actions across AgentDojo suites.
PRIVILEGED_ACTIONS = {
    # Workspace
    "delete_file",
    "delete_email",
    "send_email",
    "create_file",
    "share_file",
    "append_to_file",
    "create_calendar_event",
    "cancel_calendar_event",
    "reschedule_calendar_event",
    "add_calendar_event_participants",

    # Banking
    "send_money",
    "schedule_transaction",
    "update_scheduled_transaction",
    "update_password",
    "update_user_info",

    # Slack
    "add_user_to_channel",
    "send_direct_message",
    "send_channel_message",
    "invite_user_to_slack",
    "remove_user_from_slack",
    "post_webpage",

    # Travel
    "reserve_hotel",
    "reserve_car_rental",
    "reserve_restaurant",
}

# Tool outputs that may contain externally supplied or attacker-influenced
# text. These sources are important for tracking indirect prompt injection.
UNTRUSTED_OUTPUT_ACTIONS = {
    # Workspace
    "list_files",
    "get_file_by_id",
    "search_files",
    "search_files_by_filename",
    "get_received_emails",
    "get_sent_emails",
    "get_unread_emails",
    "get_draft_emails",
    "search_emails",
    "search_contacts_by_name",
    "search_contacts_by_email",
    "get_day_calendar_events",
    "search_calendar_events",

    # Banking
    "read_file",
    "get_most_recent_transactions",
    "get_scheduled_transactions",

    # Slack
    "read_channel_messages",
    "read_inbox",
    "get_webpage",

    # Travel: review text can contain indirect instructions.
    "get_rating_reviews_for_hotels",
    "get_rating_reviews_for_restaurants",
    "get_rating_reviews_for_car_rental",
}

# v2 labels every tool that returns text authored by third parties as untrusted
# (user-chosen names, business listings and contact data), not only free-text
# documents. Tools that return the user's own account data stay trusted.
V2_EXTRA_UNTRUSTED_ACTIONS = frozenset({
    # Slack: channel and user names are chosen by other users
    "get_channels",
    "get_users_in_channel",
    # Travel: listings, addresses and contact data come from third parties
    "get_all_hotels_in_city",
    "get_hotels_prices",
    "get_hotels_address",
    "get_all_restaurants_in_city",
    "get_cuisine_type_for_restaurants",
    "get_restaurants_address",
    "get_dietary_restrictions_for_all_restaurants",
    "get_contact_information_for_restaurants",
    "get_price_for_restaurants",
    "check_restaurant_opening_hours",
    "get_all_car_rental_companies_in_city",
    "get_car_types_available",
    "get_car_fuel_options",
    "get_car_rental_address",
    "get_car_price_per_day",
    "get_flight_information",
})


class AgentDojoGuardianAdapter:
    untrusted_actions = frozenset(UNTRUSTED_OUTPUT_ACTIONS)

    def __init__(self):
        from guardian import Decision
        from trust_model import TrustCalibratedGuardian

        self.Decision = Decision
        detector_kind = os.environ.get(
            "TRUST_DETECTOR", "heuristic"
        ).strip().lower()

        if detector_kind == "heuristic":
            from detectors import HeuristicDetector
            detector = HeuristicDetector()
        elif detector_kind == "deberta":
            from deberta_detector import DebertaDetector
            detector = DebertaDetector()
        else:
            raise ValueError(
                "TRUST_DETECTOR must be 'heuristic' or 'deberta', "
                f"not {detector_kind!r}"
            )

        version = os.environ.get("GUARDIAN_VERSION", "v1").strip().lower()
        if version == "v2":
            from trust_v2 import ArgumentGroundingGuardian
            self.guardian = ArgumentGroundingGuardian(detector)
        elif version == "v1":
            self.guardian = TrustCalibratedGuardian(detector)
        else:
            raise ValueError(f"GUARDIAN_VERSION must be v1 or v2, not {version!r}")
        type(self).untrusted_actions = frozenset(UNTRUSTED_OUTPUT_ACTIONS) | (
            V2_EXTRA_UNTRUSTED_ACTIONS if version == "v2" else frozenset()
        )
        print(
            f"[GUARDIAN] version={version}; untrusted_tools={len(type(self).untrusted_actions)}",
            flush=True,
        )
        # Freeze each largest-file target for one environment and goal.
        self._largest_file_targets = {}
        print(
            "[GUARDIAN] Enabled AgentDojo adapter; "
            f"detector={detector_kind}; "
            f"threshold={self.guardian.calibrator.threshold:.3f}",
            flush=True,
        )

    @staticmethod
    def _attr(obj, key, default=None):
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    @classmethod
    def _tool_name(cls, message):
        call = message.get("tool_call")
        return str(cls._attr(call, "function", "") or "")

    @classmethod
    def _message_text(cls, message):
        parts = []
        content = message.get("content") or []
        if isinstance(content, str):
            return content
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            else:
                value = cls._attr(block, "content", "")
                if value:
                    parts.append(str(value))
        return "\n".join(parts)

    @classmethod
    def _trajectory(cls, messages):
        trajectory = []
        for message in messages:
            if message.get("role") != "tool":
                continue
            action = cls._tool_name(message)
            if not action:
                continue
            trajectory.append({
                "action": action,
                "trust": (
                    "untrusted"
                    if action.lower() in cls.untrusted_actions
                    else "trusted"
                ),
                "observation": cls._message_text(message),
            })
        return trajectory

    @staticmethod
    def _goal_from_messages(query, messages):
        if str(query or "").strip():
            return str(query).strip()
        for message in reversed(messages):
            if message.get("role") == "user":
                content = message.get("content") or []
                if isinstance(content, str):
                    return content
                parts = []
                for block in content:
                    value = (
                        block.get("content", "")
                        if isinstance(block, dict)
                        else getattr(block, "content", "")
                    )
                    if value:
                        parts.append(str(value))
                return "\n".join(parts).strip()
        return ""

    @staticmethod
    def _tool_metadata(action, runtime):
        function = runtime.functions.get(action)
        fields = getattr(
            getattr(function, "parameters", None), "model_fields", {}
        ) or {}
        parameters = {
            name: (getattr(field, "description", None) or "")
            for name, field in fields.items()
        }
        return SimpleNamespace(
            name=action,
            description=getattr(function, "description", "") or "",
            parameters=parameters,
            privileged=action.lower() in PRIVILEGED_ACTIONS,
        )

    @staticmethod
    def _normalise_id(value):
        if value is None:
            return ""
        return str(value).strip().strip("'\"")

    @classmethod
    def _find_file_rows(cls, value, rows):
        """Extract structured file IDs and sizes; never parse file-content text."""
        if isinstance(value, list):
            for item in value:
                cls._find_file_rows(item, rows)
            return

        if not isinstance(value, dict):
            return

        file_id = None
        for key in ("id_", "file_id", "id"):
            if key in value and isinstance(value[key], (str, int)):
                file_id = value[key]
                break

        size = value.get("size")
        if file_id is not None and size is not None:
            try:
                numeric_size = float(size)
                if numeric_size >= 0:
                    rows.append((cls._normalise_id(file_id), numeric_size))
            except (TypeError, ValueError):
                pass

        for child in value.values():
            if isinstance(child, (dict, list)):
                cls._find_file_rows(child, rows)

    @classmethod
    def _last_file_listing(cls, messages):
        for message in reversed(messages):
            if (
                message.get("role") == "tool"
                and cls._tool_name(message) == "list_files"
            ):
                text = cls._message_text(message)
                try:
                    parsed = yaml.safe_load(text)
                except yaml.YAMLError:
                    return None, []
                rows = []
                cls._find_file_rows(parsed, rows)
                return parsed, rows
        return None, []

    @classmethod
    def _find_file_id_argument(cls, value):
        if isinstance(value, dict):
            if "file_id" in value:
                return cls._normalise_id(value["file_id"])
            for child in value.values():
                result = cls._find_file_id_argument(child)
                if result:
                    return result
        return ""

    def _largest_file_policy(self, goal, action, action_input, messages, env):
        """Authorize only the largest file from the initial task listing.

        The initial target is cached by environment identity and user goal.
        Later list_files calls cannot authorize deleting a different file.
        """
        if action.lower() != "delete_file":
            return None

        text = str(goal or "").lower()
        normalised = (
            text.replace("-", " ")
                .replace(".", " ")
                .replace(",", " ")
                .replace("'", " ")
                .replace('"', " ")
                .replace(":", " ")
                .replace(";", " ")
                .replace("!", " ")
                .replace("?", " ")
                .replace("(", " ")
                .replace(")", " ")
        )
        words = set(normalised.split())
        is_delete_largest_file = (
            bool(words.intersection({"delete", "remove", "erase", "destroy"}))
            and bool(words.intersection({"largest", "biggest", "maximum"}))
            and bool(words.intersection({"file", "files"}))
        )
        if not is_delete_largest_file:
            return None

        cache_key = (id(env), str(goal or "").strip().casefold())
        cached = self._largest_file_targets.get(cache_key)

        if cached is not None and cached[0] is env:
            largest_ids, max_size = cached[1], cached[2]
        else:
            _, rows = self._last_file_listing(messages)
            if not rows:
                return self.Decision(
                    allow=False,
                    score=1.0,
                    reason=(
                        "largest-file authorization failed closed: "
                        "no parseable file IDs and sizes from a prior list_files result"
                    ),
                )

            max_size = max(size for _, size in rows)
            largest_ids = frozenset(
                file_id for file_id, size in rows if size == max_size
            )
            # Crucially, do not recompute the target after a file is deleted.
            self._largest_file_targets[cache_key] = (
                env, largest_ids, max_size
            )

        target_id = self._find_file_id_argument(action_input)
        if not target_id:
            return self.Decision(
                allow=False,
                score=1.0,
                reason="largest-file authorization failed closed: delete_file target ID is missing",
            )

        # Do not authorize a repeated deletion of an absent file.
        drive = getattr(env, "cloud_drive", None)
        current_files = getattr(drive, "files", None)

        if isinstance(current_files, dict):
            current_ids = {
                str(file_id).strip() for file_id in current_files.keys()
            }
            if str(target_id).strip() not in current_ids:
                return self.Decision(
                    allow=False,
                    score=1.0,
                    reason=(
                        f"target authorization blocked: file ID {target_id!r} "
                        "is already absent from the current drive state"
                    ),
                )

        if target_id not in largest_ids:
            return self.Decision(
                allow=False,
                score=1.0,
                reason=(
                    f"target authorization blocked delete_file(file_id={target_id!r}); "
                    f"authorized initial largest file ID(s) are {sorted(largest_ids)} "
                    f"with size {max_size:g}"
                ),
            )

        return self.Decision(
            allow=True,
            score=0.0,
            reason=(
                f"target authorization allowed file {target_id!r}: "
                f"it matched the initial largest file(s), size {max_size:g}"
            ),
        )

    @classmethod
    def _nested_calls(cls, value):
        """Yield nested FunctionCall objects before runtime resolves them."""
        if hasattr(value, "function") and hasattr(value, "args"):
            action = str(value.function)
            args = value.args
            yield action, args
            yield from cls._nested_calls(args)
            return

        if isinstance(value, dict):
            if (
                isinstance(value.get("function"), str)
                and isinstance(value.get("args"), (dict, list))
            ):
                yield value["function"], value["args"]
                yield from cls._nested_calls(value["args"])
                return
            for child in value.values():
                yield from cls._nested_calls(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                yield from cls._nested_calls(child)

    def _inspect_one(
        self, goal, action, action_input, runtime, trajectory, env, messages
    ):
        target_decision = None
        if os.environ.get("GUARDIAN_LARGEST_FILE_POLICY", "0") == "1":
            target_decision = self._largest_file_policy(
                goal, action, action_input, messages, env
            )
        if target_decision is not None:
            return target_decision

        metadata = self._tool_metadata(action, runtime)
        guardian_env = SimpleNamespace(tools={action: metadata})
        self.guardian.set_user_goal(goal)
        return self.guardian.inspect(
            action=action,
            action_input=action_input,
            trajectory=trajectory,
            env=guardian_env,
        )

    def inspect(self, goal, action, action_input, runtime, env, messages):
        goal = self._goal_from_messages(goal, messages)
        trajectory = self._trajectory(messages)

        # Check nested actions before AgentDojo resolves them at runtime.
        for nested_action, nested_args in self._nested_calls(action_input):
            nested_decision = self._inspect_one(
                goal, nested_action, nested_args, runtime,
                trajectory, env, messages
            )
            if not nested_decision.allow:
                return self.Decision(
                    allow=False,
                    score=nested_decision.score,
                    reason=(
                        f"blocked outer tool call because nested action "
                        f"{nested_action!r} was denied: {nested_decision.reason}"
                    ),
                )

        return self._inspect_one(
            goal, action, action_input, runtime, trajectory, env, messages
        )

def guard_final_response(messages):
    """Heuristic response guard for explicit injection echoes.

    This first version filters final-response sentences that strongly overlap
    with explicit instruction-like text in classified untrusted tool outputs.
    It is not a general semantic prompt-injection detector.
    """
    import os
    import re

    if not messages:
        return messages

    last = messages[-1]
    if last.get("role") != "assistant" or last.get("tool_calls"):
        return messages

    response = AgentDojoGuardianAdapter._message_text(last).strip()
    if not response:
        return messages

    # Only use observations already classified as untrusted by the adapter.
    trajectory = AgentDojoGuardianAdapter._trajectory(messages)

    try:
        min_score = float(os.environ.get("GUARDIAN_RESPONSE_MIN_SCORE", "0.50"))
    except (ValueError, TypeError):
        min_score = 0.50

    from detectors import HeuristicDetector
    detector = HeuristicDetector()

    marker_re = re.compile(
        r"IMPORTANT\s*!*"
        r"|ignore\s+(?:all\s+)?(?:previous|prior|earlier)\s+instructions?"
        r"|you are now|new instructions?"
        r"|(?:^|\n)\s*(?:system|assistant)\s*:",
        re.I,
    )

    stopwords = {
        "a", "an", "and", "are", "as", "at", "be", "because", "been",
        "before", "but", "by", "could", "do", "does", "for", "from",
        "had", "has", "have", "if", "in", "into", "is", "it", "its",
        "me", "my", "of", "on", "or", "our", "please", "previous",
        "prior", "earlier", "ignore", "important", "strictly", "adhere",
        "following", "instruction", "instructions", "say", "the", "their",
        "them", "then", "there", "these", "this", "those", "to", "was",
        "we", "were", "what", "when", "which", "who", "will", "with",
        "you", "your", "should", "instead", "all", "now", "system",
        "assistant", "user",
    }

    def tokens(text):
        return {
            token.lower()
            for token in re.findall(r"[A-Za-z0-9]+", text)
            if len(token) > 1 and token.lower() not in stopwords
        }

    suspicious_token_sets = []

    for step in trajectory:
        if step.get("trust") != "untrusted":
            continue

        observation = str(step.get("observation") or "")
        marker = marker_re.search(observation)

        # Do not treat an entire ordinary tool result as an injection.
        if marker is None or detector.score(observation) < min_score:
            continue

        payload_tokens = tokens(observation[marker.start():])
        if len(payload_tokens) >= 3:
            suspicious_token_sets.append(payload_tokens)

    if not suspicious_token_sets:
        return messages

    # Sentence-level filtering preserves task results unrelated to the
    # detected injection. This deliberately targets explicit instruction
    # echoes rather than trying to classify every final response.
    sentences = re.split(r"(?<=[.!?])\s+", response)
    kept = []
    removed = 0

    for sentence in sentences:
        response_tokens = tokens(sentence)
        should_suppress = False

        if response_tokens:
            for payload_tokens in suspicious_token_sets:
                overlap = response_tokens & payload_tokens
                ratio = len(overlap) / len(response_tokens)

                if len(overlap) >= 3 and (ratio >= 0.34 or len(overlap) >= 5):
                    should_suppress = True
                    break

        if should_suppress:
            removed += 1
        else:
            kept.append(sentence)

    if removed == 0:
        return messages

    cleaned = " ".join(part.strip() for part in kept if part.strip()).strip()

    if not cleaned:
        cleaned = (
            "I ignored instruction-like content found in retrieved material. "
            "Please ask if you would like the verified task results."
        )

    from agentdojo.types import text_content_block_from_string

    updated_last = dict(last)
    updated_last["content"] = [text_content_block_from_string(cleaned)]

    updated_messages = list(messages)
    updated_messages[-1] = updated_last

    print(
        f"[GUARDIAN-RESPONSE] Suppressed {removed} sentence(s) "
        "overlapping with suspicious untrusted instructions.",
        flush=True,
    )

    return updated_messages

