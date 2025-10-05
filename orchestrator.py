import json
import os
import re
from collections import Counter
from datetime import datetime
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional


class RuleOrchestrator:
    """Rule-based orchestrator that maps input text to canned responses."""

    def __init__(self, cases_path: Optional[str] = None, enabled: bool = True) -> None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.cases_path = cases_path or os.path.join(base_dir, "cases.json")
        self.enabled = enabled
        self.match_threshold = 0.8
        self.cases: List[Dict[str, Any]] = []
        self._load_cases()

    def _load_cases(self) -> None:
        """Load orchestration cases from disk, ignoring malformed entries."""
        try:
            with open(self.cases_path, "r", encoding="utf-8") as fp:
                raw_cases = json.load(fp)
        except FileNotFoundError:
            print(f"⚠️ Orchestrator cases file not found: {self.cases_path}")
            self.cases = []
            return
        except json.JSONDecodeError as exc:
            print(f"⚠️ Failed to parse orchestrator cases: {exc}")
            self.cases = []
            return

        if not isinstance(raw_cases, list):
            print("⚠️ Orchestrator cases JSON must be a list")
            self.cases = []
            return

        parsed_cases: List[Dict[str, Any]] = []
        for index, entry in enumerate(raw_cases, start=1):
            if not isinstance(entry, dict):
                continue

            triggers = entry.get("triggers")
            response = entry.get("response")
            name = entry.get("name") or entry.get("id") or entry.get("case") or f"case_{index}"

            if not response or not isinstance(response, str):
                continue

            if not isinstance(triggers, list):
                continue

            normalized_triggers = [t.strip() for t in triggers if isinstance(t, str) and t.strip()]
            if not normalized_triggers:
                continue

            normalized_forms = [self._normalize_text(trigger) for trigger in normalized_triggers]

            parsed_cases.append({
                "name": name,
                "triggers": normalized_triggers,
                "normalized_triggers": normalized_forms,
                "response": response,
            })

        self.cases = parsed_cases

    def _match_case(self, input_text: str) -> Optional[Dict[str, Any]]:
        """Return the first case whose trigger appears in the input."""
        normalized_input = self._normalize_text(input_text)
        if not normalized_input:
            return None

        for case in self.cases:
            triggers = case.get("triggers", [])
            normalized_triggers = case.get("normalized_triggers", [])

            for trigger, normalized_trigger in zip(triggers, normalized_triggers):
                if normalized_trigger and normalized_trigger in normalized_input:
                    return case

                overlap_score = self._token_overlap_ratio(trigger, input_text)
                if overlap_score >= self.match_threshold:
                    return case

                similarity = SequenceMatcher(None, normalized_trigger, normalized_input).ratio()
                if similarity >= self.match_threshold:
                    return case
        return None

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Lowercase text and collapse whitespace, stripping punctuation."""
        cleaned = re.sub(r"[^a-z0-9\s]", " ", text.lower())
        return " ".join(cleaned.split())

    def _token_overlap_ratio(self, trigger: str, input_text: str) -> float:
        """Return the fraction of trigger tokens present in the input."""
        trigger_tokens = self._tokenize(trigger)
        if not trigger_tokens:
            return 0.0

        input_tokens = self._tokenize(input_text)
        if not input_tokens:
            return 0.0

        trigger_counter = Counter(trigger_tokens)
        input_counter = Counter(input_tokens)

        matched = sum(min(count, input_counter.get(token, 0)) for token, count in trigger_counter.items())
        total = sum(trigger_counter.values())
        return matched / total if total else 0.0

    def _tokenize(self, text: str) -> List[str]:
        return self._normalize_text(text).split()

    def process(self, input_text: str) -> Dict[str, Any]:
        """Return the orchestration decision for the provided input text."""
        timestamp = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        base_response: Dict[str, Any] = {
            "source": "unresolved",
            "input_text": input_text,
            "response": None,
            "metadata": {
                "matched_case": None,
                "timestamp": timestamp,
            },
        }

        if not self.enabled:
            return base_response

        if not input_text:
            return base_response

        matched_case = self._match_case(input_text)
        if not matched_case:
            return base_response

        return {
            "source": "orchestrator",
            "input_text": input_text,
            "response": matched_case["response"],
            "metadata": {
                "matched_case": matched_case["name"],
                "timestamp": timestamp,
            },
        }


__all__ = ["RuleOrchestrator"]
