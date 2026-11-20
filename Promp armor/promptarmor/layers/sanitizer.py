"""
Sanitizer Layer - Input normalization and encoding detection.

Theory:
    Many prompt injection attacks rely on encoding tricks to bypass filters:
    - Base64 encoding: "SWdub3JlIGFsbCBwcmV2aW91cw==" decodes to "Ignore all previous"
    - URL encoding: "%49%67%6E%6F%72%65" decodes to "Ignore"
    - Unicode tricks: Using homoglyphs or invisible characters
    - Whitespace manipulation: Hidden instructions in whitespace

    The sanitizer layer normalizes input to expose hidden content and
    remove potentially dangerous elements before further processing.

Key Concepts:
    - Normalization: Converting to a standard form
    - Encoding Detection: Identifying obfuscated content
    - Homoglyph Detection: Characters that look identical but are different

Research References:
    - "Tensor Trust: Prompt Injection Challenge" (Toyer et al., 2023)
    - Unicode Security Considerations (Unicode Consortium)
"""

import re
import base64
import urllib.parse
import unicodedata
from typing import Dict, Any, List, Tuple
import html

from promptarmor.layers.base import BaseLayer
from promptarmor.core.types import LayerResult, ThreatLevel
from promptarmor.core.config import SanitizerConfig


# Common encoding patterns
BASE64_PATTERN = re.compile(r'[A-Za-z0-9+/]{20,}={0,2}')
URL_ENCODED_PATTERN = re.compile(r'(?:%[0-9A-Fa-f]{2}){3,}')
HEX_PATTERN = re.compile(r'(?:0x[0-9A-Fa-f]{2}[\s,]*){4,}|(?:\\x[0-9A-Fa-f]{2}){4,}')

# Invisible/dangerous Unicode categories
DANGEROUS_UNICODE_CATEGORIES = {
    'Cf',  # Format characters (invisible)
    'Cc',  # Control characters
    'Co',  # Private use
    'Cn',  # Unassigned
}

# Homoglyph mappings (characters that look like ASCII but aren't)
HOMOGLYPHS = {
    '\u0430': 'a',  # Cyrillic а
    '\u0435': 'e',  # Cyrillic е
    '\u043e': 'o',  # Cyrillic о
    '\u0440': 'p',  # Cyrillic р
    '\u0441': 'c',  # Cyrillic с
    '\u0443': 'y',  # Cyrillic у
    '\u0445': 'x',  # Cyrillic х
    '\u0456': 'i',  # Cyrillic і
    '\u04cf': 'l',  # Cyrillic ԏ
    '\uff41': 'a',  # Fullwidth a
    '\uff45': 'e',  # Fullwidth e
    '\uff49': 'i',  # Fullwidth i
    '\uff4f': 'o',  # Fullwidth o
    '\u2010': '-',  # Hyphen
    '\u2011': '-',  # Non-breaking hyphen
    '\u2012': '-',  # Figure dash
    '\u2013': '-',  # En dash
    '\u2014': '-',  # Em dash
    '\u200b': '',   # Zero-width space
    '\u200c': '',   # Zero-width non-joiner
    '\u200d': '',   # Zero-width joiner
    '\ufeff': '',   # BOM
}


class SanitizerLayer(BaseLayer):
    """
    Input sanitization and normalization layer.

    Normalizes input to expose hidden content and detect encoding tricks.
    Returns both the sanitized input and detected obfuscation attempts.

    Example:
        ```python
        layer = await SanitizerLayer.create(
            config=SanitizerConfig(decode_common_encodings=True)
        )

        # Input with base64 hidden payload
        input_text = "Help me with: SWdub3JlIGFsbCBwcmV2aW91cw=="

        result = await layer.analyze(input_text)

        print(result.metadata["sanitized"])
        # "Help me with: Ignore all previous"

        print(result.metadata["detected_encodings"])
        # ["base64"]
        ```
    """

    layer_name = "sanitizer"

    def __init__(self, config: SanitizerConfig = None, armor_config: Any = None):
        super().__init__(config or SanitizerConfig(), armor_config)

    async def _analyze_impl(
        self,
        content: str,
        context: Dict[str, Any],
    ) -> LayerResult:
        """Sanitize input and detect encoding tricks."""
        detected_encodings: List[str] = []
        detected_issues: List[str] = []
        sanitized = content
        threat_level = ThreatLevel.NONE

        # Check input length
        if len(content) > self.config.max_input_length:
            sanitized = content[:self.config.max_input_length]
            detected_issues.append(f"Truncated from {len(content)} chars")
            threat_level = ThreatLevel.LOW

        # 1. Normalize Unicode
        if self.config.normalize_unicode:
            sanitized, unicode_issues = self._normalize_unicode(sanitized)
            detected_issues.extend(unicode_issues)
            if unicode_issues:
                threat_level = max(threat_level, ThreatLevel.MEDIUM, key=lambda x: x.value)

        # 2. Remove invisible characters
        if self.config.remove_invisible_chars:
            sanitized, invisible_count = self._remove_invisible(sanitized)
            if invisible_count > 0:
                detected_issues.append(f"Removed {invisible_count} invisible chars")
                threat_level = max(threat_level, ThreatLevel.MEDIUM, key=lambda x: x.value)

        # 3. Decode common encodings
        if self.config.decode_common_encodings:
            sanitized, encodings = self._decode_encodings(sanitized)
            detected_encodings.extend(encodings)
            if encodings:
                threat_level = max(threat_level, ThreatLevel.HIGH, key=lambda x: x.value)

        # 4. Strip HTML
        if self.config.strip_html:
            sanitized, had_html = self._strip_html(sanitized)
            if had_html:
                detected_issues.append("Stripped HTML tags")
                threat_level = max(threat_level, ThreatLevel.LOW, key=lambda x: x.value)

        # 5. Normalize whitespace
        sanitized = self._normalize_whitespace(sanitized)

        # Determine if we should flag this
        is_suspicious = len(detected_encodings) > 0 or len(detected_issues) > 2

        return LayerResult(
            layer_name=self.layer_name,
            passed=not is_suspicious or threat_level.value < ThreatLevel.HIGH.value,
            threat_level=threat_level,
            confidence=0.9 if detected_encodings else 0.7,
            details=f"Sanitized input. Issues: {len(detected_issues)}, Encodings: {len(detected_encodings)}",
            metadata={
                "sanitized": sanitized,
                "original_length": len(content),
                "sanitized_length": len(sanitized),
                "detected_encodings": detected_encodings,
                "detected_issues": detected_issues,
            },
        )

    def _normalize_unicode(self, text: str) -> Tuple[str, List[str]]:
        """Normalize Unicode and replace homoglyphs."""
        issues = []

        # NFKC normalization
        normalized = unicodedata.normalize('NFKC', text)
        if normalized != text:
            issues.append("Unicode normalization applied")

        # Replace homoglyphs
        result = []
        homoglyph_count = 0
        for char in normalized:
            if char in HOMOGLYPHS:
                result.append(HOMOGLYPHS[char])
                homoglyph_count += 1
            else:
                result.append(char)

        if homoglyph_count > 0:
            issues.append(f"Replaced {homoglyph_count} homoglyphs")

        return ''.join(result), issues

    def _remove_invisible(self, text: str) -> Tuple[str, int]:
        """Remove invisible Unicode characters."""
        result = []
        removed = 0

        for char in text:
            category = unicodedata.category(char)
            if category in DANGEROUS_UNICODE_CATEGORIES:
                removed += 1
            else:
                result.append(char)

        return ''.join(result), removed

    def _decode_encodings(self, text: str) -> Tuple[str, List[str]]:
        """Detect and decode common encodings."""
        detected = []
        result = text

        # Base64 detection and decoding
        for match in BASE64_PATTERN.finditer(result):
            potential_b64 = match.group()
            try:
                # Pad if necessary
                padded = potential_b64 + '=' * (4 - len(potential_b64) % 4)
                decoded = base64.b64decode(padded).decode('utf-8', errors='ignore')
                # Only replace if decoded looks like text
                if decoded.isprintable() and len(decoded) > 3:
                    result = result.replace(potential_b64, f"[DECODED_B64:{decoded}]")
                    detected.append("base64")
            except Exception:
                pass

        # URL encoding detection and decoding
        for match in URL_ENCODED_PATTERN.finditer(result):
            encoded = match.group()
            try:
                decoded = urllib.parse.unquote(encoded)
                if decoded != encoded:
                    result = result.replace(encoded, f"[DECODED_URL:{decoded}]")
                    detected.append("url_encoding")
            except Exception:
                pass

        # Hex encoding detection
        for match in HEX_PATTERN.finditer(result):
            hex_str = match.group()
            try:
                # Try to decode as hex bytes
                hex_chars = re.findall(r'[0-9A-Fa-f]{2}', hex_str)
                decoded = bytes.fromhex(''.join(hex_chars)).decode('utf-8', errors='ignore')
                if decoded.isprintable() and len(decoded) > 2:
                    result = result.replace(hex_str, f"[DECODED_HEX:{decoded}]")
                    detected.append("hex_encoding")
            except Exception:
                pass

        return result, list(set(detected))

    def _strip_html(self, text: str) -> Tuple[str, bool]:
        """Remove HTML tags from input."""
        # Check if there's any HTML
        has_html = bool(re.search(r'<[^>]+>', text))

        if has_html:
            # Unescape HTML entities
            text = html.unescape(text)
            # Remove tags
            text = re.sub(r'<[^>]+>', ' ', text)

        return text, has_html

    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace to single spaces."""
        # Replace various whitespace with regular space
        text = re.sub(r'[\t\r\n\f\v]+', ' ', text)
        # Collapse multiple spaces
        text = re.sub(r' +', ' ', text)
        return text.strip()
