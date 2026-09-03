"""Racing EV Lab core package.

The package keeps the original Markdown parser available internally while routing
public ``racing_ev.parser`` imports through the browser-resilient v2 wrapper.
"""
from __future__ import annotations

import sys

from . import parser as _legacy_parser
from . import parser_v2 as _parser_v2

# Existing application code imports ``from racing_ev.parser import ...``. Point
# that module name at the compatible v2 wrapper without rewriting the Streamlit
# app or discarding the original parser implementation.
sys.modules[__name__ + ".parser"] = _parser_v2

ParsedRace = _parser_v2.ParsedRace
parse_race = _parser_v2.parse_race

__all__ = ["ParsedRace", "parse_race"]
