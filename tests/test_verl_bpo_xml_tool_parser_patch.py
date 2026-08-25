import logging

import pytest

import scripts.apply_verl_bpo_tool_parser_patch as installer
from shopping_grpo.training.bpo.xml_tool_parser_patch import (
    PATCH_MARKER,
    patch_source,
)


SOURCE = '''
class Parser:
    def parse(self, matches):
        values = {}
        for match_text in matches:
            idx = match_text.index(">")
            param_name = match_text[:idx]
            param_value = str(match_text[idx + 1 :])
            values[param_name] = param_value
        return values
'''


def test_xml_patch_skips_malformed_parameter_and_keeps_valid_siblings():
    namespace = {"logger": logging.getLogger("test")}
    patched = patch_source(SOURCE)
    exec(patched, namespace)
    assert namespace["Parser"]().parse(["broken", "query>shoe"]) == {
        "query": "shoe"
    }
    assert patched.count(PATCH_MARKER) == 1
    assert patch_source(patched) == patched


def test_xml_patch_rejects_unknown_source():
    with pytest.raises(ValueError, match="anchor mismatch"):
        patch_source("class Parser: pass\n")


def test_xml_patch_installer_is_idempotent_and_restorable(tmp_path):
    target = tmp_path / "tool_parser.py"
    target.write_text(SOURCE, encoding="utf-8", newline="\n")
    original = target.read_bytes()

    installer.apply(target)
    installer.verify(target)
    first_patched = target.read_bytes()
    installer.apply(target)
    assert target.read_bytes() == first_patched

    installer.restore(target)
    assert target.read_bytes() == original
