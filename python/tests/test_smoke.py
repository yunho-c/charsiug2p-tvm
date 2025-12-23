from __future__ import annotations

import charsiug2p_tvm


def test_import_and_version() -> None:
    assert isinstance(charsiug2p_tvm.__version__, str)
