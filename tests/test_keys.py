import json
import os

import pytest

from leat.keys import PREFIX, Keys


def test_keys(tmp_path):
    # a key made is shown once, its file keeping its hash alone, readable by its owner alone; a
    # key's name is known by it, and none of another's
    path = tmp_path / "leat" / "keys.json"
    keys = Keys(path)
    assert keys.listed() == [] and keys.name("leat-anything") is None
    alkin, agent = keys.add("alkin"), keys.add("leat-agent")
    assert alkin.startswith(PREFIX) and len(alkin) > 40 and alkin != agent
    assert alkin not in path.read_text() and os.stat(path).st_mode & 0o777 == 0o600
    assert [k["name"] for k in keys.listed()] == ["alkin", "leat-agent"]
    assert keys.name(alkin) == "alkin" and keys.name(agent) == "leat-agent"
    for wrong in ("", None, alkin[:-1], alkin + "x", "Bearer " + alkin):
        assert keys.name(wrong) is None
    with pytest.raises(ValueError, match="there is a key named alkin already"):
        keys.add("alkin")
    for bad in ("", "a b", "x" * 65, "-dash", 'quote"'):
        with pytest.raises(ValueError, match="a key's name is"):
            keys.add(bad)


def test_keys_change_without_a_restart(tmp_path):
    # a server's keys, as another process changes the file: a key removed lets no one in from the
    # next request on, and one added does
    path = tmp_path / "keys.json"
    served, cli = Keys(path), Keys(path)
    old = cli.add("old")
    assert served.name(old) == "old"
    cli.remove("old")
    new = cli.add("new")
    assert served.name(old) is None and served.name(new) == "new"
    with pytest.raises(LookupError, match="there is no key named old"):
        cli.remove("old")


def test_broken_file(tmp_path):
    path = tmp_path / "keys.json"
    path.write_text(json.dumps({"alkin": "not a key"}))
    with pytest.raises(ValueError, match="is no file of leat's keys"):
        Keys(path).name("leat-x")
