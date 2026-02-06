from encapsulation.database.index_scoping import owner_scoped_dir, iter_owner_dirs


def test_owner_scoped_dir_sanitizes_separators(tmp_path):
    base = str(tmp_path)
    out = owner_scoped_dir(base, owner_id="a/b\\c", owner_dirname="owners", global_owner_name="__GLOBAL__")
    assert out.endswith("/owners/a_b_c") or out.endswith("\\owners\\a_b_c")


def test_iter_owner_dirs_lists_known_owners(tmp_path):
    base = tmp_path
    (base / "owners" / "__GLOBAL__").mkdir(parents=True, exist_ok=True)
    (base / "owners" / "o1").mkdir(parents=True, exist_ok=True)
    (base / "owners" / ".hidden").mkdir(parents=True, exist_ok=True)
    (base / "owners" / "notadir.txt").write_text("x", encoding="utf-8")

    out = iter_owner_dirs(str(base), owner_dirname="owners", global_owner_name="__GLOBAL__")
    # deterministic sorted: __GLOBAL__ first then o1
    assert out[0][0] is None
    assert out[1][0] == "o1"
