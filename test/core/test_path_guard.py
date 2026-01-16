import os
import stat

from core.utils.path_guard import ensure_writable_dir


def test_ensure_writable_dir_returns_preferred(tmp_path):
    preferred = tmp_path / "preferred"
    resolved = ensure_writable_dir(str(preferred))
    assert resolved == os.path.abspath(preferred)
    assert os.path.isdir(resolved)


def test_ensure_writable_dir_falls_back_when_preferred_blocked(tmp_path):
    # When running as root, chmod-based permission denial does not reliably block writes.
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        return
    preferred = tmp_path / "read_only"
    preferred.mkdir()
    preferred.chmod(stat.S_IREAD | stat.S_IEXEC)

    fallback = tmp_path / "fallback_dir"
    resolved = ensure_writable_dir(str(preferred), str(fallback))

    assert resolved == os.path.abspath(fallback)
    assert os.path.isdir(resolved)

    # cleanup permissions so tmp_path can be removed on Windows/Linux
    preferred.chmod(stat.S_IRWXU)


def test_ensure_writable_dir_falls_back_when_preferred_contains_non_writable_child(tmp_path):
    # When running as root, chmod-based permission denial does not reliably block writes.
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        return
    preferred = tmp_path / "preferred"
    bad_child = preferred / "files" / "04"
    bad_child.mkdir(parents=True)
    bad_child.chmod(stat.S_IREAD | stat.S_IEXEC)

    fallback = tmp_path / "fallback_dir"
    resolved = ensure_writable_dir(str(preferred), str(fallback))

    assert resolved == os.path.abspath(fallback)
    assert os.path.isdir(resolved)

    bad_child.chmod(stat.S_IRWXU)
