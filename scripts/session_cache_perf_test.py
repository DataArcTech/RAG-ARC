#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lightweight session Redis cache test: correctness + simple latency (miss vs hit).
Usage from repo root: PYTHONPATH=. python scripts/session_cache_perf_test.py
"""
import os
import sys
import time
import uuid

# 项目根
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _time_ms():
    return time.perf_counter() * 1000


def main():
    import logging
    logging.getLogger().setLevel(logging.WARNING)

    from framework.register import Register
    from config.application.session_config import ChatSessionConfig
    from config.application.account_config import AccountConfig
    from app_registration import _resolve_config_path

    # 初始化注册表并加载会话 + 账号配置
    register = Register()
    session_config_path = _resolve_config_path("SESSION_CONFIG_PATH", "config/json_configs/session.json")
    account_config_path = _resolve_config_path("ACCOUNT_CONFIG_PATH", "config/json_configs/account.json")
    register.register(session_config_path, "chat_session", ChatSessionConfig)
    register.register(account_config_path, "account", AccountConfig)

    session_manager = register.get_object("chat_session")
    account = register.get_object("account")
    storage = session_manager.session_storage

    # 解析可选环境变量
    user_id_s = os.getenv("TEST_USER_ID")
    session_id_s = os.getenv("TEST_SESSION_ID")
    if user_id_s:
        user_id = uuid.UUID(user_id_s)
    else:
        users = account.user_storage.list_users(limit=1)
        if not users:
            print("SKIP: No users, init account or set TEST_USER_ID")
            return
        user_id = users[0].id

    if session_id_s:
        session_id = uuid.UUID(session_id_s)
        sess = storage.get_session(session_id)
        if not sess:
            print("SKIP: Session not found: {}, check TEST_SESSION_ID".format(session_id))
            return
    else:
        sessions = storage.list_sessions_by_user(user_id, limit=5)
        if not sessions:
            session_id = uuid.UUID(storage.create_session(user_id, "perf-test"))
        else:
            session_id = sessions[0].id

    has_redis = getattr(storage, "cache_store", None) is not None
    print("--- Session Redis cache test ---")
    print("user_id={}, session_id={}, Redis={}".format(user_id, session_id, has_redis))
    print()

    # --- Consistency: write then read must see latest (cache invalidated) ---
    print("--- Consistency (write invalidation) ---")
    uid = str(uuid.uuid4())[:8]
    name_orig = "consistency-test-{}".format(uid)
    name_updated = "consistency-test-updated-{}".format(uid)
    tid = uuid.UUID(storage.create_session(user_id, name_orig))
    list_after_create = storage.list_sessions_by_user(user_id, limit=50, offset=0)
    found = any(s.id == tid and s.name == name_orig for s in list_after_create)
    ok_create = found
    print("  create_session -> list sees new session with name: {}".format("OK" if ok_create else "FAIL"))

    storage.update_session(tid, {"name": name_updated})
    sess = storage.get_session(tid)
    ok_update = sess is not None and sess.name == name_updated
    print("  update_session -> get_session sees new name: {}".format("OK" if ok_update else "FAIL"))

    storage.delete_session(tid)
    sess_after_del = storage.get_session(tid)
    list_after_del = storage.list_sessions_by_user(user_id, limit=50, offset=0)
    ok_delete = sess_after_del is None and not any(s.id == tid for s in list_after_del)
    print("  delete_session -> get_session=None, list without it: {}".format("OK" if ok_delete else "FAIL"))

    if ok_create and ok_update and ok_delete:
        print("  consistency: PASS")
    else:
        print("  consistency: FAIL")
    print()

    print("--- Latency (miss vs hit) ---")
    rounds = 6
    get_times = []
    list_times = []

    for i in range(rounds):
        t0 = _time_ms()
        storage.get_session(session_id)
        get_times.append(_time_ms() - t0)

        t0 = _time_ms()
        storage.list_sessions_by_user(user_id, limit=20, offset=0)
        list_times.append(_time_ms() - t0)

    # 第 1 次视为 miss，第 2–6 次视为 hit
    get_miss_ms = get_times[0]
    get_hit_avg_ms = sum(get_times[1:]) / (rounds - 1) if rounds > 1 else 0
    list_miss_ms = list_times[0]
    list_hit_avg_ms = sum(list_times[1:]) / (rounds - 1) if rounds > 1 else 0

    print("get_session(session_id):")
    print("  round1(miss)  {:.2f} ms".format(get_miss_ms))
    print("  round2-6(hit) avg {:.2f} ms".format(get_hit_avg_ms))
    if get_miss_ms > 0:
        print("  speedup ~{:.1f}x".format(get_miss_ms / max(get_hit_avg_ms, 0.01)))
    print()
    print("list_sessions_by_user(user_id, limit=20, offset=0):")
    print("  round1(miss)  {:.2f} ms".format(list_miss_ms))
    print("  round2-6(hit) avg {:.2f} ms".format(list_hit_avg_ms))
    if list_miss_ms > 0:
        print("  speedup ~{:.1f}x".format(list_miss_ms / max(list_hit_avg_ms, 0.01)))
    print()
    print("--- done (%d get + %d list) ---" % (rounds, rounds))


if __name__ == "__main__":
    main()
