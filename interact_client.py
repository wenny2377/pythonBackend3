"""
interact_client.py（完整版）
使用方式：
    python3 interact_client.py

指令：
    exit / quit / q  → 離開
    switch           → 切換用戶
    user             → 查看當前用戶
    clear            → 清除畫面
    help             → 顯示指令清單
"""

import requests
import os

BACKEND      = "http://127.0.0.1:5000"
DEFAULT_USER = "User_Mom"
DEFAULT_ROOM = ""

USERS = [
    ("1", "User_Mom", "媽媽"),
    ("2", "User_Dad", "爸爸"),
]

# ─────────────────────────────────────────────
# API 呼叫
# ─────────────────────────────────────────────
def ask(query, user_id, room=DEFAULT_ROOM):
    try:
        resp = requests.post(f"{BACKEND}/interact", json={
            "query":  query,
            "userID": user_id,
            "room":   room,
        }, timeout=60)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        print("❌ 無法連線到後端，請確認 app.py 已啟動")
        return None
    except requests.exceptions.Timeout:
        print("❌ 後端回應逾時（60s），LLM 可能還在跑")
        return None
    except Exception as e:
        print(f"❌ 錯誤：{e}")
        return None


def confirm(choice, nav_target, nav_label, user_id, query):
    try:
        resp = requests.post(f"{BACKEND}/interact/confirm", json={
            "choice":     choice,
            "nav_target": nav_target,
            "nav_label":  nav_label,
            "userID":     user_id,
            "query":      query,
        }, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"❌ confirm 失敗：{e}")
        return None


def check_backend():
    try:
        requests.get(f"{BACKEND}/", timeout=3)
        return True
    except Exception:
        return False


# ─────────────────────────────────────────────
# 選擇用戶
# ─────────────────────────────────────────────
def select_user(current=None):
    print("\n請選擇用戶：")
    for num, uid, name in USERS:
        print(f"  {num}. {uid}（{name}）")
    print(f"  3. 自行輸入")
    if current:
        print(f"  Enter. 維持目前（{current}）")

    choice = input("\n輸入選項：").strip()

    for num, uid, _ in USERS:
        if choice == num:
            return uid

    if choice == "3":
        custom = input("輸入用戶 ID：").strip()
        return custom if custom else (current or DEFAULT_USER)

    if choice == "" and current:
        return current

    return current or DEFAULT_USER


# ─────────────────────────────────────────────
# 顯示回答
# ─────────────────────────────────────────────
def display_result(result, user_id, query):
    if not result:
        return False

    status = result.get("status", "")
    if status == "error" or "error" in result:
        print(f"❌ 後端錯誤：{result.get('error', result)}\n")
        return False

    # 主要回答
    answer = result.get("answer", "（無回答）")
    print(f"\n🤖  {answer}")

    # 導航目標
    nav_target   = result.get("nav_target")
    nav_label    = result.get("nav_label", "")
    confidence   = result.get("confidence", 0)
    intent_type  = result.get("intent_type", "")
    personalized = result.get("is_personalized", False)

    if nav_target:
        pers_tag = "✨ 個人化" if personalized else ""
        print(f"   📍 {nav_label}  座標={nav_target}  信心度={confidence:.0%}  {pers_tag}")

    if intent_type:
        print(f"   🏷  意圖類型：{intent_type}")

    # 多選項
    options = result.get("options", [])
    if options and len(options) > 1:
        print("\n請選擇：")
        for opt in options:
            print(f"  {opt['id']}. {opt['label']}")

        try:
            sel = input("\n輸入選項編號（Enter 取消）：").strip()
            if not sel:
                print("已取消。\n")
                return True
            sel = int(sel)
        except (ValueError, KeyboardInterrupt):
            print("已取消。\n")
            return True

        confirm_result = confirm(
            choice     = sel,
            nav_target = nav_target,
            nav_label  = nav_label,
            user_id    = user_id,
            query      = query,
        )
        if confirm_result:
            msg = confirm_result.get("message", "")
            if msg:
                print(f"\n✅  {msg}")
            if confirm_result.get("status") == "navigate":
                dest = confirm_result.get("nav_label", "")
                pos  = confirm_result.get("nav_target", "")
                print(f"   🚀 前往：{dest}  {pos}")

    print()
    return True


# ─────────────────────────────────────────────
# 主程式
# ─────────────────────────────────────────────
def main():
    print("=" * 50)
    print("  🤖  Robot Brain — Interaction Client")
    print("=" * 50)

    # 確認後端是否在線
    if not check_backend():
        print("\n⚠️  後端未回應，請先啟動：")
        print("   Terminal 1: ollama serve")
        print("   Terminal 2: python3 app.py\n")

    # 選擇用戶
    user_id = select_user()
    print(f"\n✅  已連線 | 用戶：{user_id}")
    print("指令：exit 離開 | switch 切換用戶 | user 查看 | clear 清除 | help 說明\n")

    while True:
        try:
            query = input(f"[{user_id}] ❓  ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n再見！")
            break

        if not query:
            continue

        q = query.lower()

        # ── 系統指令 ──
        if q in ("exit", "quit", "q"):
            print("再見！")
            break

        if q == "user":
            print(f"   現在用戶：{user_id}\n")
            continue

        if q == "switch":
            user_id = select_user(current=user_id)
            print(f"   ✅ 切換到：{user_id}\n")
            continue

        if q == "clear":
            os.system("clear" if os.name != "nt" else "cls")
            continue

        if q == "help":
            print("""
指令清單：
  exit / quit / q  → 離開程式
  switch           → 切換用戶
  user             → 查看當前用戶
  clear            → 清除畫面
  help             → 顯示此說明

範例問題：
  我想喝東西
  我想休息
  我的手機在哪裡
  今天我做了什麼
""")
            continue

        # ── 查詢後端 ──
        print("⏳  查詢中...")
        result = ask(query, user_id=user_id)
        display_result(result, user_id=user_id, query=query)


if __name__ == "__main__":
    main()
