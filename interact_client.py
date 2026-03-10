import requests

BACKEND      = "http://127.0.0.1:5000"
DEFAULT_USER = "User_Mom"
DEFAULT_ROOM = ""


def ask(query, user_id, room=DEFAULT_ROOM):
    try:
        resp = requests.post(f"{BACKEND}/interact", json={
            "query":  query,
            "userID": user_id,
            "room":   room,
        }, timeout=40)
        return resp.json()
    except Exception as e:
        print(f"[Error] 無法連線到後端：{e}")
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
        return resp.json()
    except Exception as e:
        print(f"[Error] confirm 失敗：{e}")
        return None


def main():
    print("\n請選擇用戶：")
    print("  1. User_Mom（媽媽）")
    print("  2. User_Dad（爸爸）")
    print("  3. 自行輸入")

    choice = input("\n輸入選項（1/2/3）：").strip()
    if choice == "1":
        user_id = "User_Mom"
    elif choice == "2":
        user_id = "User_Dad"
    elif choice == "3":
        user_id = input("輸入用戶 ID：").strip() or DEFAULT_USER
    else:
        user_id = DEFAULT_USER

    print(f"\n✅ 已連線 | 用戶：{user_id}")
    print("指令：'exit' 離開 | 'switch' 切換用戶 | 'user' 查看當前用戶\n")

    while True:
        try:
            query = input(f"[{user_id}] ❓ ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n再見！")
            break

        if not query:
            continue

        # ── 指令處理 ──
        if query.lower() in ("exit", "quit", "q"):
            print("再見！")
            break

        if query.lower() == "user":
            print(f"   現在用戶：{user_id}\n")
            continue

        if query.lower() == "switch":
            print("\n請選擇用戶：")
            print("  1. User_Mom（媽媽）")
            print("  2. User_Dad（爸爸）")
            print("  3. 自行輸入")
            sc = input("\n輸入選項（1/2/3）：").strip()
            if sc == "1":
                user_id = "User_Mom"
            elif sc == "2":
                user_id = "User_Dad"
            elif sc == "3":
                user_id = input("輸入用戶 ID：").strip() or user_id
            print(f"   ✅ 切換到：{user_id}\n")
            continue

        # ── 查詢 ──
        print("⏳ 查詢中...")
        result = ask(query, user_id=user_id)

        if not result or result.get("status") == "error":
            print(f"❌ 錯誤：{result}\n")
            continue

        print(f"\n🤖 {result.get('answer', '無回答')}")

        nav_target = result.get("nav_target")
        nav_label  = result.get("nav_label")
        options    = result.get("options", [])
        confidence = result.get("confidence", 0)

        if nav_target:
            print(f"   📍 {nav_label} @ {nav_target}（信心度 {confidence:.0%}）")

        if options and len(options) > 1:
            print("\n請選擇：")
            for opt in options:
                print(f"  {opt['id']}. {opt['label']}")

            try:
                choice = int(input("\n輸入選項編號：").strip())
            except (ValueError, KeyboardInterrupt):
                print("已取消。\n")
                continue

            confirm_result = confirm(
                choice     = choice,
                nav_target = nav_target,
                nav_label  = nav_label,
                user_id    = user_id,
                query      = query,
            )
            if confirm_result:
                print(f"\n✅ {confirm_result.get('message', '')}")
                if confirm_result.get("status") == "navigate":
                    print(f"   🚀 前往：{confirm_result.get('nav_label')} {confirm_result.get('nav_target')}")

        print()


if __name__ == "__main__":
    main()