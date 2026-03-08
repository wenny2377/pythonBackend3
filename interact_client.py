import requests
import json
import sys

BACKEND = "http://127.0.0.1:5000"

DEFAULT_USER  = "User_Mom"
DEFAULT_ROOM  = ""


def ask(query, user_id=DEFAULT_USER, room=DEFAULT_ROOM):
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
    user_id = input(f"用戶 ID（直接 Enter 用 {DEFAULT_USER}）：").strip() or DEFAULT_USER
    print(f"\n✅ 已連線 | 用戶：{user_id}")
    print("輸入問題，或輸入 'exit' 離開\n")

    while True:
        try:
            query = input("❓ 請輸入問題：").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n再見！")
            break

        if query.lower() in ("exit", "quit", "q"):
            print("再見！")
            break

        if not query:
            continue

        print("\n⏳ 查詢中...")
        result = ask(query, user_id=user_id)

        if not result or result.get("status") == "error":
            print(f"❌ 錯誤：{result}")
            continue

        print(f"\n🤖 {result.get('answer', '無回答')}")

        nav_target = result.get("nav_target")
        nav_label  = result.get("nav_label")
        options    = result.get("options", [])
        confidence = result.get("confidence", 0)

        if nav_target:
            print(f"   📍 位置：{nav_label} @ {nav_target}（信心度 {confidence:.0%}）")

        if options:
            print("\n請選擇：")
            for opt in options:
                print(f"  {opt['id']}. {opt['label']}")

            try:
                choice_input = input("\n輸入選項編號：").strip()
                choice = int(choice_input)
            except (ValueError, KeyboardInterrupt):
                print("已取消。\n")
                continue

            confirm_result = confirm(
                choice=choice,
                nav_target=nav_target,
                nav_label=nav_label,
                user_id=user_id,
                query=query,
            )

            if confirm_result:
                print(f"\n✅ {confirm_result.get('message', '')}")
                if confirm_result.get("status") == "navigate":
                    print(f"   🚀 機器人前往：{confirm_result.get('nav_label')} {confirm_result.get('nav_target')}")

        print()


if __name__ == "__main__":
    main()