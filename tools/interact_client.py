import json
import os
import requests

BACKEND      = "http://127.0.0.1:5000"
DEFAULT_USER = "User_Mom"
USERS        = [("1", "User_Mom"), ("2", "User_Dad")]


def stream_query(query: str, user_id: str, room: str = "") -> dict | None:
    try:
        resp = requests.post(
            f"{BACKEND}/interact/stream",
            json={"query": query, "userID": user_id, "room": room},
            stream=True, timeout=90,
        )
        resp.raise_for_status()
    except requests.exceptions.ConnectionError:
        print("[error] Cannot connect to backend. Is app.py running?")
        return None

    print("\n[robot]  ", end="", flush=True)
    last_done = None

    for line in resp.iter_lines():
        if not line:
            continue
        line_str = line.decode("utf-8") if isinstance(line, bytes) else line
        if not line_str.startswith("data: "):
            continue
        try:
            event = json.loads(line_str[6:])
        except json.JSONDecodeError:
            continue

        if event.get("type") == "token":
            print(event.get("content", ""), end="", flush=True)
        elif event.get("type") == "done":
            print()
            last_done = event

    return last_done


def display_result(result: dict, user_id: str, query: str):
    if not result:
        return

    nav_label  = result.get("nav_label", "")
    nav_target = result.get("nav_target")
    intent     = result.get("intent_type", "")
    personalized = result.get("is_personalized", False)

    if nav_label and nav_label not in ("", "unknown"):
        tag = "  [personalized]" if personalized else ""
        if isinstance(nav_target, (list, tuple)) and len(nav_target) >= 2:
            print(f"   [location] {nav_label}  pos=[{nav_target[0]:.1f}, {nav_target[1]:.1f}]{tag}")
        else:
            print(f"   [location] {nav_label}{tag}")

    if intent:
        print(f"   [intent] {intent}")

    options = result.get("options", [])
    if options and len(options) > 1:
        print()
        for opt in options:
            print(f"  {opt['id']}. {opt['label']}")
        try:
            sel = input("\nselect (Enter to skip): ").strip()
            if sel:
                sel_int = int(sel)
                if sel_int == 1 and nav_target and nav_label:
                    if isinstance(nav_target, (list, tuple)) and len(nav_target) >= 2:
                        print(f"\n[location] {nav_label} is at [{nav_target[0]:.1f}, {nav_target[1]:.1f}]")
                    else:
                        print(f"\n[location] {nav_label}")
                elif sel_int == 2 and nav_label:
                    if isinstance(nav_target, (list, tuple)) and len(nav_target) >= 2:
                        print(f"\n[info] {nav_label} is at [{nav_target[0]:.1f}, {nav_target[1]:.1f}]")
                    else:
                        print(f"\n[info] {nav_label} — coordinates not available")
                else:
                    print("\n[ok] Cancelled.")
        except (ValueError, KeyboardInterrupt):
            pass

    print()


def select_user(current: str = None) -> str:
    print("\nSelect user:")
    for num, uid in USERS:
        print(f"  {num}. {uid}")
    print("  3. Custom")
    if current:
        print(f"  Enter. Keep current ({current})")
    choice = input("\n> ").strip()
    for num, uid in USERS:
        if choice == num:
            return uid
    if choice == "3":
        custom = input("User ID: ").strip()
        return custom if custom else (current or DEFAULT_USER)
    if not choice and current:
        return current
    return current or DEFAULT_USER


def print_help():
    print("""
Commands:
  exit/quit/q  — exit
  switch       — switch user
  clear        — clear screen
  help         — show this

Example queries:
  I'm hungry
  I'm thirsty
  Where is the remote?
  What food do we have?
  Do we have any cola?
  I don't like cola
  Yes / No, something else
""")


def main():
    print("=" * 50)
    print("  Robot Brain — Reactive Service Demo")
    print("=" * 50)

    try:
        requests.get(f"{BACKEND}/ready", timeout=3)
        print("[ok] Connected to backend")
    except Exception:
        print("[warn] Backend not responding. Start app.py first.")

    user_id = select_user()
    print(f"\n[ok] User: {user_id}")
    print("Type 'help' for commands.\n")

    while True:
        try:
            query = input(f"[{user_id}] > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye.")
            break

        if not query:
            continue

        q = query.lower()
        if q in ("exit", "quit", "q"):
            print("Goodbye.")
            break
        if q == "switch":
            user_id = select_user(current=user_id)
            print(f"  Switched to: {user_id}\n")
            continue
        if q == "clear":
            os.system("clear" if os.name != "nt" else "cls")
            continue
        if q == "help":
            print_help()
            continue

        result = stream_query(query, user_id=user_id)
        if result:
            display_result(result, user_id=user_id, query=query)


if __name__ == "__main__":
    main()