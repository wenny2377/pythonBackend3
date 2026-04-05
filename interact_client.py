import requests
import json
import os

BACKEND      = "http://127.0.0.1:5000"
DEFAULT_USER = "User_Mom"
DEFAULT_ROOM = ""

USERS = [
    ("1", "User_Mom"),
    ("2", "User_Dad"),
]


def _extract_answer(buf):
    marker = '"answer":'
    idx = buf.find(marker)
    if idx == -1:
        return None
    start = buf.find('"', idx + len(marker))
    if start == -1:
        return None
    start += 1
    result = []
    i = start
    while i < len(buf):
        c = buf[i]
        if c == '\\' and i + 1 < len(buf):
            nc = buf[i + 1]
            if nc == '"':
                result.append('"'); i += 2; continue
            elif nc == 'n':
                result.append('\n'); i += 2; continue
            elif nc == '\\':
                result.append('\\'); i += 2; continue
        elif c == '"':
            break
        result.append(c)
        i += 1
    return "".join(result)


def ask_stream(query, user_id, room=DEFAULT_ROOM):
    try:
        resp = requests.post(
            f"{BACKEND}/interact/stream",
            json={"query": query, "userID": user_id, "room": room},
            stream=True,
            timeout=90,
        )
        resp.raise_for_status()
    except requests.exceptions.ConnectionError:
        print("[error] cannot connect to backend")
        return None
    except Exception as e:
        print(f"[error] {e}")
        return None

    print("\n[robot]  ", end="", flush=True)

    raw_buf      = ""
    plain_buf    = ""
    final_result = {}
    last_len     = 0
    is_json_mode = None

    for raw_line in resp.iter_lines():
        if not raw_line:
            continue
        line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
        if not line.startswith("data: "):
            continue
        try:
            event = json.loads(line[6:])
        except json.JSONDecodeError:
            continue

        etype = event.get("type", "")

        if etype == "intent":
            intent = event.get("intent", "")
            is_json_mode = intent not in ("chat", "query", "interrupt")
            continue

        if etype == "token":
            token = event.get("content", "")

            if is_json_mode is False:
                print(token, end="", flush=True)
                plain_buf += token

            elif is_json_mode is True:
                raw_buf += token
                visible = _extract_answer(raw_buf)
                if visible is not None:
                    new_part = visible[last_len:]
                    if new_part:
                        print(new_part, end="", flush=True)
                        last_len = len(visible)

            else:
                raw_buf   += token
                plain_buf += token
                visible = _extract_answer(raw_buf)
                if visible is not None:
                    is_json_mode = True
                    new_part = visible[last_len:]
                    if new_part:
                        print(new_part, end="", flush=True)
                        last_len = len(visible)
                else:
                    print(token, end="", flush=True)

        elif etype == "done":
            if is_json_mode is False:
                answer_text = plain_buf.strip().strip('"').strip("'")
            else:
                answer_text = (
                    _extract_answer(raw_buf) or plain_buf
                ).strip().strip('"').strip("'")

            if not answer_text:
                answer_text = "(no response)"

            final_result = {
                "status":          "Success",
                "answer":          answer_text,
                "nav_target":      event.get("nav_target"),
                "nav_label":       event.get("nav_label", ""),
                "confidence":      event.get("confidence", 0.8),
                "intent_type":     event.get("intent_type", "oneshot"),
                "is_personalized": event.get("is_personalized", False),
                "options":         event.get("options", []),
                "recommendations": [],
            }
            break

    print()
    return final_result if final_result else None


def ask(query, user_id, room=DEFAULT_ROOM):
    try:
        resp = requests.post(
            f"{BACKEND}/interact",
            json={"query": query, "userID": user_id, "room": room},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        print("[error] cannot connect to backend")
        return None
    except requests.exceptions.Timeout:
        print("[error] request timed out (60s)")
        return None
    except Exception as e:
        print(f"[error] {e}")
        return None


def confirm(choice, nav_target, nav_label, user_id, query):
    try:
        resp = requests.post(
            f"{BACKEND}/interact/confirm",
            json={
                "choice":     choice,
                "nav_target": nav_target,
                "nav_label":  nav_label,
                "userID":     user_id,
                "query":      query,
            },
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"[error] confirm failed: {e}")
        return None


def check_backend():
    try:
        requests.options(f"{BACKEND}/interact/stream", timeout=3)
        return True, True
    except Exception:
        pass
    try:
        requests.get(f"{BACKEND}/", timeout=3)
        return True, False
    except Exception:
        return False, False


def display_nav(result, user_id, query, answer_printed=False):
    if not result:
        return False
    if "error" in result:
        print(f"[error] {result.get('error')}")
        return False

    if not answer_printed:
        answer = result.get("answer", "")
        if answer:
            print(f"\n[robot]  {answer}")

    nav_target   = result.get("nav_target")
    nav_label    = result.get("nav_label", "")
    confidence   = result.get("confidence", 0)
    intent_type  = result.get("intent_type", "")
    personalized = result.get("is_personalized", False)

    if nav_target:
        tag = "  [personalized]" if personalized else ""
        print(f"   [nav] {nav_label}  pos={nav_target}  conf={confidence:.0%}{tag}")

    if intent_type:
        print(f"   [intent] {intent_type}")

    options = result.get("options", [])
    if options and len(options) > 1:
        print("")
        for opt in options:
            print(f"  {opt['id']}. {opt['label']}")

        try:
            sel = input("\nselect (Enter to skip): ").strip()
            if not sel:
                print("")
                return True
            sel_int = int(sel)
        except (ValueError, KeyboardInterrupt):
            print("")
            return True

        cr = confirm(
            choice=sel_int, nav_target=nav_target,
            nav_label=nav_label, user_id=user_id, query=query,
        )
        if cr:
            msg = cr.get("message", "")
            if msg:
                print(f"\n[ok] {msg}")
            if cr.get("status") == "navigate":
                print(f"   [navigate] -> {cr.get('nav_label')}  {cr.get('nav_target')}")

    print()
    return True


def select_user(current=None):
    print("\nselect user:")
    for num, uid in USERS:
        print(f"  {num}. {uid}")
    print("  3. custom")
    if current:
        print(f"  Enter. keep current ({current})")

    choice = input("\n> ").strip()
    for num, uid in USERS:
        if choice == num:
            return uid
    if choice == "3":
        custom = input("user id: ").strip()
        return custom if custom else (current or DEFAULT_USER)
    if choice == "" and current:
        return current
    return current or DEFAULT_USER


def print_help():
    print("""
commands:
  exit / quit / q  - exit
  switch           - switch user
  user             - show current user
  clear            - clear screen
  help             - show this

examples:
  I am hungry
  I am thirsty
  where is the remote
  bring me water
  are there any fruits
""")


def main():
    print("=" * 48)
    print("  Robot Brain - Interaction Client")
    print("=" * 48)

    online, has_stream = check_backend()
    if not online:
        print("\n[warn] backend not responding")
        print("  terminal 1: ollama serve")
        print("  terminal 2: python3 app.py\n")
    elif has_stream:
        print("[ok] connected | stream mode")
    else:
        print("[warn] connected | stream not available | fallback mode")

    user_id = select_user()
    print(f"\n[ok] user: {user_id}")
    print("commands: exit / switch / user / clear / help\n")

    while True:
        try:
            query = input(f"[{user_id}] > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\ngoodbye")
            break

        if not query:
            continue

        q = query.lower()

        if q in ("exit", "quit", "q"):
            print("goodbye")
            break
        if q == "user":
            print(f"  current user: {user_id}\n")
            continue
        if q == "switch":
            user_id = select_user(current=user_id)
            print(f"  [ok] switched to: {user_id}\n")
            continue
        if q == "clear":
            os.system("clear" if os.name != "nt" else "cls")
            continue
        if q == "help":
            print_help()
            continue

        if has_stream:
            result = ask_stream(query, user_id=user_id)
            if result is not None:
                display_nav(result, user_id=user_id, query=query, answer_printed=True)
                continue

        print("[thinking]")
        result = ask(query, user_id=user_id)
        display_nav(result, user_id=user_id, query=query, answer_printed=False)


if __name__ == "__main__":
    main()