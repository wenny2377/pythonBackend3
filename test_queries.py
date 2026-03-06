import requests
import json

# 設定伺服器位址
URL = "http://localhost:5000/query"

def ask_robot(query, user_id="User_Dad"):
    payload = {
        "query": query,
        "userID": user_id
    }
    try:
        response = requests.post(URL, json=payload)
        res_data = response.json()
        
        print(f"\n🔍 問題: '{query}'")
        if response.status_code == 200:
            print(f"🤖 機器人回答: {res_data.get('answer')}")
            if res_data.get('nav_target'):
                print(f"📍 導航目標座標: {res_data.get('nav_target')}")
            else:
                print("📍 導航目標: 未知")
        else:
            print(f"❌ 錯誤: {res_data.get('error')}")
    except Exception as e:
        print(f"⚠️ 連線失敗: {e}")

if __name__ == "__main__":
    # 這裡放你想測試的各種模糊問法
    test_cases = [
        "Where is Dad?",                     # 測試基礎位置
        "Where can I find the apple?",       # 測試 all_items 模糊搜尋
        "i am hungry?",        # 測試習慣統計
        "Is there any furniture to sit on?", # 測試語意推理 (Sofa/Chair)
        "Tell me about the shelf."           # 測試特定物件
    ]

    print("=== 🤖 機器人大腦語意查詢測試 ===")
    for q in test_cases:
        ask_robot(q)