# 這段邏輯讓你不需要改 DB，就能擁有 MIT 風格的階層紀錄
def get_hierarchy_mapping(db):
    # 建立一個邏輯階層字典
    hierarchy = {}
    
    # 關聯：從 Zone 映射到 Room
    # 假設 observation_logs 裡有 zone_name 和 room 的對應
    zone_to_room = {log['zone_name']: log['room'] for log in db.observation_logs.find()}
    
    # 關聯：從 Object 映射到 Room
    obj_to_room = {item['label']: item['room'] for item in db.scene_snapshots.find()}
    
    return zone_to_room, obj_to_room