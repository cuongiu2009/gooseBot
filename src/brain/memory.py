import json
import os
import time

class Memory:
    """
    Quản lý nhật ký di chuyển (Log) và trạng thái của Bot.
    """
    def __init__(self, log_dir="data/logs"):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file = os.path.join(self.log_dir, f"log_{int(time.time())}.json")
        self.history = []

    def save_log(self, data):
        """
        Lưu một sự kiện vào lịch sử và file JSON.
        """
        event = {
            "timestamp": time.time(),
            "data": data
        }
        self.history.append(event)
        with open(self.log_file, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=4, ensure_ascii=False)

    def get_recent_history(self, n=10):
        """
        Lấy n sự kiện gần nhất.
        """
        return self.history[-n:]
