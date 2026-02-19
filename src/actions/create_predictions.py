import os
import json
import urllib.parse

# 1. Đường dẫn chứa file .txt (từ auto_label)
LABEL_DIR = r'C:\Users\PC\Desktop\Code\gooseBot\runs\detect\auto_label_results\predict\labels'
# 2. Đường dẫn gốc folder ảnh (Phải giống hệt Bước 3.3)
BASE_PATH = r'C:\Users\PC\Desktop\Code\gooseBot\unlabeled_images'
# 3. Thứ tự nhãn
CLASSES = ['dead_body', 'goose', 'interact_button', 'report_button']

output_tasks = []

for filename in os.listdir(LABEL_DIR):
    if filename.endswith('.txt'):
        image_name = filename.replace('.txt', '.jpg')
        
        # TẠO ĐƯỜNG DẪN KHỚP VỚI LOCAL STORAGE WINDOWS
        full_path = os.path.join(BASE_PATH, image_name)
        # Label Studio trên Windows thường mã hóa toàn bộ đường dẫn tuyệt đối
        encoded_path = urllib.parse.quote(full_path, safe='')
        
        predictions = []
        with open(os.path.join(LABEL_DIR, filename), 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    cls_id, x, y, w, h = map(float, parts)
                    predictions.append({
                        "from_name": "label",
                        "to_name": "image",
                        "type": "rectanglelabels",
                        "value": {
                            "rectanglelabels": [CLASSES[int(cls_id)]],
                            "x": (x - w/2) * 100,
                            "y": (y - h/2) * 100,
                            "width": w * 100,
                            "height": h * 100
                        }
                    })

        task = {
            "data": {"image": f"/data/local-files/?d={encoded_path}"},
            "predictions": [{"result": predictions}]
        }
        output_tasks.append(task)

with open('final_import.json', 'w', encoding='utf-8') as f:
    json.dump(output_tasks, f, indent=2)
print("Xong! File final_import.json đã sẵn sàng.")