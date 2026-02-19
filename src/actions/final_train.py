from ultralytics import YOLO

def main():
    # Khởi tạo model
    model = YOLO('yolov8s.pt') 

    # Bắt đầu huấn luyện
    model.train(
        data='data_final.yaml',
        epochs=100,
        imgsz=640,
        batch=16,          # RTX 5060 cân tốt mức này
        device=0,          # Chạy bằng GPU
        workers=4,         # Đây là lý do gây ra lỗi nếu không có khối 'main'
        name='goose_final_model'
    )

if __name__ == '__main__':
    main()