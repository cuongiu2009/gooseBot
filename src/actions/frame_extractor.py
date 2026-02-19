import cv2
import os
import argparse

def extract_frames(video_path, output_dir, interval):
    """
    Trích xuất khung hình từ video và lưu dưới dạng ảnh JPG chất lượng cao.

    :param video_path: Đường dẫn đến file video đầu vào.
    :param output_dir: Thư mục để lưu các khung hình được trích xuất.
    :param interval: Khoảng thời gian (giây) giữa các lần trích xuất.
    """
    # --- Tạo thư mục đầu ra cho video cụ thể này ---
    # Lấy tên video gốc để đặt tên file và thư mục con
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    specific_output_dir = os.path.join(output_dir, video_name)

    if not os.path.exists(specific_output_dir):
        print(f"Tạo thư mục con: {specific_output_dir}")
        os.makedirs(specific_output_dir)

    # --- Mở video ---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Lỗi: không thể mở video tại '{video_path}'")
        return

    # --- Lấy thông số video để tính toán việc bỏ khung hình ---
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        print("Lỗi: Không thể lấy FPS của video. Sử dụng giá trị mặc định 30.")
        fps = 30
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_to_skip = int(fps * interval)
    if frames_to_skip == 0:
        frames_to_skip = 1 # Đảm bảo luôn tiến về phía trước

    print("--- Bắt đầu trích xuất cho video ---")
    print(f"Video: {video_path}")
    print(f"Thư mục đầu ra: {specific_output_dir}")
    print(f"Tần suất: 1 ảnh / {interval} giây (~{frames_to_skip} khung hình)")
    print("------------------------------------")

    current_frame_pos = 0
    saved_count = 0

    # --- Vòng lặp trích xuất hiệu suất cao ---
    while current_frame_pos < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_pos)
        success, frame = cap.read()
        if not success:
            break

        # --- Lưu khung hình với chất lượng cao ---
        output_filename = f"{video_name}_{saved_count + 1:04d}.jpg"
        output_path = os.path.join(specific_output_dir, output_filename)
        cv2.imwrite(output_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
        saved_count += 1
        
        # --- Thông báo tiến độ (có thể bật/tắt để tránh quá nhiều output) ---
        # print(f"Đã lưu ảnh thứ: {saved_count} ({output_filename})")

        current_frame_pos += frames_to_skip

    # --- Dọn dẹp và hoàn tất ---
    cap.release()
    print(f"--- Hoàn thành cho {video_name}: Đã lưu {saved_count} ảnh ---")

def main():
    """Hàm chính để xử lý tham số dòng lệnh."""
    parser = argparse.ArgumentParser(
        description="Một script chuyên nghiệp để trích xuất khung hình từ video bằng OpenCV.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "input_path",
        type=str,
        help="Đường dẫn đến một file video hoặc một thư mục chứa các file video."
    )
    parser.add_argument(
        "-o", "--output_dir",
        type=str,
        default="frames_output",
        help="Thư mục gốc để lưu các khung hình.\nMặc định: 'frames_output'."
    )
    parser.add_argument(
        "-i", "--interval",
        type=float,
        default=1.0,
        help="Khoảng thời gian (giây) giữa các lần trích xuất.\nMặc định: 1.0 giây."
    )

    args = parser.parse_args()

    input_path = args.input_path
    output_dir = args.output_dir

    if not os.path.exists(input_path):
        print(f"Lỗi: Đường dẫn đầu vào không tồn tại: '{input_path}'")
        return

    videos_to_process = []
    # --- KIỂM TRA ĐẦU VÀO LÀ THƯ MỤC HAY FILE ---
    if os.path.isdir(input_path):
        print(f"Phát hiện đầu vào là thư mục: '{input_path}'. Đang tìm kiếm video...")
        # Định nghĩa các đuôi file video hợp lệ
        valid_extensions = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"}
        for filename in os.listdir(input_path):
            # Lấy đuôi file và chuyển sang chữ thường để so sánh
            ext = os.path.splitext(filename)[1].lower()
            if ext in valid_extensions:
                video_path = os.path.join(input_path, filename)
                videos_to_process.append(video_path)
        print(f"Tìm thấy {len(videos_to_process)} video để xử lý.")
    elif os.path.isfile(input_path):
        print(f"Phát hiện đầu vào là một file: '{input_path}'")
        videos_to_process.append(input_path)
    else:
        print(f"Lỗi: Đường dẫn '{input_path}' không phải là file hay thư mục hợp lệ.")
        return

    # --- XỬ LÝ DANH SÁCH VIDEO ---
    if not videos_to_process:
        print("Không tìm thấy video nào để xử lý.")
        return

    total_videos = len(videos_to_process)
    for index, video_path in enumerate(videos_to_process):
        print(f"\n>>> Đang xử lý video {index + 1}/{total_videos} <<<\n")
        extract_frames(video_path, output_dir, args.interval)

    print("\n======================================")
    print("Tất cả các video đã được xử lý xong!")
    print("======================================")


if __name__ == "__main__":
    main()
