from ultralytics import YOLO
import cv2
import json
import time

# Keypoint 이름 매핑
parts_path = "yolo_part.json"

with open(parts_path, "r", encoding="utf-8") as f:
    parts = json.load(f)

def write_log(data, whether_delete=False):
    """텍스트 파일에 로그 기록 """
    log_path = f"{data[:4]}.txt"
    mode = "a" if whether_delete else "w"
    with open(log_path, mode, encoding="utf-8") as f:
        f.write(f"{data}\n")

# 모델
model = YOLO("yolo11n-pose.pt")
cap = cv2.VideoCapture(0)

# 손 들기 상태
hands_up_timer = {}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    annotated_frame = frame.copy()

    for result in results:
        keypoints_obj = result.keypoints
        boxes_obj = result.boxes

        # 사람 감지 여부
        if keypoints_obj is None or keypoints_obj.data.numel() == 0:
            print("사람 감지 안됨")
            continue

        keypoints_tensor = keypoints_obj.data  # (N,17,3)
        num_people = keypoints_tensor.shape[0]

        current_time = time.time()

        for person_idx in range(num_people):
            keypoints_xy = keypoints_tensor[person_idx].cpu().numpy()  # (17,3)

            for kp_idx in range(keypoints_xy.shape[0]):
                x, y, conf = keypoints_xy[kp_idx]
                if conf < 0.3:  # confidence 낮으면 무시
                    continue
                # write_log(f"사람 {person_idx+1}, 관절 {parts[str(kp_idx)]}: x={x:.1f}, y={y:.1f}, conf={conf:.2f}", True)
                # print(f"사람 {person_idx+1}, 관절 {parts[str(kp_idx)]}: x={x:.1f}, y={y:.1f}, conf={conf:.2f}")

            # 유효성 체크
            def is_valid_point(x, y, threshold=1e-3):
                return not (abs(x) < threshold and abs(y) < threshold)

            left_shoulder_y = keypoints_xy[5][1]
            right_shoulder_y = keypoints_xy[6][1]
            left_wrist_y = keypoints_xy[9][1]
            right_wrist_y = keypoints_xy[10][1]

            if not (is_valid_point(keypoints_xy[5][0], left_shoulder_y) and
                    is_valid_point(keypoints_xy[6][0], right_shoulder_y) and
                    is_valid_point(keypoints_xy[9][0], left_wrist_y) and
                    is_valid_point(keypoints_xy[10][0], right_wrist_y)):
                continue

            # 손 들기 여부
            hands_up = (left_wrist_y <= left_shoulder_y) or (right_wrist_y <= right_shoulder_y) or \
            (left_wrist_y <= right_shoulder_y) or (right_wrist_y <= left_shoulder_y)

            if hands_up:
                if boxes_obj is not None and boxes_obj.xyxy.shape[0] > person_idx:
                    x1, y1, x2, y2 = map(int, boxes_obj.xyxy[person_idx].cpu().numpy())
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(annotated_frame, f"HandsUp {person_idx+1}",
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 255, 0), 2)
                    write_log(f"사람 {person_idx+1}, x1 : {x1}, x2 : {x2}, y1 : {y1}, y2 : {y2}, HandsUp", True)
                    print(f"사람 {person_idx+1}, x1 : {x1}, x2 : {x2}, y1 : {y1}, y2 : {y2}")
            else:
                if boxes_obj is not None and boxes_obj.xyxy.shape[0] > person_idx:
                        x1, y1, x2, y2 = map(int, boxes_obj.xyxy[person_idx].cpu().numpy())
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(annotated_frame, f"Person {person_idx+1}",
                                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6, (0, 255, 0), 2)
                        write_log(f"사람 {person_idx+1}, x1 : {x1}, x2 : {x2}, y1 : {y1}, y2 : {y2}, Person", True)
                        print(f"사람 {person_idx+1}, x1 : {x1}, x2 : {x2}, y1 : {y1}, y2 : {y2}")

    cv2.imshow("a", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
