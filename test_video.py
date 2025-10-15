from ultralytics import YOLO
import cv2
import numpy as np
import json

# YOLO Pose 모델 불러오기
model = YOLO("yolov11n-pose.pt")  # 학습된 Pose 모델 경로

# 동영상 파일 경로
video_path = "test_video.mp4"
cap = cv2.VideoCapture(video_path)

# 저장용 동영상 설정
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("output_pose.mp4", fourcc, 30, 
                      (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                       int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

# COCO Keypoint 이름
parts_path = "yolo_part.json"
with open(parts_path, "r", encoding="utf-8") as f:
    parts = json.load(f)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Pose 추론
    results = model(frame)

    for result in results:
        keypoints_data = result.keypoints.data  # Tensor (N,17,3)
        boxes = result.boxes.xyxy  # Tensor (N,4)

        if keypoints_data is not None and keypoints_data.shape[0] > 0:
            for person_idx in range(keypoints_data.shape[0]):
                keypoints = keypoints_data[person_idx].cpu().numpy()  # (17,3)

                # 신뢰도 0.3 이상 keypoints만 출력
                for kp_idx, (x, y, conf) in enumerate(keypoints):
                    if conf >= 0.3:
                        cv2.circle(frame, (int(x), int(y)), 4, (0, 0, 255), -1)
                        cv2.putText(frame, parts[str(kp_idx)], (int(x), int(y)-5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

                # 손 들기 조건: 손목(y) < 어깨(y)
                left_wrist_y = keypoints[9][1]
                right_wrist_y = keypoints[10][1]
                left_shoulder_y = keypoints[5][1]
                right_shoulder_y = keypoints[6][1]

                if (left_wrist_y <= left_shoulder_y) or (right_wrist_y <= right_shoulder_y):
                    x1, y1, x2, y2 = map(int, boxes[person_idx].cpu().numpy())
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"HandsUpPerson {person_idx+1}",
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 255, 0), 2)
    

    cv2.imshow("YOLOv11n-Pose Video", frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
