from ultralytics import YOLO
import cv2

# 학습된 모델 불러오기 (Detect 모델)
model = YOLO("runs/OBB_train/exp_small_data/weights/best.pt")

# 웹캠 연결
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO 추론
    results = model(frame)

    # 결과 처리
    for result in results:
        boxes = result.boxes  # 바운딩 박스
        if boxes is not None and len(boxes) > 0:
            for i, box in enumerate(boxes.xyxy):
                # box: [x1, y1, x2, y2]
                x1, y1, x2, y2 = map(int, box.cpu().numpy())
                conf = boxes.conf[i].cpu().numpy()
                cls = int(boxes.cls[i].cpu().numpy())

                # 바운딩 박스 그리기
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame,
                            f"Class {cls} {conf:.2f}",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 0),
                            2)

    # 결과 화면 표시
    cv2.imshow("YOLOv8 Detection", frame)

    # q 키 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
