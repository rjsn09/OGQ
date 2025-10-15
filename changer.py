import os
from shutil import copy2
from PIL import Image

# -----------------------------
# 설정
# -----------------------------
image_dirs = {
    "train": "C:/Users/User/Desktop/visual studio/test/OGQ/dataset/images/train",
    "val": "C:/Users/User/Desktop/visual studio/test/OGQ/dataset/images/val"
}

label_dirs = {
    "train": "C:/Users/User/Desktop/visual studio/test/OGQ/dataset/labels/train",
    "val": "C:/Users/User/Desktop/visual studio/test/OGQ/dataset/labels/val"
}

# 백업 폴더
backup_dir = "C:/Users/User/Desktop/visual studio/test/OGQ/dataset/labels_backup"
os.makedirs(backup_dir, exist_ok=True)

# -----------------------------
# 변환 함수
# -----------------------------
def normalize_labels(img_path, label_path, save_path):
    try:
        im = Image.open(img_path)
        W, H = im.size
    except Exception as e:
        print(f"⚠️ 이미지 열기 실패: {img_path} ({e})")
        return False

    new_lines = []
    try:
        with open(label_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"⚠️ 라벨 읽기 실패: {label_path} ({e})")
        return False

    for i, line in enumerate(lines):
        parts = line.strip().split()
        if len(parts) not in [5,6]:  # class x y w h [angle]
            print(f"⚠️ 라벨 형식 오류: {label_path} 줄{i+1}")
            continue

        cls = parts[0]
        x = float(parts[1])
        y = float(parts[2])
        w = float(parts[3])
        h = float(parts[4])
        angle = float(parts[5]) if len(parts)==6 else 0.0

        # 다크라벨 px 단위 -> 정규화
        if x>1 or y>1 or w>1 or h>1:
            x /= W
            y /= H
            w /= W
            h /= H

        # 0~1 범위 강제
        x = max(0.0, min(x, 1.0))
        y = max(0.0, min(y, 1.0))
        w = max(0.0, min(w, 1.0))
        h = max(0.0, min(h, 1.0))

        # angle을 YOLOv11-OBB 기준 -180~180도로 변환
        # 다크라벨이 0~1로 저장되었거나 라디안/px 단위일 수 있음
        # 여기서는 0~1 범위 -> -180~180 변환 예시
        if 0.0 <= angle <= 1.0:
            angle = angle * 360 - 180  # 0->-180, 0.5->0, 1->180
        angle = max(-180.0, min(angle, 180.0))  # 범위 제한

        new_lines.append(f"{cls} {x} {y} {w} {h} {angle}\n")

    # 백업
    copy2(label_path, os.path.join(backup_dir, os.path.basename(label_path)))

    # 저장
    with open(save_path, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    return True

# -----------------------------
# 전체 처리
# -----------------------------
for split in ["train", "val"]:
    img_dir = image_dirs[split]
    lbl_dir = label_dirs[split]

    for img_file in os.listdir(img_dir):
        if not img_file.lower().endswith(('.jpg','.jpeg','.png')):
            continue

        img_path = os.path.join(img_dir, img_file)
        lbl_name = os.path.splitext(img_file)[0] + ".txt"
        lbl_path = os.path.join(lbl_dir, lbl_name)

        if not os.path.exists(lbl_path):
            print(f"⚠️ 라벨 없음: {lbl_path}")
            continue

        success = normalize_labels(img_path, lbl_path, lbl_path)
        if success:
            print(f"✅ 변환 완료: {lbl_path}")
