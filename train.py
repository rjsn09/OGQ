from ultralytics import YOLO

def main():
    # OBB 모델 불러오기 (작은 모델 추천)
    model = YOLO("yolo11n-obb.pt")  # n 모델

    # Python API용 hyp dict (지원 항목만)
    hyp = {
    # 학습 / 최적화 관련
    "lr0": 0.01,                # 초기 학습률 (learning rate)  
    "lrf": 0.1,                 # 최종 lr = lr0 * lrf  
    "momentum": 0.937,          # SGD 계열 모멘텀  
    "weight_decay": 0.0005,     # 가중치 감쇠  
    "warmup_epochs": 3,         # 워밍업(epoch 수)  
    "warmup_momentum": 0.8,     # 워밍업 동안 모멘텀  
    "warmup_bias_lr": 0.1,       # 바이어스 파라미터의 워밍업 lr  

    # 손실 / 출력 관련
    "box": 7.5,                 # 바운딩 박스 예측 손실 가중치  
    "cls": 0.5,                 # 클래스 예측 손실 가중치  
    "dfl": 1.5,                 # DFL (Distribution Focal Loss) 가중치  

    # 증강 / 데이터 관련
    "hsv_h": 0.015,             # 색조 변동 범위  
    "hsv_s": 0.7,               # 채도 변동 범위  
    "hsv_v": 0.4,               # 밝기 변동 범위  
    "degrees": 180.0,           # 회전 각도 (OBB 모드에선 중요)  
    "translate": 0.1,           # 이동 변환 강도  
    "scale": 0.9,               # 스케일 변환 계수  
    "shear": 0.0,               # 전단 변환 계수  
    "perspective": 0.0,         # 원근 변환 정도  
    "flipud": 0.5,              # 상하 뒤집기 확률  
    "fliplr": 0.5,              # 좌우 뒤집기 확률  
    "mosaic": 1.0,              # 모자이크 증강 사용 확률  
    "mixup": 0.15,               # MixUp 증강 비율  
    "copy_paste": 0.0,          # 복사-붙여넣기 증강 비율  

    # 기타 설정
    "workers": 2,               # 데이터 로더 worker 개수   
    "optimizer": "SGD",         # 최적화 알고리즘 (예: SGD, AdamW)  
}


    # 학습 실행
    model.train(
        data="data.yaml",
        epochs=200,
        patience=50,
        imgsz=1024,       # VRAM 절약
        batch=16,         # VRAM 절약
        project="runs/OBB_train",
        name="exp_small_data",
        device="cpu",  # CPU
        # device=0,  # GPU
        verbose=False,
        **hyp,
        half=True        # Mixed precision (FP16)
    )

if __name__ == "__main__":
    main()

