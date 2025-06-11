from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from ultralytics import YOLO
import cv2

yolo = YOLO("yolo11m.pt")
yolo_blur = YOLO('yolov8n-face.pt')

# YOLO 모델 이용해서 frame labeling
def detect_labels_with_yolo(frame):
    results = yolo.predict(frame, verbose=False)[0]
    return results


# YOLO result를 이용해서 프레임에 사각형 표시
def draw_person_boxes(frame, results):
    for box in results.boxes:
        cls_id = int(box.cls[0])
        if yolo.model.names[cls_id] == 'person':
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # 좌표 정수 변환
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 초록색 사각형
            label = f"{results.names[cls_id]}"
            cv2.putText(frame, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame

# YOLO result를 이용해서 프레임에 사람의 비율이 일정 이상인지 확인
def detect_too_large_person(frame, results, threshold=0.2) -> bool:
    frame_area = frame.shape[0] * frame.shape[1]
    for box, cls_id in zip(results.boxes.xywh, results.boxes.cls):
        if yolo.model.names[int(cls_id)] == 'person':
            _, _, w, h = box
            person_area = w * h
            if person_area / frame_area > threshold:
                return True
    return False


# YOLO-face 를 이용해서 주어진 이미지의 얼굴 blur 처리
def blur_yolo(input_file: str, output_file: str):
    cap = cv2.VideoCapture(input_file)
    out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'),
            cap.get(cv2.CAP_PROP_FPS),
            (int(cap.get(3)), int(cap.get(4))))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = yolo_blur.predict(source=frame, verbose=False, conf=0.3)

        for r in results:
            for box in r.boxes.xyxy:
                x1, y1, x2, y2 = map(int, box)
                face = frame[y1:y2, x1:x2]
                if face.size > 0:
                    face = cv2.resize(face, (10, 10))
                    face = cv2.resize(face, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
                    frame[y1:y2, x1:x2] = face

        out.write(frame)

    cap.release()
    out.release()

# blur_yolo를 parallel하게 실행
def blur_yolo_parallel(files: list, max_workers=None):
    if not max_workers:
        max_workers = os.cpu_count()//2
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        tasks = []
        for file in files:
            blured_file_path = os.path.join(os.path.dirname(file), f"blured_{os.path.basename(file)}")
            tasks.append([file, blured_file_path])

        futures = [executor.submit(blur_yolo, *args) for args in tasks]
        for future in as_completed(futures):
            future.result()
