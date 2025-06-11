import cv2

# 시간 기준으로 특정한 위치의 frame을 반환
def extract_frame(video_path: str, sec: float):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None

# frame 번호 기준으로 특정한 위치의 frame을 반환
def extract_frame_by_number(video_path: str, frame_number: int):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None

# frame을 저장
def save_frame(frame, save_path: str):
    cv2.imwrite(save_path, frame)



# YOLO 모델로 label 판독