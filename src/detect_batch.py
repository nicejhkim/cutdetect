import json
import shutil
import os, sys
from common import utils
from frame import frames, clip, yolo
from scene import cut, split

from rich.panel import Panel
from rich import print as rprint

# 사용할 scene의 최소, 최대 길이
MIN_SEC = 5.0
MAX_SEC = 10.0

PROMPTS = [
    "a photo of a building exterior",
    "a tall skyscraper",
    "an old temple building exterior",
    "cityscape"
    "a road",
    "a road with cars",
    "a road with buildings",
    "a mountain landscape",
    "a beach or sea",
    "a river",
    "a field"
]
CLIP_THRESHOLD = 0.7

VIDEO_PATH = "samples/sinkhole.mp4"
CONFIG = {
    'FRAMES_DIR': "frames_{}",
    'SCENES_DIR': "scenes_{}",
    'DETECTION_LOG': "results_{}.json"
}

def process(input_file: str):
    #1. 씬 단위 정보 
    scenes = utils.measure_time('detect_scenes', split.split_scenes_scenedetect, input_file)
    rprint(Panel(f"[green]🔍 총 감지된 씬 수: {len(scenes)}[/green]"))

    results = []
    scene_paths = []    # blur 처리 할 scene 파일명
    cut_tasks = []      # 의미 있는 것으로 판명된, 잘라낼 씬 정보

    #2. 각 씬에 대해 
    for i, (start, end) in enumerate(scenes):
        start_sec = start.get_seconds()
        end_sec = end.get_seconds()
        duration = end_sec - start_sec

        start_frame = start.get_frames()
        end_frame = end.get_frames()

        result = {
            'scene_id': i,
            'start_time': start_sec,
            'end_time': end_sec,
            'duration': duration,
            'start_frame': start_frame,
            'end_frame': end_frame
        }

        #3. 길이 확인
        if duration < MIN_SEC or duration > MAX_SEC:
            result['error'] = 'invalid length'
        else:
            #4. 중간 프레임 추출
            mid_sec = (start_sec + end_sec) / 2
            frame = utils.measure_time('extract_frame', frames.extract_frame, input_file, mid_sec)
            if frame is None:
                result['error'] = 'failed to extract frame'
            else:
                #5. 추출한 프레임 파일로 저장
                frame_path = os.path.join(CONFIG['frames_dir'], f"frame_{i}.png")
                utils.measure_time('save_frame', frames.save_frame, frame, frame_path)

                #6. CLIP 모델 이용해서 프레임 식별
                probs = utils.measure_time('classify_with_clip', clip.classify_with_clip, frame, PROMPTS)
                best_label, best_prob = clip.get_best_from_probs(probs, PROMPTS)
                probs_f = [t.tolist() for t in probs]
                result.update({
                    'probs': probs_f,
                    'best_label': best_label,
                    'best_prob': float(best_prob),
                })

                print(f"[{i:03d}] Top match: {best_label} ({best_prob:.2%})")

                if best_prob > CLIP_THRESHOLD:

                    # 7. YOLO_face 모델 이용해서 화면에 사람이 너무 크게 나오지 않는지 판독
                    yolo_results = utils.measure_time('detect_labels_with_yolo', yolo.detect_labels_with_yolo, frame)
                    result['yolo_labels'] = [yolo_results.names[int(box.cls[0])] for box in yolo_results.boxes]
                    too_large_person = yolo.detect_too_large_person(frame, yolo_results, 0.2)

                    frame_person = yolo.draw_person_boxes(frame, yolo_results)
                    frames.save_frame(frame_person, os.path.join(CONFIG['frames_dir'], f"frame_{i}_person.png"))

                    if too_large_person:
                        result['error'] = 'too large person'
                else:
                    result['error'] = 'clip did not detect'
                    
        if 'error' not in result:
            output_path = os.path.join(CONFIG['scenes_dir'], f"scene_{i:03d}.mp4")
            scene_paths.append(output_path)
            cut_tasks.append((input_file, start_sec, end_sec, output_path))

        results.append(result)

    utils.measure_time('cut_scenes', cut.cut_video_ffmpeg_parallel, cut_tasks)
    utils.measure_time('blur_scenes', yolo.blur_yolo_parallel, scene_paths)

    utils.print_summary()

    # 결과 json으로 저장
    with open(CONFIG['detection_log'], 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    input_file = sys.argv[1] if len(sys.argv) > 1 else VIDEO_PATH
    if not os.path.isfile(input_file):
        print(f"❗ 파일이 존재하지 않음: {input_file}")
        sys.exit(1)

    base_name = utils.get_base_name(input_file)

    CONFIG['frames_dir'] = CONFIG['FRAMES_DIR'].format(base_name) # 인식할 frame 저장 경로
    CONFIG['scenes_dir'] = CONFIG['SCENES_DIR'].format(base_name) # 분리한 clip 저장 경로
    CONFIG['detection_log'] = CONFIG['DETECTION_LOG'].format(base_name)

    for folder in [CONFIG['frames_dir'], CONFIG['scenes_dir']]:
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder)

    # 작업 준비 끝.
    process(input_file)