# detection prototype
(AI 모델 학습을 위해) 원본 영상에서 특정 클립을 잘라내는 작업의 프로토타입

# modules
기본적인 작업은
    1. 원본 영상을 컷 단위로 분리
    2. 각 컷 에서 대표 frame 추출
    3. 추출한 frame의 기반으로 객체(object) 분석
    4. 목적에 맞는 컷 영상으로 저장
    5. 영상 후처리 (비식별화 처리 등)
다섯 단계로 이루어진다.

## 1. 컷 분리
- scenedetect(https://www.scenedetect.com/docs/latest/)를 이용해서 컷을 분리한다. 
- ffmpeg 을 이용해서 분리하는 것에 비해 PC기준 3배 정도 더 걸린다. 하지만 사용하기 편하고 더 정확하다.
- 컷 분리에 걸리는 시간은 전체 프로세스에서 큰 비중을 차지하지 않아서 scenedetect를 사용한다.

```
# scenedetect(https://www.scenedetect.com/docs/)를 이용해서 원본 영상을 컷 단위로 분리
def split_cuts_scenedetect(video_path: str, threshold: float = 30.0) -> list:
```

## 2. frame 추출
- 각 컷의 대표 frame을 추출한다. 대표 프레임은 임의로 컷의 가운데 프레임으로 설정한다.
- frame 추출은 cv2 를 이용하고, 정확한 프레임 단위의 지정이 필요하지는 않아서 시간 단위로 지정한다.
- 추출된 대표 frame은 추후 확인을 위해 FRAMES_DIR에 저장한다.

## 3. frame 분석
- 추출한 대표 frame에 특정한 객체가 있는지 찾아낸다.
- 첫번째로 CLIP 모델을 이용하여 원하는 프롬프트를 inference 해본다. 여기서는 아래의 프롬프트로 테스트했다. 다양한 영상으로 테스트를 해 보고 프롬프트와 threshold 를 조정할 필요 있다.
```
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
```
- 단, CLIP 모델은 멀티라벨에 대한 분류를 지원하지 않으므로 두 개 이상의 객체가 존재하는 frame의 경우 정확한 결과가 나오지 않을 수 있다. 이를 해결하기 위해서는 frame을 YOLO로 영역 분할하여 각 영역에 CLIP 적용하는 등의 방법을 사용할 수 있으나, 산이나 강 등 전체 화면을 차지하는 객체에 대해서는 효과적인 결과를 주는지 확인이 필요하다.

- CLIP 적용 이후에 YOLO 모델을 이용해서 labeling을 해본다. 여기서는 발견된 객체 중 person 에 대해서만 조건을 적용했다. (사람이 전체 화면에서 일정 비율 이상의 면적을 차지할 경우 제외하는 방식)
    - YOLO11m 이용



### YOLO11m class id
    0: 'person'
    1: 'bicycle'
    2: 'car'
    3: 'motorcycle'
    4: 'airplane'
    5: 'bus'
    6: 'train'
    7: 'truck'
    8: 'boat'
    9: 'traffic light'
    10: 'fire hydrant'
    11: 'stop sign'
    12: 'parking meter'
    13: 'bench'
    14: 'bird'
    15: 'cat'
    16: 'dog'
    17: 'horse'
    18: 'sheep'
    19: 'cow'
    20: 'elephant'
    21: 'bear'
    22: 'zebra'
    23: 'giraffe'
    24: 'backpack'
    25: 'umbrella'
    26: 'handbag'
    27: 'tie'
    28: 'suitcase'
    29: 'frisbee'
    30: 'skis'
    31: 'snowboard'
    32: 'sports ball'
    33: 'kite'
    34: 'baseball bat'
    35: 'baseball glove'
    36: 'skateboard'
    37: 'surfboard'
    38: 'tennis racket'
    39: 'bottle'
    40: 'wine glass'
    41: 'cup'
    42: 'fork'
    43: 'knife'
    44: 'spoon'
    45: 'bowl'
    46: 'banana'
    47: 'apple'
    48: 'sandwich'
    49: 'orange'
    50: 'broccoli'
    51: 'carrot'
    52: 'hot dog'
    53: 'pizza'
    54: 'donut'
    55: 'cake'
    56: 'chair'
    57: 'couch'
    58: 'potted plant'
    59: 'bed'
    60: 'dining table'
    61: 'toilet'
    62: 'tv'
    63: 'laptop'
    64: 'mouse'
    65: 'remote'
    66: 'keyboard'
    67: 'cell phone'
    68: 'microwave'
    69: 'oven'
    70: 'toaster'
    71: 'sink'
    72: 'refrigerator'
    73: 'book'
    74: 'clock'
    75: 'vase'
    76: 'scissors'
    77: 'teddy bear'
    78: 'hair drier'
    79: 'toothbrush'`

## 4. 컷 저장
- ffmpeg으로 특정 구간을 잘라서 파일로 만드는 과정. CPU 기준 이 과정이 가장 오래 걸린다.
- 하나의 원본 영상에서 15초 영상을 10개 잘라내는데 걸리는 시간은 아래와 같다. (Mac, CPU 기준)
```
  Step                    Total Time (s)   Avg per Call (ms)  
 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 
  cut_scenes                        2.57               25.66  # 1개
  cut_scenes_sequencial            60.24              602.39  # 10개 순차
  cut_scenes_parallel              34.57              345.66  # 8개 병렬(os.cpu_count()//2)
```

## 5. 컷 후처리
- 나누어진 컷 파일에 대해 기본적인 비식별화 작업을 진행한다.
- yolov8n-face 모델을 이용하면 간단하게 얼굴 blur 처리를 할 수 있다.

