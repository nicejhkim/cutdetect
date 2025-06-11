# detection prototype
(AI 모델 학습을 위해) 원본 영상에서 특정 클립을 잘라내는 작업의 프로토타입

# modules
기본적인 작업은
    1. 원본 영상을 scene 단위로 분리
    2. 각 scene 에서 대표 frame 추출
    3. 추출한 frame의 기반으로 scene의 구성(object) 분석
    4. 목적에 맞는 scene 영상으로 저장
    5. 영상 후처리 (비식별화 처리 등)
다섯 단계로 이루어진다.

## 1. scene 분리
- scenedetect(https://www.scenedetect.com/docs/latest/)를 이용해서 scene을 분리한다. 
- ffmpeg 을 이용해서 분리하는 것에 비해 PC기준 3배 정도 더 걸린다.

## 2. frame 추출
## 3. frame 분석
### YOLO
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

## 4. scene 저장
    - ffmpeg으로 특정 구간을 잘라서 파일로 만드는 과정
    - 하나의 원본 영상에서 15초 영상을 10개 잘라내는데 걸리는 시간은 아래와 같다. (Mac, CPU 기준)
```
  Step                    Total Time (s)   Avg per Call (ms)  
 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 
  cut_scenes                        2.57               25.66  # 1개
  cut_scenes_sequencial            60.24              602.39  # 10개 순차
  cut_scenes_parallel              34.57              345.66  # 8개 병렬(os.cpu_count()//2)
```

## 5. scene 후처리


