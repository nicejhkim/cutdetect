import os, sys

root_dir = os.getcwd()
sys.path.insert(0, str(root_dir))

from src.common import utils
from src.scene import cut

os.environ["TOKENIZERS_PARALLELISM"] = "false"

src_file = 'samples/health.mp4'


if __name__ == "__main__":

    ret = utils.measure_time('cut_scenes', 
                cut.cut_video_ffmpeg,
                'samples/health.mp4',
                200.0,
                215.0,
                'test/cut_health.mp4')

    ranges = [[300.0, 315.0], [350, 365], [400, 415], [450, 465], [500, 515], 
            [550, 565], [600, 615], [650, 665], [700, 715], [750, 765]]


    tasks = []
    for i, j in enumerate(ranges):
        tasks.append(('samples/health.mp4', j[0], j[1], f'test/cut_health_{i}.mp4'))


    # 순차적으로 10개 처리
    ret = utils.measure_time('cut_scenes_sequencial', 
                cut.cut_video_ffmpeg_seqential,
                tasks
                )

    # 병렬로 10개 처리
    ret = utils.measure_time('cut_scenes_parallel', 
                cut.cut_video_ffmpeg_parallel,
                tasks
                )


    print(ret)