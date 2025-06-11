
from collections import defaultdict
from time import perf_counter

from rich import box
from rich.console import Console
from rich.table import Table

import os

console = Console()
step_times = defaultdict(float)


def measure_time(key: str, func, *args, **kwargs):
    start = perf_counter()
    result = func(*args, **kwargs)
    step_times[key] += perf_counter() - start

    # print_summary()
    return result

def print_summary():
    summary = Table(title="Step Time Summary", box=box.SIMPLE_HEAVY)
    summary.add_column("Step", style="cyan")
    summary.add_column("Total Time (s)", justify="right", style="green")
    summary.add_column("Avg per Call (ms)", justify="right", style="magenta")

    for step, total_time in step_times.items():
        summary.add_row(step, f"{total_time:.2f}", f"{(total_time / 100) * 1000:.2f}")

    console.print()
    console.print(summary)

def add_postfix_to_filename(path: str, postfix: str) -> str:
    dir_name = os.path.dirname(path)
    base_name = os.path.basename(path)
    name, ext = os.path.splitext(base_name)
    return os.path.join(dir_name, f"{name}_{postfix}{ext}")

# 파일 경로에서 확장자를 제외한 기본 이름을 반환
def get_base_name(file_path):
    return os.path.splitext(os.path.basename(file_path))[0]