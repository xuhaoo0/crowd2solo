'''
将所有人物的 mask 区域填成纯白色
再将原视频的 TOP_K 个目标人物分别贴回去
'''

import subprocess
import time
from pathlib import Path

import cv2
import numpy as np


def person_number(person_dir):
    """提取 person 文件夹后的数字。"""
    return int(person_dir.name.removeprefix("person"))


def read_top_people(txt_path, top_k):
    """读取 selected_person.txt 中排名最前的人物。"""
    lines = txt_path.read_text(encoding="utf-8").splitlines()
    return [line.split(":", 1)[1].strip() for line in lines if line.startswith("rank")][
        :top_k
    ]


def start_encoder(output_path, width, height, fps, frame_count):
    """启动 ffmpeg，将处理后的画面编码为无声视频。"""
    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-frames:v",
        str(frame_count),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]
    return subprocess.Popen(command, stdin=subprocess.PIPE)


def create_single_person_videos(
    video_path,
    mask_root,
    output_dir,
    top_k,
):
    """将所有人物区域填白，再分别贴回选中的目标人物。"""
    selected_people = read_top_people(mask_root / "selected_person.txt", top_k)
    person_dirs = sorted(
        [path for path in mask_root.glob("person*") if path.is_dir()],
        key=person_number,
    )
    frame_count = len(list((person_dirs[0] / "mask").glob("*.png")))

    capture = cv2.VideoCapture(str(video_path))
    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.read()  # mask 视频去掉了原视频第一帧

    output_dir.mkdir(parents=True, exist_ok=True)
    encoders = {
        name: start_encoder(
            output_dir / f"{name}.mp4",
            width,
            height,
            fps,
            frame_count,
        )
        for name in selected_people
    }

    for frame_index in range(frame_count):
        _, frame = capture.read()
        masks = {
            person_dir.name: cv2.imread(
                str(person_dir / "mask" / f"{frame_index:06d}.png"),
                cv2.IMREAD_GRAYSCALE,
            )
            for person_dir in person_dirs
        }

        # 所有人物所在区域直接填成纯白色
        all_people_mask = np.maximum.reduce(list(masks.values()))
        white_background = frame.copy()
        white_background[all_people_mask > 0] = 255

        # 在同一张白色背景上，分别贴回每个目标人物
        for target_name in selected_people:
            single_person_frame = white_background.copy()
            target_mask = masks[target_name] > 0
            single_person_frame[target_mask] = frame[target_mask]
            encoders[target_name].stdin.write(single_person_frame.tobytes())

        print(f"\r正在处理：{frame_index + 1}/{frame_count}", end="", flush=True)

    capture.release()
    for encoder in encoders.values():
        encoder.stdin.close()
        encoder.wait()

    print(f"\n处理完成：{', '.join(selected_people)}")
    print(f"结果保存在：{output_dir}")


if __name__ == "__main__":
    start_time = time.perf_counter()

    # 处理 dance/dance 文件夹下的所有 MP4 视频
    PROJECT_ROOT = Path(__file__).resolve().parent
    VIDEO_DIR = PROJECT_ROOT / "dance/dance"
    TOP_K = 5

    for video_path in sorted(VIDEO_DIR.glob("*.mp4")):
        video_name = video_path.stem
        dance_root = video_path.parent.parent
        mask_root = dance_root / "mask" / video_name
        output_dir = dance_root / "single_white" / video_name

        print(f"开始处理：{video_name}")
        create_single_person_videos(
            video_path,
            mask_root,
            output_dir,
            TOP_K,
        )

    elapsed_seconds = time.perf_counter() - start_time
    hours, remainder = divmod(elapsed_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"运行时间：{int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}")
