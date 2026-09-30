'''
将所有人的mask向外扩展5像素，然后去掉
然后利用周围的像素进行插值来重建环境
再将原视频的目标人物贴回去
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
    """启动 ffmpeg，将修复后的画面编码为无声视频。"""
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
    mask_dilate_pixels,
    inpaint_radius,
):
    """先重建无人物环境，再分别贴回选中的目标人物。"""
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

    output_dir.mkdir(parents=True, exist_ok=True)
    kernel_size = mask_dilate_pixels * 2 + 1
    remove_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )
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

        # 所有人物的 mask 向外扩展后，一次性从画面中去掉
        all_people_mask = np.maximum.reduce(list(masks.values()))
        remove_mask = cv2.dilate(all_people_mask, remove_kernel)
        background = cv2.inpaint(
            frame,
            remove_mask,
            inpaint_radius,
            cv2.INPAINT_TELEA,
        )

        # 在同一张重建环境上，分别贴回目标人物的原始像素
        for target_name in selected_people:
            single_person_frame = background.copy()
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

    # 处理参数
    TOP_K = 3  # 选择排名前 K 的人物进行单独视频生成，或者定一个score的阈值
    MASK_DILATE_PIXELS = 5  # 所有人物的 mask 向外扩展 5 像素
    INPAINT_RADIUS = 5

    for video_path in sorted(VIDEO_DIR.glob("*.mp4")):
        video_start_time = time.perf_counter()
        video_name = video_path.stem
        dance_root = video_path.parent.parent
        mask_root = dance_root / "mask" / video_name
        output_dir = dance_root / "single" / video_name

        print(f"开始处理：{video_name}")
        create_single_person_videos(
            video_path,
            mask_root,
            output_dir,
            TOP_K,
            MASK_DILATE_PIXELS,
            INPAINT_RADIUS,
        )

        video_seconds = time.perf_counter() - video_start_time
        video_minutes, video_seconds = divmod(video_seconds, 60)
        print(
            f"{video_name} 运行时间："
            f"{int(video_minutes):02d}:{video_seconds:05.2f}"
        )

    elapsed_seconds = time.perf_counter() - start_time
    hours, remainder = divmod(elapsed_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"运行时间：{int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}")
