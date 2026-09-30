'''
输入一条多人视频，输出最主要的k个人物的单人视频和其他信息：
分为3步：用sam3提取所有人的mask、根据mask选择主要的k个人、生成单人的视频
其中最后一步：根据所有人的mask视频，在原视频里面把它们全部去掉，然后靠剩下的像素恢复环境，最后把目标人物覆盖回去，这样得到他的单人视频

args:
- input_video，例如a/b/c.mp4，表示多人视频
- device，例如0，表示用到的显卡
- sam3_device，例如[0,1]，表示使用这些卡分摊一个get_masks.py的任务，防止oom

输出：
a/b
- c.mp4  # 多人视频
- c_masks  # 存放所有人的mask
    - person1
        - mask
            00000.png
        - mask.mp4
    - person2
    ...
- c_selected_person.txt
- c_person1.mp4  # 主要人物的单人视频
- c_person3.mp4
'''

import argparse
import gc
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

# 减少长视频推理时的显存碎片
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import cv2
import numpy as np
from PIL import Image
import torch

from sam3.model_builder import build_sam3_video_predictor


def get_video_info(video_path: Path) -> tuple[str, int, int]:
    """读取原视频的帧率和分辨率。"""
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate",
        "-of",
        "json",
        str(video_path),
    ]
    stream = json.loads(
        subprocess.run(command, check=True, capture_output=True, text=True).stdout
    )["streams"][0]
    return stream["avg_frame_rate"], stream["width"], stream["height"]


def extract_frames(video_path: Path, frame_dir: Path) -> int:
    """使用 ffmpeg 将所有帧保存到临时目录。"""
    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-vsync",
        "0",
        "-start_number",
        "0",
        str(frame_dir / "%06d.png"),
    ]
    subprocess.run(command, check=True)
    return len(list(frame_dir.glob("*.png")))


def save_masks_with_sam3(
    frame_dir: Path,
    masks_dir: Path,
    checkpoint_path: Path,
    prompt: str,
    gpu_ids: list[int],
) -> dict[int, set[int]]:
    """使用文本提示检测并跟踪所有人物，逐帧保存人物 mask。"""
    predictor = build_sam3_video_predictor(
        checkpoint_path=str(checkpoint_path),
        gpus_to_use=gpu_ids,
    )

    object_to_person = {}
    saved_frames = {}
    session_id = None

    try:
        response = predictor.handle_request(
            {
                "type": "start_session",
                "resource_path": str(frame_dir),
                "offload_video_to_cpu": True,
                "offload_state_to_cpu": True,
            }
        )
        session_id = response["session_id"]

        predictor.handle_request(
            {
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": 0,
                "text": prompt,
            }
        )

        responses = predictor.handle_stream_request(
            {
                "type": "propagate_in_video",
                "session_id": session_id,
                "propagation_direction": "forward",
                "start_frame_index": 0,
            }
        )

        for response in responses:
            frame_index = response["frame_index"]
            outputs = response["outputs"]

            for object_id, mask in zip(
                outputs["out_obj_ids"], outputs["out_binary_masks"]
            ):
                object_id = int(object_id)
                if object_id not in object_to_person:
                    person_index = len(object_to_person) + 1
                    object_to_person[object_id] = person_index
                    saved_frames[person_index] = set()
                    (masks_dir / f"person{person_index}" / "mask").mkdir(
                        parents=True
                    )

                person_index = object_to_person[object_id]
                mask_path = (
                    masks_dir
                    / f"person{person_index}"
                    / "mask"
                    / f"{frame_index:06d}.png"
                )
                Image.fromarray(mask.astype("uint8") * 255).save(mask_path)
                saved_frames[person_index].add(frame_index)

            # mask 已经写入磁盘，不再保留交互式回看缓存
            inference_state = predictor._all_inference_states[session_id]["state"]
            inference_state["cached_frame_outputs"].pop(frame_index, None)
    finally:
        if session_id is not None:
            predictor.handle_request(
                {"type": "close_session", "session_id": session_id}
            )
        predictor.shutdown()
        del predictor
        gc.collect()
        torch.cuda.empty_cache()

    return saved_frames


def fill_missing_masks(
    masks_dir: Path,
    saved_frames: dict[int, set[int]],
    frame_count: int,
    width: int,
    height: int,
) -> None:
    """人物不可见的帧使用全黑 mask 补齐。"""
    empty_mask = Image.new("L", (width, height), 0)

    for person_index, frame_indices in saved_frames.items():
        mask_dir = masks_dir / f"person{person_index}" / "mask"
        for frame_index in range(frame_count):
            if frame_index not in frame_indices:
                empty_mask.save(mask_dir / f"{frame_index:06d}.png")


def create_mask_videos(
    masks_dir: Path,
    person_count: int,
    frame_count: int,
    fps: str,
    device: int,
) -> None:
    """将每个人的二值 mask 合成为 MP4 视频。"""
    for person_index in range(1, person_count + 1):
        person_dir = masks_dir / f"person{person_index}"
        mask_dir = person_dir / "mask"
        video_path = person_dir / "mask.mp4"
        command = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            fps,
            "-start_number",
            "0",
            "-i",
            str(mask_dir / "%06d.png"),
            "-frames:v",
            str(frame_count),
            "-c:v",
            "h264_nvenc",
            "-gpu",
            str(device),
            "-preset",
            "p1",
            "-rc",
            "vbr",
            "-cq",
            "23",
            "-b:v",
            "0",
            "-pix_fmt",
            "yuv420p",
            str(video_path),
        ]
        subprocess.run(command, check=True)


def save_masks(
    input_video: Path,
    masks_dir: Path,
    checkpoint_path: Path,
    prompt: str,
    gpu_ids: list[int],
    device: int,
) -> None:
    """提取所有帧，生成每个人物的 mask PNG 和 mask 视频。"""
    shutil.rmtree(masks_dir, ignore_errors=True)
    masks_dir.mkdir(parents=True)

    with tempfile.TemporaryDirectory(prefix="sam3_video_frames_") as temp_dir:
        frame_dir = Path(temp_dir)
        fps, width, height = get_video_info(input_video)
        frame_count = extract_frames(input_video, frame_dir)
        print(f"共 {frame_count} 帧，开始运行 SAM3。")

        saved_frames = save_masks_with_sam3(
            frame_dir,
            masks_dir,
            checkpoint_path,
            prompt,
            gpu_ids,
        )

    fill_missing_masks(
        masks_dir,
        saved_frames,
        frame_count,
        width,
        height,
    )
    create_mask_videos(
        masks_dir,
        len(saved_frames),
        frame_count,
        fps,
        device,
    )
    print(f"mask 生成完成，共检测到 {len(saved_frames)} 个人。")


def person_number(person_dir: Path) -> int:
    """提取 person 文件夹后的数字。"""
    return int(person_dir.name.removeprefix("person"))


def read_small_mask(mask_path: Path, analysis_scale: float) -> np.ndarray:
    """缩小 mask，加快统计速度。"""
    with Image.open(mask_path) as image:
        width = max(1, int(image.width * analysis_scale))
        height = max(1, int(image.height * analysis_scale))
        image = image.resize((width, height), Image.Resampling.NEAREST)
        return np.asarray(image) > 0


def calculate_metrics(
    mask_paths: list[Path],
    selection_count: int,
    analysis_scale: float,
) -> tuple[float, float, float]:
    """计算人物在选拔帧内的显著度、持续性和动作得分。"""
    area_ratios = []
    height_ratios = []
    motion_values = []
    visible_count = 0
    previous_mask = None

    for mask_path in mask_paths[:selection_count]:
        mask = read_small_mask(mask_path, analysis_scale)

        if mask.any():
            visible_count += 1
            rows = np.flatnonzero(mask.any(axis=1))
            area_ratios.append(mask.mean())
            height_ratios.append((rows[-1] - rows[0] + 1) / mask.shape[0])

        if previous_mask is not None and mask.any() and previous_mask.any():
            union = np.logical_or(mask, previous_mask).sum()
            intersection = np.logical_and(mask, previous_mask).sum()
            motion_values.append(1.0 - intersection / union)

        previous_mask = mask

    prominence = 0.0
    if area_ratios:
        prominence = 0.6 * np.mean(height_ratios) + 0.4 * np.mean(
            np.sqrt(area_ratios)
        )

    persistence = visible_count / selection_count
    motion = float(np.mean(motion_values)) if motion_values else 0.0
    return float(prominence), persistence, motion


def select_people(
    masks_dir: Path,
    selected_path: Path,
    keep_top_k: int,
    selection_frames: float,
    analysis_scale: float,
    prominence_weight: float,
    persistence_weight: float,
    motion_weight: float,
) -> list[str]:
    """对一个视频目录中的人物评分，并写入筛选结果。"""
    person_dirs = sorted(
        [path for path in masks_dir.glob("person*") if path.is_dir()],
        key=person_number,
    )
    if not person_dirs:
        raise RuntimeError(f"未找到人物 mask：{masks_dir}")

    frame_count = len(list((person_dirs[0] / "mask").glob("*.png")))
    selection_count = max(1, int(frame_count * selection_frames))
    metrics = []

    for person_dir in person_dirs:
        mask_paths = sorted((person_dir / "mask").glob("*.png"))
        prominence, persistence, motion = calculate_metrics(
            mask_paths,
            selection_count,
            analysis_scale,
        )
        metrics.append((person_dir.name, prominence, persistence, motion))

    max_prominence = max(item[1] for item in metrics)
    max_persistence = max(item[2] for item in metrics)
    max_motion = max(item[3] for item in metrics)
    results = []

    for name, prominence, persistence, motion in metrics:
        prominence /= max_prominence
        persistence /= max_persistence
        motion /= max_motion
        score = (
            prominence_weight * prominence
            + persistence_weight * persistence
            + motion_weight * motion
        )
        results.append((name, score, prominence, persistence, motion))

    top_results = sorted(results, key=lambda item: item[1], reverse=True)[:keep_top_k]
    selected_names = [item[0] for item in top_results]

    selected_lines = [
        f"rank{rank}: {name}"
        for rank, name in enumerate(selected_names, start=1)
    ]
    score_lines = [
        f"{name}: score={score:.4f}, "
        f"prominence_norm={prominence:.4f}, "
        f"persistence_norm={persistence:.4f}, motion_norm={motion:.4f}"
        for name, score, prominence, persistence, motion in top_results
    ]
    selected_path.write_text(
        "\n".join(selected_lines + [""] + score_lines) + "\n",
        encoding="utf-8",
    )

    print(f"\n{masks_dir.name}：使用前 {selection_count}/{frame_count} 帧")
    for name, score, prominence, persistence, motion in top_results:
        print(
            f"  {name}: score={score:.4f}, "
            f"prominence_norm={prominence:.4f}, "
            f"persistence_norm={persistence:.4f}, motion_norm={motion:.4f}"
        )
    print(f"  筛选结果：{', '.join(selected_names)}")

    return selected_names


def start_encoder(
    output_path: Path,
    width: int,
    height: int,
    fps: float,
    frame_count: int,
    device: int,
) -> subprocess.Popen:
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
        "h264_nvenc",
        "-gpu",
        str(device),
        "-preset",
        "p1",
        "-rc",
        "vbr",
        "-cq",
        "23",
        "-b:v",
        "0",
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]
    return subprocess.Popen(command, stdin=subprocess.PIPE)


def find_mask_segment(
    mask_dir: Path,
    frame_count: int,
) -> tuple[int, int] | None:
    """扫描人物的 mask，返回第一次出现到随后第一次消失之间的闭区间。"""
    start_frame = None

    for frame_index in range(frame_count):
        mask = cv2.imread(
            str(mask_dir / f"{frame_index:06d}.png"),
            cv2.IMREAD_GRAYSCALE,
        )
        if start_frame is None and mask.any():
            start_frame = frame_index
        elif start_frame is not None and not mask.any():
            return start_frame, frame_index - 1

    if start_frame is None:
        return None

    return start_frame, frame_count - 1


def create_single_person_videos(
    video_path: Path,
    masks_dir: Path,
    output_dir: Path,
    selected_people: list[str],
    mask_dilate_pixels: int,
    inpaint_radius: int,
    device: int,
) -> None:
    """先重建无人物环境，再分别贴回选中的目标人物，只保留每个人出现的帧段。"""
    person_dirs = sorted(
        [path for path in masks_dir.glob("person*") if path.is_dir()],
        key=person_number,
    )
    frame_count = len(list((person_dirs[0] / "mask").glob("*.png")))

    # 预扫描每个人物的 mask，只保留第一次出现到随后第一次消失的帧段
    segments = {}
    for name in selected_people:
        segment = find_mask_segment(masks_dir / name / "mask", frame_count)
        if segment is None:
            print(f"警告：{name} 全程不可见，跳过单人视频")
            continue
        segments[name] = segment
        start_frame, end_frame = segment
        print(
            f"{name}: frame {start_frame} ~ {end_frame}，"
            f"共 {end_frame - start_frame + 1} 帧"
        )

    if not segments:
        print("所有人物都不可见，未生成单人视频。")
        return

    min_start = min(start_frame for start_frame, _ in segments.values())
    max_end = max(end_frame for _, end_frame in segments.values())

    capture = cv2.VideoCapture(str(video_path))
    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    video_stem = video_path.stem
    kernel_size = mask_dilate_pixels * 2 + 1
    remove_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )
    encoders = {
        name: start_encoder(
            output_dir / f"{video_stem}_{name}.mp4",
            width,
            height,
            fps,
            end_frame - start_frame + 1,
            device,
        )
        for name, (start_frame, end_frame) in segments.items()
    }

    for frame_index in range(frame_count):
        print(f"\r正在处理：{frame_index + 1}/{frame_count}", end="", flush=True)
        _, frame = capture.read()

        # 所有人物帧段之外的部分直接跳过，不做重建
        if frame_index < min_start or frame_index > max_end:
            continue

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
        for target_name, (start_frame, end_frame) in segments.items():
            if start_frame <= frame_index <= end_frame:
                single_person_frame = background.copy()
                target_mask = masks[target_name] > 0
                single_person_frame[target_mask] = frame[target_mask]
                encoders[target_name].stdin.write(single_person_frame.tobytes())

    capture.release()
    for encoder in encoders.values():
        encoder.stdin.close()
        encoder.wait()

    print(f"\n单人视频生成完成：{', '.join(segments)}")


def process_video(
    input_video: Path,
    checkpoint_path: Path,
    prompt: str,
    sam3_gpu_ids: list[int],
    device: int,
    keep_top_k: int,
    selection_frames: float,
    analysis_scale: float,
    prominence_weight: float,
    persistence_weight: float,
    motion_weight: float,
    mask_dilate_pixels: int,
    inpaint_radius: int,
) -> None:
    """依次执行 mask 生成、主要人物筛选和单人视频合成。"""
    output_dir = input_video.parent
    video_stem = input_video.stem
    masks_dir = output_dir / f"{video_stem}_masks"

    if list(masks_dir.glob("person*/mask.mp4")):  # 如果 mask 结果已存在，就跳过 mask 生成
        print(f"跳过已存在的 mask 结果：{masks_dir}")
    else:
        try:
            save_masks(
                input_video,
                masks_dir,
                checkpoint_path,
                prompt,
                sam3_gpu_ids,
                device,
            )
        except Exception:
            # 失败时删除不完整结果，方便下次重新处理
            shutil.rmtree(masks_dir, ignore_errors=True)
            raise

    selected_people = select_people(
        masks_dir,
        output_dir / f"{video_stem}_selected_person.txt",
        keep_top_k,
        selection_frames,
        analysis_scale,
        prominence_weight,
        persistence_weight,
        motion_weight,
    )

    create_single_person_videos(
        input_video,
        masks_dir,
        output_dir,
        selected_people,
        mask_dilate_pixels,
        inpaint_radius,
        device,
    )

    print(f"\n处理完成，主要人物：{', '.join(selected_people)}")
    print(f"结果保存在：{output_dir}")


def parse_cli_paths(
    input_video: Path,
    device: int,
    sam3_device: list[int],
) -> tuple[Path, int, list[int]]:
    """从命令行读取参数，未传入的参数沿用测试值。"""
    parser = argparse.ArgumentParser(
        description="输入一条多人视频，输出主要人物的单人视频"
    )

    parser.add_argument(
        "--input_video",
        "--input-video",
        dest="input_video",
        type=Path,
        default=input_video,
        help="多人视频路径，例如 a/b/c.mp4",
    )

    parser.add_argument(
        "--device",
        type=int,
        default=device,
        help="GPU 编号，例如 0，用于视频编码",
    )

    parser.add_argument(
        "--sam3_device",
        "--sam3-device",
        dest="sam3_device",
        nargs="+",
        type=int,
        default=sam3_device,
        help="GPU 编号列表，例如 0 1，使用这些卡分摊 SAM3 任务，防止oom",
    )

    args = parser.parse_args()

    return args.input_video, args.device, args.sam3_device


if __name__ == "__main__":
    start_time = time.perf_counter()

    # 【用于测试】
    input_video = Path("temp_out/张资晃/张资晃/张资晃_001/张资晃_001.mp4")
    device = 6  # 用于视频编码
    sam3_device = [6, 7]  # 使用多卡分摊 SAM3 任务，防止oom

    # 从外部读取
    input_video, device, sam3_device = parse_cli_paths(
        input_video,
        device,
        sam3_device,
    )

    # 所有超参数都在这里直接修改
    PROJECT_ROOT = Path(__file__).resolve().parent
    CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints/sam3.pt"
    PROMPT = "person"

    KEEP_TOP_K = 3  # 保留最主要的3个人
    SELECTION_FRAMES = 0.5  # 使用前 50% 的帧
    ANALYSIS_SCALE = 0.5  # 缩小 mask 后计算，加快处理速度
    PROMINENCE_WEIGHT = 0.60
    PERSISTENCE_WEIGHT = 0.30
    MOTION_WEIGHT = 0.10
    MASK_DILATE_PIXELS = 5  # 所有人物的 mask 向外扩展 5 像素
    INPAINT_RADIUS = 5

    print(f"开始处理：{input_video}")
    print(f"输出目录：{input_video.parent}")
    print(f"使用 GPU：{device}，SAM3 GPU：{sam3_device}")

    process_video(
        input_video,
        CHECKPOINT_PATH,
        PROMPT,
        sam3_device,
        device,
        KEEP_TOP_K,
        SELECTION_FRAMES,
        ANALYSIS_SCALE,
        PROMINENCE_WEIGHT,
        PERSISTENCE_WEIGHT,
        MOTION_WEIGHT,
        MASK_DILATE_PIXELS,
        INPAINT_RADIUS,
    )

    elapsed_seconds = round(time.perf_counter() - start_time)
    hours, remainder = divmod(elapsed_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"crowd2solo运行时间：{hours}时{minutes}分{seconds}秒")
