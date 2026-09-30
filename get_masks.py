'''
保存多人视频里面每个人的mask

args:
- input_video，例如a/b/c.mp4
- device，例如"0,1"，表示使用这些卡分摊一个任务，防止oom

输出：
a/b/c_masks
- person1
    - mask
        00000.png
    - person1_mask.mp4
- person2

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
    output_dir: Path,
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
                    (output_dir / f"person{person_index}" / "mask").mkdir(
                        parents=True
                    )

                person_index = object_to_person[object_id]
                mask_path = (
                    output_dir
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
    output_dir: Path,
    saved_frames: dict[int, set[int]],
    frame_count: int,
    width: int,
    height: int,
) -> None:
    """人物不可见的帧使用全黑 mask 补齐。"""
    empty_mask = Image.new("L", (width, height), 0)

    for person_index, frame_indices in saved_frames.items():
        mask_dir = output_dir / f"person{person_index}" / "mask"
        for frame_index in range(frame_count):
            if frame_index not in frame_indices:
                empty_mask.save(mask_dir / f"{frame_index:06d}.png")


def create_mask_videos(
    output_dir: Path,
    person_count: int,
    frame_count: int,
    fps: str,
) -> None:
    """将每个人的二值 mask 合成为 MP4 可视化视频。"""
    for person_index in range(1, person_count + 1):
        person_dir = output_dir / f"person{person_index}"
        mask_dir = person_dir / "mask"
        video_path = person_dir / f"person{person_index}_mask.mp4"
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
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(video_path),
        ]
        subprocess.run(command, check=True)


def process_video(
    video_path: Path,
    output_dir: Path,
    checkpoint_path: Path,
    prompt: str,
    gpu_ids: list[int],
) -> None:
    """执行人物分割、mask 补齐和视频生成。"""
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True)

    with tempfile.TemporaryDirectory(prefix="sam3_video_frames_") as temp_dir:
        frame_dir = Path(temp_dir)
        fps, width, height = get_video_info(video_path)
        frame_count = extract_frames(video_path, frame_dir)
        print(f"共 {frame_count} 帧，开始运行 SAM3。")

        saved_frames = save_masks_with_sam3(
            frame_dir,
            output_dir,
            checkpoint_path,
            prompt,
            gpu_ids,
        )

    fill_missing_masks(
        output_dir,
        saved_frames,
        frame_count,
        width,
        height,
    )
    create_mask_videos(
        output_dir,
        len(saved_frames),
        frame_count,
        fps,
    )
    print(f"处理完成，共检测到 {len(saved_frames)} 个人。")
    print(f"结果保存在：{output_dir}")


def parse_cli_paths(
    input_video: Path,
    device: str,
) -> tuple[Path, str]:
    """从命令行读取参数，未传入的参数沿用测试值。"""
    parser = argparse.ArgumentParser(description="保存多人视频里面每个人的mask")

    parser.add_argument(
        "--input_video",
        "--input-video",
        dest="input_video",
        type=Path,
        default=input_video,
        help="需要分割人物的视频路径",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=device,
        help='GPU 编号，例如 "0,1"，使用这些卡分摊一个任务，防止oom',
    )

    args = parser.parse_args()

    return args.input_video, args.device


if __name__ == "__main__":
    start_time = time.perf_counter()

    # 【用于测试】
    input_video = Path("dance/dance/boomboom_iconx.mp4")
    device = "0,1"  # 使用多卡分摊一个任务，防止oom

    # 从外部读取
    input_video, device = parse_cli_paths(input_video, device)

    # 所有超参数都在这里直接修改
    PROJECT_ROOT = Path(__file__).resolve().parent
    CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints/sam3.pt"
    PROMPT = "person"
    gpu_ids = [int(gpu_id) for gpu_id in device.split(",")]

    # 输出目录：a/b/c.mp4 → a/b/c_masks
    output_dir = input_video.parent / f"{input_video.stem}_masks"

    if output_dir.exists():  # 如果存在目标文件夹，就跳过
        print(f"跳过已存在的结果：{output_dir}")
    else:
        print(f"\n开始处理：{input_video}")
        try:
            process_video(
                input_video,
                output_dir,
                CHECKPOINT_PATH,
                PROMPT,
                gpu_ids,
            )
        except Exception:
            # 失败时删除不完整结果，方便下次重新处理
            shutil.rmtree(output_dir, ignore_errors=True)
            raise

    elapsed_seconds = round(time.perf_counter() - start_time)
    hours, remainder = divmod(elapsed_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"segment_people运行时间：{hours}时{minutes}分{seconds}秒")
