'''
选择视频中主要人物，保存在txt文件
'''

from pathlib import Path

import numpy as np
from PIL import Image


def person_number(person_dir):
    """提取 person 文件夹后的数字。"""
    return int(person_dir.name.removeprefix("person"))


def read_small_mask(mask_path, analysis_scale):
    """缩小 mask，加快统计速度。"""
    with Image.open(mask_path) as image:
        width = max(1, int(image.width * analysis_scale))
        height = max(1, int(image.height * analysis_scale))
        image = image.resize((width, height), Image.Resampling.NEAREST)
        return np.asarray(image) > 0


def calculate_metrics(mask_paths, selection_count, analysis_scale):
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
    video_dir,
    keep_top_k,
    selection_frames,
    analysis_scale,
    prominence_weight,
    persistence_weight,
    motion_weight,
):
    """对一个视频目录中的人物评分，并写入筛选结果。"""
    person_dirs = sorted(
        [path for path in video_dir.glob("person*") if path.is_dir()],
        key=person_number,
    )
    if not person_dirs:
        return

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

    selected_path = video_dir / "selected_person.txt"
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

    print(f"\n{video_dir.name}：使用前 {selection_count}/{frame_count} 帧")
    for name, score, prominence, persistence, motion in top_results:
        print(
            f"  {name}: score={score:.4f}, "
            f"prominence_norm={prominence:.4f}, "
            f"persistence_norm={persistence:.4f}, motion_norm={motion:.4f}"
        )
    print(f"  筛选结果：{', '.join(selected_names)}")


if __name__ == "__main__":
    # 所有超参数都在这里直接修改
    PROJECT_ROOT = Path(__file__).resolve().parent
    OUTPUT_ROOT = PROJECT_ROOT / "dance/mask"
    KEEP_TOP_K = 5  # 保留最主要的5个人
    SELECTION_FRAMES = 0.5  # 使用前 50% 的帧
    ANALYSIS_SCALE = 0.5  # 缩小 mask 后计算，加快处理速度
    PROMINENCE_WEIGHT = 0.60
    PERSISTENCE_WEIGHT = 0.30
    MOTION_WEIGHT = 0.10

    for video_dir in sorted(path for path in OUTPUT_ROOT.iterdir() if path.is_dir()):
        select_people(
            video_dir,
            KEEP_TOP_K,
            SELECTION_FRAMES,
            ANALYSIS_SCALE,
            PROMINENCE_WEIGHT,
            PERSISTENCE_WEIGHT,
            MOTION_WEIGHT,
        )
