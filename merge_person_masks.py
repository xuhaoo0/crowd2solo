'''
合并所有人的mask，会适当向外扩展mask
'''

import subprocess
from pathlib import Path


def person_number(mask_path):
    """从 person 文件夹名称中提取人物编号。"""
    return int(mask_path.parent.name.removeprefix("person"))


def merge_person_masks(mask_root, output_path, dilate_pixels):
    """将每个人物的 mask 向外扩展后合并为一个并集视频。"""
    mask_paths = sorted(mask_root.glob("person*/person*_mask.mp4"), key=person_number)

    command = ["ffmpeg", "-y", "-loglevel", "error"]
    for mask_path in mask_paths:
        command.extend(["-i", str(mask_path)])

    # 每执行一次 dilation，mask 边缘向外扩展约一个像素
    dilation_filters = ",".join(["dilation"] * dilate_pixels)
    filter_parts = [
        f"[{index}:v]{dilation_filters}[p{index}]"
        for index in range(len(mask_paths))
    ]

    # lighten 会逐像素保留较亮的值，连续叠加后就是所有 mask 的并集
    filter_parts.append("[p0][p1]blend=all_mode=lighten[m1]")
    for index in range(2, len(mask_paths)):
        filter_parts.append(
            f"[m{index - 1}][p{index}]blend=all_mode=lighten[m{index}]"
        )

    command.extend(
        [
            "-filter_complex",
            ";".join(filter_parts),
            "-map",
            f"[m{len(mask_paths) - 1}]",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(output_path),
        ]
    )
    subprocess.run(command, check=True)
    print(f"合并完成：{output_path}")


if __name__ == "__main__":
    # 所有超参数都在这里直接修改
    PROJECT_ROOT = Path(__file__).resolve().parent
    MASK_ROOT = PROJECT_ROOT / "dance/mask/boomboom_iconx"
    OUTPUT_PATH = MASK_ROOT / "boomboom_iconx_mask.mp4"
    DILATE_PIXELS = 5  # 每个人的 mask 向外扩展的像素数

    merge_person_masks(MASK_ROOT, OUTPUT_PATH, DILATE_PIXELS)
