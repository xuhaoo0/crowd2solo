'''
基本功能：
norm_30fps：将原视频帧率统一为30 FPS，并覆盖原视频
split：切一条视频
multi2single：多人视频->主要人物的单人视频
'''

'''
输出示例：
假设处理的视频是：{input_dir}/a/b/c.mp4
{output_dir}
- a/b/c
  - c_001
    - c_001.mp4  # from split【这条视频不应该拿去重建】
    - c_001_person1.mp4
    - c_001_person3.mp4
  - c_002
    ...
'''