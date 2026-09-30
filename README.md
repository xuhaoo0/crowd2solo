功能：基于sam3，写了几个脚本，目标是从多人视频得到主要的几个单人的视频

配置：请先clone本项目，然后按照https://github.com/facebookresearch/sam3进行环境配置，模型权重文件可以按需复制：/media/data/xuhao_data/sam3-main

添加的脚本如下：【注意：没有进行小批量测试】

norm_30fps.py：将原视频降采样为30fps

split.py：按画面图片切分视频

get_masks.py：输入视频，输出每个人的mask

select_main_people.py：根据mask选择最主要的K个人

get_single_person_videos.py：根据所有人的mask视频，在原视频里面把它们全部去掉，然后靠剩下的像素恢复环境，最后把目标人物覆盖回去，这样得到他的单人视频【修改：目前是每一帧都留下了，但是应该只留下这个人的连续帧，比如他只在前半段或者后半段出现，那就应该只保存这一段的视频，具体应该根据这个人的mask来判断】

