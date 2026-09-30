功能：基于sam3，写了几个脚本，目标是从多人视频得到主要的几个单人的视频

配置：先clone本项目，然后按照 https://github.com/facebookresearch/sam3 进行环境配置

模型权重：将/media/data/xuhao_data/crowd2solo/checkpoints复制到你的项目目录下

功能测试：直接修改内部的超参数，运行单个脚本

批量运行：修改config.yml的配置，运行main.py里给的命令