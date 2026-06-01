# 原仓库地址：https://github.com/ZiqiaoPeng/SyncTalk_2D

# 进行的工作：训练优化及实时推流（虚拟摄像头）

# SyncTalk_2D 项目详细使用文档
## 文档说明
本文档基于 GitHub 开源项目 **SyncTalk_2D** 整理，该项目是一款**2D 定制化唇形同步模型**，主打高保真、低延迟的实时数字人唇形驱动，可落地商业级高清数字人场景。

## 一、项目概述
### 1.1 项目简介
SyncTalk_2D 衍生自 SyncTalk 与 Ultralight-Digital-Human 两大开源项目，是面向 2D 画面的唇形视频生成模型。
- 核心优势：画质高、延迟低，支持**实时唇形同步视频生成**；
- 版本优化：升级音频特征编码器，将画面分辨率提升至 328，适配高清输入视频，满足商业数字人使用标准。

### 1.2 项目基本信息
- 开源地址：ZiqiaoPeng/SyncTalk_2D（GitHub）
- 开发语言：Python（99.1%）、Shell（0.9%）
- 开源数据：131 Star、30 Fork、仅1位代码贡献者
- 维护状态：最后一次代码更新为11个月前，暂无正式版本发布
- 代码构成：包含数据处理、模型结构、训练脚本、推理脚本、预训练权重目录等完整工程文件

## 二、运行环境搭建
### 2.1 基础环境要求
- Python 版本：3.10
- 算力要求：支持 NVIDIA GPU + CUDA，显存不足可调小批次大小（batch_size）缓解显存溢出（OOM）问题

### 2.2 环境安装步骤
推荐使用 Conda 管理虚拟环境，执行以下命令依次部署：
1. 创建并激活虚拟环境
```bash
conda create -n synctalk_2d python=3.10
conda activate synctalk_2d
```

2. 安装深度学习框架（PyTorch 配套 CUDA 12.1）
```bash
conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=12.1 -c pytorch -c nvidia
```

3. 安装视频处理依赖（FFmpeg 为必备组件）
```bash
conda install -c conda-forge ffmpeg
```

4. 安装 Python 第三方库
```bash
pip install opencv-python transformers soundfile librosa onnxruntime-gpu configargparse
pip install numpy==1.23.5
```

## 三、训练数据集制作规范
模型训练需自行录制专属人像视频，**严格遵循以下录制要求**，否则会影响最终唇形同步效果：
1. **画面姿态**：人像正对镜头，头部无大幅度动作，全程保持固定姿态；摄像机位置、角度不可移动。
2. **光线环境**：录制环境光线充足且全程无明显明暗变化。
3. **服饰要求**：避免穿着纹理复杂的衣物，优先选择纯色服装。
4. **音频要求**：
   - 仅保留录制人声，无其他人声、环境杂音；
   - 视频开头、结尾各预留 5 秒静音片段。
5. **视频时长**：单段视频总时长 5 分钟。
6. **帧率说明**：无需手动设置帧率，代码会自动将视频转换为 25fps。

## 四、模型训练流程
### 4.1 数据存放目录
按照指定路径放置录制好的视频文件，目录格式统一为：
`dataset/自定义名称/自定义名称.mp4`
示例：将名为 May 的人像视频放入 `dataset/May/May.mp4`

### 4.2 启动训练脚本
使用 Shell 脚本执行训练，命令格式：
```bash
bash training_328.sh 数据名称 GPU卡号
```
示例（使用 0 号显卡训练 May 数据集）：
```bash
bash training_328.sh May 0
```

### 4.3 训练相关说明
1. **训练时长**：单组数据完整训练耗时约 5 小时；
2. **故障处理**：运行中出现显存溢出（OOM）报错，进入对应脚本调低 `batch_size` 参数即可；
3. **文件输出**：训练过程生成的模型权重会自动保存至项目 `model/checkpoints` 目录。

## 五、模型推理（生成唇形同步视频）
训练完成后，使用音频驱动人像视频，生成唇形同步结果。

### 5.1 推理执行命令
```bash
python inference_328.py --name 训练数据名 --audio_path 音频文件路径.wav
```

### 5.2 使用示例
调用 May 模型，使用 demo 目录下的音频生成视频：
```bash
python inference_328.py --name May --audio_path demo/talk_hb.wav
```

### 5.3 结果输出
推理生成的唇形同步视频，会自动保存到项目根目录下的 `result` 文件夹中。

## 六、项目目录结构说明
项目核心目录&文件作用一览：
1. `data_utils`：数据预处理、视频/音频解析工具代码
2. `demo`：示例音频、测试素材
3. `model/checkpoints`：训练后模型权重保存目录
4. `result`：推理生成的最终视频输出目录
5. `syncnet_ckpt`：SyncNet 相关预训练权重文件
6. 核心代码文件
   - `datasetsss.py / datasetsss_328.py`：数据集加载、数据预处理逻辑
   - `syncnet.py / syncnet_328.py`：音频特征编码网络
   - `unet.py / unet_328.py`：图像生成主干网络
   - `train.py / train_328.py`：模型训练核心代码
   - `inference.py / inference_328.py`：模型推理核心代码
   - `utils.py`：通用工具函数
7. 启动脚本
   - `training.sh / training_328.sh`：训练启动 Shell 脚本
   - `training_328.sh`：328 高清分辨率版本训练脚本（推荐使用）

## 七、致谢与引用
本项目基于 **Ultralight-Digital-Human** 和 **SyncTalk** 两个开源项目二次开发，在此对原作者的开源工作表示感谢。

## 八、常见问题补充
1. **FFmpeg 报错**：代表视频编解码依赖未安装成功，重新执行 `conda install ffmpeg`；
2. **CUDA 报错**：检查显卡驱动、CUDA 版本是否与 PyTorch 匹配；
3. **生成视频唇形错位**：优先检查原始训练视频是否符合录制规范（杂音、动作、光线问题）；
4. **运行速度慢**：确认已使用 GPU 运行，核对 CUDA、GPU 加速组件是否正常启用。