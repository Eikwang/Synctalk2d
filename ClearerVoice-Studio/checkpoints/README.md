---
frameworks:
- Pytorch
license: Apache License 2.0
tasks:
- acoustic-noise-suppression

#model-type:
##如 gpt、phi、llama、chatglm、baichuan 等
#- gpt

#domain:
##如 nlp、cv、audio、multi-modal
#- nlp

#language:
##语言代码列表 https://help.aliyun.com/document_detail/215387.html?spm=a2c4g.11186623.0.0.9f8d7467kni6Aa
#- cn

#metrics:
##如 CIDEr、Blue、ROUGE 等
#- CIDEr

#tags:
##各种自定义，包括 pretrained、fine-tuned、instruction-tuned、RL-tuned 等训练方法和其他
#- pretrained

#tools:
##如 vllm、fastchat、llamacpp、AdaSeq 等
#- vllm
-speech processing
---


ClearerVoice-Studio 是一个开源的、基于 AI 的语音处理工具包，旨在为研究人员、开发者和最终用户提供服务。
它提供了语音增强、语音分离、语音超分辨率、目标说话人提取等多种功能。该工具包提供了先进的预训练模型（涵盖。 
我们先前发布的[FRCRN](https://modelscope.cn/models/iic/speech_frcrn_ans_cirm_16k)模型、[MossFormer](https://modelscope.cn/models/iic/speech_mossformer_separation_temporal_8k)、和[MossFormer2](https://modelscope.cn/models/iic/speech_mossformer2_separation_temporal_8k)系列模型等），并附带训练和推理脚本，所有内容都可以[ClearerVoice-Studio](https://github.com/modelscope/ClearerVoice-Studio/tree/main)仓库中获得。

您可以点击以下链接进行现场演示：

#### 👉🏻[ModelScope Demo](https://modelscope.cn/studios/iic/ClearerVoice-Studio)👈🏻  | 👉🏻[Huggingface Demo](https://huggingface.co/spaces/alibabasglab/ClearVoice)👈🏻

-------------------------------------------------------------------------------------------
｜请别忘了到我们的[ClearerVoice-Studio](https://github.com/modelscope/ClearerVoice-Studio/tree/main) 仓库（右上角）点个赞支持一下哦🙏 ｜
-------------------------------------------------------------------------------------------

## 环境安装

**仓库克隆和安装**

- 克隆仓库
``` sh
git clone https://github.com/modelscope/ClearerVoice-Studio.git
```

- 安装 Conda: 请参照https://docs.conda.io/en/latest/miniconda.html
- 生成Conda环境:

``` sh
cd ClearerVoice-Studio
conda create -n ClearerVoice-Studio python=3.8
conda activate ClearerVoice-Studio
pip install -r requirements.txt
```

**模型下载**

如果您的电脑可以访问Huggingface (https://huggingface.co/), 那么您不需要手动下载预训练模型，系统会自动在运行时进行下载。
如果您的电脑不可以访问Huggingface, 我们建议您从魔搭上按照下列方法进行模型下载：

方法 1: 使用Git克隆
``` sh
# git模型下载，请确保已安装git lfs
mkdir -p checkpoints
git clone https://www.modelscope.cn/iic/ClearerVoice-Studio.git checkpoints
```

方法 2: 先安装 [modelscope](https://github.com/modelscope/modelscope/tree/master)，然后运行下面代码： 
``` python
# SDK模型下载
from modelscope import snapshot_download
snapshot_download('iic/ClearerVoice-Studio', local_dir='checkpoints/')
```


**最后，以上安装好后，就可以运行我们的演示代码：**

``` sh
cd clearvoice
python demo.py
```

**就是这么简单，就是这么自然！**

您也可以参考这个ClearerVoice-Studio详细使用教程：https://stable-learn.com/zh/clearvoice-studio-tutorial 