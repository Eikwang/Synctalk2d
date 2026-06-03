import subprocess
import os
import sys
import tempfile
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("AudioMerger")

def merge_audio_with_pingpong_loop(audio_a_path, audio_b_path, output_path, volume_ratio=0.2, sample_rate=16000):
    """
    将音频A与音频B合并，B以往复循环(正放→倒放→正放...)方式对齐A的长度，B音量为A的30%
    
    :param audio_a_path: 主音频文件路径
    :param audio_b_path: 循环音频文件路径
    :param output_path: 输出文件路径
    :param volume_ratio: B相对于A的音量比例 (0.0~1.0)
    :param sample_rate: 输出采样率 (Hz)
    """
    # 1. 验证输入文件
    for path in [audio_a, audio_b]:
        if not os.path.exists(path):
            logger.error(f"文件不存在: {path}")
            sys.exit(1)
        if not os.path.isfile(path):
            logger.error(f"路径不是文件: {path}")
            sys.exit(1)

    # 2. 创建临时文件存放往复循环单元
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as loop_unit_file:
        loop_unit_path = loop_unit_file.name

    try:
        # 3. 创建往复循环单元 (B_normal + B_reversed)
        logger.info("🔄 创建往复循环单元 (正放+倒放)...")
        loop_unit_cmd = [
            "ffmpeg", "-y",
            "-i", audio_b_path,
            "-filter_complex", 
                "[0:a]areverse[r];"   # 生成倒放版本
                "[0:a][r]concat=n=2:v=0:a=1",  # 拼接原始+倒放
            "-c:a", "pcm_s16le",
            loop_unit_path
        ]
        subprocess.run(loop_unit_cmd, 
                      stdout=subprocess.PIPE, 
                      stderr=subprocess.PIPE, 
                      check=True)

        # 4. 执行主合并操作
        logger.info(f"🔊 合并音频 (B音量={volume_ratio*100:.0f}%) → 采样率={sample_rate}Hz...")
        merge_cmd = [
            "ffmpeg", "-y",
            "-i", audio_a_path,
            "-stream_loop", "-1",  # 无限循环输入
            "-i", loop_unit_path,
            "-filter_complex",
                f"[1:a]volume={volume_ratio}[bg];"  # 设置B音量
                "[0:a][bg]amix=inputs=2:duration=first:dropout_transition=0",
            "-ar", str(sample_rate),  # 采样率
            "-ac", "1",               # 单声道
            "-c:a", "pcm_s16le",      # 16-bit WAV格式
            output_path
        ]
        
        # 执行命令并捕获错误
        result = subprocess.run(
            merge_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        if result.returncode != 0:
            logger.error(f"FFmpeg执行失败:\n{result.stderr}")
            sys.exit(1)
        
        # 5. 验证输出
        if not os.path.exists(output_path):
            logger.error("输出文件未生成")
            sys.exit(1)
            
        logger.info(f"✅ 成功! 输出文件: {os.path.abspath(output_path)}")
        return True

    except subprocess.CalledProcessError as e:
        logger.error(f"命令执行失败: {e.cmd}\n错误: {e.stderr.decode()}")
        return False
    except Exception as e:
        logger.exception(f"发生未预期错误: {str(e)}")
        return False
    finally:
        # 清理临时文件
        if os.path.exists(loop_unit_path):
            os.unlink(loop_unit_path)

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(f"用法: {sys.argv[0]} <主音频A> <循环音频B> <输出文件> [音量比例] [采样率]")
        print("  音量比例: B相对于A的音量 (0.0~1.0, 默认0.2)")
        print("  采样率: 输出采样率 (Hz, 默认16000)")
        sys.exit(1)
    
    # 获取命令行参数
    audio_a = sys.argv[1]
    audio_b = sys.argv[2]
    output = sys.argv[3]
    volume = float(sys.argv[4]) if len(sys.argv) > 4 else 0.2
    rate = int(sys.argv[5]) if len(sys.argv) > 5 else 16000

    # 执行合并
    merge_audio_with_pingpong_loop(
        audio_a_path=audio_a,
        audio_b_path=audio_b,
        output_path=output,
        volume_ratio=volume,
        sample_rate=rate
    )