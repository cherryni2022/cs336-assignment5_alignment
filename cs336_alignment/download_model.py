import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ['HF_ENDPOINT'] = 'https://xget.xi-xu.me/hf'
# 模型的官方 ID
model_id = "Qwen/Qwen2.5-Math-1.5B"
#https://huggingface.co/Qwen
# 目标保存路径 (您可以自定义)
# 作业中提到的路径是 /data/a5-alignment/models/Qwen2.5-Math-1.5B
# 这里我们演示如何下载到当前目录下的一个文件夹中
save_directory = "models/Qwen2.5-Math-1.5B-Base"

print(f"正在从 Hugging Face Hub 下载模型: {model_id}")
print(f"这可能需要一些时间，取决于您的网络速度...")

try:
    # 1. 下载模型和分词器
    #    这会将模型文件下载到 Hugging Face 的全局缓存中
    #    使用 torch_dtype=torch.bfloat16 可以节省内存，与作业要求一致
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto"  # 自动将模型加载到可用的 GPU 上
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    print("\n模型已成功加载到内存中。")

    # 2. 将模型从缓存保存到您指定的目录
    #    这样做可以使模型文件独立于缓存，方便管理和迁移
    print(f"正在将模型保存到本地目录: {save_directory}")
    model.save_pretrained(save_directory)
    tokenizer.save_pretrained(save_directory)

    print(f"\n✅ 模型和分词器已成功下载并保存到 '{save_directory}' 文件夹中。")
    print("您现在可以在代码中通过这个本地路径来加载模型。")

except Exception as e:
    print(f"\n❌ 下载或保存过程中发生错误: {e}")
    print("请检查您的网络连接、磁盘空间和 Hugging Face Hub 的访问权限。")
