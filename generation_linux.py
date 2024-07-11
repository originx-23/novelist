import time
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from tqdm import tqdm
import sys
import locale

# 确保使用正确的编码
sys.stdin.reconfigure(encoding='utf-8')
sys.stdout.reconfigure(encoding='utf-8')

# 检查CUDA是否可用，并选择设备
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 模型和分词器的加载
model_name = "NeuralNovel/Mistral-7B-Instruct-v0.2-Neural-Story"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 创建一个 pipeline，用于文本生成
nlp = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0 if device == "cuda" else -1)

# 对话循环
def main():
    print("大模型准备就绪。您可以开始提问，每次提问后请按回车键。输入 'exit' 退出对话。")

    while True:
        # 获取用户输入
        user_input = input("您: ").strip()
        if user_input.lower() == "exit":
            print("对话结束，再见！")
            break

        # 开始计时
        start_time = time.time()

        # 模拟生成进度条
        with tqdm(total=100, desc="生成中",
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} {elapsed}<{remaining}") as pbar:
            for _ in range(10):
                time.sleep(0.1)  # 模拟延迟
                pbar.update(10)

        # 生成文本
        result = nlp(user_input, max_length=200, truncation=False)

        # 结束计时
        end_time = time.time()
        duration = end_time - start_time

        # 显示生成结果和时间
        print(f"模型: {result[0]['generated_text']}")
        print(f"生成时间: {duration:.2f} 秒\n")

if __name__ == "__main__":
    main()
