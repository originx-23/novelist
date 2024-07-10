import tkinter as tk
from tkinter import messagebox
from threading import Thread
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import time

# 检查CUDA是否可用，并选择设备
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# 模型和分词器的加载
model_name = "NeuralNovel/Mistral-7B-Instruct-v0.2-Neural-Story"
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 创建一个 pipeline，用于文本生成
nlp = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0 if device == "cuda" else -1)

# 生成文本的函数 (多线程调用)
def generate_text():
    prompt = input_text.get("1.0", tk.END).strip()
    if prompt:
        output_text.delete("1.0", tk.END)
        output_text.insert(tk.END, "正在生成文本，请稍候...")

        # 创建并启动线程
        thread = Thread(target=generate_text_thread, args=(prompt,))
        thread.start()

# 生成文本的线程函数
def generate_text_thread(prompt):
    start_time = time.time()
    result = nlp(prompt, max_length=200, truncation=False)  # 指定生成文本的最大长度
    end_time = time.time()
    duration = end_time - start_time

    output_text.delete("1.0", tk.END)
    output_text.insert(tk.END, result[0]['generated_text'])
    duration_label.config(text=f"生成时间: {duration:.2f} 秒")

# 复制文本的函数
def copy_text():
    text = output_text.get("1.0", tk.END).strip()
    if text:
        root.clipboard_clear()
        root.clipboard_append(text)
        messagebox.showinfo("信息", "文本已复制到剪贴板")
        output_text.delete("1.0", tk.END)

# 创建主窗口
root = tk.Tk()
root.title("文本生成器")
root.geometry('600x600')
root.configure(bg='#f0f0f0')

# 创建输入框
input_frame = tk.Frame(root, bg='#f0f0f0', padx=10, pady=10)
input_frame.pack(fill=tk.BOTH, expand=True)

input_label = tk.Label(input_frame, text="输入：", bg='#f0f0f0')
input_label.pack(anchor='w')
input_text = tk.Text(input_frame, height=10, width=60, wrap=tk.WORD)
input_text.pack(fill=tk.BOTH, expand=True, pady=5)

# 创建输出框
output_frame = tk.Frame(root, bg='#f0f0f0', padx=10, pady=10)
output_frame.pack(fill=tk.BOTH, expand=True)

output_label = tk.Label(output_frame, text="输出：", bg='#f0f0f0')
output_label.pack(anchor='w')
output_text = tk.Text(output_frame, height=15, width=60, wrap=tk.WORD)
output_text.pack(fill=tk.BOTH, expand=True, pady=5)

# 创建持续时间标签
duration_label = tk.Label(root, text="生成时间: ", bg='#f0f0f0')
duration_label.pack()

# 创建按钮框架
button_frame = tk.Frame(root, bg='#f0f0f0', pady=10)
button_frame.pack(fill=tk.BOTH, expand=True)

generate_button = tk.Button(button_frame, text="生成文本", command=generate_text, width=20, bg='#4CAF50', fg='white')
generate_button.pack(side=tk.LEFT, padx=20)

copy_button = tk.Button(button_frame, text="复制", command=copy_text, width=20, bg='#008CBA', fg='white')
copy_button.pack(side=tk.RIGHT, padx=20)

# 运行主循环
root.mainloop()