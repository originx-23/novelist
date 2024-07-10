import tkinter as tk
from tkinter import messagebox
from threading import Thread
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

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
    result = nlp(prompt, max_length=200, truncation=False)  # 指定生成文本的最大长度
    output_text.delete("1.0", tk.END)
    output_text.insert(tk.END, result[0]['generated_text'])


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

# 创建输入框
input_label = tk.Label(root, text="输入：")
input_label.pack()
input_text = tk.Text(root, height=10, width=50)
input_text.pack()

# 创建输出框
output_label = tk.Label(root, text="输出：")
output_label.pack()
output_text = tk.Text(root, height=15, width=50)
output_text.pack()

# 创建生成按钮
generate_button = tk.Button(root, text="生成文本", command=generate_text)
generate_button.pack()

# 创建复制按钮
copy_button = tk.Button(root, text="复制", command=copy_text)
copy_button.pack()

# 运行主循环
root.mainloop()