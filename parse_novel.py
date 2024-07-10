import json
import os

def parse_novel_to_conversation(file_path):
    conversations = []
    conversation = []
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip()
            if line.startswith('human:'):
                conversation.append({
                    "from": "human",
                    "value": line.replace('human:', '').strip()
                })
            elif line.startswith('gpt:'):
                conversation.append({
                    "from": "gpt",
                    "value": line.replace('gpt:', '').strip()
                })
    return conversation

def convert_to_xtuner_format(input_file, output_file, image_folder):
    conversations = []
    conversation = parse_novel_to_conversation(input_file)
    conversation_id = os.path.splitext(os.path.basename(input_file))[0]
    conversation_data = {
        "id": conversation_id,
        "image": os.path.join(image_folder, "image_1.jpg"),  # 假设每个对话都有一张对应的图片
        "conversation": conversation
    }
    conversations.append(conversation_data)

    with open(output_file, 'w', encoding='utf-8') as outfile:
        json.dump(conversations, outfile, ensure_ascii=False, indent=4)

input_file = 'novel.txt'
output_file = 'novel_data.json'
image_folder = 'relative/path/to/images'  # 替换为实际的图片文件夹路径
convert_to_xtuner_format(input_file, output_file, image_folder)