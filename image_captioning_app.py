import gradio as gr
import numpy as np
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration

# 加载预训练的处理器和模型
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def caption_image(input_image: np.ndarray):
    # 将numpy数组转换为PIL图像并转换为RGB
    raw_image = Image.fromarray(input_image).convert('RGB')

    # 处理图像
    inputs = processor(raw_image, return_tensors="pt")

    # 为图像生成标题
    out = model.generate(**inputs,max_length=50)

    # 解码生成的标记为文本
    caption = processor.decode(out[0], skip_special_tokens=True)

    return caption

iface = gr.Interface(
    fn=caption_image, 
    inputs=gr.Image(), 
    outputs="text",
    title="图像标题生成",
    description="这是一个简单的网页应用，用于使用训练好的模型为图像生成标题。"
)

iface.launch()
