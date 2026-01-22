import os
import glob
import requests
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration #Blip2模型

# 加载预训练的处理器和模型
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")

# 指定您的图像目录
image_dir = "/path/to/your/images"
image_exts = ["jpg", "jpeg", "png"]  # 指定要搜索的图像文件扩展名

# 打开文件以写入图像说明
with open("captions.txt", "w") as caption_file:
    # 遍历目录中的每个图像文件
    for image_ext in image_exts:
        for img_path in glob.glob(os.path.join(image_dir, f"*.{image_ext}")):
            # 加载您的图像
            raw_image = Image.open(img_path).convert('RGB')

            # 图像说明不需要问题
            inputs = processor(raw_image, return_tensors="pt")

            # 为图像生成说明
            out = model.generate(**inputs, max_new_tokens=50)

            # 解码生成的标记为文本
            caption = processor.decode(out[0], skip_special_tokens=True)

            # 将说明写入文件，前面加上图像文件名
            caption_file.write(f"{os.path.basename(img_path)}: {caption}\n")
