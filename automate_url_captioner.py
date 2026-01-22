import requests
from PIL import Image
from io import BytesIO
from bs4 import BeautifulSoup
from transformers import AutoProcessor, BlipForConditionalGeneration

# 加载预训练的处理器和模型
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# 要抓取的页面的URL
url = "https://en.wikipedia.org/wiki/IBM"

# 下载页面
response = requests.get(url)
# 使用BeautifulSoup解析页面
soup = BeautifulSoup(response.text, 'html.parser')

# 找到所有img元素
img_elements = soup.find_all('img')

# 打开文件以写入标题
with open("captions.txt", "w") as caption_file:
    # 遍历每个img元素
    for img_element in img_elements:
        img_url = img_element.get('src')

        # 如果图像是SVG或太小（可能是图标），则跳过
        if 'svg' in img_url or '1x1' in img_url:
            continue

        # 如果URL格式错误则纠正
        if img_url.startswith('//'):
            img_url = 'https:' + img_url
        elif not img_url.startswith('http://') and not img_url.startswith('https://'):
            continue  # 跳过不以http://或https://开头的URL

        try:
            # 下载图像
            response = requests.get(img_url)
            # 将图像数据转换为PIL图像
            raw_image = Image.open(BytesIO(response.content))
            if raw_image.size[0] * raw_image.size[1] < 400:  # 跳过非常小的图像
                continue

            raw_image = raw_image.convert('RGB')

            # 处理图像
            inputs = processor(raw_image, return_tensors="pt")
            # 为图像生成标题
            out = model.generate(**inputs, max_new_tokens=50)
            # 解码生成的令牌为文本
            caption = processor.decode(out[0], skip_special_tokens=True)

            # 将标题写入文件，前面加上图像URL
            caption_file.write(f"{img_url}: {caption}\n")
        except Exception as e:
            print(f"处理图像 {img_url} 时出错: {e}")
            continue
