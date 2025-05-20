import re
import os
import requests
from urllib.parse import urlparse

# MD文件路径
md_file_path = '/Users/zhandaohong/vuepress/src/MMLLM/多模态模型CLIP原理与图片分类，文字搜索图像实战演练.md'

# 获取MD文件名并创建子目录
md_file_name = os.path.splitext(os.path.basename(md_file_path))[0]
image_dir = os.path.join(os.path.dirname(md_file_path), md_file_name)

# 若子目录不存在则创建
if not os.path.exists(image_dir):
    os.makedirs(image_dir)

# 读取MD文件内容
with open(md_file_path, 'r', encoding='utf-8') as f:
    md_content = f.read()

# 匹配所有HTTP图片链接
image_links = re.findall(r'!\[.*?\]\((http.*?)\)', md_content)

for link in image_links:
    try:
        # 解析URL获取文件名
        parsed_url = urlparse(link)
        file_name = os.path.basename(parsed_url.path)
        local_path = os.path.join(image_dir, file_name)

        # 下载图片
        response = requests.get(link)
        if response.status_code == 200:
            with open(local_path, 'wb') as img_file:
                img_file.write(response.content)

            # 替换MD文件中的链接
            md_content = md_content.replace(link, os.path.join(md_file_name, file_name))
        else:
            print(f'下载失败: {link}, 状态码: {response.status_code}')
    except Exception as e:
        print(f'下载失败: {link}, 错误: {e}')

# 将修改后的内容写回MD文件
with open(md_file_path, 'w', encoding='utf-8') as f:
    f.write(md_content)

print('图片下载和链接替换完成。')