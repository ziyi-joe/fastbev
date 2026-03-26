import cv2
import os
import glob
import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('path')
    parser.add_argument('--start', type=int, default=None, help='起始图片编号')
    parser.add_argument('--end', type=int, default=None, help='结束图片编号')
    args = parser.parse_args()
    return args

def images_to_mp4_cv2(image_folder, fps=24, start=None, end=None):
    """
    将文件夹中的JPG图片转换为MP4视频

    参数:
    image_folder: 包含JPG图片的文件夹路径
    output_path: 输出MP4文件的路径
    fps: 帧率，默认24帧/秒
    start: 起始图片编号（包含）
    end: 结束图片编号（包含）
    """
    # 获取所有jpg文件并按文件名排序
    output_path = f"{image_folder}/../pred.mp4"
    images = glob.glob(os.path.join(image_folder, "*.jpg"))
    images = sorted(images, key=lambda x: int(x.split('/')[-1].split('.')[0]))

    # 过滤指定范围的图片
    if start is not None or end is not None:
        filtered_images = []
        for img_path in images:
            num = int(img_path.split('/')[-1].split('.')[0])
            if start is not None and num < start:
                continue
            if end is not None and num > end:
                continue
            filtered_images.append(img_path)
        images = filtered_images
    
    if not images:
        print("未找到JPG图片！")
        return
    
    # 读取第一张图片获取尺寸
    first_image = cv2.imread(images[0])
    height, width, layers = first_image.shape
    size = (width, height)
    
    # 定义视频编码器和输出
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4编码
    out = cv2.VideoWriter(output_path, fourcc, fps, size)
    
    print(f"开始处理 {len(images)} 张图片...")
    
    for i, image_path in enumerate(images):
        img = cv2.imread(image_path)
        out.write(img)
        
        # 显示进度
        if (i + 1) % 50 == 0:
            print(f"已处理 {i + 1}/{len(images)} 张图片")
    
    out.release()
    print(f"视频已保存至: {output_path}")

# 使用示例
if __name__ == "__main__":
    args = parse_args()
    images_to_mp4_cv2(args.path, fps=6, start=args.start, end=args.end)