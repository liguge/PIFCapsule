import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

raw_num = 2000
length1 = 2048

def process_file(file_path, output_folder):
    x = []
    data = np.load(file_path)[:, 2].reshape(-1, 1)
    overlap_length = (len(data) - length1) // (raw_num - 1)
    data = data[0:length1 + (raw_num - 1) * overlap_length].reshape(-1, 1)
    x.extend(data[j:j + length1] for j in range(0, len(data) - (length1 - 1), overlap_length))
    data = np.array(x).squeeze()
    new_filename = f"{os.path.splitext(os.path.basename(file_path))[0]}_2048.npy"
    new_file_path = os.path.join(output_folder, new_filename)
    np.save(new_file_path, data)
    print(f"Saved {new_file_path}")

def load_and_save_channel_data(folder_path, output_folder):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历文件夹及其所有子文件夹
    with ThreadPoolExecutor() as executor:
        futures = []
        for root, dirs, files in os.walk(folder_path):
            for filename in files:
                if filename.endswith('.npy'):
                    file_path = os.path.join(root, filename)
                    # 提交任务到线程池
                    futures.append(executor.submit(process_file, file_path, output_folder))
        # 等待所有任务完成
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error processing file: {e}")

# 示例用法
folder_path = 'E:\Datasets\BJTU-RAO Bogie Datasets\Data\BJTU_RAO_Bogie_Datasets\BJTU_RAO_Bogie_Datasets'  # 替换为你的文件夹路径
output_folder = 'E:\博士数据集\数据集备份\地铁数据集'  # 替换为你希望保存新文件的文件夹路径
load_and_save_channel_data(folder_path, output_folder)


