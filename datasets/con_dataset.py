import os
import numpy as np
# target_strings = ['24014', '102841', '63237', '50415']
# target_strings = ['2108', '2105', '2110', '2102']
# target_strings = ['2116', '0807', '2118', '0803']
# target_strings = ['2125', '0811', '2127', '0814']
target_strings = ['2134', '0821', '2137', '0818']

labels = [0, 1, 2, 3]
output_file = 'E:\博士数据集\数据集备份\博士数据集\牵引电机数据集\H_350.npy'



#定义需要检查的字符串及其对应的标签
# target_strings = ['24014', '102841', '63237', '50415',    '40449', '53546', '02030',                    '75217']
# target_strings = ['24249', '103135', '63505', '50702',       '40729', '53813', '02314',                      '74935']
# target_strings = ['24406', '103341', '63617', '50826',          '40851', '53929', '02436',                        '74813']
# target_strings = ['24524', '103500', '63737', '51019',              '41013', '54042', '02554',                      '74634']
# target_strings = ['24805', '103748', '64006', '51257',                  '41311', '54320', '02833',                       '74341']

# target_strings = ['data_gearbox_M0_G0_LA0_RA0_20Hz_+10kN', 'data_gearbox_M0_G1_LA0_RA0_20Hz_+10kN', 'data_gearbox_M0_G2_LA0_RA0_20Hz_+10kN', 'data_gearbox_M0_G3_LA0_RA0_20Hz_+10kN', 'data_gearbox_M0_G4_LA0_RA0_20Hz_+10kN', 'data_gearbox_M0_G5_LA0_RA0_20Hz_+10kN', 'data_gearbox_M0_G6_LA0_RA0_20Hz_+10kN', 'data_gearbox_M0_G7_LA0_RA0_20Hz_+10kN', 'data_gearbox_M0_G8_LA0_RA0_20Hz_+10kN']
#
# target_strings = ['data_gearbox_M0_G0_LA0_RA0_40Hz_+10kN', 'data_gearbox_M0_G1_LA0_RA0_40Hz_+10kN', 'data_gearbox_M0_G2_LA0_RA0_40Hz_+10kN', 'data_gearbox_M0_G3_LA0_RA0_40Hz_+10kN', 'data_gearbox_M0_G4_LA0_RA0_40Hz_+10kN', 'data_gearbox_M0_G5_LA0_RA0_40Hz_+10kN', 'data_gearbox_M0_G6_LA0_RA0_40Hz_+10kN', 'data_gearbox_M0_G7_LA0_RA0_40Hz_+10kN', 'data_gearbox_M0_G8_LA0_RA0_40Hz_+10kN']

# target_strings = ['data_gearbox_M0_G0_LA0_RA0_60Hz_+10kN', 'data_gearbox_M0_G1_LA0_RA0_60Hz_+10kN', 'data_gearbox_M0_G2_LA0_RA0_60Hz_+10kN', 'data_gearbox_M0_G3_LA0_RA0_60Hz_+10kN', 'data_gearbox_M0_G4_LA0_RA0_60Hz_+10kN', 'data_gearbox_M0_G5_LA0_RA0_60Hz_+10kN', 'data_gearbox_M0_G6_LA0_RA0_60Hz_+10kN', 'data_gearbox_M0_G7_LA0_RA0_60Hz_+10kN', 'data_gearbox_M0_G8_LA0_RA0_60Hz_+10kN']

# target_strings=['data_gearbox_M0_G0_LA0_RA0_20Hz_0kN','data_gearbox_M0_G1_LA0_RA0_20Hz_0kN','data_gearbox_M0_G2_LA0_RA0_20Hz_0kN','data_gearbox_M0_G3_LA0_RA0_20Hz_0kN','data_gearbox_M0_G4_LA0_RA0_20Hz_0kN','data_gearbox_M0_G5_LA0_RA0_20Hz_0kN','data_gearbox_M0_G6_LA0_RA0_20Hz_0kN','data_gearbox_M0_G7_LA0_RA0_20Hz_0kN','data_gearbox_M0_G8_LA0_RA0_20Hz_0kN']

# target_strings=['data_gearbox_M0_G0_LA0_RA0_40Hz_0kN','data_gearbox_M0_G1_LA0_RA0_40Hz_0kN','data_gearbox_M0_G2_LA0_RA0_40Hz_0kN','data_gearbox_M0_G3_LA0_RA0_40Hz_0kN','data_gearbox_M0_G4_LA0_RA0_40Hz_0kN','data_gearbox_M0_G5_LA0_RA0_40Hz_0kN','data_gearbox_M0_G6_LA0_RA0_40Hz_0kN','data_gearbox_M0_G7_LA0_RA0_40Hz_0kN','data_gearbox_M0_G8_LA0_RA0_40Hz_0kN']

# target_strings=['data_gearbox_M0_G0_LA0_RA0_60Hz_0kN','data_gearbox_M0_G1_LA0_RA0_60Hz_0kN','data_gearbox_M0_G2_LA0_RA0_60Hz_0kN','data_gearbox_M0_G3_LA0_RA0_60Hz_0kN','data_gearbox_M0_G4_LA0_RA0_60Hz_0kN','data_gearbox_M0_G5_LA0_RA0_60Hz_0kN','data_gearbox_M0_G6_LA0_RA0_60Hz_0kN','data_gearbox_M0_G7_LA0_RA0_60Hz_0kN','data_gearbox_M0_G8_LA0_RA0_60Hz_0kN']


# 初始化数据和标签列表
data_list = []
label_list = []

# 遍历文件夹下的所有文件
folder_path = 'E:\博士数据集\数据集备份\博士数据集\牵引电机数据集'  # 替换为你的文件夹路径
files = sorted(os.listdir(folder_path))  # 按字母顺序排序文件名

for target_string, label in zip(target_strings, labels):
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            if target_string in filename and filename.endswith('.npy'):
                # print(filename)
                file_path = os.path.join(folder_path, filename)
                print(file_path)
                # 加载npy文件
                data = np.load(file_path)
                # 添加标签
                label_list.extend([label] * len(data))
                # 添加数据到列表
                data_list.append(data)
                break  # 找到匹配的文件后跳出内层循环，继续下一个目标字符串

# 合并数据和标签
combined_data = np.vstack(data_list)
print(combined_data.shape)
combined_labels = np.array(label_list)

# 保存为新的npy文件
combined_data1 = {
        'data': combined_data,

        'label': combined_labels
    }
np.save(output_file, combined_data1)

print("数据和标签已成功保存！")
