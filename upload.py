import os
import tqdm

# 指定目录路径
directory_path = '/extra/xielab0/araujog/motion-generation/HumanML3D/re-captions_refined'

# 输出文本文件路径
output_file_path = '/extra/xielab0/araujog/motion-generation/HumanML3D/recaption_test.txt'

# train.txt 文件路径
train_file_path = '/extra/xielab0/araujog/motion-generation/HumanML3D/test.txt'

# 读取 train.txt 中的文件名
with open(train_file_path, 'r') as train_file:
    train_file_names = set(line.strip() for line in train_file)

# 获取目录下所有文件的文件名（不包含后缀名）
file_names = []
for root, dirs, files in os.walk(directory_path):
    for file in files:
        file_name_without_extension = os.path.splitext(file)[0]
        file_names.append(file_name_without_extension)

# 将不在 train.txt 中的文件名写入文本文件
with open(output_file_path, 'w') as output_file:
    for file_name in file_names:
        if file_name in train_file_names:
            output_file.write(file_name + '\n')

print(f"文件名已写入 {output_file_path}")

# cnt = 0
# train_file_path = '/extra/xielab0/araujog/motion-generation/HumanML3D/all.txt'
# for name in tqdm.tqdm(open(train_file_path, 'r').readlines()):
#     name = name.strip()
#     if name[0] == 'M':
#         break
#     try:
#         with open(f'/extra/xielab0/araujog/motion-generation/HumanML3D/texts/{name}.txt', 'r') as f1:
#             text1 = f1.read()
#         with open(f'/extra/xielab0/araujog/motion-generation/HumanML3D/texts/M{name}.txt', 'r') as f2:
#             text2 = f2.read()
#         if text1 != text2:
#             cnt+=1
#             print(name)
#     except:
#         print(name)
#         continue
# print(cnt)