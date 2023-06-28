import os

current_dir = os.getcwd()  # 获取当前文件夹路径
parent_dir = os.path.dirname(current_dir)  # 获取父级文件夹路径

print(parent_dir)
