import os

# 创建目录用于存储生成的脚本文件
output_directory = "./"
os.makedirs(output_directory, exist_ok=True)

# 生成 24 个脚本文件
for year in range(2020, 2022):
    for month in range(1, 13):
        # 格式化日期参数
        process_date = f"{year}{month:02d}"  # 使用 f-string 格式化日期

        # 构建脚本内容
        script_content = f"""#!/bin/bash

python ../main_func.py --process_date {process_date}
"""

        # 生成脚本文件名
        script_filename = os.path.join(output_directory, f"script_{process_date}.sh")

        # 将脚本内容写入文件
        with open(script_filename, "w") as script_file:
            script_file.write(script_content)
            # 添加执行权限给脚本文件
            os.chmod(script_filename, 0o755)

print("脚本生成完成")
