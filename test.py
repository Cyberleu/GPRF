def split_file_into_lines(input_path, output_dir):
    import os

    try:
        # 打开输入文件
        with open(input_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        # 创建输出目录如果它不存在
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 逐行处理
        for line_number, line in enumerate(lines, start=1):
            filename = f"{line_number}.sql"
            output_path = os.path.join(output_dir, filename)

            # 写入每一行到新文件中
            with open(output_path, 'w', encoding='utf-8') as new_file:
                new_file.write(line.strip())  # 去除换行符

        print(f"成功将文件拆分为{len(lines)}个部分，保存在{output_dir}目录下。")

    except FileNotFoundError:
        print("错误：输入文件未找到。")
    except PermissionError:
        print("错误：缺少写入权限。")
    except Exception as e:
        print(f"发生了一个错误：{e}")

# 示例用法
input_path = "/data/homedata/lch/GPRF/data/job_test.txt"
output_dir = "/data/homedata/lch/GPRF/data/test_data"

split_file_into_lines(input_path, output_dir)
