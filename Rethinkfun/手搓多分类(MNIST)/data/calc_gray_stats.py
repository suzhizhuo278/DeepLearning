import pandas as pd
import glob


def calc_gray_stats(file_path):
    """计算一个CSV文件中所有灰度像素的平均值和标准差"""
    print(f"\n正在处理文件：{file_path}")
    df = pd.read_csv(file_path)

    # 如果存在标签列，则去掉
    if 'label' in df.columns:
        df = df.drop(columns=['label'])

    # 计算均值和标准差
    mean_val = df.values.mean()
    std_val = df.values.std()

    print(f"平均灰度值 (mean): {mean_val:.4f}")
    print(f"灰度标准差 (std): {std_val:.4f}")
    return mean_val, std_val


if __name__ == "__main__":
    # ====== ✅ 方法 1：单个文件 ======
    # file_path = "mnist_test.csv"
    # calc_gray_stats(file_path)

    # ====== ✅ 方法 2：批量计算 ======
    # 匹配当前目录下所有CSV文件
    for file_path in glob.glob("*.csv"):
        calc_gray_stats(file_path)
