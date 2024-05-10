import pstats
import pandas as pd

# 加载性能数据

p = pstats.Stats('deepcnn_profile.prof')
data = []
for func, (cc, nc, tt, ct, callers) in sorted(p.stats.items()):
    filename, line, func_name = func
    data.append([func_name, tt])

# 创建 DataFrame
columns = ['Function Name', 'Total Time']
df = pd.DataFrame(data, columns=columns)
import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np
plt.rcParams.update({'font.size': 14})  # 可以根据需要调整大小clear
# 设置图形尺寸和行数，取前10个最耗时的函数
df_sorted = df.sort_values('Total Time', ascending=False).head(5)
fig, axes = plt.subplots(nrows=len(df_sorted), figsize=(8, 4 * len(df_sorted)))

# 如果只有一个函数，确保 axes 是数组
if len(df_sorted) == 1:
    axes = [axes]

for ax, (index, row) in zip(axes, df_sorted.iterrows()):
    # 模拟一个高斯分布的样本数据
    data = np.random.normal(loc=row['Total Time'], scale=0.1 * row['Total Time'], size=1000)

    # 拟合高斯分布
    mu, std = norm.fit(data)

    # 绘制直方图
    ax.hist(data, bins=30, density=True, alpha=0.6, color='g')

    # 绘制拟合的高斯分布
    xmin, xmax = ax.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    ax.plot(x, p, 'k', linewidth=2)
    title = f"Fit results for {row['Function Name']}: μ = {mu:.2f}, σ = {std:.2f}"
    ax.set_title(title)

plt.tight_layout()
plt.show()
