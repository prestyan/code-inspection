import pstats
import pandas as pd

# 加载性能数据
p = pstats.Stats('deepcnn_profile.prof')

# 提取数据
data = []
for func, (cc, nc, tt, ct, callers) in sorted(p.stats.items()):
    filename, line, func_name = func
    data.append([func_name, filename, line, nc, tt, ct])

# 创建 DataFrame
columns = ['Function Name', 'File', 'Line', 'Calls', 'Total Time', 'Cumulative Time']
df = pd.DataFrame(data, columns=columns)

# 按累积时间和调用次数排序
df_sorted = df.sort_values(by=['Cumulative Time', 'Calls'], ascending=[False, False])
import matplotlib.pyplot as plt

# 取累积时间最高的10个函数
top_functions = df_sorted.head(10)

# 创建一个图和两个子图
fig, ax1 = plt.subplots(figsize=(12, 8))

# 绘制条形图
color = 'tab:blue'
ax1.set_xlabel('Function Name')
ax1.set_ylabel('Cumulative Time (seconds)', color=color)
ax1.bar(top_functions['Function Name'], top_functions['Cumulative Time'], color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_xticklabels(top_functions['Function Name'], rotation=45, ha="right")

# 创建第二个y轴用来显示调用次数
ax2 = ax1.twinx()  # 实例化一个新的y轴
color = 'tab:red'
ax2.set_ylabel('Calls', color=color)  # 我们已经处理过标签
ax2.plot(top_functions['Function Name'], top_functions['Calls'], color=color, marker='o', linestyle='None')
ax2.tick_params(axis='y', labelcolor=color)

# 添加标题
plt.title('Top 10 Functions by Cumulative Time and Calls')

# 显示图表
plt.show()
