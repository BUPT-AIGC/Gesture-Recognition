import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider

# 读取 Excel 文件
file_path = r"E:\HandTrack\mutli_viewpoints+hand+eye\Gesture-Recognition\hand-landmark-detection-pointrecord\recorded_data_20241113_221156.xlsx"
data = pd.read_excel(file_path)

# 获取数据的所有帧ID
frame_ids = data['frame_id'].unique()

# 找到数据中的x, y, z的最小值和最大值，用于动态设置坐标轴
x_min, x_max = data['x'].min(), data['x'].max()
y_min, y_max = data['y'].min(), data['y'].max()
z_min, z_max = data['z'].min(), data['z'].max()

# 输出xmin, xmax, ymin, ymax, zmin, zmax
print("xmin: ", x_min, ", xmax: ", x_max)
print("ymin: ", y_min, ", ymax: ", y_max)
print("zmin: ", z_min, ", zmax: ", z_max)

# 创建一个图形窗口和3D坐标轴
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# 定义连接的点对
connections = [
    (0, 1), (1, 2), (2, 3), (3, 4),   # 拇指
    (0, 5), (5, 6), (6, 7), (7, 8),   # 食指
    (0, 9), (9, 10), (10, 11), (11, 12),  # 中指
    (0, 13), (13, 14), (14, 15), (15, 16),  # 无名指
    (0, 17), (17, 18), (18, 19), (19, 20)  # 小指
]

# 定义一个函数来绘制某一帧的 3D 可视化
def plot_frame(frame_id):
    # 清除当前图像内容
    ax.cla()
    
    # 过滤当前帧的数据
    frame_data = data[data['frame_id'] == frame_id]
    
    # 获取不同的 hand_id
    hand_ids = frame_data['hand_id'].unique()

    # 定义每只手的颜色组合列表 (点的颜色, 连线的颜色)
    color_combinations = [
        ('r', 'g'),   # 手1: 红色点，绿色连线
        ('b', 'y'),   # 手2: 蓝色点，黄色连线
        # 可以根据需要添加更多手的颜色
    ]

    # 遍历每只手的数据
    for i, hand_id in enumerate(hand_ids):
        hand_data = frame_data[frame_data['hand_id'] == hand_id]
        
        # 获取关键点的x, y, z坐标
        x = hand_data['x'].values
        y = hand_data['y'].values
        z = hand_data['z'].values
        
        # 获取当前手的颜色组合
        point_color, line_color = color_combinations[i % len(color_combinations)]
        
        # 绘制手部的关键点
        ax.scatter(x, y, z, c=point_color, marker='o', label=f'Hand {hand_id}')
        
        # 连接关键点，使用不同颜色的线条
        for connection in connections:
            point1, point2 = connection
            ax.plot([x[point1], x[point2]], [y[point1], y[point2]], [z[point1], z[point2]], color=line_color, lw=2)
    
    # 设置轴标签和标题
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Frame {frame_id} - Hand Landmark 3D with Connections')
    
    # 调整视角：旋转180度使手腕在下面，四指在上面
    ax.view_init(elev=-90, azim=-90)
    
    # 设置坐标轴限制，水平镜像翻转X轴
    ax.set_xlim([x_max, x_min])  # 使用数据的最大值和最小值来设置 X 轴
    ax.set_ylim([y_min, y_max])
    ax.set_zlim([z_min, z_max])

    # 显示图例
    ax.legend()

    # 刷新图像
    plt.draw()

# 创建滑块
ax_slider = plt.axes([0.25, 0.01, 0.5, 0.03], facecolor='lightgoldenrodyellow')
frame_slider = Slider(
    ax=ax_slider,
    label='Frame',
    valmin=frame_ids.min(),
    valmax=frame_ids.max(),
    valinit=frame_ids[0],
    valstep=1  # 每次滑动的步长为1帧
)

# 当滑块的值发生变化时，更新显示的帧
def update(val):
    frame_id = int(frame_slider.val)
    plot_frame(frame_id)

# 绑定滑块更新事件
frame_slider.on_changed(update)

# 初始化显示第一帧
plot_frame(frame_ids[0])

# 显示图形窗口
plt.show()