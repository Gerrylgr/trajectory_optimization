import numpy as np
import math
import heapq
import matplotlib.pyplot as plt
import time
from scipy.interpolate import splprep, splev

# ------------------  ------------------
def astar(start, goal, grid):
    rows, cols = grid.shape
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}

    def heuristic(a, b):
        return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

    visited = set()
    while open_set:
        _, current = heapq.heappop(open_set)
        if current in visited:
            continue
        visited.add(current)

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]

        for dx, dy in [(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,1),(-1,-1),(1,-1)]:
            neighbor = (current[0]+dx, current[1]+dy)
            # 访问 grid 时必须 [y, x]，这是因为 Numpy 的索引顺序
            if (0 <= neighbor[0] < cols and 0 <= neighbor[1] < rows and grid[neighbor[1], neighbor[0]] == 0):
                tentative_g = g_score[current] + math.sqrt(dx**2 + dy**2)
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f, neighbor))
    return None

def compute_angles(points):
    """计算每个点的转角(弧度), 直行≈0"""
    points = np.asarray(points, dtype=float)
    n = len(points)
    if n < 3:
        return np.array([np.pi] * n, dtype=float)  # 退化情况，全部设为π，也就是全部保留
    angles = np.empty(n, dtype=float)
    angles[0] = np.pi
    angles[-1] = np.pi
    for i in range(1, n - 1):
        v1 = points[i]   - points[i - 1]
        v2 = points[i+1] - points[i]
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 1e-12 or n2 < 1e-12:
            angles[i] = 0.0
            continue
        cos_angle = np.dot(v1, v2) / (n1 * n2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angles[i] = np.arccos(cos_angle)
    return angles

def simplify_path(points, corner_deg=30, corner_dilate=1):
    """
    改进版简化：主要压缩直线段，尽量保留拐角
    - corner_deg: 认为是“拐角”的最小转角（度），推荐 20~45
    - corner_dilate: 对拐角索引做 ±k 膨胀，避免走廊在拐角断开
    """
    points = np.asarray(points, dtype=float)
    n = len(points)
    if n <= 2:
        return points.copy()

    # ---- 1) 角度阈值选拐角（注意比较方向） ----
    angles = compute_angles(points)
    theta = np.deg2rad(corner_deg)      
    corner_idx = np.where(angles >= theta)[0]     # 取出拐点索引
    # np.concatenate将[0], corner_idx, [n-1]拼接成一个大数组；np.unique 会去重并排序
    corner_idx = np.unique(np.concatenate(([0], corner_idx, [n-1])))

    # ---- 2) 膨胀拐角索引（可减少走廊缝隙） ----
    if corner_dilate > 0:
        extra = []
        for idx in corner_idx:
            for j in range(idx - corner_dilate, idx + corner_dilate + 1):
                if 0 <= j < n:
                    extra.append(j)
        corner_idx = np.unique(np.concatenate((corner_idx, np.array(extra, dtype=int))))

    out = points[corner_idx]
    return np.array(out, dtype=float)

start_time = time.time()

def bsline_smoothing_improved(points, num_points=200, smooth_factor=0.5):
    """
    改进的B样条平滑, 使用弧长参数化和平滑因子
    
    Args:
        points: 路径点集
        num_points: 输出点的数量
        smooth_factor: 平滑因子。0表示精确插值(穿过所有点)，值越大曲线越平滑。
    """
    points = np.array(points, dtype=float)
    if len(points) < 4:
        return points

    # splprep 需要点的坐标是 (M, N) 形状，M是维度(x,y)，N是点数
    # 所以需要转置一下
    points_t = points.T  

    # 使用 splprep 创建B样条表示
    # s 就是平滑因子 k=3 表示三次样条
    """
    splprep 内部在求解时，最小化以下目标函数：
        J = Σ |P_i - C(u_i)|² + s * ∫ |C''(t)|² dt
    其中，    
        Σ |P_i - C(u_i)|² (逼近项):
            P_i 是你的第 i 个输入点。
            C(u_i) 是在参数 u_i 处, B样条曲线上的点。
            这一项计算的是所有输入点与曲线之间的距离平方和。
            这个值越小，曲线越贴近原始点。

        ∫ |C''(t)|² dt (平滑项):
            C''(t) 是曲线的二阶导数，代表曲率（可以理解为“弯曲程度”）。
            这一项计算的是整条曲线的弯曲能量总和。
            这个值越小，曲线越平直、越光滑。

        s (平滑因子):
            即 smooth_factor, 它是一个权重系数, 用来平衡上述两个目标。
            s = 0: 优化目标只剩下逼近项。算法会不惜一切代价让曲线穿过所有点，结果就是精确插值，曲线可能非常崎岖。
            s > 0: 算法需要在“贴近点”和“保持平滑”之间做取舍。s 越大，“保持平滑”的权重越高，算法越倾向于生成一条光滑的曲线，即使这意味着它会离某些原始点远一点。
            s → ∞：平滑项的权重趋于无限大，算法会极力最小化曲率，最终结果可能是一条直线。
    """
    """
    tck: 
        t - 节点向量:
            是一个非递减序列,两端通常有 k+1 个重复的节点（例如，对于三次样条 k=3,开头和结尾各有4个相同的节点),
            这保证了曲线会穿过第一个和最后一个控制点。
        c - 控制点
        k - 阶数
    """
    tck, u = splprep(points_t, s=smooth_factor, k=3)

    # 在新的参数点上评估样条
    # np.linspace(0, 1, num_points) 生成均匀分布的参数
    u_new = np.linspace(0, 1, num_points)
    x_new, y_new = splev(u_new, tck)

    return np.column_stack((x_new, y_new))

# ------------------ 构造地图 (0=空地, 1=障碍物) ------------------
grid = np.zeros((60, 80))
grid[0:40, 15:19] = 1   # 一堵墙
grid[20:60, 30:33] = 1  # 一堵墙
grid[40:43, 40:60] = 1

start = (0, 0)
goal = (79, 58)
path = astar(start, goal, grid)
print(path)
if path is None:
    raise RuntimeError("No path found by A*")

print(f"路径长度：{len(path)}")
end_time_0 = time.time()
print(f"A*算法规划路径用时：{end_time_0 - start_time}")

path_simplify = simplify_path(path, corner_deg=30, corner_dilate=1)  
print(f"LENGTH of path_simplify:{len(path_simplify)}") 

path_bsline = bsline_smoothing_improved(path_simplify, round(len(path_simplify) * 1.5), smooth_factor=0.5)
print(f"LENGTH of path_bsline:{len(path_bsline)}") 

if path_bsline is None:
    raise RuntimeError("BSline Path is None!!")

path = np.array(path)
path_bsline = np.array(path_bsline)
path_simplify = np.array(path_simplify)

# ------------------ 可视化 ------------------
fig, ax = plt.subplots(figsize=(8,8))

rows, cols = grid.shape
ax.imshow(grid, origin="lower", cmap="gray_r", alpha=0.6, extent=[0, cols, 0, rows])

# 网格线
ax.set_xticks(np.arange(0, cols+1, 1))
ax.set_yticks(np.arange(0, rows+1, 1))
ax.grid(color="lightgray", linestyle="-", linewidth=0.4)
ax.set_xlim(0, cols)
ax.set_ylim(0, rows)

# ax.plot(path[:,0], path[:,1], 'b-o', label="A* path", markersize=4)
ax.plot(path_simplify[:,0], path_simplify[:,1], 'b-o', label="Simplified path", markersize=4)
ax.plot(path_bsline[:,0], path_bsline[:,1], 'r-o', label="BSline path", markersize=4)

ax.set_aspect("equal")
ax.legend()
plt.title("PATH")
plt.show()