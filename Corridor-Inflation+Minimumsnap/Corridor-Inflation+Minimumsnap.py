import numpy as np
import math
import heapq
import matplotlib.pyplot as plt
import cvxpy as cp 
from matplotlib.patches import Polygon as MplPolygon
from shapely.geometry import Polygon, LineString
from shapely import vectorized 
from shapely.ops import split
import time

# ------------------ A*算法 ------------------
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
            if (0 <= neighbor[0] < cols and 0 <= neighbor[1] < rows and grid[neighbor[1], neighbor[0]] == 0):
                tentative_g = g_score[current] + math.sqrt(dx**2 + dy**2)
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f, neighbor))
    return None

def create_normal_line(point1, point2, length=50):
    dx, dy = np.array(point2) - np.array(point1)
    vector = np.array([-dy, dx]) 
    norm = np.linalg.norm(vector)
    dir_vec = vector / norm if norm > 1e-6 else vector
    start = point2 + dir_vec * length / 2
    end = point2 - dir_vec * length / 2
    return LineString([(start[0], start[1]), (end[0], end[1])])

def convex_corridor(path, grid, max_width=10.0, extend=8.0):
    """
    修正版 convex_corridor，内部将 path (col,row) -> (x,y) 统一处理。
    返回的 rectangles 为 list of np.array of shape (4,2)（点为 (x,y) 顺序）。
    """
    rows, cols = grid.shape
    rectangles = []

    # 把 path 转为 (x, y) 语义：x = col, y = row
    path_xy = [ (float(p[0]), float(p[1])) for p in path ]  # (x, y)

    for i in range(len(path_xy) - 1):
        p1 = np.array(path_xy[i], dtype=float)    # (x, y)
        p2 = np.array(path_xy[i+1], dtype=float)  # (x, y)

        seg_vec = p2 - p1
        seg_len = np.linalg.norm(seg_vec)
        if seg_len < 1e-6:
            half = max_width * 0.5
            corners = np.array([
                [p1[0]-half, p1[1]-half],
                [p1[0]+half, p1[1]-half],
                [p1[0]+half, p1[1]+half],
                [p1[0]-half, p1[1]+half],
            ])
        else:
            unit = seg_vec / seg_len         # 沿线单位向量 (x,y)
            orth = np.array([-unit[1], unit[0]])   # 垂直单位向量 (x,y)

            half_w = max_width

            c1 = p1 + orth * half_w - unit * extend
            c2 = p1 - orth * half_w - unit * extend
            c3 = p2 - orth * half_w + unit * extend
            c4 = p2 + orth * half_w + unit * extend
            corners = np.vstack((c1, c2, c3, c4))

        corners[:, 0] = np.clip(corners[:, 0], 0, cols - 1)  # x in [0, cols-1]
        corners[:, 1] = np.clip(corners[:, 1], 0, rows - 1)  # y in [0, rows-1]

        rectangles.append(corners)

    return rectangles

def points_inside_polygon_mask_fast(poly, rows, cols):  
    if len(poly) < 3:
        return np.zeros((rows, cols), dtype=bool)

    xs = np.arange(cols) + 0.5
    ys = np.arange(rows) + 0.5
    XX, YY = np.meshgrid(xs, ys)

    mask = vectorized.contains(Polygon(poly), XX, YY)
    return mask

def corridor_generator_optimized(path, corridor, grid, max_width=10.0):
    path_xy = [ (float(p[0]), float(p[1])) for p in path ]
    rows, cols = grid.shape

    # 对每个初始膨胀多边形进行切割
    for i in range(len(corridor)):
        poly = np.array(corridor[i], dtype=float)
        if poly.size == 0:
            continue

        # 获取多边形顶点坐标
        min_x = int(np.floor(np.min(poly[:,0])))
        max_x = int(np.ceil (np.max(poly[:,0])))
        min_y = int(np.floor(np.min(poly[:,1])))
        max_y = int(np.ceil (np.max(poly[:,1])))

        # 计算子地图顶点坐标（后续切割在子地图中进行可以大量节省计算资源）
        min_x = int(max(0, min_x - int(max_width)))
        max_x = int(min(cols - 1, max_x + int(max_width)))
        min_y = int(max(0, min_y - int(max_width)))
        max_y = int(min(rows - 1, max_y + int(max_width)))

        # 提取子地图
        sub_grid = grid[min_y:max_y+1, min_x:max_x+1]
        delta_row, delta_col = sub_grid.shape

        # 将多边形坐标转换到子地图坐标系下
        poly_local = poly.copy()
        poly_local[:,0] -= min_x
        poly_local[:,1] -= min_y

        # 获取布尔掩码
        mask = points_inside_polygon_mask_fast(poly_local, delta_row, delta_col)

        # mask 中为 True 并且代价值为50的就是障碍物点
        obs_cells = np.argwhere(mask & (sub_grid == 50))
        if obs_cells.size == 0:
            continue
        # 计算世界坐标系下的障碍物点坐标
        obs_cells = obs_cells + np.array([min_y, min_x])

        del_array = []
        # 递归根据障碍物点切割多边形
        while obs_cells.size > 0:
            r, c = obs_cells[0]     # 障碍物点坐标
            print(f"Point ({c}, {r}) is inside rectangle {i}")

            # 路径中点
            point1 = ((path_xy[i][0] + path_xy[i+1][0]) / 2.0,
                    (path_xy[i][1] + path_xy[i+1][1]) / 2.0)
            point2 = (float(c), float(r))
            # 创建切割线
            line = create_normal_line(point1, point2, length=50)
            del_array.append((r, c))

            poly_full = Polygon(corridor[i])
            # 获取切割结果
            result = split(poly_full, line)

            # 遍历切割结果
            for m in range(len(result.geoms)):
                candi_polygon = result.geoms[m]  
                # 顶点坐标   
                coords = np.array(candi_polygon.exterior.coords[:-1])
                coords_local = coords.copy()
                # 顶点坐标转换到子地图坐标系下
                coords_local[:,0] -= min_x
                coords_local[:,1] -= min_y

                # 新的布尔掩码
                mask1 = points_inside_polygon_mask_fast(coords_local, delta_row, delta_col)

                # 路径中点在子地图中坐标
                mid_x = (path_xy[i][0] + path_xy[i+1][0]) / 2.0 - min_x
                mid_y = (path_xy[i][1] + path_xy[i+1][1]) / 2.0 - min_y
                
                # 确保路径中点在子地图中
                if not (0 <= int(np.floor(mid_y)) < delta_row and 0 <= int(np.floor(mid_x)) < delta_col):
                    continue

                # 包含路径中点的多边形是要保留的多边形
                if mask1[int(np.floor(mid_y)), int(np.floor(mid_x))]:
                    corridor[i] = coords    # 更新多边形
                    mask = mask1    # 更新布尔掩码
                    obs_cells_local = np.argwhere(mask & (sub_grid == 50))      # 更新障碍物点
                    obs_cells = obs_cells_local + np.array([min_y, min_x])
                    # 删除切割过的点
                    if del_array:
                        obs_cells = np.array([row for row in obs_cells if not any((row == np.array(del_array)).all(axis=1))])
                    break
    return corridor

def compute_angles(points):
    points = np.asarray(points, dtype=float)
    n = len(points)
    if n < 3:
        return np.array([np.pi] * n, dtype=float)
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
    points = np.asarray(points, dtype=float)
    n = len(points)
    if n <= 2:
        return points.copy()

    angles = compute_angles(points)
    theta = np.deg2rad(corner_deg)      
    corner_idx = np.where(angles >= theta)[0]
    corner_idx = np.unique(np.concatenate(([0], corner_idx, [n-1])))

    if corner_dilate > 0:
        extra = []
        for idx in corner_idx:
            for j in range(idx - corner_dilate, idx + corner_dilate + 1):
                if 0 <= j < n:
                    extra.append(j)
        corner_idx = np.unique(np.concatenate((corner_idx, np.array(extra, dtype=int))))

    out = points[corner_idx]
    return np.array(out, dtype=float)

def polygon_to_inequalities(vertices):
        n = len(vertices)
        A, b = [], []
        for i in range(n):
            p1 = vertices[i]    # 当前顶点 
            p2 = vertices[(i + 1) % n]      # 下一个顶点
            edge = p2 - p1
            normal = np.array([edge[1], -edge[0]])  # 法向量

            center = np.mean(vertices, axis=0)      # 计算所有顶点的几何中心（质心）
            # 点积 > 0，说明法向量与内部向量的夹角小于90度，即法向量指向了多边形内部（此时将法向量取反 normal = -normal，确保它始终指向外部）
            if np.dot(normal, center - p1) > 0:
                normal = -normal

            A.append(normal)
            b.append(np.dot(normal, p1))
        return np.array(A), np.array(b)

def minimum_snap_solver(corridor, grid, path, N, dim, solver="OSQP", lambda_center=2.5):
        """
        Minimumsnap问题求解函数
        输入: corridor:凸多边形膨胀走廊
            grid:栅格代价地图
            path: 路径点
            N: 路径长度
            dim: 维度数
            solver: 求解器选择
            lambda_center: 中点偏移量权重系数
        输出: 优化后的路径点
        """
        # path = [ (float(p[1]), float(p[0])) for p in path ] 
        # ---------------- 差分矩阵 ----------------
        # 构造差分矩阵 S（用于 snap，即 4 阶差分），若 N<5 用 2 阶差分矩阵（加速度最小化）
        def make_diff_matrix(order, N):
            # 差分阶数 order 大于或等于点的数量 N，我们无法计算任何差分。例如，你不能用3个点去计算4阶差分
            if order >= N:
                return np.zeros((0, N))
            S = np.zeros((N - order, N))
            coeff = np.array([1.])
            for _ in range(order):
                # 通过卷积来递归地计算差分系数
                coeff = np.convolve(coeff, np.array([1, -1]))
            # 循环 N - order 次，对应矩阵的每一行
            for i in range(N - order):
                S[i, i:i+order+1] = coeff       # 选中了第 i 行，从第 i 列到第 i+order 列（共 order+1 个元素）
            return S

        if N >= 5:
            S = make_diff_matrix(4, N)   # snap
            # X[:,0] 和 X[:,1] 分别取路径点的 x 和 y 坐标；sum_squares 用于计算平方和
            obj_expr = lambda X: cp.sum_squares(S @ X[:,0]) + cp.sum_squares(S @ X[:,1])
        elif N >= 3:
            S = make_diff_matrix(2, N)   # accel
            obj_expr = lambda X: cp.sum_squares(S @ X[:,0]) + cp.sum_squares(S @ X[:,1])
        else:
            S = make_diff_matrix(1, N)   # velocity
            obj_expr = lambda X: cp.sum_squares(S @ X[:,0]) + cp.sum_squares(S @ X[:,1])

        # ---------------- 变量 ----------------
        # N个点，二维坐标
        traj = cp.Variable((N, dim))

        constraints = []
        # 起点终点硬约束
        constraints += [traj[0, :] == path[0]]
        constraints += [traj[-1, :] == path[-1]]

        # ---------------- 走廊约束 ----------------
        # 每个轨迹点落在对应的 corridor[i] 内
        for i in range(N-1):    # corridor 数量是 N-1（每段一个多边形）
            poly = corridor[i]
            if poly is None or len(poly) < 3:
                # 限制在网格边界内
                constraints += [traj[i,0] >= 0, traj[i,0] <= grid.shape[1]-1]
                constraints += [traj[i,1] >= 0, traj[i,1] <= grid.shape[0]-1]
            else:
                A_poly, b_poly = polygon_to_inequalities(np.array(poly))
                # 对 A_poly @ p <= b_poly 添加约束（允许一点容差）
                # cvxpy 需要将每行单独加约束
                for row_idx in range(A_poly.shape[0]):
                    arow = A_poly[row_idx]
                    brow = b_poly[row_idx] + 1e-6  # 少量容差
                    # 约束在走廊内部（Ax <= b）
                    constraints += [arow[0]*traj[i,0] + arow[1]*traj[i,1] <= brow]

        # 也可以对最后一点 traj[-1] 附加最后段的走廊（保证终点可行）
        if len(corridor) >= 1:
            A_poly, b_poly = polygon_to_inequalities(np.array(corridor[-1]))
            for row_idx in range(A_poly.shape[0]):
                arow = A_poly[row_idx]
                brow = b_poly[row_idx] + 1e-6
                constraints += [arow[0]*traj[-1,0] + arow[1]*traj[-1,1] <= brow]

        # ---------------- 目标函数 ----------------
        # 计算质心坐标
        def compute_centroid(polygon):
            x_coords = [p[0] for p in polygon]      # x坐标
            y_coords = [p[1] for p in polygon]      # y坐标
            n = len(polygon)
            area = 0.0
            cx = 0.0
            cy = 0.0
            for i in range(n):
                x0, y0 = polygon[i]
                x1, y1 = polygon[(i + 1) % n]
                # |OA × OB| 的值等于由向量 OA 和 OB 构成的平行四边形的面积。这个平行四边形的面积，正好是由三角形 OAB 构成的面积的两倍。
                cross = (x0 * y1) - (x1 * y0)
                area += cross
                """
                三角形 OAB 的质心 x 坐标是 (x0 + x1 + 0) / 3
                三角形 OAB 的面积是 |cross| / 2
                所以 (质心x * 面积) 这一项正比于 ( (x0+x1)/3 * |cross|/2 )
                公式中的 (x0 + x1) * cross 是这个值的6倍(包含了符号和常数因子)。常数因子不影响最终结果
                """
                cx += (x0 + x1) * cross
                cy += (y0 + y1) * cross
            area *= 0.5
            if area == 0:       # 处理退化情况（例如所有点共线，面积为0）
                return (sum(x_coords) / n, sum(y_coords) / n)
            cx /= (6 * area)        # 质心x坐标 = (累加的x分量) / (6 * 总面积)
            cy /= (6 * area)        # 质心y坐标 = (累加的y分量) / (6 * 总面积)
            return (cx, cy)

        ### 虽然前边已经有了硬约束，但是即使落在走廊内，轨迹可能贴边走，在数值误差或走廊过窄时容易“蹭到”障碍物
        ### 因此加入下边的软约束
        # 计算每个轨迹点对应的走廊中心
        centers = []
        for i in range(len(corridor)):
            poly = corridor[i]
            if len(poly) < 3:       # 三条边以内
                center_x = (path[i][0] + path[i+1][0]) / 2
                center_y = (path[i][1] + path[i+1][1]) / 2
                centers.append((center_x, center_y))
            else:
                centroid = compute_centroid(poly)
                centers.append(centroid)

        # 之所以将最后一段走廊单独再处理一遍，是因为traj有N个点，那么centers也必须有N个值
        poly = corridor[-1]
        if len(poly) < 3:
            # --- 最后一个多边形退化 ---
            # 逻辑：取连接该走廊的两个原始路径点的中点
            # path[-2] 是倒数第二个点，path[-1] 是最后一个点（终点）
            center_x = (path[-2][0] + path[-1][0]) / 2
            center_y = (path[-2][1] + path[-1][1]) / 2
            centers.append((center_x, center_y))
        else:
            centroid = compute_centroid(poly)
            centers.append(centroid)
        centers = np.array(centers)

        # 添加中心约束到优化目标
        centers_const = cp.Parameter(shape=(N, 2))      # 代表一个在优化问题中值是已知且固定的符号变量
        centers_const.value = centers
        # traj - centers_const: 这个结果矩阵代表了每个轨迹点与其对应参考点之间的坐标差向量
        center_term = cp.sum(cp.sum_squares(traj - centers_const))

        objective = cp.Minimize(obj_expr(traj) + lambda_center*center_term)

        # ---------------- 求解 ----------------
        prob = cp.Problem(objective, constraints)
        try:
            if solver == "ECOS":
                prob.solve(solver=cp.ECOS, verbose=False)
            elif solver == "OSQP":
                prob.solve(solver=cp.OSQP, verbose=False, eps_abs=1e-3, eps_rel=1e-3, max_iter=20000)
            else:
                print(f"INVALID solver-param: {solver}")
                raise NameError("Minimum-snap FAILED, returning!!!")
        except Exception as e:
            print("Primary solver failed:", e)
            # fallback
            prob.solve(solver=cp.SCS, verbose=False)

        if traj.value is None:
            print(f"Optimization failed, traj.value is None! Returning original path!")
            path = [ (float(p[1]), float(p[0])) for p in path ] 
            return path
        return traj.value

# ------------------ 主程序 ------------------
start_time = time.time()

# 构造地图 (0=空地, 50=障碍物)
grid = np.zeros((60, 80))
grid[0:40, 15:19] = 50   # 一堵墙
grid[20:60, 30:33] = 50  # 一堵墙
grid[40:43, 40:60] = 50

start = (0, 0)  # (col, row)
goal = (79, 58)  # (col, row)
path = astar(start, goal, grid)

if path is None:
    raise RuntimeError("No path found by A*")

print(f"原始路径长度：{len(path)}")
end_time_0 = time.time()
print(f"A*算法规划路径用时：{end_time_0 - start_time:.3f}秒")

# 转换为numpy数组并简化路径
path_array = np.array(path, dtype=float)
path_simplified = simplify_path(path_array, corner_deg=30, corner_dilate=1)
print(f"简化后路径长度：{len(path_simplified)}")

### 在应用到项目中时，可以在此处检测路径是否有变化；如果没有变化就不需要重复走廊构建+Minimumsnap求解

# 构造凸多边形走廊
max_width = 7.0 
corridor = convex_corridor(path_simplified, grid, max_width, extend=8)

corridor = corridor_generator_optimized(path_simplified, corridor, grid, max_width)

end_time_1 = time.time()
print(f"构造膨胀走廊用时：{end_time_1 - end_time_0:.3f}秒")

# Minimumsnap 求解
N = len(path_simplified)
dim = 2
traj_result = minimum_snap_solver(corridor, grid, path, N, dim, solver="OSQP")
opt_traj = np.array(traj_result) 

end_time_2 = time.time()
print(f"Minimumsnap求解用时: {end_time_2 - end_time_1:.3f}秒")

# ------------------ 可视化 ------------------
fig, ax = plt.subplots(figsize=(12, 9))

rows, cols = grid.shape

# 显示地图（修正extent参数）
ax.imshow(grid, origin="lower", cmap="gray_r", alpha=0.6, extent=[0, cols, 0, rows])

# 设置坐标轴范围
ax.set_xlim(-1, cols)
ax.set_ylim(-1, rows)

# 绘制原始A*路径（黄色线）
if len(path) > 1:
    path_coords = np.array(path)
    ax.plot(path_coords[:, 0], path_coords[:, 1], 'y--', alpha=0.3, linewidth=1, label="Original A* path")

# 绘制简化后路径（红色线）
if len(path_simplified) > 1:
    ax.plot(path_simplified[:, 0], path_simplified[:, 1], 'ro--', linewidth=1, markersize=6, 
            label="Simplified path", markerfacecolor='red', markeredgecolor='darkred')
    
# 绘制动力学优化后路径（蓝色粗线）
if len(opt_traj) > 1:
    ax.plot(opt_traj[:, 0], opt_traj[:, 1], 'bo-', linewidth=2, markersize=6, 
            label="Minimumsnap path", markerfacecolor='blue', markeredgecolor='darkred')

# 绘制走廊多边形
for i, rect in enumerate(corridor):
    if rect is not None and len(rect) >= 3:
        poly_patch = MplPolygon(rect, closed=True, alpha=0.3, facecolor='orange', 
                               edgecolor='red', linewidth=1.5)
        ax.add_patch(poly_patch)

# 标记起点和终点
ax.plot(start[0], start[1], 'go', markersize=12, label='Start', markeredgecolor='darkgreen', markeredgewidth=2)
ax.plot(goal[0], goal[1], 'r*', markersize=15, label='Goal', markeredgecolor='darkred', markeredgewidth=1)

# 设置网格
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_aspect("equal")
ax.legend(loc='upper left', fontsize=10)
ax.set_xlabel('X (columns)', fontsize=12)
ax.set_ylabel('Y (rows)', fontsize=12)
plt.title("Convex Corridor Generation for Path Planning", fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()

print(f"总用时：{end_time_1 - start_time:.3f}秒")
