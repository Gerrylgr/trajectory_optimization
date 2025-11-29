/**
 *  A* + (Corner-Preserving Path Simplification) + Improved Uniform B-Spline Smoothing
 *  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 *  Copyright (C) 2025 Gerry (刘耕睿)
 *  GitHub    : https://github.com/Gerrylgr
 *  Bilibili  : https://space.bilibili.com/673367025
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *
 *  Author         : 刘耕睿 (Gerry)
 *  Contact        : 2717915639@qq.com
 *  Created        : 2025-11
 *  Last Modified  : 2025-11-30
 *
 *  Description    : A* + (Corner-Preserving Path Simplification) + Improved Uniform B-Spline Smoothing
 */

#include <iostream>
#include <vector>
#include <queue>
#include <cmath>
#include <unordered_map>
#include <set>
#include <algorithm>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace Eigen;

struct Node {
    int x, y;
    /*
    f = g + h
        g：从起点到当前节点的实际代价
        h：从当前节点到终点的预估代价（启发函数值）
    */
    double f;   // 从起点经过当前节点到终点的预估总代价
    bool operator>(const Node &other) const { return f > other.f; }
    // 第一个const: 不能修改other的值
    // 第二个const: 不能修改当前结构体中成员变量的数值；常量对象也能调用这个函数
};

// 自定义哈希运算结构体（解释见最后注释）
struct PairHash {
    size_t operator()(const pair<int,int> &p) const noexcept {
        return hash<long long>()(((long long)p.first << 32) ^ (long long)p.second);
    }
};

// =====================================================
// A* 搜索算法
// =====================================================
vector<pair<int,int>> astar(pair<int,int> start, pair<int,int> goal, const vector<vector<int>> &grid) {
    int rows = grid.size();     // 外层 vector 的大小（行数）
    int cols = grid[0].size();  // 列数

    auto heuristic = [&](pair<int,int> a, pair<int,int> b) {
        // 欧几里德距离
        return sqrt((a.first - b.first)*(a.first - b.first) + (a.second - b.second)*(a.second - b.second));
    };

    // 优先队列的第一个参数是要存储的数据类型，第二个是用于存储数据的底层容器（用默认的vector就行），
    // 第三个表示创建的是大顶堆还是小顶堆，这里的greater<Node>表示创建小顶堆，即优先 pop 出最小的节点
    priority_queue<Node, vector<Node>, greater<Node>> open_set; // 存储所有已知但未被探索过的节点
    unordered_map<pair<int,int>, pair<int,int>, PairHash> came_from;
    unordered_map<pair<int,int>, double, PairHash> g_score;     // 记录了从起点到各个点的最小实际代价值
    set<pair<int,int>> visited;

    g_score[start] = 0.0;
    open_set.push({start.first, start.second, 0.0});    // 第一个要探索的就是start

    vector<pair<int,int>> dirs = {
        // 邻居节点向量
        {1,0},{-1,0},{0,1},{0,-1},{1,1},{-1,1},{-1,-1},{1,-1}
    };

    while (!open_set.empty()) {
        Node current = open_set.top();
        open_set.pop();
        pair<int,int> cur = {current.x, current.y};     // cur 表示当前节点的坐标

        if (visited.count(cur)) continue;   // 访问过则跳过
        visited.insert(cur);    // 之前没访问过，那这把就访问过了

        // 到达终点了
        if (cur == goal) {
            vector<pair<int,int>> path;     // 用于存储路径
            while (came_from.find(cur) != came_from.end()) {
                // 路径存在就回溯存储
                path.push_back(cur);
                cur = came_from[cur];
            }
            path.push_back(start);
            reverse(path.begin(), path.end());
            return path;
        }

        for (auto &d : dirs) {
            // 计算出邻居节点的坐标
            int nx = cur.first + d.first;
            int ny = cur.second + d.second;
            if (nx < 0 || nx >= cols || ny < 0 || ny >= rows) continue;     // 确保点在地图内部
            if (grid[ny][nx] != 0) continue;    // 不能走到障碍物上

            // 新邻居点的代价
            double tentative_g = g_score[cur] + sqrt(d.first*d.first + d.second*d.second);
            pair<int,int> neighbor = {nx, ny};
            // 如果这条路没有走过，或者比刚才的路代价更小，就更新
            if (!g_score.count(neighbor) || tentative_g < g_score[neighbor]) {
                g_score[neighbor] = tentative_g;
                double f = tentative_g + heuristic(neighbor, goal);
                open_set.push({nx, ny, f});
                came_from[neighbor] = cur;      // 记住来时节点
            }
        }
    }

    return {}; // 无路径
}

// =====================================================
// 计算路径上所有线段向量的夹角 (弧度)
// =====================================================
vector<double> compute_angles(const vector<Vector2d> &points) {
    int n = points.size();
    vector<double> angles(n, M_PI);
    if (n < 3) return angles;

    for (int i = 1; i < n - 1; ++i) {
        // 表示两个相邻向量
        Vector2d v1 = points[i] - points[i-1];
        Vector2d v2 = points[i+1] - points[i];
        // 计算两个相邻向量的长度
        double n1 = v1.norm();
        double n2 = v2.norm();
        if (n1 < 1e-12 || n2 < 1e-12) { angles[i] = 0.0; continue; }
        // 计算夹角
        double cos_angle = v1.dot(v2) / (n1 * n2);
        cos_angle = std::clamp(cos_angle, -1.0, 1.0);
        angles[i] = acos(cos_angle);
    }
    return angles;
}

// =====================================================
// 路径简化：保留拐角
// =====================================================
vector<Vector2d> simplify_path(const vector<Vector2d> &points, double corner_deg=30.0, int corner_dilate=1) {
    /*
    路径简化函数，输入：
        points: 路径点向量
        corner_deg: 简化路径的角度阈值
        corner_dilate: 在拐点处上采样的点数
    */
    int n = points.size();
    if (n <= 2) return points;

    vector<double> angles = compute_angles(points);
    double theta = corner_deg * M_PI / 180.0;

    vector<int> corner_idx;     // 用于保存幸存下来的点（特征点）
    for (int i = 0; i < n; ++i) {
        if (i == 0 || i == n-1 || angles[i] >= theta)
            corner_idx.push_back(i);
    }

    // 膨胀拐角索引
    if (corner_dilate > 0) {
        set<int> expanded;
        for (int idx : corner_idx) {
            for (int j = idx - corner_dilate; j <= idx + corner_dilate; ++j)
                if (j >= 0 && j < n) expanded.insert(j);    // 不越界就加入
        }
        // 用膨胀后的点集替换掉原来的点集
        corner_idx.assign(expanded.begin(), expanded.end());
    }

    vector<Vector2d> out;
    for (int idx : corner_idx)
        out.push_back(points[idx]);
    return out;
}

/**
 * 改进版B样条平滑函数
 * @param ctrl_pts      原始控制点序列
 * @param num_points    生成的平滑点数量
 * @param smooth_factor 平滑因子 (0 表示精确插值，越大越平滑)
 */
vector<Vector2d> bspline_smooth(
    const vector<Vector2d> &ctrl_pts,
    int num_points = 100,
    double smooth_factor = 0.0)
{
    int n = ctrl_pts.size();
    if (n < 4) return ctrl_pts;

    // === Step 1: 复制控制点，便于处理 ===
    vector<Vector2d> smoothed_ctrl = ctrl_pts;

    // === Step 2: 平滑控制点（模拟smoothing spline）===
    // 这里用一个简单的二阶滤波迭代，类似 ∫|C''|² 的正则化效果
    if (smooth_factor > 0.0) {
        double alpha = std::min(1.0, smooth_factor); // 防止过大
        for (int iter = 0; iter < 5; ++iter) {       // 多次迭代增强效果
            vector<Vector2d> tmp = smoothed_ctrl;
            for (int i = 1; i < n - 1; ++i) {
                /*
                下边的公式就是：
                    Pi' = (1-α) * Pi + α/2 * (P(i-1) + P(i+1))
                    Pi: 当前点坐标
                    P(i-1)和P(i+1): 邻居点坐标

                α=0 时：
                    Pi' = Pi (没有任何变化，原样保留)
                α=1 时：
                    Pi' = 1/2 * (P(i-1) + P(i+1))   (直接变为两个邻居点的中点，曲线直接被拉平了)
                0<α<1 时：
                    Pi' = (1-α) * Pi + α/2 * (邻居点均值)   （会逐渐向“平滑方向”移动）
                */
                tmp[i] = (1 - alpha) * smoothed_ctrl[i] + 0.5 * alpha * (smoothed_ctrl[i - 1] + smoothed_ctrl[i + 1]);
            }
            smoothed_ctrl.swap(tmp);
        }
    }

    // === Step 3: 分段生成B样条点 ===
    vector<Vector2d> result;
    result.reserve(num_points);     // 预先分配足够的内存空间，但并不实际创建元素

    // 三次B样条每次只依赖4个控制点（i, i+1, i+2, i+3），所以总共能生成 (n - 3) 段曲线
    for (int i = 0; i < n - 3; ++i) {
        // num_points / (n - 3) 是均匀采样；比如说200个点、5段曲线，那么就是一段曲线40个点
        for (int step = 0; step < num_points / (n - 3); ++step) {
            // 参数t，
            double t = (double)step / (num_points / (n - 3)); // [0,1]

            // 直接代入三次均匀B样条基函数公式
            double b0 = pow(1 - t, 3) / 6.0;
            double b1 = (3 * pow(t, 3) - 6 * t * t + 4) / 6.0;
            double b2 = (-3 * pow(t, 3) + 3 * t * t + 3 * t + 1) / 6.0;
            double b3 = pow(t, 3) / 6.0;

            Vector2d p = b0 * smoothed_ctrl[i] +
                         b1 * smoothed_ctrl[i + 1] +
                         b2 * smoothed_ctrl[i + 2] +
                         b3 * smoothed_ctrl[i + 3];
            result.push_back(p);
        }
    }

    result.push_back(smoothed_ctrl.back());
    return result;
}


// =================== 可视化函数 ===================
void visualize(const vector<vector<int>> &grid,
    const vector<Vector2d> &path_simplify,
    const vector<Vector2d> &path_bspline)
{
    int rows = grid.size();
    int cols = grid[0].size();
    int scale = 10; // 放大比例

    cv::Mat img(rows * scale, cols * scale, CV_8UC3, cv::Scalar(255,255,255));

    // --- 新增部分 1: 绘制网格线 ---
    // 使用浅灰色绘制网格线，让背景不那么单调
    cv::Scalar grid_color(220, 220, 220);
    for (int r = 0; r <= rows; ++r) {
        // 画横线
        cv::line(img, cv::Point(0, r * scale), cv::Point(cols * scale, r * scale), grid_color, 1);
    }
    for (int c = 0; c <= cols; ++c) {
        // 画竖线
        cv::line(img, cv::Point(c * scale, 0), cv::Point(c * scale, rows * scale), grid_color, 1);
    }

    // 绘制障碍物
    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            if (grid[y][x] == 1) {
                cv::rectangle(img,
                            cv::Rect(x*scale, (rows-1-y)*scale, scale, scale),
                            cv::Scalar(50,50,50), cv::FILLED);
            }
        }
    }

    // 绘制简化路径（蓝色）
    for (size_t i = 1; i < path_simplify.size(); ++i) {
        cv::line(img,
            cv::Point(path_simplify[i-1].x()*scale, (rows-1 - path_simplify[i-1].y())*scale),
            cv::Point(path_simplify[i].x()*scale,   (rows-1 - path_simplify[i].y())*scale),
            cv::Scalar(150,0,0), 2);
    }

    // --- 新增部分 2: 绘制简化路径的点 ---
    // 用深蓝色小圆点标记简化路径的节点
    for (const auto& pt : path_simplify) {
        cv::circle(img, 
                   cv::Point(pt.x()*scale, (rows-1-pt.y())*scale), 
                   3, // 半径
                   cv::Scalar(255, 0, 0), // 深蓝色
                   cv::FILLED);
    }

    // 绘制B样条路径（红色）
    for (size_t i = 1; i < path_bspline.size(); ++i) {
        cv::line(img,
            cv::Point(path_bspline[i-1].x()*scale, (rows-1 - path_bspline[i-1].y())*scale),
            cv::Point(path_bspline[i].x()*scale,   (rows-1 - path_bspline[i].y())*scale),
            cv::Scalar(0,0,150), 2);
    }
    
    // 绘制B样条路径的点
    for (const auto& pt : path_bspline) {
        cv::circle(img, cv::Point(pt.x()*scale, (rows-1-pt.y())*scale), 3, cv::Scalar(0, 0, 255), cv::FILLED);
    }


    cv::imshow("Path Visualization", img);
    cv::waitKey(0);
}

// =====================================================
// 主函数
// =====================================================
int main() {
    const int rows = 60, cols = 80;
    vector<vector<int>> grid(rows, vector<int>(cols, 0));

    // 墙体
    for (int y = 0; y < 40; ++y)
        for (int x = 15; x < 19; ++x)
            grid[y][x] = 1;
    for (int y = 20; y < 60; ++y)
        for (int x = 30; x < 33; ++x)
            grid[y][x] = 1;
    for (int y = 40; y < 43; ++y)
        for (int x = 40; x < 60; ++x)
            grid[y][x] = 1;

    pair<int,int> start = {0,0}, goal = {79,58};
    auto path_int = astar(start, goal, grid);
    if (path_int.empty()) {
        cerr << "No path found by A*" << endl;
        return -1;
    }

    vector<Vector2d> path;
    for (auto &p : path_int)
        path.emplace_back(p.first, p.second);

    auto simplified = simplify_path(path, 30.0, 1);
    auto bspline = bspline_smooth(simplified, (int)(simplified.size() * 1.5), 0.15);

    visualize(grid, simplified, bspline);

    return 0;
}


/*
unordered_map 使用：
    template<
        class Key,                                    // 1. 键的类型
        class T,                                      // 2. 值的类型
        class Hash = std::hash<Key>,                  // 3. 哈希函数
        class KeyEqual = std::equal_to<Key>,          // 4. 键比较函数
        class Allocator = std::allocator<std::pair<const Key, T>> // 5. 分配器 (很少见)
    > class unordered_map;

    所以其模板一般最多接受4个参数（最后一个用不上），其中：
        1. Key (键)
        2. T (值)：键对应存储的数据
        3. Hash (哈希函数)：把 Key 转换成一个叫做“哈希值”的整数。unordered_map 内部通过这个哈希值来快速定位数据应该存放在哪个“桶”里。
        4. KeyEqual (键比较函数)：当两个不同的键通过哈希函数计算出了相同的哈希值（“哈希冲突”），unordered_map 就需要用这个比较函数来判断这两个键是不是真的相等。     
*/
/*
对于 PairHash：
    该结构体是一个自定义的哈希函数，专门用来计算 std::pair<int, int> 类型的哈希值（unordered_map 没有预设 pair 的哈希计算函数）：
        (long long)p.first << 32：将 pair 的第一个 int 类型数据转换为 long long（64位长整型） 并向高位移动32位
        ^ (long long)p.second：
                ^ 是按位异或（XOR）运算符（如果两个对应位的值不同，结果为1；相同则为0）
                则这一句将 pair 的第二个 int 数据“嵌入” (long long)p.first 中（将二者合并了）
        hash<long long>()(...)：将前边得到的组合数转换为哈希值
*/