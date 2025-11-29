
#include <bits/stdc++.h>
#include <Eigen/Dense>
#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point.hpp>
#include <boost/geometry/geometries/polygon.hpp>
#include <boost/geometry/geometries/linestring.hpp>
#include <boost/geometry/algorithms/intersects.hpp>
#include <boost/geometry/algorithms/within.hpp>
#include <boost/geometry/algorithms/area.hpp>
#include <boost/geometry/algorithms/centroid.hpp>
#include <boost/geometry/algorithms/intersection.hpp>
#include <boost/geometry/algorithms/difference.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>

// OSQP-Eigen headers (需要事先安装并能被 CMake 找到)
#include <OsqpEigen/OsqpEigen.h>

using namespace std;
using Eigen::VectorXd;
using Eigen::MatrixXd;

namespace bg = boost::geometry;
using Point = bg::model::d2::point_xy<double>;
using Polygon = bg::model::polygon<Point>;      // 封闭的多边形区域
using Linestring = bg::model::linestring<Point>;

// ------------------ A* ------------------
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

// 自定义哈希运算结构体
struct PairHash {
    size_t operator()(const pair<int,int> &p) const noexcept {
        return hash<long long>()(((long long)p.first << 32) ^ (long long)p.second);
    }
};

// A* 搜索算法
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

// ------------------ 几何工具 ------------------

// 点是否在多边形内部（包括边界）
bool point_in_polygon(const Polygon &poly, double x, double y){
    Point p(x,y);
    // boost::geometry::within 不把边界视为内部，所以使用 intersects/covered_by as alternative
    // 使用 bg::within 或 bg::covered_by，可根据需要调节
    return bg::covered_by(p, poly);
    // return bg::within(p, poly);
}

// 计算多边形质心（如果退化，降级为顶点平均）
pair<double,double> polygon_centroid(const Polygon &poly){
    if(bg::area(poly) == 0)
    {
        double sx=0, sy=0; int cnt=0;
        for(auto const &pt: poly.outer())
        {
            sx += bg::get<0>(pt);
            sy += bg::get<1>(pt);
            cnt++;
        }
        if(cnt==0) return {0,0};
        return {sx/cnt, sy/cnt};
    }
    Point c; 
    bg::centroid(poly, c);
    return {bg::get<0>(c), bg::get<1>(c)};
}

// 重载版本：接受Linestring作为切割线
vector<Polygon> split_polygon_by_line(const Polygon &poly, const Linestring &cut_line){
    // 使用 Boost.Geometry 的 intersection 来模拟分割
    // 注意：Boost.Geometry 没有直接的 split 函数，我们需要用 intersection + difference 来模拟
    
    // 方法：将多边形与切割线相交，然后基于交点进行分割
    // 这里使用一个简化的方法：构造一个很窄的多边形条带
    
    // 获取切割线的两个端点
    if (cut_line.size() < 2) return {poly};
    
    Point p1 = cut_line[0];
    Point p2 = cut_line[1];
    
    // 计算切割线方向和法向量
    double dx = bg::get<0>(p2) - bg::get<0>(p1);
    double dy = bg::get<1>(p2) - bg::get<1>(p1);
    double len = hypot(dx, dy);
    
    if (len < 1e-9) return {poly};
    
    // 构造一个窄的多边形条带（用于分割）
    double half_width = 1e-3;  // 很窄的条带
    double ux = dx / len;
    double uy = dy / len;
    double ox = -uy;  // 法向量
    double oy = ux;
    
    Polygon band;
    vector<Point> outer;
    double L = len;  // 使用切割线本身的长度
    
    outer.emplace_back(bg::get<0>(p1) + ox*half_width, bg::get<1>(p1) + oy*half_width);
    outer.emplace_back(bg::get<0>(p2) + ox*half_width, bg::get<1>(p2) + oy*half_width);
    outer.emplace_back(bg::get<0>(p2) - ox*half_width, bg::get<1>(p2) - oy*half_width);
    outer.emplace_back(bg::get<0>(p1) - ox*half_width, bg::get<1>(p1) - oy*half_width);
    outer.emplace_back(outer.front());
    
    bg::assign_points(band, outer);
    bg::correct(band);
    
    // 使用 difference 来分割多边形
    vector<Polygon> result;
    try {
        bg::difference(poly, band, result);
    } catch (...) {
        return {poly};
    }
    
    // 过滤并返回面积非零的子多边形
    vector<Polygon> res;
    for(auto &pp: result){
        if(fabs(bg::area(pp)) > 1e-9) res.push_back(pp);
    }
    
    if(res.empty()) res.push_back(poly);
    return res;
}

// 将多边形用直线切割（line 由两个点给出），返回切割后的多边形集合
vector<Polygon> split_polygon_by_line(const Polygon &poly, const pair<double,double>& p1, const pair<double,double>& p2){
    // 用 Boost.Geometry 做 intersection: 将切割直线转换为长线段（足够长以保证穿透）
    Linestring cut;
    // line 长度设置较大，跨越可能的地图区域
    double dx = p2.first - p1.first;
    double dy = p2.second - p1.second;
    double len = hypot(dx,dy);      // 计算两个切割点距离
    if(len < 1e-9) 
        return {poly};
    double ux = dx / len, uy = dy / len;    // 切割线的单位向量
    double L = 1e4; // 很长的线段
    // 向两端延伸，构造切割线
    cut.push_back(Point(p1.first - ux*L, p1.second - uy*L));
    cut.push_back(Point(p1.first + ux*L, p1.second + uy*L));

    // Boost.Geometry没有直接split，下面用 difference/union近似
    // 方法：将 polygon 分成两部分，方法是构造一条窄的多边形条带（宽度 very small），再对 poly 做差集。
    // 这里使用一个简单且常用的策略：把切割线缓冲成窄带，并求差集的两个连通区域

    // 1) 将切割线缓冲成窄带（一个很薄的长矩形）
    double half_w = 1e-3; // 极薄
    // 计算 orth
    double ox = -uy, oy = ux;
    Polygon band;
    vector<Point> outer;
    // 计算长薄矩形的顶点坐标
    outer.emplace_back(p1.first - ux*L + ox*half_w, p1.second - uy*L + oy*half_w);
    outer.emplace_back(p1.first + ux*L + ox*half_w, p1.second + uy*L + oy*half_w);
    outer.emplace_back(p1.first + ux*L - ox*half_w, p1.second + uy*L - oy*half_w);
    outer.emplace_back(p1.first - ux*L - ox*half_w, p1.second - uy*L - oy*half_w);
    outer.emplace_back(outer.front());
    bg::assign_points(band, outer);
    bg::correct(band);

    // 2) boost::geometry::difference 会求几何体的差集
    vector<Polygon> result1;
    try{
        bg::difference(poly, band, result1);
    } catch(...) {
        // 若 difference 失败，直接返回原多边形
        return {poly};
    }
    // result1 里包含若干多边形连通分量，理论上两边的部分都会在 result1 中
    // 过滤并返回面积非零的子多边形
    vector<Polygon> res;
    for(auto &pp: result1){
        if(fabs(bg::area(pp)) > 1e-9) 
            res.push_back(pp);
    }
    if(res.empty()) 
        res.push_back(poly);
    return res;
}

// ------------------ Convex corridor ------------------
vector<array<double,8>> convex_corridor(const vector<pair<double,double>>& path_xy, int rows, int cols, double max_width=10.0, double extend=8.0){
    // 返回每段的矩形4角 (x,y) 顺序: 4点 -> 8个double
    vector<array<double,8>> rects;
    for(size_t i=0; i+1 < path_xy.size(); ++i)
    {
        Eigen::Vector2d p1(path_xy[i].first, path_xy[i].second);
        Eigen::Vector2d p2(path_xy[i+1].first, path_xy[i+1].second);
        Eigen::Vector2d seg = p2 - p1;
        double seg_len = seg.norm();    // 两点间距离
        array<double,8> corners;
        if(seg_len < 1e-6)
        {
            // 路径太短的话，膨胀半径减半
            double half = max_width*0.5;
            corners = {p1.x()-half, p1.y()-half, p1.x()+half, p1.y()-half, p1.x()+half, p1.y()+half, p1.x()-half, p1.y()+half};
        } 
        else 
        {
            Eigen::Vector2d unit = seg/seg_len;     // 沿路径方向的单位向量
            auto orth_unit = Eigen::Vector2d(-unit.y(), unit.x());      // 获取法向量（orthogonal [ɔr'θɑgənəl]）
            double half_w = max_width;
            // 计算膨胀后矩形的顶点坐标
            Eigen::Vector2d c1 = p1 + orth_unit*half_w - unit*extend;
            Eigen::Vector2d c2 = p1 - orth_unit*half_w - unit*extend;
            Eigen::Vector2d c3 = p2 - orth_unit*half_w + unit*extend;
            Eigen::Vector2d c4 = p2 + orth_unit*half_w + unit*extend;
            // 防止越界
            auto clampx = [&](double x){ return min<double>(max<double>(x, 0.0), cols-1); };
            auto clampy = [&](double y){ return min<double>(max<double>(y, 0.0), rows-1); };
            corners = {clampx(c1.x()), clampy(c1.y()), clampx(c2.x()), clampy(c2.y()), clampx(c3.x()), clampy(c3.y()), clampx(c4.x()), clampy(c4.y())};
        }
        rects.push_back(corners);
    }
    return rects;
}

// 将 C++ 的 rect -> Polygon
Polygon rect_to_polygon(const array<double,8>& rect){
    Polygon poly;
    vector<Point> pts;
    pts.emplace_back(rect[0], rect[1]);
    pts.emplace_back(rect[2], rect[3]);
    pts.emplace_back(rect[4], rect[5]);
    pts.emplace_back(rect[6], rect[7]);
    pts.emplace_back(pts.front());
    bg::assign_points(poly, pts);   // 用点为多边形赋值
    bg::correct(poly);      // 检查并修复多边形使其符合boost::geometry 的内部规范，防止后续计算出错
    return poly;
}

// 遍历子地图中所有障碍物点，返回在多边形内部的障碍物坐标点集
vector<pair<int,int>> polygon_obstacles_in_subgrid(const Polygon &poly, const vector<vector<int>>& subgrid, int min_x, int min_y){
    int rows = (int)subgrid.size();
    int cols = (int)subgrid[0].size();
    vector<pair<int,int>> obs;
    for(int r = 0; r < rows; ++r){
        for(int c = 0; c < cols; ++c){
            if(subgrid[r][c] != 50) 
                continue;
            // 将栅格的角点坐标转换为中心点坐标，这样点在多边形内的判断更准确
            double x = c + 0.5 + min_x;
            double y = r + 0.5 + min_y;
            if(point_in_polygon(poly, x, y)) 
                obs.emplace_back(r + min_y, c + min_x);
        }
    }
    return obs;
}

// 创建法线切割线（对应Python的create_normal_line）
Linestring create_normal_line(const pair<double,double>& point1, 
                              const pair<double,double>& point2, 
                              double length = 50.0) {
    // 计算从point1到point2的向量
    double dx = point2.first - point1.first;
    double dy = point2.second - point1.second;
    
    // 旋转90度得到法向量 (-dy, dx)
    double nx = -dy;
    double ny = dx;
    
    // 归一化
    double norm = sqrt(nx*nx + ny*ny);
    if (norm < 1e-6) {
        // 如果两点重合，返回默认线段
        return Linestring{Point(point2.first - 25, point2.second), 
                          Point(point2.first + 25, point2.second)};
    }
    nx /= norm;
    ny /= norm;
    
    // 构造足够长的切割线段
    double half_len = length / 2.0;
    Point start(point2.first + nx * half_len, point2.second + ny * half_len);
    Point end(point2.first - nx * half_len, point2.second - ny * half_len);
    
    return Linestring{start, end};
}

// corridor generator (optimized) 
vector<Polygon> corridor_generator_optimized(const vector<pair<double,double>>& path_xy, vector<array<double,8>>& corridor_rects, const vector<vector<int>>& grid, double max_width){
    int rows = (int)grid.size();
    int cols = (int)grid[0].size();
    vector<Polygon> corridor;
    for(size_t i=0; i < corridor_rects.size(); ++i){
        Polygon poly = rect_to_polygon(corridor_rects[i]);
        corridor.push_back(poly);
    }

    for(size_t i=0; i < corridor.size(); ++i){
        Polygon current = corridor[i];
        // compute bbox
        double minx, miny, maxx, maxy;
        bg::model::box<Point> box;
        // bg::envelope() 会找到能够完全包围给定几何图形的最小矩形区域，这个矩形的边与坐标轴平行（此处将计算结果存入box）
        bg::envelope(current, box);
        // bg::min_corner 表示最小角点（左下角），bg::max_corner 表示最大角点（右上角）
        // 0 表示 x 坐标，1 表示 y 坐标，（如果是3D几何，2 表示 z 坐标）
        minx = bg::get<bg::min_corner,0>(box);
        miny = bg::get<bg::min_corner,1>(box);
        maxx = bg::get<bg::max_corner,0>(box);
        maxy = bg::get<bg::max_corner,1>(box);

        // 计算子地图顶点坐标（后续切割在子地图中进行可以大量节省计算资源）
        int ix0 = max(0, int(floor(minx - max_width)));
        int iy0 = max(0, int(floor(miny - max_width)));
        int ix1 = min(cols-1, int(ceil (maxx + max_width)));
        int iy1 = min(rows-1, int(ceil (maxy + max_width)));

        // 提取子地图
        vector<vector<int>> subgrid(iy1 - iy0 + 1, vector<int>(ix1 - ix0 + 1));
        for(int r = iy0; r <= iy1; ++r) 
        {
            for(int c = ix0; c <= ix1; ++c) 
                subgrid[r-iy0][c-ix0] = grid[r][c];
        }

        // 获取在多边形内部的障碍物坐标集
        auto obs = polygon_obstacles_in_subgrid(current, subgrid, ix0, iy0);
        if(obs.empty()) 
            continue;

        // 分割循环：直到没有障碍物落在当前保留多边形中
        vector<pair<int,int>> obs_cells = obs;
        vector<pair<int,int>> del_array;
        // 当（当前）多边形内部还有障碍物时
        while(!obs_cells.empty()){
            auto rc = obs_cells.front();
            int r = rc.first, c = rc.second;
            cout << "Point ("<< c << ", "<< r <<") is inside rectangle " << i << "\n";

            // 获取路径中点坐标
            double midx = (path_xy[i].first + path_xy[i+1].first)/2.0;
            double midy = (path_xy[i].second + path_xy[i+1].second)/2.0;
            pair<double,double> point1 = {midx, midy};
            pair<double,double> point2 = {double(c), double(r)};  

            // 计算从路径中点到障碍物的向量
            double dx = point2.first - point1.first;
            double dy = point2.second - point1.second;

            // 旋转90度得到法向量
            double nx = -dy;
            double ny = dx;

            // 归一化法向量
            double norm = sqrt(nx*nx + ny*ny);
            if (norm > 1e-6) {
                nx /= norm;
                ny /= norm;
            }

            // 以障碍物点为基准，沿法向量方向取距离d的两个点
            double d = 25.0;  // 半长度，总长度为2d
            pair<double,double> normal_point1 = {
                point2.first + nx * d,    // 以障碍物点为基准
                point2.second + ny * d
            };
            pair<double,double> normal_point2 = {
                point2.first - nx * d,    // 以障碍物点为基准
                point2.second - ny * d
            };

            // 创建法线切割线
            // auto cutting_line = create_normal_line(point1, point2, 50.0);

            auto pieces = split_polygon_by_line(current, normal_point1, normal_point2);

            // 使用法线切割线进行分割（重载函数）
            // auto pieces = split_polygon_by_line(current, cutting_line);

            // 找包含路径中点的 piece
            Polygon keep = current;
            bool found = false;
            for(auto &piece: pieces){
                double localx = midx;
                double localy = midy;
                if(point_in_polygon(piece, localx, localy)){
                    keep = piece; found = true; break;
                }
            }
            if(!found){
                // 若没找到包含路径中点的 piece，尝试选面积最大的
                double bestA = -1; int idx = 0;
                for(size_t t=0;t<pieces.size();++t){ double a = fabs(bg::area(pieces[t])); if(a>bestA){bestA=a; idx=t;} }
                if(pieces.size()>0) keep = pieces[idx];
            }
            current = keep;
            corridor[i] = current;

            // 计算切割后多边形内部的障碍物点集
            obs_cells = polygon_obstacles_in_subgrid(current, subgrid, ix0, iy0);

            // 下边这段代码看似是多余的，但实际上非常重要，因为实际上可能存在切割失败的情况；
            // 如果没有下边代码（强行将已经处理过的障碍物剔除当前障碍物点集），在切割失败时，
            // 代码在下一轮循环中仍会尝试对之前失败的障碍物点进行切割（当然还是会失败），从而导致死循环
            // 而下边的（强行将已经处理过的障碍物剔除当前障碍物点集）可以避免处理重复障碍物，从而避免死循环
            // remove del_array points
            if(!del_array.empty()){
                vector<pair<int,int>> tmp;
                for(auto &oc: obs_cells)
                { 
                    bool drop=false; 
                    for(auto &d: del_array) 
                        if(d==oc) 
                        {
                            drop=true; 
                            break;
                        } 
                    if(!drop) 
                        tmp.push_back(oc); 
                }
                obs_cells.swap(tmp);
            }
            // mark current obstacle as deleted
            del_array.push_back(rc);
        }
    }
    return corridor;
}

// ------------------ Minimum-snap -> QP ------------------

// make difference matrix of given order
MatrixXd make_diff_matrix(int order, int N){
    if(order >= N) return MatrixXd::Zero(0, N);
    MatrixXd S = MatrixXd::Zero(N-order, N);
    // compute coeff via convolution
    vector<int> coeff = {1};
    for(int k=0;k<order;++k){
        vector<int> nc(coeff.size()+1);
        for(size_t i=0;i<coeff.size();++i){ nc[i] += coeff[i]; nc[i+1] -= coeff[i]; }
        coeff.swap(nc);
    }
    for(int i=0;i<N-order;++i){
        for(size_t j=0;j<coeff.size();++j) S(i, i+j) = coeff[j];
    }
    return S;
}

// 将 polygon 转换为 A x <= b 形式（每条边一个不等式）
void polygon_to_inequalities(const Polygon &poly, MatrixXd &A, VectorXd &b){
    const auto &coords = poly.outer();      // 多边形外环的点集
    int n = (int)coords.size()-1; // 最后一个是重复点
    A.resize(n, 2);
    b.resize(n);
    // 计算质心坐标
    auto cen = polygon_centroid(poly);
    for(int i = 0; i < n; ++i)
    {
        // 获取相邻边的两个顶点坐标
        double x0 = bg::get<0>(coords[i]);
        double y0 = bg::get<1>(coords[i]);
        double x1 = bg::get<0>(coords[(i+1)%n]);
        double y1 = bg::get<1>(coords[(i+1)%n]);
        // 计算x、y方向向量
        double ex = x1-x0, ey = y1-y0;
        // 计算该边的法向量
        Eigen::Vector2d normal(ey, -ex);
        // 质心坐标与第一个点的连线向量
        Eigen::Vector2d center_vec(cen.first - x0, cen.second - y0);
        if(normal.dot(center_vec) > 0) 
            normal = -normal; // 确保法向量朝向多边形外部

        // 计算A、b矩阵
        A(i,0) = normal.x(); 
        A(i,1) = normal.y();
        b(i) = normal.x()*x0 + normal.y()*y0;
    }
}

/*
看 README 解释
*/
// Solve minimum-snap QP using OSQP-Eigen
// minimize  (Sx)^T(Sx) + lambda * sum ||x - centers||^2  s.t. A_i * x_i <= b_i  (每个点的走廊约束)
vector<pair<double,double>> minimum_snap_solver(const vector<Polygon>& corridor, const vector<vector<int>>& grid, const vector<pair<int,int>>& path, int N, int dim, const string &solver="OSQP", double lambda_center=2.5){
    // construct S
    MatrixXd S;
    if(N >= 5) 
        S = make_diff_matrix(4, N);     // 要计算四阶差分，至少需要五个数
    else if(N >=3) 
        S = make_diff_matrix(2, N);     // 3个数计算2阶
    else 
        S = make_diff_matrix(1, N);

    // decision variables: traj: N x 2 -> flatten为 [x0,y0,x1,y1,...]
    int varN = N * dim;

    // objective quadratic part Q and linear part c
    // obj = sum_squares(S * X_col) + lambda * sum ||X - centers||^2
    // Expand: for x-coords: x^T (S^T S) x + lambda * (x^T x - 2 x^T centers_x + centers_x^T centers_x)
    MatrixXd Qt = MatrixXd::Zero(varN, varN);
    VectorXd c = VectorXd::Zero(varN);

    MatrixXd STS = MatrixXd::Zero(N,N);
    if(S.rows() > 0) 
        STS = S.transpose()*S;

    // centers 是安全走廊的质心坐标
    vector<pair<double,double>> centers;
    for(size_t i = 0; i < corridor.size(); ++i)
    {
        if(bg::area(corridor[i]) == 0)
        {
            double cx = (path[i].first + path[i+1].first)/2.0;
            double cy = (path[i].second + path[i+1].second)/2.0;
            centers.emplace_back(cx, cy);
        } 
        else 
        {
            auto ct = polygon_centroid(corridor[i]); 
            centers.emplace_back(ct.first, ct.second);
        }
    }
    // 最后一个center
    if(!corridor.empty())
    {
        if(bg::area(corridor.back()) == 0)
        {
            centers.emplace_back((path[path.size()-2].first + path[path.size()-1].first)/2.0, (path[path.size()-2].second + path[path.size()-1].second)/2.0);
        } 
        else 
        {
            auto ct = polygon_centroid(corridor.back()); 
            centers.emplace_back(ct.first, ct.second);
        }
    }
    if((int)centers.size() != N){
        // 若数量不匹配，补齐或截断
        while((int)centers.size() < N) 
            centers.emplace_back(centers.back());
        if((int)centers.size() > N) 
            centers.resize(N);      // 截断丢弃末尾多余元素
    }

    // assemble Qt and c
    /*
    Qt = STS + lambda * I; 
    c = -2 * lambda * centers;
    */
    for(int i = 0; i < N; ++i)
    {
        int xi = i*dim + 0;     // 第i个点的x坐标在总向量x中的索引
        int yi = i*dim + 1;     // 第i个点的y坐标在总向量x中的索引
        for(int j=0;j<N;++j)
        {
            double v = (S.rows()>0) ? STS(i,j) : 0.0;
            /*
            Qt = [STS  0
                  0   STS​]
            */
            Qt(xi, j*dim + 0) += v;
            Qt(yi, j*dim + 1) += v;
        }
        Qt(xi, xi) += lambda_center;
        Qt(yi, yi) += lambda_center;
        c(xi) += -2.0 * lambda_center * centers[i].first;
        c(yi) += -2.0 * lambda_center * centers[i].second;
    }

    // OSQP expects 1/2 x^T P x + q^T x, so we must set P = 2*Qt to match cp formulation 
    MatrixXd P = 2.0 * Qt;
    VectorXd q = c;

    // Constraints: equality constraints for start/end, and inequality A x <= b for each point inside corridor polygons
    // Arows、brows 存储每个点对应的多边形的A、b矩阵（以用作约束条件）
    vector<VectorXd> Arows; 
    vector<double> brows;   // brows存储了每个点的约束范围

    // start
    VectorXd Arow = VectorXd::Zero(varN);       // 创建了一个全0的行向量
    // x0 的系数是1
    Arow(0*dim + 0) = 1.0;      // 0*dim + 0 = 0 * 2 + 0 = 0；此处索引 0 对应的就是第一个轨迹点的x坐标 x0
    brows.push_back(path[0].first);     // 起点位置不能改变；1*x <= path[0].first
    Arows.push_back(Arow);

    Arow = VectorXd::Zero(varN); 
    Arow(0*dim + 1) = 1.0;      // 同理此处索引 1 对应的就是第一个轨迹点的y坐标 y0
    brows.push_back(path[0].second);    
    Arows.push_back(Arow);

    // end
    Arow = VectorXd::Zero(varN); 
    Arow((N-1)*dim + 0) = 1.0; 
    brows.push_back(path.back().first); 
    Arows.push_back(Arow);

    Arow = VectorXd::Zero(varN); 
    Arow((N-1)*dim + 1) = 1.0; 
    brows.push_back(path.back().second); 
    Arows.push_back(Arow);

    // corridor inequalities
    for(int i = 0; i < N-1; ++i){
        const Polygon &poly = corridor[i];
        if(bg::area(poly) == 0)
        {
            // 此处没有生成安全走廊
            VectorXd ar = VectorXd::Zero(varN); 
            ar(i * dim + 0) = 1;          // 该点的A矩阵为1，与下边结合就是 1 * xi ≤ maxX
            Arows.push_back(ar); 
            // 即 brows.push_back(maxX);
            brows.push_back(grid[0].size()-1);      // 这段路径很窄，就用整个栅格地图的边界作为约束（grid[0].size()-1）

            ar = VectorXd::Zero(varN); 
            // 这个同理，即 -1 * xi ≤ 0，也就是 xi ≥ 0
            ar(i * dim + 0) = -1; 
            Arows.push_back(ar); 
            brows.push_back(0);

            ar = VectorXd::Zero(varN); 
            ar(i * dim + 1) = 1; 
            Arows.push_back(ar); 
            brows.push_back(grid.size()-1);

            ar = VectorXd::Zero(varN); 
            ar(i * dim + 1) = -1; 
            Arows.push_back(ar); 
            brows.push_back(0);
        } 
        else 
        {
            MatrixXd A_poly; VectorXd b_poly;
            polygon_to_inequalities(poly, A_poly, b_poly);  // 获取当前多边形的A、b矩阵
            for(int row = 0; row < A_poly.rows(); ++row)
            {
                VectorXd ar = VectorXd::Zero(varN);
                // 赋值A、b矩阵
                ar(i*dim + 0) = A_poly(row,0);
                ar(i*dim + 1) = A_poly(row,1);
                Arows.push_back(ar);
                brows.push_back(b_poly(row) + 1e-6);    // Ax <= b
            }
        }
    }
    // 手动填充最后一个值为 corridor[-1]
    if(!corridor.empty()){
        MatrixXd A_poly; VectorXd b_poly;
        polygon_to_inequalities(corridor.back(), A_poly, b_poly);

        for(int row = 0; row < A_poly.rows(); ++row)
        {
            VectorXd ar = VectorXd::Zero(varN);
            ar((N-1)*dim + 0) = A_poly(row,0);
            ar((N-1)*dim + 1) = A_poly(row,1);
            Arows.push_back(ar); brows.push_back(b_poly(row) + 1e-6);
        }
    }

    // OSQP 使用 l ≤ Ax ≤ u 形式的约束
    int m = (int)Arows.size();      // 计算总共有多少条约束
    MatrixXd Aineq = MatrixXd::Zero(m, varN);       // 约束矩阵 A:创建一个 m x varN 的全零矩阵(m 个约束，varN 个决策变量)
    // 下界向量 l
    VectorXd lb = VectorXd::Constant(m, -1e20);     // 创建一个长度为 m 的向量，所有元素初始化为一个非常小的数（负无穷大）
    // 上界向量 u
    VectorXd ub = VectorXd::Zero(m);    // 创建一个长度为 m 的全零向量
    for(int i = 0; i < m; ++i)
    {
        Aineq.row(i) = Arows[i];    // 把 A 矩阵中第 i 个约束（A矩阵）赋值给 Aineq（约束矩阵）的第 i 行
        ub(i) = brows[i];   // 把 brows 中的第 i 个约束值（b），赋给 ub 的第 i 个元素
        lb(i) = -1e20; // 不等式约束没有下限
    }

    // 前 4 个约束是等式约束（起点、终点必须固定）
    for(int i = 0; i < 4; ++i)
        lb(i) = ub(i); 
    

    // Setup OSQP-Eigen
    OsqpEigen::Solver solverOSQP;
    // 转换为稀疏矩阵
    Eigen::SparseMatrix<double> Psp = P.sparseView();
    Eigen::SparseMatrix<double> Asp = Aineq.sparseView();

    solverOSQP.data()->setNumberOfVariables(varN);      // 告诉 solver 决策变量的个数
    solverOSQP.data()->setNumberOfConstraints(m);       // 告诉 solver 约束的行数
    // 设置二次项矩阵 P
    if(!solverOSQP.data()->setHessianMatrix(Psp)) 
        cerr<<"Failed set Hessian\n"; 
    // 设置线性项向量 q
    if(!solverOSQP.data()->setGradient(q)) 
        cerr<<"Failed set gradient\n"; 
    // 设置线性约束矩阵 A (Asp.rows() 必须等于之前的 m，Asp.cols() 必须等于 varN)
    if(!solverOSQP.data()->setLinearConstraintsMatrix(Asp)) 
        cerr<<"Failed set A\n"; 
    // 设置约束的下界 l 和上界 u
    if(!solverOSQP.data()->setLowerBound(lb)) 
        cerr<<"Failed set lb\n"; 
    if(!solverOSQP.data()->setUpperBound(ub)) 
        cerr<<"Failed set ub\n"; 

    // settings
    solverOSQP.settings()->setVerbosity(false);     // 关闭日志
    solverOSQP.settings()->setWarmStart(true);      // 允许使用热启动（如果有上一次的解或想重用初值，会加速收敛）

    // 初始化求解器
    if(!solverOSQP.initSolver()){
        cerr<<"OSQP init failed"<<endl;
        return {};
    }
    
    // 使用静态转换检查错误（非 0 表示出现问题或未达到收敛）
    if(static_cast<int>(solverOSQP.solveProblem()) != 0){
        cerr<<"OSQP solve failed"<<endl; 
        return {};
    }
    
    VectorXd x = solverOSQP.getSolution();

    vector<pair<double,double>> traj;
    for(int i=0;i<N;++i)
        traj.emplace_back(x(i*dim+0), x(i*dim+1)); 
    return traj;
}

// ------------------ 辅助：path 简化 ------------------
// 计算路径上所有线段向量的夹角 (弧度)
vector<double> compute_angles(const vector<Eigen::Vector2d> &points) {
    int n = points.size();
    vector<double> angles(n, M_PI);
    if (n < 3) return angles;

    for (int i = 1; i < n - 1; ++i) {
        // 表示两个相邻向量
        Eigen::Vector2d v1 = points[i] - points[i-1];
        Eigen::Vector2d v2 = points[i+1] - points[i];
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

// 路径简化：保留拐角
vector<pair<double,double>> simplify_path(
        const vector<Eigen::Vector2d> &points,
        double corner_deg = 30.0,
        int corner_dilate = 1)
{
    int n = points.size();
    if (n <= 2) {
        vector<pair<double,double>> out;
        for (auto &p : points)
            out.emplace_back(p.x(), p.y());
        return out;
    }

    vector<double> angles = compute_angles(points);
    double theta = corner_deg * M_PI / 180.0;

    // 特征点索引集合
    vector<int> corner_idx;
    corner_idx.reserve(n);

    for (int i = 0; i < n; ++i) {
        if (i == 0 || i == n - 1 || angles[i] >= theta)
            corner_idx.push_back(i);
    }

    // 拐角膨胀
    if (corner_dilate > 0) {
        set<int> expanded;
        for (int idx : corner_idx) {
            for (int j = idx - corner_dilate; j <= idx + corner_dilate; ++j) {
                if (j >= 0 && j < n) 
                    expanded.insert(j);     // set 会自动去重排序
            }
        }
        corner_idx.assign(expanded.begin(), expanded.end());    // 将set集合的内容重新赋给vector
    }

    // 输出为 pair<double,double>
    vector<pair<double,double>> out;
    out.reserve(corner_idx.size());

    for (int idx : corner_idx) {
        out.emplace_back(points[idx].x(), points[idx].y());
    }

    return out;
}

// ------------------ 可视化函数 ------------------
void visualize_results(const vector<vector<int>>& grid,
                       const vector<pair<int,int>>& raw_path,
                       const vector<pair<double,double>>& simplified_path,
                       const vector<Polygon>& corridors,
                       const vector<pair<double,double>>& optimized_traj,
                       const vector<array<double,8>>& initial_rects,    // 初始走廊膨胀矩形
                       pair<int,int> start_pos,      // 起点
                       pair<int,int> goal_pos,       // 终点
                       int scale = 12)                // 大一点更清楚
{
    int rows = grid.size();
    int cols = grid[0].size();
    cv::Mat img(rows * scale, cols * scale, CV_8UC3, cv::Scalar(255, 255, 255)); // 白底

    // 1. 障碍物（深灰）
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            if (grid[r][c] == 50) {
                cv::rectangle(img,
                    cv::Point(c * scale, r * scale),
                    cv::Point((c + 1) * scale - 1, (r + 1) * scale - 1),
                    cv::Scalar(70, 70, 70), cv::FILLED);
            }
        }
    }

    // // 2. 初始膨胀矩形（浅黄填充 + 橙色边框）
    // for (const auto& rect : initial_rects) {
    //     vector<cv::Point> pts;
    //     for (int i = 0; i < 4; ++i) {
    //         int x = static_cast<int>(rect[i*2] * scale + 0.5);
    //         int y = static_cast<int>(rect[i*2+1] * scale + 0.5);
    //         pts.emplace_back(x, y);
    //     }
    //     cv::fillPoly(img, pts, cv::Scalar(255, 255, 200));        // 浅黄填充
    //     cv::polylines(img, pts, true, cv::Scalar(0, 150, 255), 2); // 橙色边框
    // }

    // 2. 安全走廊（浅蓝填充 + 深蓝边框）
    for (const auto& poly : corridors) {
        if (bg::area(poly) < 1e-6) continue;

        vector<cv::Point> pts;
        for (const auto& p : poly.outer()) {
            int x = static_cast<int>(bg::get<0>(p) * scale + 0.5);
            int y = static_cast<int>(bg::get<1>(p) * scale + 0.5);
            pts.emplace_back(x, y);
        }
        if (pts.size() < 3) continue;

        // OpenCV fillPoly 推荐逆时针
        if (cv::contourArea(pts) < 0) {
            std::reverse(pts.begin(), pts.end());
        }

        cv::fillPoly(img, pts, cv::Scalar(230, 230, 255));        // 浅蓝填充
        cv::polylines(img, pts, true, cv::Scalar(80, 80, 255), 2);   // 深蓝边框
    }

    // 3. A* 原始路径（红色）
    for (size_t i = 1; i < raw_path.size(); ++i) {
        cv::line(img,
                 cv::Point(raw_path[i-1].first * scale + scale/2, raw_path[i-1].second * scale + scale/2),
                 cv::Point(raw_path[i].first   * scale + scale/2, raw_path[i].second   * scale + scale/2),
                 cv::Scalar(0, 0, 180), 3);
    }

    // 4. 简化后的关键点路径（绿色粗线）
    for (size_t i = 1; i < simplified_path.size(); ++i) {
        cv::line(img,
                 cv::Point(static_cast<int>(simplified_path[i-1].first * scale + 0.5),
                           static_cast<int>(simplified_path[i-1].second * scale + 0.5)),
                 cv::Point(static_cast<int>(simplified_path[i].first * scale + 0.5),
                           static_cast<int>(simplified_path[i].second * scale + 0.5)),
                 cv::Scalar(0, 200, 0), 4);
    }

    // 5. 最小snap优化轨迹
    for (size_t i = 1; i < optimized_traj.size(); ++i) {
        cv::line(img,
                 cv::Point(static_cast<int>(optimized_traj[i-1].first * scale + 0.5),
                           static_cast<int>(optimized_traj[i-1].second * scale + 0.5)),
                 cv::Point(static_cast<int>(optimized_traj[i].first * scale + 0.5),
                           static_cast<int>(optimized_traj[i].second * scale + 0.5)),
                 cv::Scalar(0, 0, 139), 3);   // 
    }

    // 1. 简化后的关键点（绿色大圆 + 白边）
    for (const auto& p : simplified_path) {
        int x = static_cast<int>(p.first * scale + 0.5);
        int y = static_cast<int>(p.second * scale + 0.5);
        cv::circle(img, cv::Point(x, y), 5, cv::Scalar(0, 255, 0), -1);     // 绿填充
        // cv::circle(img, cv::Point(x, y), 18, cv::Scalar(255, 255, 255), 4);  // 白边框超醒目
        // cv::circle(img, cv::Point(x, y), 8,  cv::Scalar(0, 100, 0), -1);    // 深绿内圈
    }

    // 2. 最小snap优化轨迹的每个点（橙色超大圆 + 黑边 + 序号）
    for (size_t i = 0; i < optimized_traj.size(); ++i) {
        int x = static_cast<int>(optimized_traj[i].first * scale + 0.5);
        int y = static_cast<int>(optimized_traj[i].second * scale + 0.5);

        // 大橙圈 + 黑边
        cv::circle(img, cv::Point(x, y), 5, cv::Scalar(30, 30, 30), -1);    // 填充
        // cv::circle(img, cv::Point(x, y), 22, cv::Scalar(0, 0, 0), 5);         // 黑边超粗

        // // 写序号（从0开始）
        // std::string text = std::to_string(i);
        // cv::putText(img, text, cv::Point(x - 15, y + 10),
        //             cv::FONT_HERSHEY_DUPLEX, 1.2, cv::Scalar(255, 255, 255), 3);
        // cv::putText(img, text, cv::Point(x - 15, y + 10),
        //             cv::FONT_HERSHEY_DUPLEX, 1.2, cv::Scalar(0, 0, 0), 2);  // 黑边文字
    }

    // // 3. A* 原始路径的每一个格子点（红色小方块）
    // for (const auto& p : raw_path) {
    //     cv::rectangle(img,
    //         cv::Point(p.first * scale + scale/4, p.second * scale + scale/4),
    //         cv::Point(p.first * scale + 3*scale/4, p.second * scale + 3*scale/4),
    //         cv::Scalar(0, 0, 255), -1);
    // }

    // 6. 起点终点大圆圈
    cv::circle(img, cv::Point(start_pos.first * scale + scale/2, start_pos.second * scale + scale/2),
               10, cv::Scalar(0, 255, 255), cv::FILLED);   // 黄
    cv::circle(img, cv::Point(goal_pos.first  * scale + scale/2, goal_pos.second  * scale + scale/2),
               10, cv::Scalar(255, 0, 255), cv::FILLED);   // 紫

    cv::flip(img, img, 0);   // 0 = 上下翻转 → 坐标系变成左下角为原点！

    // 7. 图例
    int y0 = 30;
    auto put = [&](const string& s, cv::Scalar col) {
        cv::putText(img, s, cv::Point(15, y0 += 28), cv::FONT_HERSHEY_SIMPLEX, 0.75, col, 2);
    };
    put("Initial Rects",         cv::Scalar(0, 150, 255));
    put("Optimized Corridors",   cv::Scalar(80, 80, 255));
    put("A* Raw Path",          cv::Scalar(0, 0, 200));
    put("Simplified Waypoints", cv::Scalar(0, 200, 0));
    put("Min-Snap Trajectory",  cv::Scalar(0, 0, 139));
    put("Start",                cv::Scalar(0, 255, 255));
    put("Goal",                 cv::Scalar(255, 0, 255));

    cv::namedWindow("Convex Corridor + Minimum-Snap Trajectory", cv::WINDOW_NORMAL);  // 关键！必须是 WINDOW_NORMAL
    cv::resizeWindow("Convex Corridor + Minimum-Snap Trajectory", 1200, 800);           // 初始开一个超大窗口（根据你屏幕改）
    // cv::setWindowProperty("Convex Corridor + Minimum-Snap Trajectory", cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN); // 甚至可以直接全屏（可选）

    cv::imshow("Convex Corridor + Minimum-Snap Trajectory", img);

    // 顺便保存高清图
    cv::imwrite("result_full_hd.png", img);

    std::cout << "\n可视化窗口已打开，支持鼠标拖拽边框自由调整大小！\n"
              << "按任意键关闭窗口...\n";

    cv::waitKey(0);
    cv::destroyAllWindows();
}

// ------------------ 主程序 ------------------
int main(){
    using namespace std::chrono;
    auto t0 = high_resolution_clock::now();

    int rows = 60, cols = 80;
    vector<vector<int>> grid(rows, vector<int>(cols, 0));
    for(int r=0;r<40;++r) for(int c=15;c<19;++c) grid[r][c] = 50;
    for(int r=20;r<60;++r) for(int c=30;c<33;++c) grid[r][c] = 50;
    for(int r=40;r<43;++r) for(int c=40;c<60;++c) grid[r][c] = 50;

    pair<int,int> start = {0,0};
    pair<int,int> goal  = {79,59};

    // --------------------- A* ---------------------
    auto path_Astar = astar(start, goal, grid);
    if(path_Astar.empty())
        {cerr<<"No path found by A*"<<endl; return -1; }
    cout<<"Original path length: "<<path_Astar.size()<<"\n";
    auto t1 = high_resolution_clock::now();
    cout<<"A* time: "<<duration<double>(t1-t0).count()<<"s\n";

    // --------------------- Simplified ---------------------
    // 转换为vector<Eigen::Vector2d>
    vector<Eigen::Vector2d> path;
    for (auto &p : path_Astar)
        path.emplace_back(p.first, p.second);

    auto simplified = simplify_path(path, 30.0, 1);
    cout<<"Simplified length: "<<simplified.size()<<"\n";

    // --------------------- build corridor rectangles ---------------------
    double max_width = 7; double extend = 8;
    vector<pair<double,double>> path_xy;
    for(auto &p: simplified) 
        path_xy.emplace_back(p.first, p.second);

    // 走廊初始膨胀
    // 注意：当四个顶点只有一个被裁剪时会生成梯形
    auto rects = convex_corridor(path_xy, rows, cols, max_width, extend);
    // for (const auto& rect : rects) {
    //     for (const double& value : rect) {
    //         std::cout << value << " ";
    //     }
    //     std::cout << std::endl;
    // }

    // convert and refine by splitting
    auto corridors = corridor_generator_optimized(path_xy, rects, grid, max_width);

    auto t2 = high_resolution_clock::now();
    cout<<"Corridor build time: "<<duration<double>(t2-t1).count()<<"s\n";

    // --------------------- Minimumsnap ---------------------
    int N = (int)simplified.size();
    vector<pair<int,int>> path_int;
    for(auto &p: simplified) 
        path_int.emplace_back(int(round(p.first)), int(round(p.second)));

    auto traj = minimum_snap_solver(corridors, grid, path_int, N, 2, "OSQP", 1.0);
    auto t3 = high_resolution_clock::now();  
    cout<<"Minimum-snap time: "<<duration<double>(t3-t2).count()<<"s\n";
    
    // --------------------- visualization ---------------------
    visualize_results(grid, path_Astar, simplified, corridors, traj, rects, start, goal, 12);
    
    // 输出轨迹点
    // cout<<"Optimized trajectory: \n";
    // for(auto &pt: traj) 
    //     cout<<pt.first<<", "<<pt.second<<"\n";

    return 0;
}
