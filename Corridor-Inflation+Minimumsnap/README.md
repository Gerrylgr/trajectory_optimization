# (A*+)安全走廊构建 + Minimumsnap 求解算法实现
## 1.项目介绍

本项目实现了 (A*+) 安全走廊构建 + Minimumsnap 算法，提供 C++ 与 Python 两种版本，可用于机器人路径规划；
对于 Minimumsnap 问题构建的简单解释，可见文件：[Minimumsnap问题构建解释.md](Minimumsnap问题构建解释.md)

项目文件可视化截图：

Python:

<img src="https://github.com/Gerrylgr/trajectory_optimization/blob/master/Corridor-Inflation%2BMinimumsnap/image/image9.png" width="500px">

C++:

<img src="https://github.com/Gerrylgr/trajectory_optimization/blob/master/Corridor-Inflation%2BMinimumsnap/image/image10.png" width="500px">

## 2.使用方法

对于Linux操作系统：

运行 Corridor-Inflation+Minimumsnap.py:

    python3 Corridor-Inflation+Minimumsnap/Corridor-Inflation+Minimumsnap.py

<br/>
运行 Corridor-Inflation+Minimumsnap.cpp:

配置编译规则：(请提前安装好 OpenCV、OSQP、OSQP-Eigen，如果找不到包请自行到 CMakeLists.txt 中修改路径)

    mkdir Corridor-Inflation+Minimumsnap/build
    cd Corridor-Inflation+Minimumsnap/build/
    cmake ..

编译：

    make -j8

运行：

    ./corridor_minimum_snap  

（代码中函数具体用法可见注释）

### 讲解视频可见个人主页：

https://b23.tv/bTT1iDU

### 声明 ※※※
    
If you use this work in your research(or paper,video,etc.), you must cite it as follows:

[刘耕睿]. (2025). *trajectory_optimization* [Computer software]. GitHub. https://github.com/Gerrylgr/trajectory_optimization


## 3.作者

- [Gerry Liu](https://github.com/Gerrylgr?tab=repositories)