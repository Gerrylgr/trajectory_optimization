# 如何将 Minimumsnap 问题构建为二次规划问题，以丢给求解器（OSQP）求解？

Minimumsnap 的核心思想是让加加速度的变化整体最小化；<br/>

## 一、Minimumsnap 问题的原始表达式
在这个轨迹优化问题中，表达式为：<br/>

| <img src="https://github.com/Gerrylgr/trajectory_optimization/blob/master/Corridor-Inflation%2BMinimumsnap/image/image0.png" width="350px"> |
|:---:|

<br/>

其中第一部分（ (Sx)^T*(Sx) ）是平滑项，第二部分（ (xi​−ci)的平方积分 ）是约束项（让轨迹尽量靠近 corridor 中心线）；
<br/><br/>
而 S 就是（四阶）差分矩阵，x 是要优化的变量（轨迹点），(Sx)^T*(Sx) 等价于 （Sx）^2，这是一个平滑性代价，它越小轨迹越平滑；
<br/><br/>
xi 代表每一个路径点，ci 是对应走廊质心坐标，第二部分越小，轨迹越靠近走廊中心；
<br/>

## 二、写为二次规划形式
要将上边的表达式写为标准二次规划的形式，目标函数要变成：<br/>

| <img src="https://github.com/Gerrylgr/trajectory_optimization/blob/master/Corridor-Inflation%2BMinimumsnap/image/image1.png" width="200px"> |
|:---:|

<br/>
也就是把目标函数拆成 "二次项 + 一次项"；
<br/><br/>
**将一、中的表达式展开：**<br/>
对于平滑项：<br/>

| <img src="https://github.com/Gerrylgr/trajectory_optimization/blob/master/Corridor-Inflation%2BMinimumsnap/image/image2.png" width="250px"> |
|:---:|

<br/>
所以平滑项的二次项（系数）是 S^T*S，且没有线性项。
<br/>
对于约束项，展开为：<br/>

| <img src="https://github.com/Gerrylgr/trajectory_optimization/blob/master/Corridor-Inflation%2BMinimumsnap/image/image3.png" width="240px"> |
|:---:|

<br/>
所以二次项是 λ*xi^2，线性项是 −2λ*ci*xi	，常数项不用管（对求最小值没影响）
<br/><br/>
总结可知，二次项矩阵为：<br/>

| <img src="https://github.com/Gerrylgr/trajectory_optimization/blob/master/Corridor-Inflation%2BMinimumsnap/image/image4.png" width="210px"> |
|:---:|

<br/>
线性项为：<br/>

| <img src="https://github.com/Gerrylgr/trajectory_optimization/blob/master/Corridor-Inflation%2BMinimumsnap/image/image5.png" width="190px"> |
|:---:|

<br/>
这就是代码中的 Qt（Q total） 和 c（的组成）
<br/><br/>
**那如何把 STS 和 λ 填入到 Q 矩阵中？**
<br/>
首先，轨迹是二维曲线，所以需要为每个维度（x、y）分别加入平滑项：<br/>

| <img src="https://github.com/Gerrylgr/trajectory_optimization/blob/master/Corridor-Inflation%2BMinimumsnap/image/image7.png" width="230px"> |
|:---:|

<br/>
所以此时 Q（Qt）的形状为：<br/>

| <img src="https://github.com/Gerrylgr/trajectory_optimization/blob/master/Corridor-Inflation%2BMinimumsnap/image/image8.png" width="230px"> |
|:---:|

<br/>
在此基础上，直接加上 λ 即可得到最终的 Q 矩阵
<br/>

## 三、把 Q 和 c 转成 OSQP 所需要的形式
OSQP 的定义是：<br/>

| <img src="https://github.com/Gerrylgr/trajectory_optimization/blob/master/Corridor-Inflation%2BMinimumsnap/image/image6.png" width="200px"> |
|:---:|

<br/>
而上边的 Q 矩阵没有 1/2 系数，因此 P = 2Q
<br/><br/>
对应代码中：

    MatrixXd P = 2.0 * Qt;
    VectorXd q = c;
<br/>
此时 OSQP 需要的都已经构造完毕，在代码中后续只需要再增加走廊边界的约束即可。
