cmake_minimum_required(VERSION 3.16)
project(corridor_minimum_snap)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找依赖包
find_package(Eigen3 REQUIRED)
find_package(Boost REQUIRED COMPONENTS system filesystem thread)

# --- 查找OSQP和OSQP-Eigen ---
message(STATUS "Searching for OSQP and OSQP-Eigen...")

# 查找OSQP核心库
find_path(OSQP_INCLUDE_DIR 
    NAMES osqp/osqp.h
    PATHS /usr/local/include /usr/include
    DOC "OSQP include directory"
)
find_library(OSQP_LIBRARY 
    NAMES osqp
    PATHS /usr/local/lib /usr/lib
    DOC "OSQP library"
)

# 查找OSQP-Eigen库
find_library(OSQP_EIGEN_LIBRARY 
    NAMES OsqpEigen
    PATHS /usr/local/lib /usr/lib
    DOC "OSQP-Eigen library"
)

# --- 添加OpenCV查找 ---
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")

# --- 打印查找结果，用于调试 ---
message(STATUS "OSQP_INCLUDE_DIR: ${OSQP_INCLUDE_DIR}")
message(STATUS "OSQP_LIBRARY: ${OSQP_LIBRARY}")
message(STATUS "OSQP_EIGEN_LIBRARY: ${OSQP_EIGEN_LIBRARY}")

# 检查是否找到所有必需的库和头文件
if(NOT OSQP_INCLUDE_DIR OR NOT OSQP_LIBRARY)
    message(FATAL_ERROR "OSQP library not found. Please ensure OSQP is installed.")
endif()

if(NOT OSQP_EIGEN_LIBRARY)
    message(FATAL_ERROR "OSQP-Eigen library not found. Please ensure OSQP-Eigen is installed.")
endif()

if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV not found. Please ensure OpenCV is installed.")
endif()

# --- 创建导入目标来管理依赖 ---
# 1. 为OSQP创建一个接口库
add_library(osqp_lib INTERFACE)
target_include_directories(osqp_lib INTERFACE 
    ${OSQP_INCLUDE_DIR}
    ${OSQP_INCLUDE_DIR}/osqp
)
target_link_libraries(osqp_lib INTERFACE ${OSQP_LIBRARY})

# 2. 为OSQP-Eigen创建一个接口库，并让它依赖于osqp_lib
add_library(osqp_eigen_lib INTERFACE)
target_include_directories(osqp_eigen_lib INTERFACE 
    ${OSQP_INCLUDE_DIR}
    ${OSQP_INCLUDE_DIR}/osqp
)
target_link_libraries(osqp_eigen_lib INTERFACE osqp_lib ${OSQP_EIGEN_LIBRARY})

# --- 添加可执行文件并链接 ---
add_executable(corridor_minimum_snap Corridor-Inflation+Minimumsnap.cpp)

target_link_libraries(corridor_minimum_snap 
    Eigen3::Eigen 
    Boost::boost
    osqp_eigen_lib
    ${OpenCV_LIBS}  # ⭐ 添加OpenCV链接
)

# 可选：设置OpenCV的包含目录（如果需要）
target_include_directories(corridor_minimum_snap PRIVATE ${OpenCV_INCLUDE_DIRS})
