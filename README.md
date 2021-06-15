# Loam_MeiyuanXiang
Loam相关论文、代码中文注释以及代码改动

# 参考
https://github.com/cuitaixiang/LOAM_NOTED  

# 环境
1. Ubuntu（测试了Ubuntu16.04.5、Ubuntu18.04）
2. ROS (测试了kinetic、melodic)
3. PCL（测试了pcl1.7）
4. Opencv（测试了opencv3.4.3）

# 编译
1. 下载源码 git clone https://github.com/MeiyuanXiang/Loam_MeiyuanXiang.git
2. 将Loam_MeiyuanXiang\src下的loam_velodyne或loam_velodyne_modified拷贝到ros工程空间src文件夹内，例如~/catkin_ws/src/
3. cd ~/catkin_ws
4. catkin_make
5. source ~/catkin_ws/devel/setup.bash

# 数据
链接：https://pan.baidu.com/s/1S2PTKo3T8ugWyHyULqwzag  
提取码：gfen

# 运行
roslaunch loam_velodyne loam_velodyne.launch  
rosbag play 2018-05-18-14-49-12_0.bag
