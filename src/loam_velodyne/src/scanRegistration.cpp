// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// This is an implementation of the algorithm described in the following paper:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014.

/*
  scanRegistration主要对点云和imu数据进行预处理，用于特征点的配准。
  1.对单帧点云（sweep）进行分线束（VLP16分为16束），每束称为一个scan，并记录每个点所属线束和每个点在此帧点云内的相对扫描时间；
  2.有imu数据时，对对应的激光点进行运动补偿；
  3.针对单个scan，根据激光点的曲率，将激光点划分为不同特征类别（边特征/面特征/不是特征）；
  4.发布处理结果。
*/

/*******************************************************************************
  ROS坐标系：x轴向前，y轴向左，z轴向上的右手坐标系
  欧拉角坐标系：z轴向前，X轴向左，y轴向上的右手坐标系
  imu坐标系：x轴向前，y轴向左，z轴向上的右手坐标系
  velodyne lidar安装坐标系：x轴向前，y轴向左，z轴向上的右手坐标系
  scanRegistration会把两者通过交换坐标轴，都统一到z轴向前，x轴向左，y轴向上的欧拉角坐标系，这是J. Zhang的论文里面使用的坐标系
*******************************************************************************/

#include <cmath>
#include <vector>

#include <loam_velodyne/common.h>
#include <opencv/cv.h>
#include <nav_msgs/Odometry.h>
#include <opencv/cv.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>

using std::sin;
using std::cos;
using std::atan2;

const double scanPeriod = 0.1; // 一个scan的周期, velodyne频率10Hz，周期0.1s

// 初始化变量
const int systemDelay = 20; // 弃用前20帧初始数据
int systemInitCount = 0;
bool systemInited = false;

const int N_SCANS = 16; // 激光雷达线数

// 一帧点云最多存放40000个点，雷达一圈为32256个点
float cloudCurvature[40000]; // 曲率
int cloudSortInd[40000]; // 曲率对应的序号
int cloudNeighborPicked[40000]; // 是否筛选过，0没有，1有
int cloudLabel[40000]; // 2曲率很大，1曲率较大，0曲率较小，-1曲率很小(其中1包含了2，0包含了1，0和-1构成了点云全部的点)

int imuPointerFront = 0; // 时间戳大于当前点云时间戳的imu数据在各个imu数组中的位置
int imuPointerLast = -1; // 最新收到的imu数据在各个imu数组中的位置
const int imuQueLength = 200; // imu更新队列的长度

// 点云数据开始第一个点的位移/速度/欧拉角
float imuRollStart = 0, imuPitchStart = 0, imuYawStart = 0;
float imuRollCur = 0, imuPitchCur = 0, imuYawCur = 0;

float imuVeloXStart = 0, imuVeloYStart = 0, imuVeloZStart = 0;
float imuShiftXStart = 0, imuShiftYStart = 0, imuShiftZStart = 0;

// 当前点的速度，位移信息
float imuVeloXCur = 0, imuVeloYCur = 0, imuVeloZCur = 0;
float imuShiftXCur = 0, imuShiftYCur = 0, imuShiftZCur = 0;

// 每次点云数据当前点相对于开始第一个点的畸变位移，速度
float imuShiftFromStartXCur = 0, imuShiftFromStartYCur = 0, imuShiftFromStartZCur = 0;
float imuVeloFromStartXCur = 0, imuVeloFromStartYCur = 0, imuVeloFromStartZCur = 0;

// imu信息
double imuTime[imuQueLength] = {0};
float imuRoll[imuQueLength] = {0};
float imuPitch[imuQueLength] = {0};
float imuYaw[imuQueLength] = {0};

float imuAccX[imuQueLength] = {0};
float imuAccY[imuQueLength] = {0};
float imuAccZ[imuQueLength] = {0};

float imuVeloX[imuQueLength] = {0};
float imuVeloY[imuQueLength] = {0};
float imuVeloZ[imuQueLength] = {0};

float imuShiftX[imuQueLength] = {0};
float imuShiftY[imuQueLength] = {0};
float imuShiftZ[imuQueLength] = {0};

ros::Publisher pubLaserCloud; // 发布按线分类后的点云
ros::Publisher pubCornerPointsSharp; // 发布边界线上锐角特征点云
ros::Publisher pubCornerPointsLessSharp; // 发布边界线上钝角特征点云
ros::Publisher pubSurfPointsFlat; // 发布平面上特征点云
ros::Publisher pubSurfPointsLessFlat; // 发布平面上不太平的特征点云
ros::Publisher pubImuTrans; // 发布处理后的imu数据

/*
  求当前点的位移相对于点云起始点的位移畸变，先计算全局坐标系下的位移畸变，然后再转换到IMU起始点的坐标系中。 
  首先计算畸变位移，再根据rpy的反向，绕y，x，z轴分别旋转，即可将位移畸变从世界坐标系转移到局部坐标系。
  rpy即roll，pitch，yaw分别是绕着z，x，y轴旋转。
*/
void ShiftToStartIMU(float pointTime)
{
  // 相对于第一个点由于加减速产生的畸变位移(全局坐标系下畸变位移量delta_Tg)
  imuShiftFromStartXCur = imuShiftXCur - imuShiftXStart - imuVeloXStart * pointTime;
  imuShiftFromStartYCur = imuShiftYCur - imuShiftYStart - imuVeloYStart * pointTime;
  imuShiftFromStartZCur = imuShiftZCur - imuShiftZStart - imuVeloZStart * pointTime;
  /********************************************************************************
    rpy:Rz(roll).inverse * Rx(pitch).inverse * Ry(yaw).inverse * delta_Vg
    transfrom from the global frame to the local frame
  *********************************************************************************/

  float x1 = cos(imuYawStart) * imuShiftFromStartXCur - sin(imuYawStart) * imuShiftFromStartZCur;
  float y1 = imuShiftFromStartYCur;
  float z1 = sin(imuYawStart) * imuShiftFromStartXCur + cos(imuYawStart) * imuShiftFromStartZCur;

  // 绕x轴旋转(-imuPitchStart)，即Rx(pitch).inverse
  float x2 = x1;
  float y2 = cos(imuPitchStart) * y1 + sin(imuPitchStart) * z1;
  float z2 = -sin(imuPitchStart) * y1 + cos(imuPitchStart) * z1;

  // 绕z轴旋转(-imuRollStart)，即Rz(pitch).inverse
  // 位移偏差由局部地球坐标系旋转到初始点位姿坐标系
  imuShiftFromStartXCur = cos(imuRollStart) * x2 + sin(imuRollStart) * y2;
  imuShiftFromStartYCur = -sin(imuRollStart) * x2 + cos(imuRollStart) * y2;
  imuShiftFromStartZCur = z2;
}

/*
  计算局部坐标系下点云中的点相对第一个开始点由于加减速产生的的速度畸变（增量）
*/
void VeloToStartIMU()
{
  // 相对于第一个点由于加减速产生的畸变速度(全局坐标系下畸变速度增量delta_Vg)
  imuVeloFromStartXCur = imuVeloXCur - imuVeloXStart;
  imuVeloFromStartYCur = imuVeloYCur - imuVeloYStart;
  imuVeloFromStartZCur = imuVeloZCur - imuVeloZStart;
  /********************************************************************************
    rpy:Rz(roll).inverse * Rx(pitch).inverse * Ry(yaw).inverse * delta_Vg
    transfrom from the global frame to the local frame
  *********************************************************************************/

  // 绕y轴旋转(-imuYawStart)，即Ry(yaw).inverse
  float x1 = cos(imuYawStart) * imuVeloFromStartXCur - sin(imuYawStart) * imuVeloFromStartZCur;
  float y1 = imuVeloFromStartYCur;
  float z1 = sin(imuYawStart) * imuVeloFromStartXCur + cos(imuYawStart) * imuVeloFromStartZCur;

  // 绕x轴旋转(-imuPitchStart)，即Rx(pitch).inverse
  float x2 = x1;
  float y2 = cos(imuPitchStart) * y1 + sin(imuPitchStart) * z1;
  float z2 = -sin(imuPitchStart) * y1 + cos(imuPitchStart) * z1;

  // 绕z轴旋转(-imuRollStart)，即Rz(pitch).inverse
  imuVeloFromStartXCur = cos(imuRollStart) * x2 + sin(imuRollStart) * y2;
  imuVeloFromStartYCur = -sin(imuRollStart) * x2 + cos(imuRollStart) * y2;
  imuVeloFromStartZCur = z2;
}

/*
  去除点云加减速产生的位移畸变。 
  首先rpy轴将点转换到世界坐标系，
  然后再ypr由世界坐标系转换到IMU起始点坐标系，
  最后减去加减速造成的非匀速畸变的值。
*/
void TransformToStartIMU(PointType *p)
{
  /********************************************************************************
    Ry*Rx*Rz*Pl, transform point to the global frame
  *********************************************************************************/
  
  // 绕z轴旋转(imuRollCur)
  float x1 = cos(imuRollCur) * p->x - sin(imuRollCur) * p->y;
  float y1 = sin(imuRollCur) * p->x + cos(imuRollCur) * p->y;
  float z1 = p->z;

  // 绕x轴旋转(imuPitchCur)
  float x2 = x1;
  float y2 = cos(imuPitchCur) * y1 - sin(imuPitchCur) * z1;
  float z2 = sin(imuPitchCur) * y1 + cos(imuPitchCur) * z1;

  // 绕y轴旋转(imuYawCur)
  float x3 = cos(imuYawCur) * x2 + sin(imuYawCur) * z2;
  float y3 = y2;
  float z3 = -sin(imuYawCur) * x2 + cos(imuYawCur) * z2;

  /********************************************************************************
    Rz(pitch).inverse * Rx(pitch).inverse * Ry(yaw).inverse * Pg
    transfrom global points to the local frame
  *********************************************************************************/

  // 局部地球坐标系下激光点旋转至该帧初始点坐标系并消除imu加速度造成的位移偏差
  // 绕y轴旋转(-imuYawStart)
  float x4 = cos(imuYawStart) * x3 - sin(imuYawStart) * z3;
  float y4 = y3;
  float z4 = sin(imuYawStart) * x3 + cos(imuYawStart) * z3;

  // 绕x轴旋转(-imuPitchStart)
  float x5 = x4;
  float y5 = cos(imuPitchStart) * y4 + sin(imuPitchStart) * z4;
  float z5 = -sin(imuPitchStart) * y4 + cos(imuPitchStart) * z4;

  // 绕z轴旋转(-imuRollStart)，然后加上imu位置运动补偿
  p->x = cos(imuRollStart) * x5 + sin(imuRollStart) * y5 + imuShiftFromStartXCur;
  p->y = -sin(imuRollStart) * x5 + cos(imuRollStart) * y5 + imuShiftFromStartYCur;
  p->z = z5 + imuShiftFromStartZCur;
}

/*
  主要是积分速度与位移
  首先读取imu回调函数中存储的各个变量消息，
  然后通过绕z/x/y轴分别旋转将加速度从imu坐标系转换至局部地球坐标系，
  timeDiff表示相邻两个惯导数据的时间差，假设在这个时间差内载体做匀加速运动，根据前一时刻三个坐标轴方向位移imuShift和速度imuVelo求当前时刻位移和速度消息，
  此时imu消息就转换到了局部地球坐标系。
*/
void AccumulateIMUShift()
{
  // 积分速度与位移
  float roll = imuRoll[imuPointerLast];
  float pitch = imuPitch[imuPointerLast];
  float yaw = imuYaw[imuPointerLast];
  float accX = imuAccX[imuPointerLast];
  float accY = imuAccY[imuPointerLast];
  float accZ = imuAccZ[imuPointerLast];

  // 绕RPY旋转转换得到世界坐标系下的加速度值
  // 绕z轴旋转(roll)
  float x1 = cos(roll) * accX - sin(roll) * accY;
  float y1 = sin(roll) * accX + cos(roll) * accY;
  float z1 = accZ;
  // 绕x轴旋转(pitch)
  float x2 = x1;
  float y2 = cos(pitch) * y1 - sin(pitch) * z1;
  float z2 = sin(pitch) * y1 + cos(pitch) * z1;
  // 绕y轴旋转(yaw)
  accX = cos(yaw) * x2 + sin(yaw) * z2;
  accY = y2;
  accZ = -sin(yaw) * x2 + cos(yaw) * z2;

  int imuPointerBack = (imuPointerLast + imuQueLength - 1) % imuQueLength; // 前一时刻的imu
  double timeDiff = imuTime[imuPointerLast] - imuTime[imuPointerBack]; // 前一时刻与后一时刻时间差，即计算imu测量周期
  // 要求imu的频率至少比lidar高，这样的imu信息才使用，后面校正也才有意义
  if (timeDiff < scanPeriod) { // （隐含从静止开始运动）
    // 位移计算：X（t+1）= X（t）+ v * dt + 1/2 * a * t^2;
    imuShiftX[imuPointerLast] = imuShiftX[imuPointerBack] + imuVeloX[imuPointerBack] * timeDiff 
                              + accX * timeDiff * timeDiff / 2;
    imuShiftY[imuPointerLast] = imuShiftY[imuPointerBack] + imuVeloY[imuPointerBack] * timeDiff 
                              + accY * timeDiff * timeDiff / 2;
    imuShiftZ[imuPointerLast] = imuShiftZ[imuPointerBack] + imuVeloZ[imuPointerBack] * timeDiff 
                              + accZ * timeDiff * timeDiff / 2;

    // 速度计算：V(t+1) = V(t) + a * t；
    imuVeloX[imuPointerLast] = imuVeloX[imuPointerBack] + accX * timeDiff; 
    imuVeloY[imuPointerLast] = imuVeloY[imuPointerBack] + accY * timeDiff;
    imuVeloZ[imuPointerLast] = imuVeloZ[imuPointerBack] + accZ * timeDiff;
  }
}

// 对接收到的点云进行预处理，完成分类（1.按照不同线，保存点云；2.对其进行特征分类）
void laserCloudHandler(const sensor_msgs::PointCloud2ConstPtr& laserCloudMsg)
{
  /* 延迟systemDelay帧数据后读取，以保证传感器都正常工作后进行下一步 */
  if (!systemInited) { // 丢弃前20个点云数据
    systemInitCount++;
    if (systemInitCount >= systemDelay) {
      systemInited = true;
    }
    return;
  }

  /* 记录每个scan 有曲率点的起始序号索引 */
  std::vector<int> scanStartInd(N_SCANS, 0); // 起始位置
  std::vector<int> scanEndInd(N_SCANS, 0); // 终止位置
  
  double timeScanCur = laserCloudMsg->header.stamp.toSec(); // 获取时间戳，toSec()转化成相应的浮点秒数，timeScanCur是当前点云帧的起始时刻

  /* 剔除异常点 */
  pcl::PointCloud<pcl::PointXYZ> laserCloudIn;
  pcl::fromROSMsg(*laserCloudMsg, laserCloudIn); // 将ros消息转换成pcl点云
  std::vector<int> indices;
  pcl::removeNaNFromPointCloud(laserCloudIn, laserCloudIn, indices); // 剔除坐标中包含NaN的无效点
  int cloudSize = laserCloudIn.points.size(); // 点云的点数

  /*
    获取点云的开始和结束水平角度, 确定一帧中点的角度范围
    此处需要注意一帧扫描角度不一定<2pi, 可能大于2pi, 角度需特殊处理
    角度范围用于确定每个点的相对扫描时间, 用于运动补偿
  */

  /*
    反正切函数 atan2() 和正切函数 tan() 的功能恰好相反：tan() 是已知一个角的弧度值，求该角的正切值；
    而 atan2() 是已知一个角的正切值（也就是 y/x），求该角的弧度值。
    atan2 以弧度表示的 y/x 的反正切的值，取值范围介于 -pi 到 pi 之间（不包括 -pi）
    而atan(a/b)的取值范围介于-pi/2到pi/2之间（不包括 ±pi/2)
  */

  // 扫描开始点的旋转角，atan2范围[-pi, +pi]，计算旋转角时取负号是因为velodyne是顺时针增大, 而坐标轴中的yaw是逆时针增加, 所以这里要取负号
  float startOri = -atan2(laserCloudIn.points[0].y, laserCloudIn.points[0].x);
  // 扫描结束点的旋转角，加2*pi使点云旋转周期为2*pi
  float endOri = -atan2(laserCloudIn.points[cloudSize - 1].y, laserCloudIn.points[cloudSize - 1].x) + 2 * M_PI;

  // 结束方位角与开始方位角差值控制在(PI,3*PI)范围，允许lidar不是一个圆周扫描
  // 正常情况下在这个范围内：pi < endOri - startOri < 3*pi，异常则修正
  if (endOri - startOri > 3 * M_PI) {
    endOri -= 2 * M_PI;
  } else if (endOri - startOri < M_PI) {
    endOri += 2 * M_PI;
  }

  /*
     三维扫描仪并不像二维那样按照角度给出距离值，从而保证每次的扫描都有相同的数据量。
     PointCloud2接受到的点云的大小在变化，因此在数据到达时需要一些运算来判断点的一些特征。
  */

  bool halfPassed = false;
  int count = cloudSize;
  PointType point;
  std::vector<pcl::PointCloud<PointType> > laserCloudScans(N_SCANS); // 每一线存储为一个单独的线(SCAN), 针对单个线计算特征点
  
  // 根据几何角度(竖直)，把激光点分配到线中
  for (int i = 0; i < cloudSize; i++) {
    // 将ros坐标系转换为欧拉角坐标系
    point.x = laserCloudIn.points[i].y;
    point.y = laserCloudIn.points[i].z;
    point.z = laserCloudIn.points[i].x;

    float angle = atan(point.y / sqrt(point.x * point.x + point.z * point.z)) * 180 / M_PI; // 俯仰角，上正下负
    int scanID; // 由竖直角度映射而得
    int roundedAngle = int(angle + (angle < 0.0 ? -0.5 : +0.5)); // 计算激光点垂直角，加减0.5实现四舍五入
    /*
      Laser ID    |    Vertical Angle
        0                  -15°
        1                   1°
        2                  -13°
        3                   3°
        4                  -11°
        5                   5°
        6                  -9°         
        7                   7°
        8                  -7°
        9                   9°
        10                 -5°
        11                  11°
        12                 -3°
        13                  13°
        14                 -1°
        15                  15°
    */
	
    // 根据上表进行对应激光点所述的SCAN行号
    if (roundedAngle > 0){
      scanID = roundedAngle; // 该点对应的scanID
    }
    else {
      scanID = roundedAngle + (N_SCANS - 1);
    } // 角度大于零，由小到大划入偶数线（0->16）；角度小于零，由大到小划入奇数线(15->1)

    // 过滤点，只挑选[-15度，+15度]范围内的点,scanID属于[0,15]，剔除16线以外的杂点
    if (scanID > (N_SCANS - 1) || scanID < 0 ){
      count--; // 将16线以外的杂点剔除
      continue;
    }

    float ori = -atan2(point.x, point.z); // 计算该点的水平旋转角
    if (!halfPassed) { // 根据扫描线是否旋转过半选择与起始位置还是终止位置进行差值计算，从而进行补偿
      // 确保-pi/2 < ori - startOri < 3*pi/2
      if (ori < startOri - M_PI / 2) {
        ori += 2 * M_PI;
      } else if (ori > startOri + M_PI * 3 / 2) {
        ori -= 2 * M_PI;
      }
      // 判断是否过半
      if (ori - startOri > M_PI) {
        halfPassed = true;
      }
    } else {
      ori += 2 * M_PI;

      // 确保-3*pi/2 < ori - endOri < pi/2
      if (ori < endOri - M_PI * 3 / 2) {
        ori += 2 * M_PI;
      } else if (ori > endOri + M_PI / 2) {
        ori -= 2 * M_PI;
      } 
    }
    // 插补计算时间，并存在属性里，整数位为行号，小数位为起始点到当前点的时间差
    // 该点在一帧数据中的相对时间，-0.5 < relTime < 1.5（点旋转的角度与整个周期旋转角度的比率, 即点云中点的相对时间）
    float relTime = (ori - startOri) / (endOri - startOri);
    // 点强度=线号+点相对时间（即一个整数+一个小数，整数部分是线号，小数部分是该点的相对时间）,匀速扫描：根据当前扫描的角度和扫描周期计算相对扫描起始位置的时间
    point.intensity = scanID + scanPeriod * relTime; // 不是真的反射率

    // 插入imu数据，假设机器人匀速运动，因此需要先消除激光点imu加速度影响
    if (imuPointerLast >= 0) {
      float pointTime = relTime * scanPeriod; // 当前点在该帧数据中的偏移时刻，其中第一个点的时刻为0，最后一个点的时刻为scanPeriod
      // 当前激光点和imu数据的时间同步，即在imu数组中找到与当前激光点时间一致的索引imuPointerFront，imuPointerLast表示imu循环数组最新索引。
      // 如果当前激光点的绝对时间大于imuPointerFront对应的imu时间，则将imuPointerFront向imuPointerLast靠拢。
      // 如果激光点时间比imu最新数据对应的时间还大，即imuPointerFront=imuPointerLast，则与之对应的imu就选最新值。
	  // 寻找是否有当前点的时间戳小于IMU的时间戳的IMU数据位置:imuPointerFront，以保证IMU数据信息可用
      while (imuPointerFront != imuPointerLast) {
        if (timeScanCur + pointTime < imuTime[imuPointerFront]) {
          break; // 遍历imu数组寻找是否有imu的时间戳大于当前点的时间戳，imu位置:imuPointerFront，如果有，则开始修正
        }
        imuPointerFront = (imuPointerFront + 1) % imuQueLength;
      }
	  
      // 如果没找到,此时imuPointerFront==imuPointerLast,
      // 只能以当前收到的最新的IMU的速度，位移，欧拉角作为当前点的速度，位移，欧拉角使用
      if (timeScanCur + pointTime > imuTime[imuPointerFront]) {
        imuRollCur = imuRoll[imuPointerFront];
        imuPitchCur = imuPitch[imuPointerFront];
        imuYawCur = imuYaw[imuPointerFront];

        imuVeloXCur = imuVeloX[imuPointerFront];
        imuVeloYCur = imuVeloY[imuPointerFront];
        imuVeloZCur = imuVeloZ[imuPointerFront];

        imuShiftXCur = imuShiftX[imuPointerFront];
        imuShiftYCur = imuShiftY[imuPointerFront];
        imuShiftZCur = imuShiftZ[imuPointerFront];
      } else { // 否则插值前后两个imu数据点，得到与雷达数据点对应的imu参数
        int imuPointerBack = (imuPointerFront + imuQueLength - 1) % imuQueLength;
        // 按时间距离计算权重分配比率，即线性插值
        float ratioFront = (timeScanCur + pointTime - imuTime[imuPointerBack]) 
                         / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);
        float ratioBack = (imuTime[imuPointerFront] - timeScanCur - pointTime) 
                        / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);

        imuRollCur = imuRoll[imuPointerFront] * ratioFront + imuRoll[imuPointerBack] * ratioBack;
        imuPitchCur = imuPitch[imuPointerFront] * ratioFront + imuPitch[imuPointerBack] * ratioBack;
        if (imuYaw[imuPointerFront] - imuYaw[imuPointerBack] > M_PI) { // 保证航向角按照小的一侧进行插值
          imuYawCur = imuYaw[imuPointerFront] * ratioFront + (imuYaw[imuPointerBack] + 2 * M_PI) * ratioBack;
        } else if (imuYaw[imuPointerFront] - imuYaw[imuPointerBack] < -M_PI) {
          imuYawCur = imuYaw[imuPointerFront] * ratioFront + (imuYaw[imuPointerBack] - 2 * M_PI) * ratioBack;
        } else {
          imuYawCur = imuYaw[imuPointerFront] * ratioFront + imuYaw[imuPointerBack] * ratioBack;
        }
        // 速度，本质:imuVeloXCur = imuVeloX[imuPointerback] + (imuVelX[imuPointerFront]-imuVelX[imuPoniterBack])*ratioFront
        imuVeloXCur = imuVeloX[imuPointerFront] * ratioFront + imuVeloX[imuPointerBack] * ratioBack;
        imuVeloYCur = imuVeloY[imuPointerFront] * ratioFront + imuVeloY[imuPointerBack] * ratioBack;
        imuVeloZCur = imuVeloZ[imuPointerFront] * ratioFront + imuVeloZ[imuPointerBack] * ratioBack;
        // 位移
        imuShiftXCur = imuShiftX[imuPointerFront] * ratioFront + imuShiftX[imuPointerBack] * ratioBack;
        imuShiftYCur = imuShiftY[imuPointerFront] * ratioFront + imuShiftY[imuPointerBack] * ratioBack;
        imuShiftZCur = imuShiftZ[imuPointerFront] * ratioFront + imuShiftZ[imuPointerBack] * ratioBack;
      }

      // 如果是第一个点，记住点云起始位置的速度，位移，欧拉角
      if (i == 0) {
        imuRollStart = imuRollCur;
        imuPitchStart = imuPitchCur;
        imuYawStart = imuYawCur;

        imuVeloXStart = imuVeloXCur;
        imuVeloYStart = imuVeloYCur;
        imuVeloZStart = imuVeloZCur;

        imuShiftXStart = imuShiftXCur;
        imuShiftYStart = imuShiftYCur;
        imuShiftZStart = imuShiftZCur;
      } else {
       // 计算之后每个点相对于第一个点的由于加减速非匀速运动产生的位移、速度畸变，
       // 并对点云中的每个点位置信息重新补偿矫正
       // Lidar位移、速度转移到IMU起始坐标系下
        ShiftToStartIMU(pointTime);
        VeloToStartIMU();
        TransformToStartIMU(&point);
      }
    }

    laserCloudScans[scanID].push_back(point); 
  }
  
  // 更新点云的数量，之前剔除了不是0-15行的一些点，所以数量会降低
  cloudSize = count; // 正负15度范围内的点数

  // 更新总的点云laserCloud
  pcl::PointCloud<PointType>::Ptr laserCloud(new pcl::PointCloud<PointType>());
  
  // 将16个环的点云拼接，按照行号放入一个容器
  for (int i = 0; i < N_SCANS; i++) {
    *laserCloud += laserCloudScans[i];
  }

  /*
    针对按线分类后的点云，通过激光点左右各5个点进行曲率计算
    曲率计算方法为求相邻10个点与当前点i在x/y/z方向的坐标差，并求平方和，存放在cloudCurvature[i]
    通过当前点intensity中存放的scanID信息，求每一个线的起始点和终止点索引，其中第一线的起始索引和最后线的终止索引直接赋值即可
  */
  int scanCount = -1;
  // 计算去除 前五个点  后五个点的点云的曲率
  // 该点与周围10个点的偏差，参考论文公式
  for (int i = 5; i < cloudSize - 5; i++) {
    float diffX = laserCloud->points[i - 5].x + laserCloud->points[i - 4].x 
                + laserCloud->points[i - 3].x + laserCloud->points[i - 2].x 
                + laserCloud->points[i - 1].x - 10 * laserCloud->points[i].x 
                + laserCloud->points[i + 1].x + laserCloud->points[i + 2].x
                + laserCloud->points[i + 3].x + laserCloud->points[i + 4].x
                + laserCloud->points[i + 5].x;
    float diffY = laserCloud->points[i - 5].y + laserCloud->points[i - 4].y 
                + laserCloud->points[i - 3].y + laserCloud->points[i - 2].y 
                + laserCloud->points[i - 1].y - 10 * laserCloud->points[i].y 
                + laserCloud->points[i + 1].y + laserCloud->points[i + 2].y
                + laserCloud->points[i + 3].y + laserCloud->points[i + 4].y
                + laserCloud->points[i + 5].y;
    float diffZ = laserCloud->points[i - 5].z + laserCloud->points[i - 4].z 
                + laserCloud->points[i - 3].z + laserCloud->points[i - 2].z 
                + laserCloud->points[i - 1].z - 10 * laserCloud->points[i].z 
                + laserCloud->points[i + 1].z + laserCloud->points[i + 2].z
                + laserCloud->points[i + 3].z + laserCloud->points[i + 4].z
                + laserCloud->points[i + 5].z;

    /*
      计算以某点与其相邻的10个点所构成的平面在该点出的曲率：
      由曲率公式知：K=1/R，因此为简化计算可通过10个向量的和向量的模长表示其在该点处曲率半径的长，因此R×R可用来表示曲率的大小，
      R×R越大，该点处越不平坦。
    */

    cloudCurvature[i] = diffX * diffX + diffY * diffY + diffZ * diffZ; // 当前点的曲率
    cloudSortInd[i] = i; // 当前点在点云中索引
    cloudNeighborPicked[i] = 0; // 用于标记是否可作为特征点
    cloudLabel[i] = 0; // 用于标记特征点为边界线还是平面特征点

    // 计算每一线的起始点和终止点，即扣除每一线始末各5个点
    // 每个scan，只有第一个符合的点会进来，因为每个scan的点的intensity数据结构里存放的是扫描时间，同一scan的整数部分是一样的，但是15度和0度一样，14度和-1度一样，1度和-14度一样
    if (int(laserCloud->points[i].intensity) != scanCount) {
      scanCount = int(laserCloud->points[i].intensity); // 更新本次scan的scanID，遇到下次scan的第一个点时，会再次更新
      if (scanCount > 0 && scanCount < N_SCANS) { // 点云的排列方式是水平的，一次scan内的点，竖直的点由于分辨率很大，不认为信息足够作为特征点提取
        scanStartInd[scanCount] = i + 5; // 该scan的起始点位置的索引（滤出前5个点）
        scanEndInd[scanCount - 1] = i - 5; // 该scan的终止点位置的索引(滤出后5个点)
      }
    }
  }

  // 曲率计算完毕
  scanStartInd[0] = 5; // 第一条线的起始位置
  scanEndInd.back() = cloudSize - 5; // 最后一条线的终止位置

  /*
  	排除瑕疵点：避免周围点已被选择从而保证特征点分布均匀，或者局部平行于激光束的局部平面上的点和 被遮挡点
    瑕疵点的筛选条件：1.平面/直线与激光近似平行的点不能要；2.被遮挡的边缘点不能要
    遍历所有点（除去前五个和后六个），判断该点及其周边点是否可以作为特征点位，
    当某点及其后点间的距离平方大于某阈值a（说明这两点有一定距离），且两向量夹角小于某阈值b时（夹角小就可能存在遮挡），将其一侧的临近6个点设为不可标记为特征点的点；
    若某点到其前后两点的距离均大于c倍的该点深度，则该点判定为不可标记特征点的点（入射角越小，点间距越大，即激光发射方向与投射到的平面越近似水平）。
  */

  /*
    判断当前点和周边点是否可以作为特征点
    选其点云前后各5个点，逐个计算当前激光点与后一个激光点的距离平方，如果距离平方diff大于阈值，则表示可能存在遮挡。
    进一步计算两个激光点的深度信息depth1和depth2，如果depth1>depth2，则当前点可能被遮挡。
    进一步以短边depth2构成等腰三角形并计算三角形的底边长，根据θ≈sin(θ)≈底/边，求得当前激光点与后一点的夹角，
    如果夹角小于阈值，则认为存在遮挡，从而被遮挡的当前激光点及其往前5个点不可作为特征点。
    同理，如果depth1<depth2，则后一点可能被遮挡。
    考虑完遮挡点情况后，再考虑激光入射角，即论文fig4(a)所示情况。
    如果当前点与前后两个点的距离都大于c倍当前点深度，则认为当前点入射角与物理表面平行，当前点标记为不可作特征点
  */
  for (int i = 5; i < cloudSize - 6; i++) { // 涉及到与后一个点i+1差值，所以cloudSize-6不是-5
    float diffX = laserCloud->points[i + 1].x - laserCloud->points[i].x;
    float diffY = laserCloud->points[i + 1].y - laserCloud->points[i].y;
    float diffZ = laserCloud->points[i + 1].z - laserCloud->points[i].z;
    float diff = diffX * diffX + diffY * diffY + diffZ * diffZ; // 有效曲率点与后一个点之间的距离平方和

    if (diff > 0.1) { // 前后两点距离的平方大于阈值，两个点之间距离要大于0.1（sqrt(0.1)=0.32m）
      float depth1 = sqrt(laserCloud->points[i].x * laserCloud->points[i].x + 
                     laserCloud->points[i].y * laserCloud->points[i].y +
                     laserCloud->points[i].z * laserCloud->points[i].z); // 前一个点的深度值

      float depth2 = sqrt(laserCloud->points[i + 1].x * laserCloud->points[i + 1].x + 
                     laserCloud->points[i + 1].y * laserCloud->points[i + 1].y +
                     laserCloud->points[i + 1].z * laserCloud->points[i + 1].z); // 后一个点的深度值

      // 论文中fig4中(b)情况，被遮挡的边缘点不能要
      if (depth1 > depth2) { // 当前点可能被遮挡
        // 以短边i+1构成等腰三角形，diffX/diffY/diffZ平方表示等腰三角形的底
        diffX = laserCloud->points[i + 1].x - laserCloud->points[i].x * depth2 / depth1;
        diffY = laserCloud->points[i + 1].y - laserCloud->points[i].y * depth2 / depth1;
        diffZ = laserCloud->points[i + 1].z - laserCloud->points[i].z * depth2 / depth1;

        // 实际表示i与i+1夹角小于5.732度，sin(5.732) ~= 0.1，认为被遮挡
        if (sqrt(diffX * diffX + diffY * diffY + diffZ * diffZ) / depth2 < 0.1) {
          // 边长比即弧度值，若小于0.1，说明夹角小于5.73度，斜面陡峭，点深度变化剧烈，点处在近似与激光束平行的斜面上
          // 当前点i被遮挡，则i及其往前5个点都不能作为特征点
          // cloudNeighborPicked 是考虑一个特征点周围不能再设置成特征约束的判断标志位
          cloudNeighborPicked[i - 5] = 1;
          cloudNeighborPicked[i - 4] = 1;
          cloudNeighborPicked[i - 3] = 1;
          cloudNeighborPicked[i - 2] = 1;
          cloudNeighborPicked[i - 1] = 1;
          cloudNeighborPicked[i] = 1;
        }
      } else { // 后一点可能被遮挡
        diffX = laserCloud->points[i + 1].x * depth1 / depth2 - laserCloud->points[i].x;
        diffY = laserCloud->points[i + 1].y * depth1 / depth2 - laserCloud->points[i].y;
        diffZ = laserCloud->points[i + 1].z * depth1 / depth2 - laserCloud->points[i].z;

        if (sqrt(diffX * diffX + diffY * diffY + diffZ * diffZ) / depth1 < 0.1) {
          // 边长比即弧度值，若小于0.1，说明夹角小于5.73度，斜面陡峭，点深度变化剧烈，点处在近似与激光束平行的斜面上
          // 当前点后一点i+1被遮挡，则i+1及其往前5个点都不能作为特征点
          cloudNeighborPicked[i + 1] = 1;
          cloudNeighborPicked[i + 2] = 1;
          cloudNeighborPicked[i + 3] = 1;
          cloudNeighborPicked[i + 4] = 1;
          cloudNeighborPicked[i + 5] = 1;
          cloudNeighborPicked[i + 6] = 1;
        }
      }
    }

    // 论文fig4中(a)情况，平面/直线与激光近似平行的点不能要
    float diffX2 = laserCloud->points[i].x - laserCloud->points[i - 1].x;
    float diffY2 = laserCloud->points[i].y - laserCloud->points[i - 1].y;
    float diffZ2 = laserCloud->points[i].z - laserCloud->points[i - 1].z;
    float diff2 = diffX2 * diffX2 + diffY2 * diffY2 + diffZ2 * diffZ2; // 当前点与前一个点的距离平方和

    float dis = laserCloud->points[i].x * laserCloud->points[i].x
              + laserCloud->points[i].y * laserCloud->points[i].y
              + laserCloud->points[i].z * laserCloud->points[i].z; // 当前点深度平方和

    if (diff > 0.0002 * dis && diff2 > 0.0002 * dis) {
      cloudNeighborPicked[i] = 1; // 当前点入射角与物理表面平行，当前点标记为不可作特征点
    }
  }

  pcl::PointCloud<PointType> cornerPointsSharp;
  pcl::PointCloud<PointType> cornerPointsLessSharp;
  pcl::PointCloud<PointType> surfPointsFlat;
  pcl::PointCloud<PointType> surfPointsLessFlat;

  /*
    特征点分类，上一步剔除特征不符的点后，从剩下的激光点中选择平面特征点/边缘特征点
    特征点按照每一线进行选取，首先将该线分为6段，根据每一线的始末激光点索引计算6段的始末点索引，然后每一段内部激光点按照曲率升序排序
  */
  for (int i = 0; i < N_SCANS; i++) { // 按线处理
    pcl::PointCloud<PointType>::Ptr surfPointsLessFlatScan(new pcl::PointCloud<PointType>);
    // 将每个线等分为六段，分别进行处理（sp、ep分别为各段的起始和终止位置）
    for (int j = 0; j < 6; j++) {
      // 各段起始位置：sp = scanStartInd + (scanEndInd - scanStartInd)*j/6
      int sp = (scanStartInd[i] * (6 - j)  + scanEndInd[i] * j) / 6;
      // 各段终止位置：ep = scanStartInd - 1 + (scanEndInd - scanStartInd)*(j+1)/6
      int ep = (scanStartInd[i] * (5 - j)  + scanEndInd[i] * (j + 1)) / 6 - 1;

      // 将曲率排序--从小到大  冒泡排序法
      for (int k = sp + 1; k <= ep; k++) {
        for (int l = k; l >= sp + 1; l--) {
          // 如果后面曲率点大于前面，则交换
          if (cloudCurvature[cloudSortInd[l]] < cloudCurvature[cloudSortInd[l - 1]]) {
            int temp = cloudSortInd[l - 1];
            cloudSortInd[l - 1] = cloudSortInd[l];
            cloudSortInd[l] = temp;
          }
        }
      }

      // 选取角点，从每一段最大曲率（每一段末尾）处往前判断，挑选每个分段的曲率很大和比较大的点，作为边缘特征
      int largestPickedNum = 0;
      for (int k = ep; k >= sp; k--) {
        int ind = cloudSortInd[k];  // 排序后的点在点云中的索引
        if (cloudNeighborPicked[ind] == 0 && cloudCurvature[ind] > 0.1) { // 没有被标记为不可取特征点且曲率大于0.1
          largestPickedNum++;
          // 挑选曲率最大的前2个点放入sharp点集合
          if (largestPickedNum <= 2) { // 2个特征点
            cloudLabel[ind] = 2; // 2代表点曲率很大，最优
            cornerPointsSharp.push_back(laserCloud->points[ind]);
            cornerPointsLessSharp.push_back(laserCloud->points[ind]);
          } else if (largestPickedNum <= 20) { // 挑选曲率最大的前20个点放入less sharp点集合
            cloudLabel[ind] = 1; // 1代表点曲率比较尖锐，次优
            cornerPointsLessSharp.push_back(laserCloud->points[ind]);
          } else {
            break; // 20个点以上不要，每个区域只取2+20个
          }

          cloudNeighborPicked[ind] = 1; // 该点被选为特征点后，标记为已选择
		  
		  // 将曲率比较大的点的前后各5个连续距离比较近的点筛选出去，以防止特征点聚集，使得特征点在每个方向上尽量分布均匀
          // 对ind点周围的点是否能作为特征点进行判断，除非距离大于阈值，否则前后各5各点不能作为特征点
          for (int l = 1; l <= 5; l++) { // 后5个点
            float diffX = laserCloud->points[ind + l].x 
                        - laserCloud->points[ind + l - 1].x;
            float diffY = laserCloud->points[ind + l].y 
                        - laserCloud->points[ind + l - 1].y;
            float diffZ = laserCloud->points[ind + l].z 
                        - laserCloud->points[ind + l - 1].z;
            if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05) { // 相邻两点距离大于阈值，则不用标记
              break;
            }

            cloudNeighborPicked[ind + l] = 1; // 否则标记为不可用特征点
          }
          for (int l = -1; l >= -5; l--) { // 前5个点
            float diffX = laserCloud->points[ind + l].x 
                        - laserCloud->points[ind + l + 1].x;
            float diffY = laserCloud->points[ind + l].y 
                        - laserCloud->points[ind + l + 1].y;
            float diffZ = laserCloud->points[ind + l].z 
                        - laserCloud->points[ind + l + 1].z;
            if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05) {
              break;
            }

            cloudNeighborPicked[ind + l] = 1;
          }
        }
      }

      /*
        对于平面特征点，通过每一段线中曲率小端开始查找。
        如果当前点没有被标记且曲率小于阈值，则该点为FlatPoints，保存在surfPointsFlat中，平面特征点每段提取4个。
        同样的，需要判断前后各5个点是否需要标记为不可用特征点。
        最后，除了被标记过的点(本身不能作为特征点以及被标记为边界线的特征点)，其它点都作为lessFlatPoints存储到surfPointsLessFlatScan中，并进一步滤波降采样(lessFlat点太多)，最后存储到surfPointsLessFlat中。
      */

      // 选取平面点，从每一段曲率最小（前端）开始查找，用于确定平面点
      int smallestPickedNum = 0;
      for (int k = sp; k <= ep; k++) {
        int ind = cloudSortInd[k];
        if (cloudNeighborPicked[ind] == 0 && cloudCurvature[ind] < 0.1) { // 没有被标记为不可取特征点，且曲率小于0.1
          cloudLabel[ind] = -1; // -1代表曲率很小的点
          surfPointsFlat.push_back(laserCloud->points[ind]);

          smallestPickedNum++;
          if (smallestPickedNum >= 4) { // 只选最小的四个，剩下的Label==0,就都是曲率比较小的
            break;
          }

		  // 同样防止特征点聚集
          cloudNeighborPicked[ind] = 1;
          // 对ind点周围的点是否能作为特征点进行判断
          for (int l = 1; l <= 5; l++) {
            float diffX = laserCloud->points[ind + l].x 
                        - laserCloud->points[ind + l - 1].x;
            float diffY = laserCloud->points[ind + l].y 
                        - laserCloud->points[ind + l - 1].y;
            float diffZ = laserCloud->points[ind + l].z 
                        - laserCloud->points[ind + l - 1].z;
            if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05) {
              break;
            }

            cloudNeighborPicked[ind + l] = 1;
          }
          for (int l = -1; l >= -5; l--) {
            float diffX = laserCloud->points[ind + l].x 
                        - laserCloud->points[ind + l + 1].x;
            float diffY = laserCloud->points[ind + l].y 
                        - laserCloud->points[ind + l + 1].y;
            float diffZ = laserCloud->points[ind + l].z 
                        - laserCloud->points[ind + l + 1].z;
            if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05) {
              break;
            }

            cloudNeighborPicked[ind + l] = 1;
          }
        }
      }

      // 最后把剩余点（包括之前被排除的点）全部归入平面点中less flat类别中
      for (int k = sp; k <= ep; k++) {
        if (cloudLabel[k] <= 0) {
          surfPointsLessFlatScan->push_back(laserCloud->points[k]);
        }
      }
      // 一个六等分段处理完毕
    }

    // 由于less flat点最多，对每个分段less flat的点进行降采样，简化了点的数量，又保证了整体点云的基本形状
    pcl::PointCloud<PointType> surfPointsLessFlatScanDS;
    pcl::VoxelGrid<PointType> downSizeFilter;
    downSizeFilter.setInputCloud(surfPointsLessFlatScan);
    downSizeFilter.setLeafSize(0.2, 0.2, 0.2);
    downSizeFilter.filter(surfPointsLessFlatScanDS);
    
    surfPointsLessFlat += surfPointsLessFlatScanDS;
    // 一个scan处理完毕
  }

  // 所有scan处理完毕，发布不同的点云消息包括： 原始、最优边缘、次优边缘、最优平面、次优平
  // 发布按线分类后的点云，原始点
  sensor_msgs::PointCloud2 laserCloudOutMsg;
  pcl::toROSMsg(*laserCloud, laserCloudOutMsg);
  laserCloudOutMsg.header.stamp = laserCloudMsg->header.stamp;
  laserCloudOutMsg.header.frame_id = "/camera";
  pubLaserCloud.publish(laserCloudOutMsg);

  // 发布边界线上锐角特征点云，最优边缘
  sensor_msgs::PointCloud2 cornerPointsSharpMsg;
  pcl::toROSMsg(cornerPointsSharp, cornerPointsSharpMsg);
  cornerPointsSharpMsg.header.stamp = laserCloudMsg->header.stamp;
  cornerPointsSharpMsg.header.frame_id = "/camera";
  pubCornerPointsSharp.publish(cornerPointsSharpMsg);

  // 发布边界线上钝角特征点云，次优边缘
  sensor_msgs::PointCloud2 cornerPointsLessSharpMsg;
  pcl::toROSMsg(cornerPointsLessSharp, cornerPointsLessSharpMsg);
  cornerPointsLessSharpMsg.header.stamp = laserCloudMsg->header.stamp;
  cornerPointsLessSharpMsg.header.frame_id = "/camera";
  pubCornerPointsLessSharp.publish(cornerPointsLessSharpMsg);

  // 发布平面上特征点云，最优平面
  sensor_msgs::PointCloud2 surfPointsFlat2;
  pcl::toROSMsg(surfPointsFlat, surfPointsFlat2);
  surfPointsFlat2.header.stamp = laserCloudMsg->header.stamp;
  surfPointsFlat2.header.frame_id = "/camera";
  pubSurfPointsFlat.publish(surfPointsFlat2);

  // 发布平面上不太平的特征点云，次优平
  sensor_msgs::PointCloud2 surfPointsLessFlat2;
  pcl::toROSMsg(surfPointsLessFlat, surfPointsLessFlat2);
  surfPointsLessFlat2.header.stamp = laserCloudMsg->header.stamp;
  surfPointsLessFlat2.header.frame_id = "/camera";
  pubSurfPointsLessFlat.publish(surfPointsLessFlat2);

  // 发布imu消息,由于循环到了最后，因此是最后一个点的欧拉角，畸变位移及一个点云周期增加的速度
  pcl::PointCloud<pcl::PointXYZ> imuTrans(4, 1);
  // 起始点欧拉角
  imuTrans.points[0].x = imuPitchStart;
  imuTrans.points[0].y = imuYawStart;
  imuTrans.points[0].z = imuRollStart;

  // 最后一个点的欧拉角
  imuTrans.points[1].x = imuPitchCur;
  imuTrans.points[1].y = imuYawCur;
  imuTrans.points[1].z = imuRollCur;

  // 最后一个点相对于第一个点的畸变位移和速度
  imuTrans.points[2].x = imuShiftFromStartXCur;
  imuTrans.points[2].y = imuShiftFromStartYCur;
  imuTrans.points[2].z = imuShiftFromStartZCur;

  imuTrans.points[3].x = imuVeloFromStartXCur;
  imuTrans.points[3].y = imuVeloFromStartYCur;
  imuTrans.points[3].z = imuVeloFromStartZCur;

  // 发布处理后的imu数据
  sensor_msgs::PointCloud2 imuTransMsg;
  pcl::toROSMsg(imuTrans, imuTransMsg);
  imuTransMsg.header.stamp = laserCloudMsg->header.stamp;
  imuTransMsg.header.frame_id = "/camera";
  pubImuTrans.publish(imuTransMsg);
}

/*
  接收imu消息
  首先读取方位角四元素到orientation，
  然后转换为欧拉角roll/pitch/yaw，
  接着计算消除重力加速度影响的各个方向的线加速度，此时加速度是在imu自身坐标系中，imuPointerLast作为imu各个参数数组的索引，循环累加，
  然后将当前imu数据及对应的时刻保存在各自数组中，
  最后调用AccumulateIMUShift将imu转换到局部地球坐标系。
*/
void imuHandler(const sensor_msgs::Imu::ConstPtr& imuIn)
{
  double roll, pitch, yaw;
  tf::Quaternion orientation;
  tf::quaternionMsgToTF(imuIn->orientation, orientation); // 将Quaternion消息转换为tf的Quaternion数据
  tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw); // 将orientation转换为欧拉角roll/pitch/yaw

  // 减去重力的影响，求出xyz方向的加速度实际值，并进行坐标轴交换，统一到z轴向前，x轴向左的右手坐标系, 交换过后RPY对应fixed axes ZXY(RPY---ZXY)。Now R = Ry(yaw)*Rx(pitch)*Rz(roll).
  float accX = imuIn->linear_acceleration.y - sin(roll) * cos(pitch) * 9.81;
  float accY = imuIn->linear_acceleration.z - cos(roll) * cos(pitch) * 9.81;
  float accZ = imuIn->linear_acceleration.x + sin(pitch) * 9.81;

  // 循环移位效果，形成环形数组
  imuPointerLast = (imuPointerLast + 1) % imuQueLength;

  imuTime[imuPointerLast] = imuIn->header.stamp.toSec();
  imuRoll[imuPointerLast] = roll;
  imuPitch[imuPointerLast] = pitch;
  imuYaw[imuPointerLast] = yaw;
  imuAccX[imuPointerLast] = accX;
  imuAccY[imuPointerLast] = accY;
  imuAccZ[imuPointerLast] = accZ;
  // 位姿估算 
  AccumulateIMUShift();
}

/*
  首先注册scanRegistration节点，
  然后订阅雷达话题/velodyne_points和惯导话题/imu/data，
  接着发布按线分类后的点云、边界特征点点云、面特征特征点点云以及处理后的imu数据
*/
int main(int argc, char** argv)
{
  ros::init(argc, argv, "scanRegistration"); // 注册scanRegistration节点
  ros::NodeHandle nh; // 创建管理节点句柄

  ros::Subscriber subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>("/velodyne_points", 2, laserCloudHandler); // 订阅雷达数据，处理队列大小为2，laserCloudHandler回调函数进行处理
  ros::Subscriber subImu = nh.subscribe<sensor_msgs::Imu> ("/imu/data", 50, imuHandler); // 订阅imu数据，处理队列大小为50，imuHandler回调函数进行处理

  pubLaserCloud = nh.advertise<sensor_msgs::PointCloud2>("/velodyne_cloud_2", 2); // 发布按线分类后的点云
  pubCornerPointsSharp = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_sharp", 2); // 发布边缘锐角特征点云
  pubCornerPointsLessSharp = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_less_sharp", 2); // 发布边缘钝角特征点云
  pubSurfPointsFlat = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_flat", 2); // 发布平面上特征点云
  pubSurfPointsLessFlat = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_less_flat", 2); // 发布平面上不太平的特征点云
  pubImuTrans = nh.advertise<sensor_msgs::PointCloud2> ("/imu_trans", 5); // 发布处理后的imu数据

  /*
    在使用ros::spin()的情况下，一般来说初始化时已经设置好所有消息的回调，并且不需要其他背景程序运行。
    这样一来消息每次到达时会执行用户的回调函数进行操作，相当于程序是消息事件驱动的。
    而在使用spinOnce()的情况下，一般来说仅仅使用回调函数不足以完成任务，还需要其他辅助程序的执行：比如定时任务，数据处理，用户界面等。
    二者最大的区别在于spin()调用后不会返回，而spinOnce()调用后还可以继续执行之后的程序。
  */
  ros::spin();

  return 0;
}
