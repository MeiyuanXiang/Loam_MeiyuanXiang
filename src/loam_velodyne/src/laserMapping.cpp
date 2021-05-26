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

#include <math.h>

#include <loam_velodyne/common.h>
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

const float scanPeriod = 0.1; // 扫描周期

const int stackFrameNum = 1; // 控制接收到的点云数据，每隔1帧处理一次
const int mapFrameNum = 5;   // 控制处理得到的点云map，每隔5次publich给rviz显示

// 时间戳
double timeLaserCloudCornerLast = 0;
double timeLaserCloudSurfLast = 0;
double timeLaserCloudFullRes = 0;
double timeLaserOdometry = 0;

// 接收标志
bool newLaserCloudCornerLast = false;
bool newLaserCloudSurfLast = false;
bool newLaserCloudFullRes = false;
bool newLaserOdometry = false;

int laserCloudCenWidth = 10; // 中心cube索引
int laserCloudCenHeight = 5; // 中心cube索引
int laserCloudCenDepth = 10; // 中心cube索引
const int laserCloudWidth = 21;
const int laserCloudHeight = 11;
const int laserCloudDepth = 21;

const int laserCloudNum = laserCloudWidth * laserCloudHeight * laserCloudDepth; // 点云方块集合最大数量，4851个长宽高为 cm × cm × cm 的立方体

int laserCloudValidInd[125]; // lidar视域范围内(FOV)的点云集索引
int laserCloudSurroundInd[125]; // lidar周围的点云集索引

// 最新接收到的边沿点
pcl::PointCloud<PointType>::Ptr laserCloudCornerLast(new pcl::PointCloud<PointType>());
// 最新接收到的平面点
pcl::PointCloud<PointType>::Ptr laserCloudSurfLast(new pcl::PointCloud<PointType>());
// 存放当前收到的下采样之后的边沿点(in the local frame)
pcl::PointCloud<PointType>::Ptr laserCloudCornerStack(new pcl::PointCloud<PointType>());
// 存放当前收到的下采样之后的平面点(in the local frame)
pcl::PointCloud<PointType>::Ptr laserCloudSurfStack(new pcl::PointCloud<PointType>());
// 存放当前收到的边沿点，作为下采样的数据源
pcl::PointCloud<PointType>::Ptr laserCloudCornerStack2(new pcl::PointCloud<PointType>());
// 存放当前收到的平面点，作为下采样的数据源
pcl::PointCloud<PointType>::Ptr laserCloudSurfStack2(new pcl::PointCloud<PointType>());
// 原始点云坐标
pcl::PointCloud<PointType>::Ptr laserCloudOri(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr coeffSel(new pcl::PointCloud<PointType>());
// 匹配使用的特征点（下采样之后的）
pcl::PointCloud<PointType>::Ptr laserCloudSurround(new pcl::PointCloud<PointType>());
// 匹配使用的特征点（下采样之前的）
pcl::PointCloud<PointType>::Ptr laserCloudSurround2(new pcl::PointCloud<PointType>());
// map中提取的匹配使用的边沿点
pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMap(new pcl::PointCloud<PointType>());
// map中提取的匹配使用的平面点
pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMap(new pcl::PointCloud<PointType>());
// 点云全部点
pcl::PointCloud<PointType>::Ptr laserCloudFullRes(new pcl::PointCloud<PointType>());
// array都是以50米为单位的立方体地图，运行过程中会一直保存(有需要的话可考虑优化，只保存近邻的，或者直接数组开小一点)
// 存放边沿点的cube
pcl::PointCloud<PointType>::Ptr laserCloudCornerArray[laserCloudNum];
// 存放平面点的cube
pcl::PointCloud<PointType>::Ptr laserCloudSurfArray[laserCloudNum];
// 中间变量，存放下采样过的边沿点
pcl::PointCloud<PointType>::Ptr laserCloudCornerArray2[laserCloudNum];
// 中间变量，存放下采样过的平面点
pcl::PointCloud<PointType>::Ptr laserCloudSurfArray2[laserCloudNum];

// kd-tree
pcl::KdTreeFLANN<PointType>::Ptr kdtreeCornerFromMap(new pcl::KdTreeFLANN<PointType>());
pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurfFromMap(new pcl::KdTreeFLANN<PointType>());

/*************高频转换量**************/
float transformSum[6] = {0}; // odometry计算得到的到世界坐标系下的转移矩阵
float transformIncre[6] = {0}; // 转移增量，只使用了后三个平移增量

/*************低频转换量*************/
float transformTobeMapped[6] = {0}; // 以起始位置为原点的世界坐标系下的转换矩阵（猜测与调整的对象）
float transformBefMapped[6] = {0}; // 存放mapping之前的Odometry计算的世界坐标系的转换矩阵（注：低频量，不一定与transformSum一样）
float transformAftMapped[6] = {0}; // 存放mapping之后的经过mapping微调之后的转换矩阵

int imuPointerFront = 0;
int imuPointerLast = -1;
const int imuQueLength = 200;

double imuTime[imuQueLength] = {0};
float imuRoll[imuQueLength] = {0};
float imuPitch[imuQueLength] = {0};

/*
  首先将每次地图匹配前的位姿transformBefMapped与订阅的初始位姿transformSum之间的平移偏差按照yxz顺序旋转到transformSum坐标系方向，得到transformIncre。
  然后和第二部分PluginIMURotation()函数一样，通过地图匹配前后的姿态transformBefMapped/transformAftMapped以及初始姿态transformSum计算得到地图匹配时transformTobeMapped的姿态。
*/
// 基于匀速模型，根据上次微调的结果和odometry这次与上次计算的结果，猜测一个新的世界坐标系的转换矩阵transformTobeMapped[6]
void transformAssociateToMap()
{
  float x1 = cos(transformSum[1]) * (transformBefMapped[3] - transformSum[3]) - sin(transformSum[1]) * (transformBefMapped[5] - transformSum[5]);
  float y1 = transformBefMapped[4] - transformSum[4];
  float z1 = sin(transformSum[1]) * (transformBefMapped[3] - transformSum[3]) + cos(transformSum[1]) * (transformBefMapped[5] - transformSum[5]);

  float x2 = x1;
  float y2 = cos(transformSum[0]) * y1 + sin(transformSum[0]) * z1;
  float z2 = -sin(transformSum[0]) * y1 + cos(transformSum[0]) * z1;

  // 平移增量
  transformIncre[3] = cos(transformSum[2]) * x2 + sin(transformSum[2]) * y2;
  transformIncre[4] = -sin(transformSum[2]) * x2 + cos(transformSum[2]) * y2;
  transformIncre[5] = z2;

  float sbcx = sin(transformSum[0]);
  float cbcx = cos(transformSum[0]);
  float sbcy = sin(transformSum[1]);
  float cbcy = cos(transformSum[1]);
  float sbcz = sin(transformSum[2]);
  float cbcz = cos(transformSum[2]);

  float sblx = sin(transformBefMapped[0]);
  float cblx = cos(transformBefMapped[0]);
  float sbly = sin(transformBefMapped[1]);
  float cbly = cos(transformBefMapped[1]);
  float sblz = sin(transformBefMapped[2]);
  float cblz = cos(transformBefMapped[2]);

  float salx = sin(transformAftMapped[0]);
  float calx = cos(transformAftMapped[0]);
  float saly = sin(transformAftMapped[1]);
  float caly = cos(transformAftMapped[1]);
  float salz = sin(transformAftMapped[2]);
  float calz = cos(transformAftMapped[2]);

  float srx = -sbcx*(salx*sblx + calx*cblx*salz*sblz + calx*calz*cblx*cblz)
            - cbcx*sbcy*(calx*calz*(cbly*sblz - cblz*sblx*sbly)
            - calx*salz*(cbly*cblz + sblx*sbly*sblz) + cblx*salx*sbly)
            - cbcx*cbcy*(calx*salz*(cblz*sbly - cbly*sblx*sblz) 
            - calx*calz*(sbly*sblz + cbly*cblz*sblx) + cblx*cbly*salx);
  transformTobeMapped[0] = -asin(srx);

  float srycrx = sbcx*(cblx*cblz*(caly*salz - calz*salx*saly)
               - cblx*sblz*(caly*calz + salx*saly*salz) + calx*saly*sblx)
               - cbcx*cbcy*((caly*calz + salx*saly*salz)*(cblz*sbly - cbly*sblx*sblz)
               + (caly*salz - calz*salx*saly)*(sbly*sblz + cbly*cblz*sblx) - calx*cblx*cbly*saly)
               + cbcx*sbcy*((caly*calz + salx*saly*salz)*(cbly*cblz + sblx*sbly*sblz)
               + (caly*salz - calz*salx*saly)*(cbly*sblz - cblz*sblx*sbly) + calx*cblx*saly*sbly);
  float crycrx = sbcx*(cblx*sblz*(calz*saly - caly*salx*salz)
               - cblx*cblz*(saly*salz + caly*calz*salx) + calx*caly*sblx)
               + cbcx*cbcy*((saly*salz + caly*calz*salx)*(sbly*sblz + cbly*cblz*sblx)
               + (calz*saly - caly*salx*salz)*(cblz*sbly - cbly*sblx*sblz) + calx*caly*cblx*cbly)
               - cbcx*sbcy*((saly*salz + caly*calz*salx)*(cbly*sblz - cblz*sblx*sbly)
               + (calz*saly - caly*salx*salz)*(cbly*cblz + sblx*sbly*sblz) - calx*caly*cblx*sbly);
  transformTobeMapped[1] = atan2(srycrx / cos(transformTobeMapped[0]), 
                                 crycrx / cos(transformTobeMapped[0]));
  
  float srzcrx = (cbcz*sbcy - cbcy*sbcx*sbcz)*(calx*salz*(cblz*sbly - cbly*sblx*sblz)
               - calx*calz*(sbly*sblz + cbly*cblz*sblx) + cblx*cbly*salx)
               - (cbcy*cbcz + sbcx*sbcy*sbcz)*(calx*calz*(cbly*sblz - cblz*sblx*sbly)
               - calx*salz*(cbly*cblz + sblx*sbly*sblz) + cblx*salx*sbly)
               + cbcx*sbcz*(salx*sblx + calx*cblx*salz*sblz + calx*calz*cblx*cblz);
  float crzcrx = (cbcy*sbcz - cbcz*sbcx*sbcy)*(calx*calz*(cbly*sblz - cblz*sblx*sbly)
               - calx*salz*(cbly*cblz + sblx*sbly*sblz) + cblx*salx*sbly)
               - (sbcy*sbcz + cbcy*cbcz*sbcx)*(calx*salz*(cblz*sbly - cbly*sblx*sblz)
               - calx*calz*(sbly*sblz + cbly*cblz*sblx) + cblx*cbly*salx)
               + cbcx*cbcz*(salx*sblx + calx*cblx*salz*sblz + calx*calz*cblx*cblz);
  transformTobeMapped[2] = atan2(srzcrx / cos(transformTobeMapped[0]), 
                                 crzcrx / cos(transformTobeMapped[0]));

  x1 = cos(transformTobeMapped[2]) * transformIncre[3] - sin(transformTobeMapped[2]) * transformIncre[4];
  y1 = sin(transformTobeMapped[2]) * transformIncre[3] + cos(transformTobeMapped[2]) * transformIncre[4];
  z1 = transformIncre[5];

  x2 = x1;
  y2 = cos(transformTobeMapped[0]) * y1 - sin(transformTobeMapped[0]) * z1;
  z2 = sin(transformTobeMapped[0]) * y1 + cos(transformTobeMapped[0]) * z1;

  transformTobeMapped[3] = transformAftMapped[3] - (cos(transformTobeMapped[1]) * x2 + sin(transformTobeMapped[1]) * z2);
  transformTobeMapped[4] = transformAftMapped[4] - y2;
  transformTobeMapped[5] = transformAftMapped[5] - (-sin(transformTobeMapped[1]) * x2 + cos(transformTobeMapped[1]) * z2);
}

// 记录odometry发送的转换矩阵与mapping之后的转换矩阵，下一帧点云会使用(有IMU的话会使用IMU进行补偿)
void transformUpdate()
{
  if (imuPointerLast >= 0) { // 有imu数据
    float imuRollLast = 0, imuPitchLast = 0;
    // 查找点云时间戳小于imu时间戳的imu位置
    while (imuPointerFront != imuPointerLast) { // front变量调整到与last变量一致，或者到激光点最近的imu数据位置
      if (timeLaserOdometry + scanPeriod < imuTime[imuPointerFront]) {
        break;
      }
      imuPointerFront = (imuPointerFront + 1) % imuQueLength;
    }

    if (timeLaserOdometry + scanPeriod > imuTime[imuPointerFront]) { // 未找到，此时imuPointerFront==imuPointerLast
      imuRollLast = imuRoll[imuPointerFront];
      imuPitchLast = imuPitch[imuPointerFront];
    } else { // 否则插值imu相邻的back与front数据
      int imuPointerBack = (imuPointerFront + imuQueLength - 1) % imuQueLength;
      float ratioFront = (timeLaserOdometry + scanPeriod - imuTime[imuPointerBack]) / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);
      float ratioBack = (imuTime[imuPointerFront] - timeLaserOdometry - scanPeriod) / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);

      // 按时间比例求翻滚角和俯仰角
      imuRollLast = imuRoll[imuPointerFront] * ratioFront + imuRoll[imuPointerBack] * ratioBack;
      imuPitchLast = imuPitch[imuPointerFront] * ratioFront + imuPitch[imuPointerBack] * ratioBack;
    }

    // imu稍微补偿俯仰角和翻滚角
    transformTobeMapped[0] = 0.998 * transformTobeMapped[0] + 0.002 * imuPitchLast;
    transformTobeMapped[2] = 0.998 * transformTobeMapped[2] + 0.002 * imuRollLast;
  }

  // 记录LM优化之前与之后的位姿转换矩阵
  for (int i = 0; i < 6; i++) {
    transformBefMapped[i] = transformSum[i]; // 初始位姿
    transformAftMapped[i] = transformTobeMapped[i]; // 补偿后的位姿为最终位姿
  }
}

// 根据调整计算后的转移矩阵transformTobeMapped，将点注册到全局世界坐标系下
void pointAssociateToMap(PointType const *const pi, PointType *const po)
{
  // 绕z轴旋转（transformTobeMapped[2]）
  float x1 = cos(transformTobeMapped[2]) * pi->x - sin(transformTobeMapped[2]) * pi->y;
  float y1 = sin(transformTobeMapped[2]) * pi->x + cos(transformTobeMapped[2]) * pi->y;
  float z1 = pi->z;

  // 绕x轴旋转（transformTobeMapped[0]）
  float x2 = x1;
  float y2 = cos(transformTobeMapped[0]) * y1 - sin(transformTobeMapped[0]) * z1;
  float z2 = sin(transformTobeMapped[0]) * y1 + cos(transformTobeMapped[0]) * z1;

  // 绕y轴旋转（transformTobeMapped[1]），再平移
  po->x = cos(transformTobeMapped[1]) * x2 + sin(transformTobeMapped[1]) * z2 + transformTobeMapped[3];
  po->y = y2 + transformTobeMapped[4];
  po->z = -sin(transformTobeMapped[1]) * x2 + cos(transformTobeMapped[1]) * z2 + transformTobeMapped[5];
  po->intensity = pi->intensity;
}

// 点转移到局部坐标系下
void pointAssociateTobeMapped(PointType const * const pi, PointType * const po)
{
  // 平移后绕y轴旋转（-transformTobeMapped[1]）
  float x1 = cos(transformTobeMapped[1]) * (pi->x - transformTobeMapped[3]) - sin(transformTobeMapped[1]) * (pi->z - transformTobeMapped[5]);
  float y1 = pi->y - transformTobeMapped[4];
  float z1 = sin(transformTobeMapped[1]) * (pi->x - transformTobeMapped[3]) + cos(transformTobeMapped[1]) * (pi->z - transformTobeMapped[5]);

  // 绕x轴旋转（-transformTobeMapped[0]）
  float x2 = x1;
  float y2 = cos(transformTobeMapped[0]) * y1 + sin(transformTobeMapped[0]) * z1;
  float z2 = -sin(transformTobeMapped[0]) * y1 + cos(transformTobeMapped[0]) * z1;

  // 绕z轴旋转（-transformTobeMapped[2]）
  po->x = cos(transformTobeMapped[2]) * x2 + sin(transformTobeMapped[2]) * y2;
  po->y = -sin(transformTobeMapped[2]) * x2 + cos(transformTobeMapped[2]) * y2;
  po->z = z2;
  po->intensity = pi->intensity;
}

// 接收边沿点
void laserCloudCornerLastHandler(const sensor_msgs::PointCloud2ConstPtr& laserCloudCornerLast2)
{
  timeLaserCloudCornerLast = laserCloudCornerLast2->header.stamp.toSec();

  laserCloudCornerLast->clear();
  pcl::fromROSMsg(*laserCloudCornerLast2, *laserCloudCornerLast);

  newLaserCloudCornerLast = true;
}

// 接收平面点
void laserCloudSurfLastHandler(const sensor_msgs::PointCloud2ConstPtr& laserCloudSurfLast2)
{
  timeLaserCloudSurfLast = laserCloudSurfLast2->header.stamp.toSec();

  laserCloudSurfLast->clear();
  pcl::fromROSMsg(*laserCloudSurfLast2, *laserCloudSurfLast);

  newLaserCloudSurfLast = true;
}

// 接收点云全部点
void laserCloudFullResHandler(const sensor_msgs::PointCloud2ConstPtr& laserCloudFullRes2)
{
  timeLaserCloudFullRes = laserCloudFullRes2->header.stamp.toSec();

  laserCloudFullRes->clear();
  pcl::fromROSMsg(*laserCloudFullRes2, *laserCloudFullRes);

  newLaserCloudFullRes = true;
}

// 接收旋转平移信息
void laserOdometryHandler(const nav_msgs::Odometry::ConstPtr& laserOdometry)
{
  timeLaserOdometry = laserOdometry->header.stamp.toSec();

  double roll, pitch, yaw;
  // 四元数转换为欧拉角
  geometry_msgs::Quaternion geoQuat = laserOdometry->pose.pose.orientation;
  tf::Matrix3x3(tf::Quaternion(geoQuat.z, -geoQuat.x, -geoQuat.y, geoQuat.w)).getRPY(roll, pitch, yaw);

  transformSum[0] = -pitch;
  transformSum[1] = -yaw;
  transformSum[2] = roll;

  transformSum[3] = laserOdometry->pose.pose.position.x;
  transformSum[4] = laserOdometry->pose.pose.position.y;
  transformSum[5] = laserOdometry->pose.pose.position.z;

  newLaserOdometry = true;
}

// 接收imu信息，只使用了翻滚角和俯仰角
void imuHandler(const sensor_msgs::Imu::ConstPtr& imuIn)
{
  double roll, pitch, yaw;
  tf::Quaternion orientation;
  tf::quaternionMsgToTF(imuIn->orientation, orientation);
  tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);

  imuPointerLast = (imuPointerLast + 1) % imuQueLength;

  imuTime[imuPointerLast] = imuIn->header.stamp.toSec();
  imuRoll[imuPointerLast] = roll;
  imuPitch[imuPointerLast] = pitch;
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "laserMapping");
  ros::NodeHandle nh;

  ros::Subscriber subLaserCloudCornerLast = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_corner_last", 2, laserCloudCornerLastHandler);
  ros::Subscriber subLaserCloudSurfLast = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_surf_last", 2, laserCloudSurfLastHandler);
  ros::Subscriber subLaserOdometry = nh.subscribe<nav_msgs::Odometry>("/laser_odom_to_init", 5, laserOdometryHandler);

  ros::Subscriber subLaserCloudFullRes = nh.subscribe<sensor_msgs::PointCloud2>("/velodyne_cloud_3", 2, laserCloudFullResHandler);
  ros::Subscriber subImu = nh.subscribe<sensor_msgs::Imu> ("/imu/data", 50, imuHandler);
  
  ros::Publisher pubLaserCloudSurround = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_surround", 1);
  ros::Publisher pubLaserCloudFullRes = nh.advertise<sensor_msgs::PointCloud2>("/velodyne_cloud_registered", 2);
  ros::Publisher pubOdomAftMapped = nh.advertise<nav_msgs::Odometry> ("/aft_mapped_to_init", 5);
  nav_msgs::Odometry odomAftMapped;
  odomAftMapped.header.frame_id = "/camera_init";
  odomAftMapped.child_frame_id = "/aft_mapped";

  tf::TransformBroadcaster tfBroadcaster;
  tf::StampedTransform aftMappedTrans;
  aftMappedTrans.frame_id_ = "/camera_init";
  aftMappedTrans.child_frame_id_ = "/aft_mapped";

  std::vector<int> pointSearchInd;
  std::vector<float> pointSearchSqDis;

  PointType pointOri, pointSel, pointProj, coeff;

  cv::Mat matA0(5, 3, CV_32F, cv::Scalar::all(0));
  cv::Mat matB0(5, 1, CV_32F, cv::Scalar::all(-1));
  cv::Mat matX0(3, 1, CV_32F, cv::Scalar::all(0));

  cv::Mat matA1(3, 3, CV_32F, cv::Scalar::all(0));
  cv::Mat matD1(1, 3, CV_32F, cv::Scalar::all(0));
  cv::Mat matV1(3, 3, CV_32F, cv::Scalar::all(0));

  bool isDegenerate = false;
  cv::Mat matP(6, 6, CV_32F, cv::Scalar::all(0));

  // 创建VoxelGrid滤波器（体素栅格滤波器），设置体素大小
  pcl::VoxelGrid<PointType> downSizeFilterCorner;
  // 设置体素大小
  downSizeFilterCorner.setLeafSize(0.2, 0.2, 0.2);

  pcl::VoxelGrid<PointType> downSizeFilterSurf;
  downSizeFilterSurf.setLeafSize(0.4, 0.4, 0.4);

  pcl::VoxelGrid<PointType> downSizeFilterMap;
  downSizeFilterMap.setLeafSize(0.6, 0.6, 0.6);

  // 指针初始化
  for (int i = 0; i < laserCloudNum; i++) {
    laserCloudCornerArray[i].reset(new pcl::PointCloud<PointType>());
    laserCloudSurfArray[i].reset(new pcl::PointCloud<PointType>());
    laserCloudCornerArray2[i].reset(new pcl::PointCloud<PointType>());
    laserCloudSurfArray2[i].reset(new pcl::PointCloud<PointType>());
  }

  int frameCount = stackFrameNum - 1; //0
  int mapFrameCount = mapFrameNum - 1; //4
  ros::Rate rate(100); // 运行频率100Hz
  bool status = ros::ok();
  while (status) { // while循环首先进行一次回调，并确保时间一致性，然后重置标志位，帧数量+1
    ros::spinOnce();

    if (newLaserCloudCornerLast && newLaserCloudSurfLast && newLaserCloudFullRes && newLaserOdometry &&
        fabs(timeLaserCloudCornerLast - timeLaserOdometry) < 0.005 &&
        fabs(timeLaserCloudSurfLast - timeLaserOdometry) < 0.005 &&
        fabs(timeLaserCloudFullRes - timeLaserOdometry) < 0.005) {
      newLaserCloudCornerLast = false;
      newLaserCloudSurfLast = false;
      newLaserCloudFullRes = false;
      newLaserOdometry = false;

      frameCount++;
      /*坐标变换*/
      if (frameCount >= stackFrameNum) {
        transformAssociateToMap(); // 地图匹配坐标转换

        // 将最新接收到的平面点和边沿点进行旋转平移转换到世界坐标系下(这里和后面的逆转换应无必要)
        int laserCloudCornerLastNum = laserCloudCornerLast->points.size();
        for (int i = 0; i < laserCloudCornerLastNum; i++) {
          pointAssociateToMap(&laserCloudCornerLast->points[i], &pointSel);
          laserCloudCornerStack2->push_back(pointSel);
        }

        int laserCloudSurfLastNum = laserCloudSurfLast->points.size(); // 面特征点
        for (int i = 0; i < laserCloudSurfLastNum; i++) {
          pointAssociateToMap(&laserCloudSurfLast->points[i], &pointSel);
          laserCloudSurfStack2->push_back(pointSel);
        }
      }

	    /* 优化处理,找当前估计的lidar位姿属于哪个cube，I/J/K对应cube的索引 */
      if (frameCount >= stackFrameNum) {
        frameCount = 0;
        // 当前lidar坐标系{L}y轴一点(0,10,0)
        PointType pointOnYAxis;
        pointOnYAxis.x = 0.0;
        pointOnYAxis.y = 10.0;
        pointOnYAxis.z = 0.0;
        pointAssociateToMap(&pointOnYAxis, &pointOnYAxis); // 获取雷达坐标系下，y方向上10米高位置的点在世界坐标系下的坐标

        // 立方体中点在世界坐标系下的（原点）位置，[-25m,25m]处于中心cube，即(10,5,10)的cube
        // 过半取一（以50米进行四舍五入的效果），由于数组下标只能为正数，而地图可能建立在原点前后，因此
        // 每一维偏移一个laserCloudCenWidth（该值会动态调整，以使得数组利用最大化，初始值为该维数组长度1/2）的量
        int centerCubeI = int((transformTobeMapped[3] + 25.0) / 50.0) + laserCloudCenWidth;
        int centerCubeJ = int((transformTobeMapped[4] + 25.0) / 50.0) + laserCloudCenHeight;
        int centerCubeK = int((transformTobeMapped[5] + 25.0) / 50.0) + laserCloudCenDepth;

        // 由于计算机求余是向零取整，为了不使（-50.0,50.0）求余后都向零偏移，当被求余数为负数时求余结果统一向左偏移一个单位，也即减一
        if (transformTobeMapped[3] + 25.0 < 0) centerCubeI--;
        if (transformTobeMapped[4] + 25.0 < 0) centerCubeJ--;
        if (transformTobeMapped[5] + 25.0 < 0) centerCubeK--;

        // 调整之后取值范围:3 < centerCubeI < 18， 3 < centerCubeJ < 8, 3 < centerCubeK < 18
        // 如果当前帧lidar位姿对应的cube在整个大cube边缘则将索引向中心方向挪动一个单位
        while (centerCubeI < 3) { // width方向的小端，将帧cube指针向中心方向平移
          for (int j = 0; j < laserCloudHeight; j++) {
            for (int k = 0; k < laserCloudDepth; k++) { // 实现一次循环移位效果
              int i = laserCloudWidth - 1;
              // 指向width最大处的cube点云
              pcl::PointCloud<PointType>::Ptr laserCloudCubeCornerPointer =
              laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k]; // that's [i + 21 * j + 231 * k]
              pcl::PointCloud<PointType>::Ptr laserCloudCubeSurfPointer =
              laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
              for (; i >= 1; i--) { // 将cube由width大段开始依次向增大方向移动一个单位
                laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                laserCloudCornerArray[i - 1 + laserCloudWidth*j + laserCloudWidth * laserCloudHeight * k];
                laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                laserCloudSurfArray[i - 1 + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
              }
              // 移动后最小width索引指向原来最大width处
              laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] = 
              laserCloudCubeCornerPointer;
              laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] = 
              laserCloudCubeSurfPointer;
              laserCloudCubeCornerPointer->clear();
              laserCloudCubeSurfPointer->clear();
            }
          }

          centerCubeI++; // 帧所在的cube在width方向相应增大一个单位
          laserCloudCenWidth++; // 中心位置索引同样增大
        }

        // 如果处于上边界，表明地图向正方向延伸的可能性比较大，则循环移位，将数组中心点向下边界调整一个单位
        while (centerCubeI >= laserCloudWidth - 3) {//18
          for (int j = 0; j < laserCloudHeight; j++) {
            for (int k = 0; k < laserCloudDepth; k++) {
              int i = 0;
              pcl::PointCloud<PointType>::Ptr laserCloudCubeCornerPointer =
              laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
              pcl::PointCloud<PointType>::Ptr laserCloudCubeSurfPointer =
              laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
              // I维度上依次前移
              for (; i < laserCloudWidth - 1; i++) {
                laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                laserCloudCornerArray[i + 1 + laserCloudWidth*j + laserCloudWidth * laserCloudHeight * k];
                laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                laserCloudSurfArray[i + 1 + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
              }
              laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] = 
              laserCloudCubeCornerPointer;
              laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] = 
              laserCloudCubeSurfPointer;
              laserCloudCubeCornerPointer->clear();
              laserCloudCubeSurfPointer->clear();
            }
          }

          centerCubeI--;
          laserCloudCenWidth--;
        }

        while (centerCubeJ < 3) {
          for (int i = 0; i < laserCloudWidth; i++) {
            for (int k = 0; k < laserCloudDepth; k++) {
              int j = laserCloudHeight - 1;
              pcl::PointCloud<PointType>::Ptr laserCloudCubeCornerPointer =
              laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
              pcl::PointCloud<PointType>::Ptr laserCloudCubeSurfPointer =
              laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
              // J维度上，依次后移
              for (; j >= 1; j--) {
                laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                laserCloudCornerArray[i + laserCloudWidth*(j - 1) + laserCloudWidth * laserCloudHeight*k];
                laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                laserCloudSurfArray[i + laserCloudWidth * (j - 1) + laserCloudWidth * laserCloudHeight*k];
              }
              laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] = 
              laserCloudCubeCornerPointer;
              laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] = 
              laserCloudCubeSurfPointer;
              laserCloudCubeCornerPointer->clear();
              laserCloudCubeSurfPointer->clear();
            }
          }
 
          centerCubeJ++;
          laserCloudCenHeight++;
        } 

        while (centerCubeJ >= laserCloudHeight - 3) {
          for (int i = 0; i < laserCloudWidth; i++) {
            for (int k = 0; k < laserCloudDepth; k++) {
              int j = 0;
              pcl::PointCloud<PointType>::Ptr laserCloudCubeCornerPointer =
              laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
              pcl::PointCloud<PointType>::Ptr laserCloudCubeSurfPointer =
              laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
              // J维度上一次前移
              for (; j < laserCloudHeight - 1; j++) {
                laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                laserCloudCornerArray[i + laserCloudWidth*(j + 1) + laserCloudWidth * laserCloudHeight*k];
                laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                laserCloudSurfArray[i + laserCloudWidth * (j + 1) + laserCloudWidth * laserCloudHeight*k];
              }
              laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] = 
              laserCloudCubeCornerPointer;
              laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] = 
              laserCloudCubeSurfPointer;
              laserCloudCubeCornerPointer->clear();
              laserCloudCubeSurfPointer->clear();
            }
          }

          centerCubeJ--;
          laserCloudCenHeight--;
        }

        while (centerCubeK < 3) {
          for (int i = 0; i < laserCloudWidth; i++) {
            for (int j = 0; j < laserCloudHeight; j++) {
              int k = laserCloudDepth - 1;
              pcl::PointCloud<PointType>::Ptr laserCloudCubeCornerPointer =
              laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
              pcl::PointCloud<PointType>::Ptr laserCloudCubeSurfPointer =
              laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
              // K维度上依次后移
              for (; k >= 1; k--) {
                laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                laserCloudCornerArray[i + laserCloudWidth*j + laserCloudWidth * laserCloudHeight*(k - 1)];
                laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight*(k - 1)];
              }
              laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] = 
              laserCloudCubeCornerPointer;
              laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] = 
              laserCloudCubeSurfPointer;
              laserCloudCubeCornerPointer->clear();
              laserCloudCubeSurfPointer->clear();
            }
          }

          centerCubeK++;
          laserCloudCenDepth++;
        }
      
        while (centerCubeK >= laserCloudDepth - 3) {
          for (int i = 0; i < laserCloudWidth; i++) {
            for (int j = 0; j < laserCloudHeight; j++) {
              int k = 0;
              pcl::PointCloud<PointType>::Ptr laserCloudCubeCornerPointer =
              laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
              pcl::PointCloud<PointType>::Ptr laserCloudCubeSurfPointer =
              laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
              // K维度上依次前移
              for (; k < laserCloudDepth - 1; k++) {
                laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                laserCloudCornerArray[i + laserCloudWidth*j + laserCloudWidth * laserCloudHeight*(k + 1)];
                laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight*(k + 1)];
              }
              laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] = 
              laserCloudCubeCornerPointer;
              laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] = 
              laserCloudCubeSurfPointer;
              laserCloudCubeCornerPointer->clear();
              laserCloudCubeSurfPointer->clear();
            }
          }

          centerCubeK--;
          laserCloudCenDepth--;
        }

        // 在当前帧lidar位姿所在cube附近5×5×5的邻域cube内找有效的cube
        int laserCloudValidNum = 0; // 125个领域cube中有效领域的num
        int laserCloudSurroundNum = 0; // RS最远距离确认
        // 在每一维附近5个cube(前2个，后2个，中间1个)里进行查找（前后250米范围内，总共500米范围），三个维度总共125个cube
        // 在这125个cube里面进一步筛选在视域范围内的cube
        for (int i = centerCubeI - 2; i <= centerCubeI + 2; i++) { // centerCube为当前帧所在cube的索引
          for (int j = centerCubeJ - 2; j <= centerCubeJ + 2; j++) {
            for (int k = centerCubeK - 2; k <= centerCubeK + 2; k++) {
              if (i >= 0 && i < laserCloudWidth && 
                  j >= 0 && j < laserCloudHeight && 
                  k >= 0 && k < laserCloudDepth) { // 由于之前设计了循环结构，所以要防止索引越界

                // 计算各邻域cube中心点坐标，换算成实际比例，在世界坐标系下的坐标
                float centerX = 50.0 * (i - laserCloudCenWidth);
                float centerY = 50.0 * (j - laserCloudCenHeight);
                float centerZ = 50.0 * (k - laserCloudCenDepth);

                // 取cube的8个顶点
                bool isInLaserFOV = false; // 判断是否在lidar视线范围内的标志（is in Laser's Field of View）
                for (int ii = -1; ii <= 1; ii += 2) {
                  for (int jj = -1; jj <= 1; jj += 2) {
                    for (int kk = -1; kk <= 1; kk += 2) {
                      // 上下左右八个顶点坐标
                      float cornerX = centerX + 25.0 * ii; // 当前cube的边缘坐标
                      float cornerY = centerY + 25.0 * jj;
                      float cornerZ = centerZ + 25.0 * kk; // 八次循环得到八个顶点坐标

                      // 原点到顶点距离的平方和
                      float squaredSide1 = (transformTobeMapped[3] - cornerX) 
                                         * (transformTobeMapped[3] - cornerX) 
                                         + (transformTobeMapped[4] - cornerY) 
                                         * (transformTobeMapped[4] - cornerY)
                                         + (transformTobeMapped[5] - cornerZ) 
                                         * (transformTobeMapped[5] - cornerZ); // 当前帧lidar全局位姿与邻域cube顶点的距离平方

                      // 当前帧中(0,10,0)与顶点距离方
                      float squaredSide2 = (pointOnYAxis.x - cornerX) * (pointOnYAxis.x - cornerX) 
                                         + (pointOnYAxis.y - cornerY) * (pointOnYAxis.y - cornerY)
                                         + (pointOnYAxis.z - cornerZ) * (pointOnYAxis.z - cornerZ);

                      float check1 = 100.0 + squaredSide1 - squaredSide2
                                   - 10.0 * sqrt(3.0) * sqrt(squaredSide1);

                      float check2 = 100.0 + squaredSide1 - squaredSide2
                                   + 10.0 * sqrt(3.0) * sqrt(squaredSide1);

                      // 该cube的8个顶点只要有一个符合距离相关的条件，则该cube为有效邻域
                      if (check1 < 0 && check2 > 0) { // if |100 + squaredSide1 - squaredSide2| < 10.0 * sqrt(3.0) * sqrt(squaredSide1)
                        isInLaserFOV = true; // 有一个顶点满足即可
                      }
                    }
                  }
                }
                // 八个顶点遍历结束

                if (isInLaserFOV) { // 当前lidar全局位姿的有效邻域cube在地图中的索引，匹配用
                  laserCloudValidInd[laserCloudValidNum] = i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k;
                  laserCloudValidNum++;
                }
                // 125个领域cube索引(包括不符合距离条件的cube），显示用
                laserCloudSurroundInd[laserCloudSurroundNum] = i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k;
                laserCloudSurroundNum++;
              }
            }
          }
        }

        // 拼接有效领域cube内的特征点云作为map点云用来匹配
        laserCloudCornerFromMap->clear();
        laserCloudSurfFromMap->clear();
        // 构建激光雷达扫描区域的特征点地图，查找匹配更加高效
        for (int i = 0; i < laserCloudValidNum; i++) { // 有效的邻域cube
          *laserCloudCornerFromMap += *laserCloudCornerArray[laserCloudValidInd[i]]; // 有效cube构成匹配地图
          *laserCloudSurfFromMap += *laserCloudSurfArray[laserCloudValidInd[i]];
        }
        int laserCloudCornerFromMapNum = laserCloudCornerFromMap->points.size(); // corner点云中特征点数量
        int laserCloudSurfFromMapNum = laserCloudSurfFromMap->points.size(); // surf点云特征点数量

        /***********************************************************************
		      将世界坐标系下的当前帧特征点转到当前Lidar坐标系下
          此处将特征点转移回local坐标系，是为了voxel grid filter的下采样操作不越
          界？好像不是！后面还会转移回世界坐标系，这里是前面的逆转换，和前面一样
          应无必要，可直接对laserCloudCornerLast和laserCloudSurfLast进行下采样
        ***********************************************************************/
        int laserCloudCornerStackNum2 = laserCloudCornerStack2->points.size(); // 当前帧边特征点点数
        for (int i = 0; i < laserCloudCornerStackNum2; i++) {
          pointAssociateTobeMapped(&laserCloudCornerStack2->points[i], &laserCloudCornerStack2->points[i]);
        }

        int laserCloudSurfStackNum2 = laserCloudSurfStack2->points.size(); // 当前帧平面特征点点数
        for (int i = 0; i < laserCloudSurfStackNum2; i++) {
          pointAssociateTobeMapped(&laserCloudSurfStack2->points[i], &laserCloudSurfStack2->points[i]);
        }

        // 对当前帧特征点进行滤波处理
        laserCloudCornerStack->clear();
        downSizeFilterCorner.setInputCloud(laserCloudCornerStack2); // 设置滤波对象
        downSizeFilterCorner.filter(*laserCloudCornerStack); // 执行滤波处理
        int laserCloudCornerStackNum = laserCloudCornerStack->points.size(); // 获取滤波后体素点尺寸

        laserCloudSurfStack->clear();
        downSizeFilterSurf.setInputCloud(laserCloudSurfStack2);
        downSizeFilterSurf.filter(*laserCloudSurfStack);
        int laserCloudSurfStackNum = laserCloudSurfStack->points.size();

        laserCloudCornerStack2->clear();
        laserCloudSurfStack2->clear();

        if (laserCloudCornerFromMapNum > 10 && laserCloudSurfFromMapNum > 100) { // 地图特征点足够多
          kdtreeCornerFromMap->setInputCloud(laserCloudCornerFromMap); // 地图特征点云存入kd树
          kdtreeSurfFromMap->setInputCloud(laserCloudSurfFromMap);

          for (int iterCount = 0; iterCount < 10; iterCount++) { // 最多迭代10次
            laserCloudOri->clear();
            coeffSel->clear();

            // 当前帧边特征点，地图KD树搜索当前帧边特征点最邻近的5个点
            for (int i = 0; i < laserCloudCornerStackNum; i++) {
              pointOri = laserCloudCornerStack->points[i];
              pointAssociateToMap(&pointOri, &pointSel); // 特征点转到全局坐标系下
              kdtreeCornerFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis); // 寻找最近距离五个点
              
              if (pointSearchSqDis[4] < 1.0) { // 第5远的点与当前特征点的距离在阈值内
                // 求出5个邻近点坐标均值
                float cx = 0;
                float cy = 0; 
                float cz = 0;
                for (int j = 0; j < 5; j++) {
                  cx += laserCloudCornerFromMap->points[pointSearchInd[j]].x;
                  cy += laserCloudCornerFromMap->points[pointSearchInd[j]].y;
                  cz += laserCloudCornerFromMap->points[pointSearchInd[j]].z;
                }
                cx /= 5;
                cy /= 5; 
                cz /= 5;

                // 构建协方差矩阵
                float a11 = 0;
                float a12 = 0; 
                float a13 = 0;
                float a22 = 0;
                float a23 = 0; 
                float a33 = 0;
                for (int j = 0; j < 5; j++) {
                  float ax = laserCloudCornerFromMap->points[pointSearchInd[j]].x - cx;
                  float ay = laserCloudCornerFromMap->points[pointSearchInd[j]].y - cy;
                  float az = laserCloudCornerFromMap->points[pointSearchInd[j]].z - cz;

                  a11 += ax * ax;
                  a12 += ax * ay;
                  a13 += ax * az;
                  a22 += ay * ay;
                  a23 += ay * az;
                  a33 += az * az;
                }
                a11 /= 5;
                a12 /= 5; 
                a13 /= 5;
                a22 /= 5;
                a23 /= 5; 
                a33 /= 5;

                // 3X3协方差矩阵  
                matA1.at<float>(0, 0) = a11;
                matA1.at<float>(0, 1) = a12;
                matA1.at<float>(0, 2) = a13;
                matA1.at<float>(1, 0) = a12;
                matA1.at<float>(1, 1) = a22;
                matA1.at<float>(1, 2) = a23;
                matA1.at<float>(2, 0) = a13;
                matA1.at<float>(2, 1) = a23;
                matA1.at<float>(2, 2) = a33;

                // 特征根matD1，对应的特征向量matV1
                cv::eigen(matA1, matD1, matV1);
				        // 点云协方差矩阵进行主成分分析：若这五个点分布在直线上，协方差矩阵的特征值包含一个元素显著大于其余两个，与该特征值相关的特征向量表示所处直线的方向

                if (matD1.at<float>(0, 0) > 3 * matD1.at<float>(0, 1)) { // 如果最大的特征值大于三倍的第二大特征值

                  float x0 = pointSel.x;
                  float y0 = pointSel.y;
                  float z0 = pointSel.z;
                  // 根据特征向量的方向构建x1和x2，两者连线代表最近点构成的直线 
                  float x1 = cx + 0.1 * matV1.at<float>(0, 0);
                  float y1 = cy + 0.1 * matV1.at<float>(0, 1);
                  float z1 = cz + 0.1 * matV1.at<float>(0, 2);
                  float x2 = cx - 0.1 * matV1.at<float>(0, 0);
                  float y2 = cy - 0.1 * matV1.at<float>(0, 1);
                  float z2 = cz - 0.1 * matV1.at<float>(0, 2);

                  // 向量OA = (x0 - x1, y0 - y1, z0 - z1), 向量OB = (x0 - x2, y0 - y2, z0 - z2)，向量AB = （x1 - x2, y1 - y2, z1 - z2）
                  // 向量OA OB的向量积(即叉乘)为：
                  // |  i      j      k  |
                  // |x0-x1  y0-y1  z0-z1|
                  // |x0-x2  y0-y2  z0-z2|
                  // 向量积的模为a012
                  float a012 = sqrt(((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1))
                             * ((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) 
                             + ((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1))
                             * ((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1)) 
                             + ((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1))
                             * ((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1)));

                  float l12 = sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2));

                  float la = ((y1 - y2)*((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) 
                           + (z1 - z2)*((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1))) / a012 / l12;

                  float lb = -((x1 - x2)*((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) 
                           - (z1 - z2)*((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1))) / a012 / l12;

                  float lc = -((x1 - x2)*((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1)) 
                           + (y1 - y2)*((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1))) / a012 / l12;

                  float ld2 = a012 / l12;

                  // 没用到
                  pointProj = pointSel;
                  pointProj.x -= la * ld2;
                  pointProj.y -= lb * ld2;
                  pointProj.z -= lc * ld2;

                  float s = 1 - 0.9 * fabs(ld2); // 权重系数计算，距离越小，s越大

                  coeff.x = s * la;
                  coeff.y = s * lb;
                  coeff.z = s * lc;
                  coeff.intensity = s * ld2;

                  if (s > 0.1) { // 只保留权重大的即距离比较小的点
                    laserCloudOri->push_back(pointOri);
                    coeffSel->push_back(coeff);
                  }
                }
              }
            }
            // 本次迭代中的边缘特征点处理完毕

            for (int i = 0; i < laserCloudSurfStackNum; i++) {
              pointOri = laserCloudSurfStack->points[i];
              pointAssociateToMap(&pointOri, &pointSel); 
              kdtreeSurfFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);

              if (pointSearchSqDis[4] < 1.0) {
                // 构建五个最近点的坐标矩阵
                for (int j = 0; j < 5; j++) {
                  matA0.at<float>(j, 0) = laserCloudSurfFromMap->points[pointSearchInd[j]].x;
                  matA0.at<float>(j, 1) = laserCloudSurfFromMap->points[pointSearchInd[j]].y;
                  matA0.at<float>(j, 2) = laserCloudSurfFromMap->points[pointSearchInd[j]].z;
                }
                // 求解matA0*matX0=matB0
                cv::solve(matA0, matB0, matX0, cv::DECOMP_QR);
				        // matB0全为-1，可以理解为寻找一个法向量X0，使得与A0的五个点组成的曲面垂直

                float pa = matX0.at<float>(0, 0);
                float pb = matX0.at<float>(1, 0);
                float pc = matX0.at<float>(2, 0);
                float pd = 1;
				
				        // 向量归一化
                float ps = sqrt(pa * pa + pb * pb + pc * pc);
                pa /= ps;
                pb /= ps;
                pc /= ps;
                pd /= ps;

                bool planeValid = true;
                for (int j = 0; j < 5; j++) {
                  if (fabs(pa * laserCloudSurfFromMap->points[pointSearchInd[j]].x +
                      pb * laserCloudSurfFromMap->points[pointSearchInd[j]].y +
                      pc * laserCloudSurfFromMap->points[pointSearchInd[j]].z + pd) > 0.2) {
                    planeValid = false; // 曲面不是很平坦
                    break;
                  }
                }

                if (planeValid) { // 如果找到的五个附近的可以组成类似的平面
                  float pd2 = pa * pointSel.x + pb * pointSel.y + pc * pointSel.z + pd;

                  // 没有用到
                  pointProj = pointSel;
                  pointProj.x -= pa * pd2;
                  pointProj.y -= pb * pd2;
                  pointProj.z -= pc * pd2;

                  float s = 1 - 0.9 * fabs(pd2) / sqrt(sqrt(pointSel.x * pointSel.x + pointSel.y * pointSel.y + pointSel.z * pointSel.z));

                  coeff.x = s * pa;
                  coeff.y = s * pb;
                  coeff.z = s * pc;
                  coeff.intensity = s * pd2;

                  if (s > 0.1) {
                    laserCloudOri->push_back(pointOri);
                    coeffSel->push_back(coeff);
                  }
                }
              }
            }
			      // 本次迭代中的平面特征点处理完毕
			
            float srx = sin(transformTobeMapped[0]);
            float crx = cos(transformTobeMapped[0]);
            float sry = sin(transformTobeMapped[1]);
            float cry = cos(transformTobeMapped[1]);
            float srz = sin(transformTobeMapped[2]);
            float crz = cos(transformTobeMapped[2]);

            int laserCloudSelNum = laserCloudOri->points.size();
            if (laserCloudSelNum < 50) {
              continue; // 如果符合权重条件的边缘和平面特征点少于50个，放弃本次迭代
            }

            cv::Mat matA(laserCloudSelNum, 6, CV_32F, cv::Scalar::all(0));
            cv::Mat matAt(6, laserCloudSelNum, CV_32F, cv::Scalar::all(0));
            cv::Mat matAtA(6, 6, CV_32F, cv::Scalar::all(0));
            cv::Mat matB(laserCloudSelNum, 1, CV_32F, cv::Scalar::all(0));
            cv::Mat matAtB(6, 1, CV_32F, cv::Scalar::all(0));
            cv::Mat matX(6, 1, CV_32F, cv::Scalar::all(0));
            for (int i = 0; i < laserCloudSelNum; i++) {
              pointOri = laserCloudOri->points[i]; // 当前帧特征点
              coeff = coeffSel->points[i]; // 对应距离的偏导

              // 雷达坐标系下的k+1帧点云xyz经过pointAssociateToMap()函数变换得到世界坐标系下的k+1帧点云x'y'z'，x'y'z'是transformTobeMapped[6]的函数
              // 点云x'y'z'在世界坐标系下的点云KD树中寻找最近点得到距离值，完美的transformTobeMapped[6]使得距离值为0
              // 所以距离值对transformTobeMapped[6]求导使用LM算法求极值
              // 距离值对点云x'y'z'求导得到coeff
              // 点云x'y'z'对transform[6]求导可以由pointAssociateToMap()函数计算得到
              // 从而得到距离值对transform[6]的导数arx,ary,arz,atx,aty,atz
              // 其中点云x'y'z'对atx,aty,atz的导数均为1，所以距离值对atx,aty,atz的导数就是距离值对点云x'y'z'的对应导数
              float arx = (crx*sry*srz*pointOri.x + crx*crz*sry*pointOri.y - srx*sry*pointOri.z) * coeff.x
                        + (-srx*srz*pointOri.x - crz*srx*pointOri.y - crx*pointOri.z) * coeff.y
                        + (crx*cry*srz*pointOri.x + crx*cry*crz*pointOri.y - cry*srx*pointOri.z) * coeff.z;

              float ary = ((cry*srx*srz - crz*sry)*pointOri.x 
                        + (sry*srz + cry*crz*srx)*pointOri.y + crx*cry*pointOri.z) * coeff.x
                        + ((-cry*crz - srx*sry*srz)*pointOri.x 
                        + (cry*srz - crz*srx*sry)*pointOri.y - crx*sry*pointOri.z) * coeff.z;

              float arz = ((crz*srx*sry - cry*srz)*pointOri.x + (-cry*crz-srx*sry*srz)*pointOri.y)*coeff.x
                        + (crx*crz*pointOri.x - crx*srz*pointOri.y) * coeff.y
                        + ((sry*srz + cry*crz*srx)*pointOri.x + (crz*sry-cry*srx*srz)*pointOri.y)*coeff.z;

              matA.at<float>(i, 0) = arx;
              matA.at<float>(i, 1) = ary;
              matA.at<float>(i, 2) = arz;
              matA.at<float>(i, 3) = coeff.x;
              matA.at<float>(i, 4) = coeff.y;
              matA.at<float>(i, 5) = coeff.z;
              matB.at<float>(i, 0) = -coeff.intensity;  // 负号使得拟合效果越好，matB越小
            }
            cv::transpose(matA, matAt);
            matAtA = matAt * matA;
            matAtB = matAt * matB;
            cv::solve(matAtA, matAtB, matX, cv::DECOMP_QR);
			
            if (iterCount == 0) { // 如果是首次迭代，检查矩阵退化，和laserOdometry相同
              cv::Mat matE(1, 6, CV_32F, cv::Scalar::all(0));
              cv::Mat matV(6, 6, CV_32F, cv::Scalar::all(0));
              cv::Mat matV2(6, 6, CV_32F, cv::Scalar::all(0));

              cv::eigen(matAtA, matE, matV);
              matV.copyTo(matV2);

              isDegenerate = false;
              float eignThre[6] = {100, 100, 100, 100, 100, 100};
              for (int i = 5; i >= 0; i--) {
                if (matE.at<float>(0, i) < eignThre[i]) {
                  for (int j = 0; j < 6; j++) {
                    matV2.at<float>(i, j) = 0;
                  }
                  isDegenerate = true;
                } else {
                  break;
                }
              }
              matP = matV.inv() * matV2;
            }

            if (isDegenerate) {
              cv::Mat matX2(6, 1, CV_32F, cv::Scalar::all(0));
              matX.copyTo(matX2);
              matX = matP * matX2;
            }

            // 积累每次的调整量
            transformTobeMapped[0] += matX.at<float>(0, 0);
            transformTobeMapped[1] += matX.at<float>(1, 0);
            transformTobeMapped[2] += matX.at<float>(2, 0);
            transformTobeMapped[3] += matX.at<float>(3, 0);
            transformTobeMapped[4] += matX.at<float>(4, 0);
            transformTobeMapped[5] += matX.at<float>(5, 0);

            float deltaR = sqrt(
                                pow(rad2deg(matX.at<float>(0, 0)), 2) +
                                pow(rad2deg(matX.at<float>(1, 0)), 2) +
                                pow(rad2deg(matX.at<float>(2, 0)), 2));
            float deltaT = sqrt(
                                pow(matX.at<float>(3, 0) * 100, 2) +
                                pow(matX.at<float>(4, 0) * 100, 2) +
                                pow(matX.at<float>(5, 0) * 100, 2));

            // 旋转平移量足够小就停止迭代
            if (deltaR < 0.05 && deltaT < 0.05) {
              break; // 步长足够小就结束所有迭代
            }
			      // 结束本次迭代
          }
		      // 结束所有迭代

          // 迭代结束更新相关的转移矩阵
          transformUpdate();
        }

        /* 前面完成了当前帧与地图的匹配优化，现在将当前帧特征点云保存在地图中 */
        // 将边缘特征点按距离（比例尺缩小）归入相应的立方体
        for (int i = 0; i < laserCloudCornerStackNum; i++) {
          pointAssociateToMap(&laserCloudCornerStack->points[i], &pointSel); // 转移到世界坐标系

          // 按50的比例尺缩小，四舍五入，偏移laserCloudCen*的量，计算索引
          int cubeI = int((pointSel.x + 25.0) / 50.0) + laserCloudCenWidth; // 特征点width方向索引
          int cubeJ = int((pointSel.y + 25.0) / 50.0) + laserCloudCenHeight;
          int cubeK = int((pointSel.z + 25.0) / 50.0) + laserCloudCenDepth;

          if (pointSel.x + 25.0 < 0) cubeI--;
          if (pointSel.y + 25.0 < 0) cubeJ--;
          if (pointSel.z + 25.0 < 0) cubeK--;

          if (cubeI >= 0 && cubeI < laserCloudWidth && 
              cubeJ >= 0 && cubeJ < laserCloudHeight && 
              cubeK >= 0 && cubeK < laserCloudDepth) { // 只挑选-laserCloudCenWidth * 50.0 < point.x < laserCloudCenWidth * 50.0范围内的点，y和z同理
              // 按照尺度放进不同的组，每个组的点数量各异
            int cubeInd = cubeI + laserCloudWidth * cubeJ + laserCloudWidth * laserCloudHeight * cubeK; // 边特征点对应的cube索引号
            laserCloudCornerArray[cubeInd]->push_back(pointSel); // 特征点存入cube中
          }
        }

        // 将平面特征点按距离（比例尺缩小）归入相应的立方体
        for (int i = 0; i < laserCloudSurfStackNum; i++) {
          pointAssociateToMap(&laserCloudSurfStack->points[i], &pointSel);

          int cubeI = int((pointSel.x + 25.0) / 50.0) + laserCloudCenWidth;
          int cubeJ = int((pointSel.y + 25.0) / 50.0) + laserCloudCenHeight;
          int cubeK = int((pointSel.z + 25.0) / 50.0) + laserCloudCenDepth;

          if (pointSel.x + 25.0 < 0) cubeI--;
          if (pointSel.y + 25.0 < 0) cubeJ--;
          if (pointSel.z + 25.0 < 0) cubeK--;

          if (cubeI >= 0 && cubeI < laserCloudWidth && 
              cubeJ >= 0 && cubeJ < laserCloudHeight && 
              cubeK >= 0 && cubeK < laserCloudDepth) {
            int cubeInd = cubeI + laserCloudWidth * cubeJ + laserCloudWidth * laserCloudHeight * cubeK;
            laserCloudSurfArray[cubeInd]->push_back(pointSel);
          }
        }

        // 对当前帧而言，存在有限个邻域cube用于配准，特征点下采样
        for (int i = 0; i < laserCloudValidNum; i++) { // 对当前帧有效的邻域cube数量
          int ind = laserCloudValidInd[i];
		  
		      // 有特征点的cube下采样滤波，放到新的数组2里面
          // 对cube内边特征点云降采样
          laserCloudCornerArray2[ind]->clear();
          downSizeFilterCorner.setInputCloud(laserCloudCornerArray[ind]);
          downSizeFilterCorner.filter(*laserCloudCornerArray2[ind]); // 滤波输出到Array2

          // 对cube内面特征点云降采样
          laserCloudSurfArray2[ind]->clear();
          downSizeFilterSurf.setInputCloud(laserCloudSurfArray[ind]);
          downSizeFilterSurf.filter(*laserCloudSurfArray2[ind]);

          // Array与Array2交换，即滤波后自我更新
          pcl::PointCloud<PointType>::Ptr laserCloudTemp = laserCloudCornerArray[ind];
          laserCloudCornerArray[ind] = laserCloudCornerArray2[ind];
          laserCloudCornerArray2[ind] = laserCloudTemp;

          laserCloudTemp = laserCloudSurfArray[ind];
          laserCloudSurfArray[ind] = laserCloudSurfArray2[ind];
          laserCloudSurfArray2[ind] = laserCloudTemp;
        }

        mapFrameCount++;
        if (mapFrameCount >= mapFrameNum) { // 每隔五帧，特征点汇总后进行下采样并publish一次，从第一次开始
          mapFrameCount = 0;

          laserCloudSurround2->clear();
          for (int i = 0; i < laserCloudSurroundNum; i++) {
            int ind = laserCloudSurroundInd[i];
            *laserCloudSurround2 += *laserCloudCornerArray[ind];
            *laserCloudSurround2 += *laserCloudSurfArray[ind];
          }

          laserCloudSurround->clear();
          downSizeFilterCorner.setInputCloud(laserCloudSurround2);
          downSizeFilterCorner.filter(*laserCloudSurround);

          sensor_msgs::PointCloud2 laserCloudSurround3;
          pcl::toROSMsg(*laserCloudSurround, laserCloudSurround3);
          laserCloudSurround3.header.stamp = ros::Time().fromSec(timeLaserOdometry);
          laserCloudSurround3.header.frame_id = "/camera_init";
          pubLaserCloudSurround.publish(laserCloudSurround3);
        }

        // 将laserOdometry发来的点云中的全部点转移到世界坐标系下
        int laserCloudFullResNum = laserCloudFullRes->points.size();
        for (int i = 0; i < laserCloudFullResNum; i++) {
          pointAssociateToMap(&laserCloudFullRes->points[i], &laserCloudFullRes->points[i]);
        }

        sensor_msgs::PointCloud2 laserCloudFullRes3;
        pcl::toROSMsg(*laserCloudFullRes, laserCloudFullRes3);
        laserCloudFullRes3.header.stamp = ros::Time().fromSec(timeLaserOdometry);
        laserCloudFullRes3.header.frame_id = "/camera_init";
        pubLaserCloudFullRes.publish(laserCloudFullRes3);

        geometry_msgs::Quaternion geoQuat = tf::createQuaternionMsgFromRollPitchYaw
                                  (transformAftMapped[2], -transformAftMapped[0], -transformAftMapped[1]);

        odomAftMapped.header.stamp = ros::Time().fromSec(timeLaserOdometry);
        odomAftMapped.pose.pose.orientation.x = -geoQuat.y;
        odomAftMapped.pose.pose.orientation.y = -geoQuat.z;
        odomAftMapped.pose.pose.orientation.z = geoQuat.x;
        odomAftMapped.pose.pose.orientation.w = geoQuat.w;
        odomAftMapped.pose.pose.position.x = transformAftMapped[3];
        odomAftMapped.pose.pose.position.y = transformAftMapped[4];
        odomAftMapped.pose.pose.position.z = transformAftMapped[5];
        // 扭转量
        odomAftMapped.twist.twist.angular.x = transformBefMapped[0];
        odomAftMapped.twist.twist.angular.y = transformBefMapped[1];
        odomAftMapped.twist.twist.angular.z = transformBefMapped[2];
        odomAftMapped.twist.twist.linear.x = transformBefMapped[3];
        odomAftMapped.twist.twist.linear.y = transformBefMapped[4];
        odomAftMapped.twist.twist.linear.z = transformBefMapped[5];
        pubOdomAftMapped.publish(odomAftMapped);

        // 广播坐标系旋转平移参量
        aftMappedTrans.stamp_ = ros::Time().fromSec(timeLaserOdometry);
        aftMappedTrans.setRotation(tf::Quaternion(-geoQuat.y, -geoQuat.z, geoQuat.x, geoQuat.w));
        aftMappedTrans.setOrigin(tf::Vector3(transformAftMapped[3], 
                                             transformAftMapped[4], transformAftMapped[5]));
        tfBroadcaster.sendTransform(aftMappedTrans);

      }
    }

    status = ros::ok();
    rate.sleep();
  }

  return 0;
}

