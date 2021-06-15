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
  laserOdometry主要利用相邻两帧的点云数据进行配准，即完成t时刻和t+1时刻点云数据的光联，并估计雷达的相对运动关系，输出高频率odom。
*/

#include <cmath>

#include <loam_velodyne/common.h>
#include <nav_msgs/Odometry.h>
#include <opencv/cv.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>

const float scanPeriod = 0.1; // 一个scan的周期, velodyne频率10Hz，周期0.1s

const int skipFrameNum = 1; // 控制接收到的点云数据，每隔1帧处理一次
bool systemInited = false;

//时间戳信息
double timeCornerPointsSharp = 0;
double timeCornerPointsLessSharp = 0;
double timeSurfPointsFlat = 0;
double timeSurfPointsLessFlat = 0;
double timeLaserCloudFullRes = 0;
double timeImuTrans = 0;

//消息接收标志
bool newCornerPointsSharp = false;
bool newCornerPointsLessSharp = false;
bool newSurfPointsFlat = false;
bool newSurfPointsLessFlat = false;
bool newLaserCloudFullRes = false;
bool newImuTrans = false;

// 边缘锐角特征点云
pcl::PointCloud<PointType>::Ptr cornerPointsSharp(new pcl::PointCloud<PointType>());
// 边缘钝角特征点云
pcl::PointCloud<PointType>::Ptr cornerPointsLessSharp(new pcl::PointCloud<PointType>());
// 平面上特征点云
pcl::PointCloud<PointType>::Ptr surfPointsFlat(new pcl::PointCloud<PointType>());
// 平面上不太平的特征点云
pcl::PointCloud<PointType>::Ptr surfPointsLessFlat(new pcl::PointCloud<PointType>());
// 最后一帧的边缘点云
pcl::PointCloud<PointType>::Ptr laserCloudCornerLast(new pcl::PointCloud<PointType>());
// 最后一帧的平面点云
pcl::PointCloud<PointType>::Ptr laserCloudSurfLast(new pcl::PointCloud<PointType>());
// 保存前一个节点发过来的未经处理过的特征点
pcl::PointCloud<PointType>::Ptr laserCloudOri(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr coeffSel(new pcl::PointCloud<PointType>());
// 接收所有点云
pcl::PointCloud<PointType>::Ptr laserCloudFullRes(new pcl::PointCloud<PointType>());
// 接收imu数据
pcl::PointCloud<pcl::PointXYZ>::Ptr imuTrans(new pcl::PointCloud<pcl::PointXYZ>());
// 最后一帧的边缘点云构建的kd-tree
pcl::KdTreeFLANN<PointType>::Ptr kdtreeCornerLast(new pcl::KdTreeFLANN<PointType>());
// 最后一帧的平面点云构建的kd-tree
pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurfLast(new pcl::KdTreeFLANN<PointType>());

int laserCloudCornerLastNum;
int laserCloudSurfLastNum;

int pointSelCornerInd[40000]; // 没有用到
// 存储两个搜索到的边缘点的索引
float pointSearchCornerInd1[40000];
float pointSearchCornerInd2[40000];

int pointSelSurfInd[40000]; // 没有用到
// 存储三个搜索到的平面点的索引
float pointSearchSurfInd1[40000];
float pointSearchSurfInd2[40000];
float pointSearchSurfInd3[40000];

float transform[6] = {0}; // 当前帧相对上一帧的状态转移量（in the local frame）
float transformSum[6] = {0}; // 当前帧相对于第一帧的状态转移量（in the global frame）

// 点云第一个点的Roll\Pitcg\Yaw
float imuRollStart = 0, imuPitchStart = 0, imuYawStart = 0;
// 点云最后一个点的Roll\Pitcg\Yaw
float imuRollLast = 0, imuPitchLast = 0, imuYawLast = 0;
// 点云最后一个点相对于第一个点由于加减速产生的畸变位移
float imuShiftFromStartX = 0, imuShiftFromStartY = 0, imuShiftFromStartZ = 0;
// 点云最后一个点相对于第一个点由于加减速产生的畸变速度
float imuVeloFromStartX = 0, imuVeloFromStartY = 0, imuVeloFromStartZ = 0;

/*****************************************************************************
  将当前帧点云TransformToStart和将上一帧点云TransformToEnd的作用：
  去除畸变，并将两帧点云数据统一到同一个坐标系下计算
*****************************************************************************/

/*
  首先通过intensity信息求得当前激光点在该帧点云中的相对位置0～1，
  然后线性插值位姿变换矩阵transform，得到该帧初始点坐标系相对当前激光点坐标系位姿关系，并将当前激光点转换至初始点坐标系
*/
// 三维位姿转换，pi坐标转换至该帧初始点位姿坐标系得到po
void TransformToStart(PointType const * const pi, PointType * const po)
{
  // 插值系数计算，云中每个点的相对时间/点云周期（0.1）相当于*10
  float s = 10 * (pi->intensity - int(pi->intensity)); // s=0~1为激光点在一帧数据中的相对位置，插值系数计算，还原了reltime变量

  // 线性插值：根据每个点在点云中的相对位置关系，乘以相应的旋转平移系数
  float rx = s * transform[0];
  float ry = s * transform[1];
  float rz = s * transform[2];
  float tx = s * transform[3];
  float ty = s * transform[4];
  float tz = s * transform[5];

  // 平移后绕z轴旋转（-rz）
  float x1 = cos(rz) * (pi->x - tx) + sin(rz) * (pi->y - ty);
  float y1 = -sin(rz) * (pi->x - tx) + cos(rz) * (pi->y - ty);
  float z1 = (pi->z - tz);

  // 绕x轴旋转（-rx）
  float x2 = x1;
  float y2 = cos(rx) * y1 + sin(rx) * z1;
  float z2 = -sin(rx) * y1 + cos(rx) * z1;

  // 绕y轴旋转（-ry）
  po->x = cos(ry) * x2 - sin(ry) * z2;
  po->y = y2;
  po->z = sin(ry) * x2 + cos(ry) * z2;
  po->intensity = pi->intensity;
}

// 当前点云中的点相对第一个点去除因匀速运动产生的畸变，效果相当于得到在点云扫描结束位置坐标系下  的扫描得到的点云
void TransformToEnd(PointType const * const pi, PointType * const po)
{
  // 插值系数计算
  float s = 10 * (pi->intensity - int(pi->intensity));

  float rx = s * transform[0];
  float ry = s * transform[1];
  float rz = s * transform[2];
  float tx = s * transform[3];
  float ty = s * transform[4];
  float tz = s * transform[5];

  // 平移后绕z轴旋转（-rz）
  float x1 = cos(rz) * (pi->x - tx) + sin(rz) * (pi->y - ty);
  float y1 = -sin(rz) * (pi->x - tx) + cos(rz) * (pi->y - ty);
  float z1 = (pi->z - tz);

  // 绕x轴旋转（-rx）
  float x2 = x1;
  float y2 = cos(rx) * y1 + sin(rx) * z1;
  float z2 = -sin(rx) * y1 + cos(rx) * z1;

  // 绕y轴旋转（-ry）
  float x3 = cos(ry) * x2 - sin(ry) * z2;
  float y3 = y2;
  float z3 = sin(ry) * x2 + cos(ry) * z2; // 求出了相对于起始点校正的坐标

  rx = transform[0];
  ry = transform[1];
  rz = transform[2];
  tx = transform[3];
  ty = transform[4];
  tz = transform[5];

  // 绕y轴旋转（ry）
  float x4 = cos(ry) * x3 + sin(ry) * z3;
  float y4 = y3;
  float z4 = -sin(ry) * x3 + cos(ry) * z3;

  // 绕x轴旋转（rx）
  float x5 = x4;
  float y5 = cos(rx) * y4 - sin(rx) * z4;
  float z5 = sin(rx) * y4 + cos(rx) * z4;

  // 绕z轴旋转（rz），再平移
  float x6 = cos(rz) * x5 - sin(rz) * y5 + tx;
  float y6 = sin(rz) * x5 + cos(rz) * y5 + ty;
  float z6 = z5 + tz;

  // 平移后绕z轴旋转（imuRollStart）
  float x7 = cos(imuRollStart) * (x6 - imuShiftFromStartX) - sin(imuRollStart) * (y6 - imuShiftFromStartY);
  float y7 = sin(imuRollStart) * (x6 - imuShiftFromStartX) + cos(imuRollStart) * (y6 - imuShiftFromStartY);
  float z7 = z6 - imuShiftFromStartZ;

  // 绕x轴旋转（imuPitchStart）
  float x8 = x7;
  float y8 = cos(imuPitchStart) * y7 - sin(imuPitchStart) * z7;
  float z8 = sin(imuPitchStart) * y7 + cos(imuPitchStart) * z7;

  // 绕y轴旋转（imuYawStart）
  float x9 = cos(imuYawStart) * x8 + sin(imuYawStart) * z8;
  float y9 = y8;
  float z9 = -sin(imuYawStart) * x8 + cos(imuYawStart) * z8;

  // 绕y轴旋转（-imuYawLast）
  float x10 = cos(imuYawLast) * x9 - sin(imuYawLast) * z9;
  float y10 = y9;
  float z10 = sin(imuYawLast) * x9 + cos(imuYawLast) * z9;

  // 绕x轴旋转（-imuPitchLast）
  float x11 = x10;
  float y11 = cos(imuPitchLast) * y10 + sin(imuPitchLast) * z10;
  float z11 = -sin(imuPitchLast) * y10 + cos(imuPitchLast) * z10;

  // 绕z轴旋转（-imuRollLast）
  po->x = cos(imuRollLast) * x11 + sin(imuRollLast) * y11;
  po->y = -sin(imuRollLast) * x11 + cos(imuRollLast) * y11;
  po->z = z11;
  // 只保留线号
  po->intensity = int(pi->intensity);
}

// 利用IMU修正旋转量，根据起始欧拉角，当前点云的欧拉角修正
void PluginIMURotation(float bcx, float bcy, float bcz, float blx, float bly, float blz, 
                       float alx, float aly, float alz, float &acx, float &acy, float &acz)
{
  float sbcx = sin(bcx);
  float cbcx = cos(bcx);
  float sbcy = sin(bcy);
  float cbcy = cos(bcy);
  float sbcz = sin(bcz);
  float cbcz = cos(bcz);

  float sblx = sin(blx);
  float cblx = cos(blx);
  float sbly = sin(bly);
  float cbly = cos(bly);
  float sblz = sin(blz);
  float cblz = cos(blz);

  float salx = sin(alx);
  float calx = cos(alx);
  float saly = sin(aly);
  float caly = cos(aly);
  float salz = sin(alz);
  float calz = cos(alz);

  float srx = -sbcx*(salx*sblx + calx*caly*cblx*cbly + calx*cblx*saly*sbly) 
            - cbcx*cbcz*(calx*saly*(cbly*sblz - cblz*sblx*sbly) 
            - calx*caly*(sbly*sblz + cbly*cblz*sblx) + cblx*cblz*salx) 
            - cbcx*sbcz*(calx*caly*(cblz*sbly - cbly*sblx*sblz) 
            - calx*saly*(cbly*cblz + sblx*sbly*sblz) + cblx*salx*sblz);
  acx = -asin(srx);

  float srycrx = (cbcy*sbcz - cbcz*sbcx*sbcy)*(calx*saly*(cbly*sblz - cblz*sblx*sbly) 
               - calx*caly*(sbly*sblz + cbly*cblz*sblx) + cblx*cblz*salx) 
               - (cbcy*cbcz + sbcx*sbcy*sbcz)*(calx*caly*(cblz*sbly - cbly*sblx*sblz) 
               - calx*saly*(cbly*cblz + sblx*sbly*sblz) + cblx*salx*sblz) 
               + cbcx*sbcy*(salx*sblx + calx*caly*cblx*cbly + calx*cblx*saly*sbly);
  float crycrx = (cbcz*sbcy - cbcy*sbcx*sbcz)*(calx*caly*(cblz*sbly - cbly*sblx*sblz) 
               - calx*saly*(cbly*cblz + sblx*sbly*sblz) + cblx*salx*sblz) 
               - (sbcy*sbcz + cbcy*cbcz*sbcx)*(calx*saly*(cbly*sblz - cblz*sblx*sbly) 
               - calx*caly*(sbly*sblz + cbly*cblz*sblx) + cblx*cblz*salx) 
               + cbcx*cbcy*(salx*sblx + calx*caly*cblx*cbly + calx*cblx*saly*sbly);
  acy = atan2(srycrx / cos(acx), crycrx / cos(acx));
  
  float srzcrx = sbcx*(cblx*cbly*(calz*saly - caly*salx*salz) 
               - cblx*sbly*(caly*calz + salx*saly*salz) + calx*salz*sblx) 
               - cbcx*cbcz*((caly*calz + salx*saly*salz)*(cbly*sblz - cblz*sblx*sbly) 
               + (calz*saly - caly*salx*salz)*(sbly*sblz + cbly*cblz*sblx) 
               - calx*cblx*cblz*salz) + cbcx*sbcz*((caly*calz + salx*saly*salz)*(cbly*cblz 
               + sblx*sbly*sblz) + (calz*saly - caly*salx*salz)*(cblz*sbly - cbly*sblx*sblz) 
               + calx*cblx*salz*sblz);
  float crzcrx = sbcx*(cblx*sbly*(caly*salz - calz*salx*saly) 
               - cblx*cbly*(saly*salz + caly*calz*salx) + calx*calz*sblx) 
               + cbcx*cbcz*((saly*salz + caly*calz*salx)*(sbly*sblz + cbly*cblz*sblx) 
               + (caly*salz - calz*salx*saly)*(cbly*sblz - cblz*sblx*sbly) 
               + calx*calz*cblx*cblz) - cbcx*sbcz*((saly*salz + caly*calz*salx)*(cblz*sbly 
               - cbly*sblx*sblz) + (caly*salz - calz*salx*saly)*(cbly*cblz + sblx*sbly*sblz) 
               - calx*calz*cblx*sblz);
  acz = atan2(srzcrx / cos(acx), crzcrx / cos(acx));
}

// 相对于第一个点云即原点，积累旋转量
void AccumulateRotation(float cx, float cy, float cz, float lx, float ly, float lz, 
                        float &ox, float &oy, float &oz)
{
  // 对应R(w,k)第(1,2)元素
  float srx = cos(lx)*cos(cx)*sin(ly)*sin(cz) - cos(cx)*cos(cz)*sin(lx) - cos(lx)*cos(ly)*sin(cx);
  ox = -asin(srx);

  // (0,2)元素
  float srycrx = sin(lx)*(cos(cy)*sin(cz) - cos(cz)*sin(cx)*sin(cy)) + cos(lx)*sin(ly)*(cos(cy)*cos(cz) 
               + sin(cx)*sin(cy)*sin(cz)) + cos(lx)*cos(ly)*cos(cx)*sin(cy);
  // (2,2)元素
  float crycrx = cos(lx)*cos(ly)*cos(cx)*cos(cy) - cos(lx)*sin(ly)*(cos(cz)*sin(cy) 
               - cos(cy)*sin(cx)*sin(cz)) - sin(lx)*(sin(cy)*sin(cz) + cos(cy)*cos(cz)*sin(cx));
  oy = atan2(srycrx / cos(ox), crycrx / cos(ox));

  // (1,0)元素
  float srzcrx = sin(cx)*(cos(lz)*sin(ly) - cos(ly)*sin(lx)*sin(lz)) + cos(cx)*sin(cz)*(cos(ly)*cos(lz) 
               + sin(lx)*sin(ly)*sin(lz)) + cos(lx)*cos(cx)*cos(cz)*sin(lz);
  // (1,1）元素
  float crzcrx = cos(lx)*cos(lz)*cos(cx)*cos(cz) - cos(cx)*sin(cz)*(cos(ly)*sin(lz) 
               - cos(lz)*sin(lx)*sin(ly)) - sin(cx)*(sin(ly)*sin(lz) + cos(ly)*cos(lz)*sin(lx));
  oz = atan2(srzcrx / cos(ox), crzcrx / cos(ox));
}

void laserCloudSharpHandler(const sensor_msgs::PointCloud2ConstPtr& cornerPointsSharp2)
{
  timeCornerPointsSharp = cornerPointsSharp2->header.stamp.toSec();

  cornerPointsSharp->clear();
  pcl::fromROSMsg(*cornerPointsSharp2, *cornerPointsSharp);
  std::vector<int> indices;
  pcl::removeNaNFromPointCloud(*cornerPointsSharp,*cornerPointsSharp, indices);
  newCornerPointsSharp = true;
}

void laserCloudLessSharpHandler(const sensor_msgs::PointCloud2ConstPtr& cornerPointsLessSharp2)
{
  timeCornerPointsLessSharp = cornerPointsLessSharp2->header.stamp.toSec();

  cornerPointsLessSharp->clear();
  pcl::fromROSMsg(*cornerPointsLessSharp2, *cornerPointsLessSharp);
  std::vector<int> indices;
  pcl::removeNaNFromPointCloud(*cornerPointsLessSharp,*cornerPointsLessSharp, indices);
  newCornerPointsLessSharp = true;
}

void laserCloudFlatHandler(const sensor_msgs::PointCloud2ConstPtr& surfPointsFlat2)
{
  timeSurfPointsFlat = surfPointsFlat2->header.stamp.toSec();

  surfPointsFlat->clear();
  pcl::fromROSMsg(*surfPointsFlat2, *surfPointsFlat);
  std::vector<int> indices;
  pcl::removeNaNFromPointCloud(*surfPointsFlat,*surfPointsFlat, indices);
  newSurfPointsFlat = true;
}

void laserCloudLessFlatHandler(const sensor_msgs::PointCloud2ConstPtr& surfPointsLessFlat2)
{
  timeSurfPointsLessFlat = surfPointsLessFlat2->header.stamp.toSec();

  surfPointsLessFlat->clear();
  pcl::fromROSMsg(*surfPointsLessFlat2, *surfPointsLessFlat);
  std::vector<int> indices;
  pcl::removeNaNFromPointCloud(*surfPointsLessFlat,*surfPointsLessFlat, indices);
  newSurfPointsLessFlat = true;
}

// 接收全部点
void laserCloudFullResHandler(const sensor_msgs::PointCloud2ConstPtr& laserCloudFullRes2)
{
  timeLaserCloudFullRes = laserCloudFullRes2->header.stamp.toSec();

  laserCloudFullRes->clear();
  pcl::fromROSMsg(*laserCloudFullRes2, *laserCloudFullRes);
  std::vector<int> indices;
  pcl::removeNaNFromPointCloud(*laserCloudFullRes,*laserCloudFullRes, indices);
  newLaserCloudFullRes = true;
}

// 接收imu消息
void imuTransHandler(const sensor_msgs::PointCloud2ConstPtr& imuTrans2)
{
  timeImuTrans = imuTrans2->header.stamp.toSec();

  imuTrans->clear();
  pcl::fromROSMsg(*imuTrans2, *imuTrans);

  // 根据发来的消息提取imu信息
  imuPitchStart = imuTrans->points[0].x;
  imuYawStart = imuTrans->points[0].y;
  imuRollStart = imuTrans->points[0].z;

  imuPitchLast = imuTrans->points[1].x;
  imuYawLast = imuTrans->points[1].y;
  imuRollLast = imuTrans->points[1].z;

  imuShiftFromStartX = imuTrans->points[2].x;
  imuShiftFromStartY = imuTrans->points[2].y;
  imuShiftFromStartZ = imuTrans->points[2].z;

  imuVeloFromStartX = imuTrans->points[3].x;
  imuVeloFromStartY = imuTrans->points[3].y;
  imuVeloFromStartZ = imuTrans->points[3].z;

  newImuTrans = true;
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "laserOdometry"); // 注册laserOdometry节点
  ros::NodeHandle nh; // 创建管理节点的句柄

  /* 订阅并处理消息 */
  ros::Subscriber subCornerPointsSharp = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_sharp", 2, laserCloudSharpHandler);
  ros::Subscriber subCornerPointsLessSharp = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_less_sharp", 2, laserCloudLessSharpHandler);
  ros::Subscriber subSurfPointsFlat = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_flat", 2, laserCloudFlatHandler);
  ros::Subscriber subSurfPointsLessFlat = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_less_flat", 2, laserCloudLessFlatHandler);
  ros::Subscriber subLaserCloudFullRes = nh.subscribe<sensor_msgs::PointCloud2>("/velodyne_cloud_2", 2, laserCloudFullResHandler); // 上一节点里的所有点
  ros::Subscriber subImuTrans = nh.subscribe<sensor_msgs::PointCloud2>("/imu_trans", 5, imuTransHandler);

  /* 发布消息 */
  ros::Publisher pubLaserCloudCornerLast = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_corner_last", 2);
  ros::Publisher pubLaserCloudSurfLast = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_surf_last", 2);
  ros::Publisher pubLaserCloudFullRes = nh.advertise<sensor_msgs::PointCloud2>("/velodyne_cloud_3", 2); // 本次节点里的变换到扫描结束时刻的所有点
  ros::Publisher pubLaserOdometry = nh.advertise<nav_msgs::Odometry> ("/laser_odom_to_init", 5);

  // 设置里程计和TF的坐标系
  nav_msgs::Odometry laserOdometry;
  laserOdometry.header.frame_id = "/camera_init";
  laserOdometry.child_frame_id = "/laser_odom";

  tf::TransformBroadcaster tfBroadcaster;
  tf::StampedTransform laserOdometryTrans;
  laserOdometryTrans.frame_id_ = "/camera_init";
  laserOdometryTrans.child_frame_id_ = "/laser_odom";

  std::vector<int> pointSearchInd; // 搜索到的点序
  std::vector<float> pointSearchSqDis; // 搜索到的点平方距离

  PointType pointOri, pointSel/*选中的特征点*/, tripod1, tripod2, tripod3/*特征点的对应点*/, pointProj/*没用到*/, coeff;

  bool isDegenerate = false; // 退化标志
  cv::Mat matP(6, 6, CV_32F, cv::Scalar::all(0)); // P矩阵，预测矩阵

  int frameCount = skipFrameNum;
  ros::Rate rate(100); // 运行频率100Hz
  bool status = ros::ok();
  while (status) {
    ros::spinOnce(); // 执行回调函数
    // 读取新数据后改变标志位，数据是否都接收，而且时差小于0.005
    if (newCornerPointsSharp && newCornerPointsLessSharp && newSurfPointsFlat && 
        newSurfPointsLessFlat && newLaserCloudFullRes && newImuTrans &&
        fabs(timeCornerPointsSharp - timeSurfPointsLessFlat) < 0.005 &&
        fabs(timeCornerPointsLessSharp - timeSurfPointsLessFlat) < 0.005 &&
        fabs(timeSurfPointsFlat - timeSurfPointsLessFlat) < 0.005 &&
        fabs(timeLaserCloudFullRes - timeSurfPointsLessFlat) < 0.005 &&
        fabs(timeImuTrans - timeSurfPointsLessFlat) < 0.005) {  // 同步作用，确保同时收到同一个点云的特征点以及imu信息才进入
      newCornerPointsSharp = false;
      newCornerPointsLessSharp = false;
      newSurfPointsFlat = false;
      newSurfPointsLessFlat = false;
      newLaserCloudFullRes = false;
      newImuTrans = false;

      /***********************************
       ************* 1.初始化 ************
       ***********************************/

      if (!systemInited) {
        // 将订阅的数据保存为上一时刻的数据
        pcl::PointCloud<PointType>::Ptr laserCloudTemp = cornerPointsLessSharp;
        cornerPointsLessSharp = laserCloudCornerLast;
        laserCloudCornerLast = laserCloudTemp;

        laserCloudTemp = surfPointsLessFlat;
        surfPointsLessFlat = laserCloudSurfLast;
        laserCloudSurfLast = laserCloudTemp;

        // 构建kd-tree
        kdtreeCornerLast->setInputCloud(laserCloudCornerLast); // 所有的边缘点集合
        kdtreeSurfLast->setInputCloud(laserCloudSurfLast); // 所有的平面点集合

        // 将上一时刻的数据直接发布出去
        sensor_msgs::PointCloud2 laserCloudCornerLast2;
        pcl::toROSMsg(*laserCloudCornerLast, laserCloudCornerLast2);
        laserCloudCornerLast2.header.stamp = ros::Time().fromSec(timeSurfPointsLessFlat);
        laserCloudCornerLast2.header.frame_id = "/camera";
        pubLaserCloudCornerLast.publish(laserCloudCornerLast2);

        sensor_msgs::PointCloud2 laserCloudSurfLast2;
        pcl::toROSMsg(*laserCloudSurfLast, laserCloudSurfLast2);
        laserCloudSurfLast2.header.stamp = ros::Time().fromSec(timeSurfPointsLessFlat);
        laserCloudSurfLast2.header.frame_id = "/camera";
        pubLaserCloudSurfLast.publish(laserCloudSurfLast2);

        // 第一帧数据的初始点姿态角作为全局坐标系姿态角
        transformSum[0] += imuPitchStart;
        transformSum[2] += imuRollStart;

        systemInited = true; // 初始化完成
        continue;
      }

      // 012旋转，345平移，平移量的初值赋值
      transform[3] -= imuVeloFromStartX * scanPeriod;
      transform[4] -= imuVeloFromStartY * scanPeriod;
      transform[5] -= imuVeloFromStartZ * scanPeriod;

      /***********************************
       ****** 2.点云配准与运动估计 *******
       ***********************************/

      if (laserCloudCornerLastNum > 10 && laserCloudSurfLastNum > 100) { // 上一时刻特征边(曲率大)上的点云个数大于10， 特征面内的点云大于100，保证足够多的特征点可用于t+1时刻的匹配
        std::vector<int> indices;
        pcl::removeNaNFromPointCloud(*cornerPointsSharp, *cornerPointsSharp, indices); // 去除坐标包含NaN的无效点
        int cornerPointsSharpNum = cornerPointsSharp->points.size(); // 当前时刻特征边上的点云个数
        int surfPointsFlatNum = surfPointsFlat->points.size(); // 前时刻特征面上的点云个数
        
        // Levenberg-Marquardt算法(L-M method)，非线性最小二乘算法，最优化算法的一种
        // 最多迭代25次
        for (int iterCount = 0; iterCount < 25; iterCount++) {
          laserCloudOri->clear();
          coeffSel->clear();

          /*
           1. 特征边上的点配准并构建Jaccobian
           利用KD树找点i在t时刻点云中最近的一点j，并在j周围(上下几条线的范围内)找次近点I，于是我们把(j, I)称为点i在t时刻点云中的对应。
          */

          /*
            对于每一个线特征点调用TransformToStart()函数，将其转换至初始点位姿坐标系。
            每迭代五次，重新配准当前特征点，先搜索当前帧线特征点在targetcloud中的最邻近点，最邻近点与当前点距离pointSearchSqDis[0]在阈值范围内，则继续搜索次邻近点。
            此处的次邻近点不是当前帧激光点在targetcloud的绝对次邻近点，所以不能直接由kd树搜索得到。

            次邻近点从当前最邻近点往上下三条线内搜索，往上搜索时索引j上限源代码中是当前帧的线特征点数量cornerPointsSharpNum，
            本人认为应该是上一帧即target点云的线特征点数量laserCloudCornerLastNum？
            遍历target点云特征线点，计算与当前帧特征线点的距离方，求出距离小于阈值minPointsSqDis2且最短的target配准点即为次邻近点，次邻近点与最邻近点不同线。
            向下搜索三条线时与向上搜索方法一致，最终得到当前帧线特征点的最邻近和次邻近点索引pointSearchCornerInd1[i]/pointSearchCornerInd2[i]。
          */

          // 特征线上的点配准
          for (int i = 0; i < cornerPointsSharpNum; i++) {
		  	    // 遍历边缘特征点寻找最近点和次近点
            TransformToStart(&cornerPointsSharp->points[i], &pointSel); // 对于每一个线特征点，将其转换至初始点位姿坐标系。

            if (iterCount % 5 == 0) { // 每迭代五次，搜索一次最近点和次近点(降采样)
              std::vector<int> indices;
              pcl::removeNaNFromPointCloud(*laserCloudCornerLast,*laserCloudCornerLast, indices);
              // kd-tree查找一个最近距离点，边沿点未经过体素栅格滤波，一般边沿点本来就比较少，不做滤波
              kdtreeCornerLast->nearestKSearch(pointSel, 1, pointSearchInd, pointSearchSqDis); // 寻找当前边缘特征点pointSel的最近点，搜索数目为1，搜索到的下标，它和查询点的距离平方
              // 在最近点附近(向上下三条扫描线以内）找到次近点
              int closestPointInd = -1, minPointInd2 = -1;

              // 寻找相邻线距离目标点距离最小的点
              // 注意：velodyne是2度一线，scanID相邻并不代表线号相邻，相邻线度数相差2度，也即线号scanID相差2
              if (pointSearchSqDis[0] < 25) { // 最近点与特征点距离在一定阈值内
                closestPointInd = pointSearchInd[0]; // 最邻近点
                int closestPointScan = int(laserCloudCornerLast->points[closestPointInd].intensity);  // 最近点所在的scan_ID
                float pointSqDis, minPointSqDis2 = 25;
                // 从找得到的最邻近点开始，向上搜索，遍历所有边特征点
                for (int j = closestPointInd + 1; j < cornerPointsSharpNum; j++) { // 向scanID增大的方向查找
                  if (int(laserCloudCornerLast->points[j].intensity) > closestPointScan + 2.5) { // 找到与最邻近点相距3条线的特征点时跳出
                    break;
                  }

                  // 计算遍历点与最邻近点的距离(平方)
                  pointSqDis = (laserCloudCornerLast->points[j].x - pointSel.x) * 
                               (laserCloudCornerLast->points[j].x - pointSel.x) + 
                               (laserCloudCornerLast->points[j].y - pointSel.y) * 
                               (laserCloudCornerLast->points[j].y - pointSel.y) + 
                               (laserCloudCornerLast->points[j].z - pointSel.z) * 
                               (laserCloudCornerLast->points[j].z - pointSel.z);

                  if (int(laserCloudCornerLast->points[j].intensity) > closestPointScan) { // 次邻近点与最近点不同线
                    if (pointSqDis < minPointSqDis2) { // 比较法求出距离最短点，从而得到次邻近点 
                      // 更新最小距离与点序
                      minPointSqDis2 = pointSqDis;
                      minPointInd2 = j;
                    }
                  }
                }

                // 向下搜索三条线，找次邻近点
                for (int j = closestPointInd - 1; j >= 0; j--) { // 向scanID减小的方向查找
                  if (int(laserCloudCornerLast->points[j].intensity) < closestPointScan - 2.5) {
                    break;
                  }

                  pointSqDis = (laserCloudCornerLast->points[j].x - pointSel.x) * 
                               (laserCloudCornerLast->points[j].x - pointSel.x) + 
                               (laserCloudCornerLast->points[j].y - pointSel.y) * 
                               (laserCloudCornerLast->points[j].y - pointSel.y) + 
                               (laserCloudCornerLast->points[j].z - pointSel.z) * 
                               (laserCloudCornerLast->points[j].z - pointSel.z);

                  if (int(laserCloudCornerLast->points[j].intensity) < closestPointScan) {
                    if (pointSqDis < minPointSqDis2) {
                      minPointSqDis2 = pointSqDis; // 保证scanID更小的同时，搜索次近点
                      minPointInd2 = j;
                    }
                  }
                }
                // 搜索到了当前边缘特征点在不同scanID里的最近点和次近点
              }

              pointSearchCornerInd1[i] = closestPointInd; // 当前所有边特征点在上一时刻边特征点云中对应的最邻近点的索引
              pointSearchCornerInd2[i] = minPointInd2;    // 当前所有边特征点在上一时刻边特征点云中对应的次邻近点的索引
            }

            /*
              当前帧线特征点有新的配准点后，构建点到直线距离，即论文公式(2)。
              公式中分子表示平行四边行的面积，对应代码中a012向量叉乘求模。
              分母为四边行的底，对应l12。ld2即为当前帧特征线点到target帧对应直线的距离，la/lb/lc分别为距离ld2对x0/y0/z0的偏导。
              程序中给当前特征点设置了权重s，并保存满足距离条件的匹配结果。
            */

            // 构建Jaccobian矩阵，构建点到直线的距离约束，求偏导
            if (pointSearchCornerInd2[i] >= 0) { // 特征点存在次邻近点，即有配准点
              tripod1 = laserCloudCornerLast->points[pointSearchCornerInd1[i]]; // 最邻近点
              tripod2 = laserCloudCornerLast->points[pointSearchCornerInd2[i]]; // 次邻近点

              // 选择的特征点记为O，kd-tree最近距离点记为A，另一个最近距离点记为B
              float x0 = pointSel.x; // 当前帧特征点
              float y0 = pointSel.y;
              float z0 = pointSel.z;
              float x1 = tripod1.x; // 上一帧最邻近点
              float y1 = tripod1.y;
              float z1 = tripod1.z;
              float x2 = tripod2.x; // 上一帧次邻近点
              float y2 = tripod2.y;
              float z2 = tripod2.z;

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
                         * ((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1))); // 文章公式(2)中的分子部分->分别作差并叉乘后的向量模长，表示平行四边形面积 

              float l12 = sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2)); // 公式（2）分母部分->两个最近距离点之间的距离，即向量AB的模

              // AB方向的单位向量与OAB平面的单位法向量的向量积在各轴上的分量（d的方向）
              // x轴分量i
              float la = ((y1 - y2)*((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) 
                       + (z1 - z2)*((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1))) / a012 / l12;

              // y轴分量j
              float lb = -((x1 - x2)*((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) 
                       - (z1 - z2)*((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1))) / a012 / l12;

              // z轴分量k
              float lc = -((x1 - x2)*((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1)) 
                       + (y1 - y2)*((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1))) / a012 / l12;

              // d = |(X(k+1,i) - X(k,j)) * (X(k+1,i) - X(k,l))| / | X(k,j) - X(k,l)|
              // 特征点到前一帧最近点和次近点构成直线的距离，d = |向量OA 叉乘 向量OB|/|AB|
              float ld2 = a012 / l12;

              // 没用到
              pointProj = pointSel;
              pointProj.x -= la * ld2;
              pointProj.y -= lb * ld2;
              pointProj.z -= lc * ld2;

              // 权重计算，距离越大权重越小，距离越小权重越大，得到的权重范围<=1
              float s = 1; // 阻尼因子
              if (iterCount >= 5) { // 5次迭代之后开始增加权重因素
                s = 1 - 1.8 * fabs(ld2); // 点到直线距离越小配准越可信，则分配权重越大，距离越小，s越大
              }

              // 考虑权重
              coeff.x = s * la; // 偏导乘以系数
              coeff.y = s * lb;
              coeff.z = s * lc;
              coeff.intensity = s * ld2;

              if (s > 0.1 && ld2 != 0) { // 满足0 < ld2 < 0.5 则保存配准结果，只保留权重大的即距离比较小的点，同时也舍弃距离为零的点，放入laserCloudOri
                laserCloudOri->push_back(cornerPointsSharp->points[i]); // 保存特征点
                coeffSel->push_back(coeff); // 保存距离值和距离的偏导
              }
            }
			      // 一次迭代中的一个边缘特征点处理完毕
          }
		      // 一次迭代中的所有边缘特征点处理完毕

          /*
            2. 特征面上的点配准并构建Jaccobian
            与特征线类似，先找最近点j，在j周围找I，在j周围找m，将(j,I,m)称为点i在t时刻点云中的对应
          */

          /*
            特征面上的点配准与特征线类似，差异在于线特征点需要target点云中与之匹配的2个线特征点，而面特征点需要找3个匹配点，其中最近点与次近点在同一条scan线上。
            最后，将找到的当前帧特征点i在上一帧中的配准点保存在pointSearchSurfInd1[i]、pointSearchSurfInd2[i]、pointSearchSurfInd3[i]中。

            面特征点到配准面的距离见论文公式(3)，即dh=(pointSel-t1)*((t1-t2)X(t1-t3))/|(t1-t2)X(t1-t3)|，其中pointSel为当前帧面特征点，t1/t2/t3为上一帧与之配准的面特征点。
            公式主要思想为利用四面体的边在高度方向的投影来计算，简单的画个图就能看出来。
            代码中有了点到面的距离pd2后，求距离对当前特征点的偏导pa/pb/pc，保存相关变量。
          */

          // 面特征点配准
          for (int i = 0; i < surfPointsFlatNum; i++) {
            TransformToStart(&surfPointsFlat->points[i], &pointSel);
            if (iterCount % 5 == 0) { // 每迭代五次，重新寻找最近点和次近点
              // kd-tree最近点查找，在经过体素栅格滤波之后的平面特征点中查找，一般平面点太多，滤波后最近点查找的数据量小
              kdtreeSurfLast->nearestKSearch(pointSel, 1, pointSearchInd, pointSearchSqDis);
              // 寻找当前平面特征点pointSel的最近点，搜索数目为1，搜索到的下标，它和查询点的距离
              int closestPointInd = -1, minPointInd2 = -1, minPointInd3 = -1;
              // 第一个使用kd-tree查找，第二个在同一线上查找满足要求的，第三个在不同线上查找满足要求的
              if (pointSearchSqDis[0] < 25) { // 找到的最近点距离确实很近
                closestPointInd = pointSearchInd[0];
                int closestPointScan = int(laserCloudSurfLast->points[closestPointInd].intensity); // 提取最近点所属的scanID
                float pointSqDis, minPointSqDis2 = 25, minPointSqDis3 = 25;
                for (int j = closestPointInd + 1; j < surfPointsFlatNum; j++) { // 向后搜索
                  if (int(laserCloudSurfLast->points[j].intensity) > closestPointScan + 2.5) { // 如果相差超过2.5度则不再向后搜索
                    break;
                  }

                  pointSqDis = (laserCloudSurfLast->points[j].x - pointSel.x) * 
                               (laserCloudSurfLast->points[j].x - pointSel.x) + 
                               (laserCloudSurfLast->points[j].y - pointSel.y) * 
                               (laserCloudSurfLast->points[j].y - pointSel.y) + 
                               (laserCloudSurfLast->points[j].z - pointSel.z) * 
                               (laserCloudSurfLast->points[j].z - pointSel.z);

                  if (int(laserCloudSurfLast->points[j].intensity) <= closestPointScan) { // 如果点的线号小于等于最近点的线号(应该最多取等，也即同一线上的点)，最近距离点更新在2中
                     if (pointSqDis < minPointSqDis2) {
                       minPointSqDis2 = pointSqDis;
                       minPointInd2 = j;
                     }
                  } else { // 如果点的线号大于最近点的线号，最近距离点更新在3中
                     if (pointSqDis < minPointSqDis3) {
                       minPointSqDis3 = pointSqDis;
                       minPointInd3 = j;
                     }
                  }
                }

                for (int j = closestPointInd - 1; j >= 0; j--) { // 向前搜索
                  if (int(laserCloudSurfLast->points[j].intensity) < closestPointScan - 2.5) { // 如果相差超过2.5度则不再向前搜索
                    break;
                  }

                  pointSqDis = (laserCloudSurfLast->points[j].x - pointSel.x) * 
                               (laserCloudSurfLast->points[j].x - pointSel.x) + 
                               (laserCloudSurfLast->points[j].y - pointSel.y) * 
                               (laserCloudSurfLast->points[j].y - pointSel.y) + 
                               (laserCloudSurfLast->points[j].z - pointSel.z) * 
                               (laserCloudSurfLast->points[j].z - pointSel.z);

                  if (int(laserCloudSurfLast->points[j].intensity) >= closestPointScan) { // 如果点的线号大于等于最近点的线号(应该最多取等，也即同一线上的点)，最近距离点更新在2中
                    if (pointSqDis < minPointSqDis2) {
                      minPointSqDis2 = pointSqDis;
                      minPointInd2 = j;
                    }
                  } else { // 如果点的线号小于最近点的线号，最近距离点更新在3中
                    if (pointSqDis < minPointSqDis3) {
                      minPointSqDis3 = pointSqDis;
                      minPointInd3 = j;
                    }
                  }
                }
                // 向后向前搜索完毕，总归是相同线的放在2，不同线的放在3
              }

              pointSearchSurfInd1[i] = closestPointInd; // kd-tree最近距离点,-1表示未找到满足要求的点
              pointSearchSurfInd2[i] = minPointInd2; // 同一线号上的距离最近的点，-1表示未找到满足要求的点
              pointSearchSurfInd3[i] = minPointInd3; // 不同线号上的距离最近的点，-1表示未找到满足要求的点
            }
            // “每迭代五次，重新寻找最近点和次近点”这个过程结束

            // 构建Jaccobian矩阵
            if (pointSearchSurfInd2[i] >= 0 && pointSearchSurfInd3[i] >= 0) { // 不为-1表示找到了满足要求的点
              tripod1 = laserCloudSurfLast->points[pointSearchSurfInd1[i]]; // A点
              tripod2 = laserCloudSurfLast->points[pointSearchSurfInd2[i]]; // B点
              tripod3 = laserCloudSurfLast->points[pointSearchSurfInd3[i]]; // C点

              // 向量AB = (tripod2.x - tripod1.x, tripod2.y - tripod1.y, tripod2.z - tripod1.z)
              // 向量AC = (tripod3.x - tripod1.x, tripod3.y - tripod1.y, tripod3.z - tripod1.z)

              // 向量AB AC的向量积(即叉乘)为：
              // |          i                      j                      k          |
              // |tripod2.x - tripod1.x  tripod2.y - tripod1.y  tripod2.z - tripod1.z|
              // |tripod3.x - tripod1.x  tripod3.y - tripod1.y  tripod3.z - tripod1.z|
              // pd2=AB×AC·OA/|AB×AC|，对O点坐标求导，分母为系数，分子为AB×AC·OA，AB×AC为系数，OA=(x0-x1)i+(y0-y1)j+(z0-z1)k
              // 对x0求导是AB×AC中的i部分，对y0求导是AB×AC中的j部分，对z0求导是k部分
              float pa = (tripod2.y - tripod1.y) * (tripod3.z - tripod1.z) 
                       - (tripod3.y - tripod1.y) * (tripod2.z - tripod1.z); // x轴方向分向量i
              float pb = (tripod2.z - tripod1.z) * (tripod3.x - tripod1.x) 
                       - (tripod3.z - tripod1.z) * (tripod2.x - tripod1.x); // y轴方向分向量j
              float pc = (tripod2.x - tripod1.x) * (tripod3.y - tripod1.y) 
                       - (tripod3.x - tripod1.x) * (tripod2.y - tripod1.y); // z轴方向分向量k
              float pd = -(pa * tripod1.x + pb * tripod1.y + pc * tripod1.z);

              float ps = sqrt(pa * pa + pb * pb + pc * pc); // 向量AB AC的向量积的模
              pa /= ps; // i方向单位分量
              pb /= ps; // j方向单位分量
              pc /= ps; // k方向单位分量
              pd /= ps;

              float pd2 = pa * pointSel.x + pb * pointSel.y + pc * pointSel.z + pd; // pointSel到ABC平面的距离：向量OA与与法向量的点积除以法向量的模

              // 没用到
              pointProj = pointSel;
              pointProj.x -= pa * pd2;
              pointProj.y -= pb * pd2;
              pointProj.z -= pc * pd2;

              float s = 1;
              if (iterCount >= 5) {
                s = 1 - 1.8 * fabs(pd2) / sqrt(sqrt(pointSel.x * pointSel.x
                  + pointSel.y * pointSel.y + pointSel.z * pointSel.z)); // 距离越小，平面特征点和雷达的距离越大，s越大
              }

              // 考虑权重
              coeff.x = s * pa;
              coeff.y = s * pb;
              coeff.z = s * pc;
              coeff.intensity = s * pd2;

              if (s > 0.1 && pd2 != 0) { // 只保留权重大的即距离比较小的点，同时也舍弃距离为零的点，放入laserCloudOri
                // 保存原始点与相应的系数
                laserCloudOri->push_back(surfPointsFlat->points[i]);
                coeffSel->push_back(coeff);
              }
            }
            // 一次迭代中的一个平面特征点处理完毕
          }
          // 一次迭代中的所有平面特征点处理完毕

          /* 3. L-M运动估计求解 */

          int pointSelNum = laserCloudOri->points.size(); // 匹配到的点的个数(即存在多少个约束) 
          if (pointSelNum < 10) {
            continue; // 如果符合权重条件的边缘和平面特征点少于10个，放弃本次迭代
          }

          cv::Mat matA(pointSelNum, 6, CV_32F, cv::Scalar::all(0));
          cv::Mat matAt(6, pointSelNum, CV_32F, cv::Scalar::all(0));
          cv::Mat matAtA(6, 6, CV_32F, cv::Scalar::all(0));
          cv::Mat matB(pointSelNum, 1, CV_32F, cv::Scalar::all(0));
          cv::Mat matAtB(6, 1, CV_32F, cv::Scalar::all(0));
          cv::Mat matX(6, 1, CV_32F, cv::Scalar::all(0));

          //计算matA，matB矩阵
          for (int i = 0; i < pointSelNum; i++) {
            pointOri = laserCloudOri->points[i]; // 当前时刻点坐标
            coeff = coeffSel->points[i]; // 该点所对应的偏导数

            float s = 1;
            // 采用Levenberg-Marquardt计算
            // 首先建立当前时刻Lidar坐标系下提取到的特征点与点到直线/平面的约束方程，而后对约束方程求对坐标变换(3旋转+3平移)的偏导，偏导保存在coeff
            // 公式参见论文(2)-(8)  
            // 0/1/2对应pitch yaw roll
            float srx = sin(s * transform[0]);  // sin(rx)
            float crx = cos(s * transform[0]);  // cos(rx)
            float sry = sin(s * transform[1]);  // sin(ry)
            float cry = cos(s * transform[1]);  // cos(ry)
            float srz = sin(s * transform[2]);  // sin(rz)
            float crz = cos(s * transform[2]);  // cos(rz)
            float tx = s * transform[3];
            float ty = s * transform[4];
            float tz = s * transform[5];

            // k+1帧的点云xyz经过TransformToStart()函数变换得到k+1帧起始位置的点云x'y'z'，x'y'z'是transform[6]的函数
            // 点云x'y'z'在k帧点云中寻找最近点得到距离值，完美的transform[6]使得距离值为0
            // 所以距离值对transform[6]求导使用LM算法求极值
            // 距离值对点云x'y'z'求导得到coeff
            // 点云x'y'z'对transform[6]求导可以由TransformToStart()函数计算得到
            // 从而得到距离值对transform[6]的导数arx,ary,arz,atx,aty,atz
            float arx = (-s*crx*sry*srz*pointOri.x + s*crx*crz*sry*pointOri.y + s*srx*sry*pointOri.z 
                      + s*tx*crx*sry*srz - s*ty*crx*crz*sry - s*tz*srx*sry) * coeff.x
                      + (s*srx*srz*pointOri.x - s*crz*srx*pointOri.y + s*crx*pointOri.z
                      + s*ty*crz*srx - s*tz*crx - s*tx*srx*srz) * coeff.y
                      + (s*crx*cry*srz*pointOri.x - s*crx*cry*crz*pointOri.y - s*cry*srx*pointOri.z
                      + s*tz*cry*srx + s*ty*crx*cry*crz - s*tx*crx*cry*srz) * coeff.z;

            float ary = ((-s*crz*sry - s*cry*srx*srz)*pointOri.x 
                      + (s*cry*crz*srx - s*sry*srz)*pointOri.y - s*crx*cry*pointOri.z 
                      + tx*(s*crz*sry + s*cry*srx*srz) + ty*(s*sry*srz - s*cry*crz*srx) 
                      + s*tz*crx*cry) * coeff.x
                      + ((s*cry*crz - s*srx*sry*srz)*pointOri.x 
                      + (s*cry*srz + s*crz*srx*sry)*pointOri.y - s*crx*sry*pointOri.z
                      + s*tz*crx*sry - ty*(s*cry*srz + s*crz*srx*sry) 
                      - tx*(s*cry*crz - s*srx*sry*srz)) * coeff.z;

            float arz = ((-s*cry*srz - s*crz*srx*sry)*pointOri.x + (s*cry*crz - s*srx*sry*srz)*pointOri.y
                      + tx*(s*cry*srz + s*crz*srx*sry) - ty*(s*cry*crz - s*srx*sry*srz)) * coeff.x
                      + (-s*crx*crz*pointOri.x - s*crx*srz*pointOri.y
                      + s*ty*crx*srz + s*tx*crx*crz) * coeff.y
                      + ((s*cry*crz*srx - s*sry*srz)*pointOri.x + (s*crz*sry + s*cry*srx*srz)*pointOri.y
                      + tx*(s*sry*srz - s*cry*crz*srx) - ty*(s*crz*sry + s*cry*srx*srz)) * coeff.z;

            float atx = -s*(cry*crz - srx*sry*srz) * coeff.x + s*crx*srz * coeff.y 
                      - s*(crz*sry + cry*srx*srz) * coeff.z;
  
            float aty = -s*(cry*srz + crz*srx*sry) * coeff.x - s*crx*crz * coeff.y 
                      - s*(sry*srz - cry*crz*srx) * coeff.z;
  
            float atz = s*crx*sry * coeff.x - s*srx * coeff.y - s*crx*cry * coeff.z;

            float d2 = coeff.intensity;

            matA.at<float>(i, 0) = arx;
            matA.at<float>(i, 1) = ary;
            matA.at<float>(i, 2) = arz;
            matA.at<float>(i, 3) = atx;
            matA.at<float>(i, 4) = aty;
            matA.at<float>(i, 5) = atz;
            matB.at<float>(i, 0) = -0.05 * d2; // 负号使得拟合效果越好，matB越小，0.05是LM算法的矫正系数
          }

          // 最小二乘计算(QR分解法)
          cv::transpose(matA, matAt); // 生成转置矩阵
          matAtA = matAt * matA;
          matAtB = matAt * matB;
          cv::solve(matAtA, matAtB, matX, cv::DECOMP_QR); // 求解matAtA * matX = matAtB，LM解算公式:(Jt·J)·delta=Jt·(Y-f(delta))，对应此处(At·A)·X=At·B

          if (iterCount == 0) { // 如果是首次迭代
            cv::Mat matE(1, 6, CV_32F, cv::Scalar::all(0)); // 特征值1*6矩阵
            cv::Mat matV(6, 6, CV_32F, cv::Scalar::all(0)); // 特征向量6*6矩阵
            cv::Mat matV2(6, 6, CV_32F, cv::Scalar::all(0));

            cv::eigen(matAtA, matE, matV); // 计算矩阵的特征向量E及特征向量的反对称阵V
            matV.copyTo(matV2);

            isDegenerate = false;
            float eignThre[6] = {10, 10, 10, 10, 10, 10}; // 特征值阈值
            for (int i = 5; i >= 0; i--) { // matE是从大到小排列的，迅速查找最小值
              if (matE.at<float>(0, i) < eignThre[i]) { // 如果AtA的某个特征值小于10，发生一定程度的退化
                for (int j = 0; j < 6; j++) { 
                  matV2.at<float>(i, j) = 0; // 对应特征向量置为0
                }
                isDegenerate = true; // 存在比10小的特征值则出现退化
              } else {
                break;
              }
            }

            matP = matV.inv() * matV2; // 计算P矩阵
          }

          if (isDegenerate) { // 如果发生退化，只使用预测矩阵P计算
            cv::Mat matX2(6, 1, CV_32F, cv::Scalar::all(0));
            matX.copyTo(matX2);
            matX = matP * matX2;
          }

          // 累加每次迭代的旋转平移量
          transform[0] += matX.at<float>(0, 0);
          transform[1] += matX.at<float>(1, 0);
          transform[2] += matX.at<float>(2, 0);
          transform[3] += matX.at<float>(3, 0);
          transform[4] += matX.at<float>(4, 0);
          transform[5] += matX.at<float>(5, 0);

          for(int i=0; i<6; i++){
            if(isnan(transform[i])) // 判断是否非数字
              transform[i]=0;
          }

          // 判断是否已收敛, 如果很小就停止迭代
          float deltaR = sqrt(
                              pow(rad2deg(matX.at<float>(0, 0)), 2) +
                              pow(rad2deg(matX.at<float>(1, 0)), 2) +
                              pow(rad2deg(matX.at<float>(2, 0)), 2)); // 旋转姿态角判断
          float deltaT = sqrt(
                              pow(matX.at<float>(3, 0) * 100, 2) +
                              pow(matX.at<float>(4, 0) * 100, 2) +
                              pow(matX.at<float>(5, 0) * 100, 2)); // 平移向量长度判断

          if (deltaR < 0.1 && deltaT < 0.1) { // 步长小于阈值
            break; // 结束所有迭代
          }
          // 结束本次迭代
        }
        // 结束所有迭代
      }

      /**********************************
       *********** 3.坐标转换 ***********
       *********************************/

      float rx, ry, rz, tx, ty, tz; // 当前帧lidar全局位姿
      // 计算旋转角的累计变化量
      AccumulateRotation(transformSum[0], transformSum[1], transformSum[2], 
                         -transform[0], -transform[1] * 1.05, -transform[2], rx, ry, rz);

      // 帧间相对位移去除imu重力加速度影响 旋转到全局坐标系  
      float x1 = cos(rz) * (transform[3] - imuShiftFromStartX) 
               - sin(rz) * (transform[4] - imuShiftFromStartY);
      float y1 = sin(rz) * (transform[3] - imuShiftFromStartX) 
               + cos(rz) * (transform[4] - imuShiftFromStartY);
      float z1 = transform[5] * 1.05 - imuShiftFromStartZ;

      float x2 = x1;
      float y2 = cos(rx) * y1 - sin(rx) * z1;
      float z2 = sin(rx) * y1 + cos(rx) * z1;

      // 求相对于原点的平移量
      tx = transformSum[3] - (cos(ry) * x2 + sin(ry) * z2);
      ty = transformSum[4] - y2;
      tz = transformSum[5] - (-sin(ry) * x2 + cos(ry) * z2);

      // 考虑惯导当前帧首末时刻姿态差得到最终lidar全局姿态
      PluginIMURotation(rx, ry, rz, imuPitchStart, imuYawStart, imuRollStart, 
                        imuPitchLast, imuYawLast, imuRollLast, rx, ry, rz);

      // 得到世界坐标系下的转移矩阵
      transformSum[0] = rx;
      transformSum[1] = ry;
      transformSum[2] = rz;
      transformSum[3] = tx;
      transformSum[4] = ty;
      transformSum[5] = tz;

      // 欧拉角转换成四元数
      geometry_msgs::Quaternion geoQuat = tf::createQuaternionMsgFromRollPitchYaw(rz, -rx, -ry);

      laserOdometry.header.stamp = ros::Time().fromSec(timeSurfPointsLessFlat);
      laserOdometry.pose.pose.orientation.x = -geoQuat.y;
      laserOdometry.pose.pose.orientation.y = -geoQuat.z;
      laserOdometry.pose.pose.orientation.z = geoQuat.x;
      laserOdometry.pose.pose.orientation.w = geoQuat.w;
      laserOdometry.pose.pose.position.x = tx;
      laserOdometry.pose.pose.position.y = ty;
      laserOdometry.pose.pose.position.z = tz;
      pubLaserOdometry.publish(laserOdometry); // 发布四元数和平移量

      laserOdometryTrans.stamp_ = ros::Time().fromSec(timeSurfPointsLessFlat);
      laserOdometryTrans.setRotation(tf::Quaternion(-geoQuat.y, -geoQuat.z, geoQuat.x, geoQuat.w));
      laserOdometryTrans.setOrigin(tf::Vector3(tx, ty, tz));
      tfBroadcaster.sendTransform(laserOdometryTrans); // 广播新的平移旋转之后的坐标系(rviz)

      // 对点云的曲率比较大和比较小的点，即所有特征点，投影到扫描结束位置，但是相比于laserCloudFullRes，少了剔除的特征点以及体素栅格滤波的点
      int cornerPointsLessSharpNum = cornerPointsLessSharp->points.size();
      for (int i = 0; i < cornerPointsLessSharpNum; i++) {
        TransformToEnd(&cornerPointsLessSharp->points[i], &cornerPointsLessSharp->points[i]);
      }

      int surfPointsLessFlatNum = surfPointsLessFlat->points.size();
      for (int i = 0; i < surfPointsLessFlatNum; i++) {
        TransformToEnd(&surfPointsLessFlat->points[i], &surfPointsLessFlat->points[i]);
      }

      frameCount++;
      if (frameCount >= skipFrameNum + 1) { // 每间隔一个点云数据帧，相对点云最后一个点进行畸变校正
        int laserCloudFullResNum = laserCloudFullRes->points.size();
        for (int i = 0; i < laserCloudFullResNum; i++) {
          TransformToEnd(&laserCloudFullRes->points[i], &laserCloudFullRes->points[i]); // 畸变校正，上一节点里的所有点
        }
      }

      // 畸变校正之后的点作为last点保存等下个点云进来进行匹配
      pcl::PointCloud<PointType>::Ptr laserCloudTemp = cornerPointsLessSharp;
      cornerPointsLessSharp = laserCloudCornerLast;
      laserCloudCornerLast = laserCloudTemp;

      laserCloudTemp = surfPointsLessFlat;
      surfPointsLessFlat = laserCloudSurfLast;
      laserCloudSurfLast = laserCloudTemp;

      laserCloudCornerLastNum = laserCloudCornerLast->points.size();
      laserCloudSurfLastNum = laserCloudSurfLast->points.size();
      // 点足够多就构建kd-tree，否则弃用此帧，沿用上一帧数据的kd-tree
      if (laserCloudCornerLastNum > 10 && laserCloudSurfLastNum > 100) {
        kdtreeCornerLast->setInputCloud(laserCloudCornerLast);
        kdtreeSurfLast->setInputCloud(laserCloudSurfLast);
      }

      if (frameCount >= skipFrameNum + 1) { // 每间隔一个点云数据帧，发布边沿点，平面点以及全部点 done 第一帧怎么变换 :第一帧把所有点作为target，同时发布给下一节点，第二帧提取的特征点在第一帧中匹配
        frameCount = 0;

        sensor_msgs::PointCloud2 laserCloudCornerLast2;
        pcl::toROSMsg(*laserCloudCornerLast, laserCloudCornerLast2);
        laserCloudCornerLast2.header.stamp = ros::Time().fromSec(timeSurfPointsLessFlat);
        laserCloudCornerLast2.header.frame_id = "/camera";
        pubLaserCloudCornerLast.publish(laserCloudCornerLast2);

        sensor_msgs::PointCloud2 laserCloudSurfLast2;
        pcl::toROSMsg(*laserCloudSurfLast, laserCloudSurfLast2);
        laserCloudSurfLast2.header.stamp = ros::Time().fromSec(timeSurfPointsLessFlat);
        laserCloudSurfLast2.header.frame_id = "/camera";
        pubLaserCloudSurfLast.publish(laserCloudSurfLast2);

        sensor_msgs::PointCloud2 laserCloudFullRes3;
        pcl::toROSMsg(*laserCloudFullRes, laserCloudFullRes3);
        laserCloudFullRes3.header.stamp = ros::Time().fromSec(timeSurfPointsLessFlat);
        laserCloudFullRes3.header.frame_id = "/camera";
        pubLaserCloudFullRes.publish(laserCloudFullRes3);
      }
    }

    status = ros::ok();
    rate.sleep();
  }

  return 0;
}
