/*
 * Software License Agreement (BSD License)
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2011, Willow Garage, Inc.
 *
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Willow Garage, Inc. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 */

#include <iostream>
#include <algorithm>

#include <pcl/common/time.h>
#include <pcl/gpu/kinfu_large_scale/kinfu.h>
#include "estimate_combined.h"
#include "internal.h"

#include <Eigen/Core>
#include <Eigen/SVD>
#include <Eigen/Cholesky>
#include <Eigen/Geometry>
#include <Eigen/LU>

#ifdef HAVE_OPENCV
  #include <opencv2/opencv.hpp>
  //~ #include <opencv2/gpu/gpu.hpp>
#endif

using namespace std;
using namespace pcl::device::kinfuLS;

using pcl::device::kinfuLS::device_cast;
using Eigen::AngleAxisf;
using Eigen::Array3f;
using Eigen::Vector3i;
using Eigen::Vector3f;

namespace pcl
{
  namespace gpu
  {
    namespace kinfuLS
    {
      Eigen::Vector3f rodrigues2(const Eigen::Matrix3f& matrix);
    }
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
pcl::gpu::kinfuLS::KinfuTracker::KinfuTracker (const Eigen::Vector3f &volume_size, const float shiftingDistance) :
  cyclical_( DISTANCE_THRESHOLD, VOLUME_SIZE, VOLUME_X), rows_(480), cols_(680), focal_length_(FOCAL_LENGTH), global_time_(0), max_icp_distance_(0), integration_metric_threshold_(0.f), perform_last_scan_ (false), finished_(false), lost_ (false), disable_icp_ (false), use_external_poses_(false)
{
  //const Vector3f volume_size = Vector3f::Constant (VOLUME_SIZE);
  const Vector3i volume_resolution (VOLUME_X, VOLUME_Y, VOLUME_Z);

  volume_size_ = volume_size(0);

  tsdf_volume_ = TsdfVolume::Ptr ( new TsdfVolume(volume_resolution) );
  tsdf_volume_->setSize (volume_size);

  shifting_distance_ = shiftingDistance;

  // set cyclical buffer values
  cyclical_.setDistanceThreshold (shifting_distance_);
  cyclical_.setVolumeSize (volume_size_, volume_size_, volume_size_);

  init_Rcam_ = Eigen::Matrix3f::Identity ();// * AngleAxisf(-30.f/180*3.1415926, Vector3f::UnitX());
  init_tcam_ = volume_size * 0.5f - Vector3f (0, 0, volume_size (2) / 2 * 1.2f);

  const int iters[] = {10, 5, 4};
  std::copy (iters, iters + LEVELS, icp_iterations_);

  const float default_distThres = 0.10f; //meters
  const float default_angleThres = sin (20.f * 3.14159254f / 180.f);
  const float default_tranc_dist = 0.03f; //meters

  setIcpCorespFilteringParams (default_distThres, default_angleThres);
  tsdf_volume_->setTsdfTruncDist (default_tranc_dist);

  rmats_.reserve (30000);
  tvecs_.reserve (30000);

  reset ();

  // initialize cyclical buffer
  cyclical_.initBuffer(tsdf_volume_);

  last_estimated_rotation_= Eigen::Matrix3f::Identity ();
  last_estimated_translation_= volume_size * 0.5f - Vector3f (0, 0, volume_size (2) / 2 * 1.2f);

}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl::gpu::kinfuLS::KinfuTracker::setDepthIntrinsics (int rows, int cols, float fx, float fy, float cx, float cy)
{
  rows_ = rows;
  cols_ = cols;
  fx_ = fx;
  fy_ = fy;
  cx_ = (cx == -1) ? cols_/2-0.5f : cx;
  cy_ = (cy == -1) ? rows_/2-0.5f : cy;

  allocateBufffers (rows_, cols_);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl::gpu::kinfuLS::KinfuTracker::setInitialCameraPose (const Eigen::Affine3f& pose)
{
  init_Rcam_ = pose.rotation ();
  init_tcam_ = pose.translation ();
  //reset (); // (already called in constructor)
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl::gpu::kinfuLS::KinfuTracker::setDepthTruncationForICP (float max_icp_distance)
{
  max_icp_distance_ = max_icp_distance;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl::gpu::kinfuLS::KinfuTracker::setCameraMovementThreshold(float threshold)
{
  integration_metric_threshold_ = threshold;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl::gpu::kinfuLS::KinfuTracker::setRelativeLeftCameraPosition(float x, float y, float z) {
  relative_R_R_L_ = Eigen::Matrix3f::Identity();
  relative_t_R_L_ = Vector3f(x, y, z);

  relative_R_L_R_ = relative_R_R_L_.inverse();
  relative_t_L_R_ = -relative_R_L_R_ * relative_t_R_L_;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl::gpu::kinfuLS::KinfuTracker::getRelativeLeftCameraPose(Matrix3frm& relative_R_L_R, Vector3f& relative_t_L_R) {
  relative_R_L_R = relative_R_L_R_;
  relative_t_L_R = relative_t_L_R_;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl::gpu::kinfuLS::KinfuTracker::setGlobalCameraPoses(vector< Matrix3frm > Rs_cam_g, vector< Vector3f > ts_cam_g) {
  Matrix3frm R_c1_w = init_Rcam_.inverse();
  for(int i = 0; i < Rs_cam_g.size(); i++) {
    Matrix3frm R_R_cam_g = Rs_cam_g.at(i);
    Vector3f t_R_cam_g = ts_cam_g.at(i);

    // Move camera according to init camera pose
    t_R_cam_g = R_R_cam_g * (-R_c1_w * init_tcam_) + t_R_cam_g;
    R_R_cam_g = R_R_cam_g * R_c1_w;

    Matrix3frm R_R_g_cam = R_R_cam_g.inverse();
    Vector3f t_R_g_cam = -R_R_g_cam * t_R_cam_g;

    Matrix3frm R_L_cam_g = relative_R_L_R_ * R_R_cam_g;
    Vector3f t_L_cam_g = relative_R_L_R_ * t_R_cam_g + relative_t_L_R_;

    Matrix3frm R_L_g_cam = R_L_cam_g.inverse();
    Vector3f t_L_g_cam = -R_L_g_cam * t_L_cam_g;

    R_L_cam_g_.push_back(R_L_cam_g);
    R_R_cam_g_.push_back(R_R_cam_g);
    R_L_g_cam_.push_back(R_L_g_cam);
    R_R_g_cam_.push_back(R_R_g_cam);

    t_L_cam_g_.push_back(t_L_cam_g);
    t_R_cam_g_.push_back(t_R_cam_g);
    t_L_g_cam_.push_back(t_L_g_cam);
    t_R_g_cam_.push_back(t_R_g_cam);
  }
  use_external_poses_ = true;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl::gpu::kinfuLS::KinfuTracker::getGlobalLeftCameraRotation(Matrix3frm& rotation, int index) {
  if (index != -1 && index >= 0 && index < R_L_g_cam_.size() - 1) {
    rotation = R_L_g_cam_.at(index);
  } else {
    rotation = current_R_L_g_cam_;
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl::gpu::kinfuLS::KinfuTracker::getGlobalRightCameraRotation(Matrix3frm& rotation, int index) {
  if (index != -1 && index >= 0 && index < R_R_g_cam_.size() - 1) {
    rotation = R_R_g_cam_.at(index);
  } else {
    rotation = current_R_R_g_cam_;
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl::gpu::kinfuLS::KinfuTracker::getGlobalLeftCameraTranslation(Vector3f& translation, int index) {
  if (index != -1 && index >= 0 && index < t_L_g_cam_.size() - 1) {
    translation = t_L_g_cam_.at(index);
  } else {
    translation = current_t_L_g_cam_;
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl::gpu::kinfuLS::KinfuTracker::getGlobalRightCameraTranslation(Vector3f& translation, int index) {
  if (index != -1 && index >= 0 && index < t_R_g_cam_.size() - 1) {
    translation = t_R_g_cam_.at(index);
  } else {
    translation = current_t_R_g_cam_;
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl::gpu::kinfuLS::KinfuTracker::getLeftCameraRotation(Matrix3frm& rotation, int index) {
  if (index != -1 && index >= 0 && index < R_L_cam_g_.size() - 1) {
    rotation = R_L_cam_g_.at(index);
  } else {
    rotation = current_R_L_cam_g_;
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl::gpu::kinfuLS::KinfuTracker::getRightCameraRotation(Matrix3frm& rotation, int index) {
  if (index != -1 && index >= 0 && index < R_R_cam_g_.size() - 1) {
    rotation = R_R_cam_g_.at(index);
  } else {
    rotation = current_R_R_cam_g_;
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl::gpu::kinfuLS::KinfuTracker::getLeftCameraTranslation(Vector3f& translation, int index) {
  if (index != -1 && index >= 0 && index < t_L_cam_g_.size() - 1) {
    translation = t_L_cam_g_.at(index);
  } else {
    translation = current_t_L_cam_g_;
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl::gpu::kinfuLS::KinfuTracker::getRightCameraTranslation(Vector3f& translation, int index) {
  if (index != -1 && index >= 0 && index < t_R_cam_g_.size() - 1) {
    translation = t_R_cam_g_.at(index);
  } else {
    translation = current_t_R_cam_g_;
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void pcl::gpu::kinfuLS::KinfuTracker::getVMapL(MapArr& vmaps) {
  vmaps = vmaps_g_prev_L_[0];
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void pcl::gpu::kinfuLS::KinfuTracker::getVMapR(MapArr& vmaps) {
  vmaps = vmaps_g_prev_R_[0];
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void pcl::gpu::kinfuLS::KinfuTracker::getNMapL(MapArr& nmaps) {
  nmaps = nmaps_g_prev_L_[0];
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void pcl::gpu::kinfuLS::KinfuTracker::getNMapR(MapArr& nmaps) {
  nmaps = nmaps_g_prev_R_[0];
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void pcl::gpu::kinfuLS::KinfuTracker::getVNMaps(Matrix3frm& R_g_cam, Vector3f& t_g_cam, MapArr& vmaps, MapArr& nmaps) {
  // Intrisics of the camera
  Intr intr (fx_, fy_, cx_, cy_);

  // Physical volume size (meters)
  float3 device_volume_size = device_cast<const float3> (tsdf_volume_->getSize());

  // Allocate maps
  std::vector<MapArr> vmaps_tmp;
  std::vector<MapArr> nmaps_tmp;
  vmaps_tmp.resize(LEVELS);
  nmaps_tmp.resize(LEVELS);
  int rows = 480;
  int cols = 640;
  for (int i = 0; i < LEVELS; ++i)
  {
    int pyr_rows = rows >> i;
    int pyr_cols = cols >> i;
    vmaps_tmp[i].create (pyr_rows*3, pyr_cols);
    nmaps_tmp[i].create (pyr_rows*3, pyr_cols);
  }

  // Convert pose
  Mat33 R_device_g_cam;
  float3 t_device_g_cam;
  convertTransforms(R_g_cam,
                    t_g_cam,
                    R_device_g_cam,
                    t_device_g_cam);

  // Generate synthetic vertex and normal maps
  raycast(intr, R_device_g_cam, t_device_g_cam, tsdf_volume_->getTsdfTruncDist (), device_volume_size, tsdf_volume_->data (), getCyclicalBufferStructure (), vmaps_tmp[0], nmaps_tmp[0]);
  vmaps = vmaps_tmp[0];
  nmaps = nmaps_tmp[0];
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl::gpu::kinfuLS::KinfuTracker::setIcpCorespFilteringParams (float distThreshold, float sineOfAngle)
{
  distThres_  = distThreshold; //mm
  angleThres_ = sineOfAngle;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int
pcl::gpu::kinfuLS::KinfuTracker::cols ()
{
  return (cols_);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int
pcl::gpu::kinfuLS::KinfuTracker::rows ()
{
  return (rows_);
}

void
pcl::gpu::kinfuLS::KinfuTracker::extractAndSaveWorld ()
{

  //extract current volume to world model
  PCL_INFO("Extracting current volume...");
  cyclical_.checkForShift(tsdf_volume_, getCameraPose (), 0.6 * volume_size_, true, true, true); // this will force the extraction of the whole cube.
  PCL_INFO("Done\n");

  finished_ = true; // TODO maybe we could add a bool param to prevent kinfuLS from exiting after we saved the current world model

  int cloud_size = 0;

  cloud_size = cyclical_.getWorldModel ()->getWorld ()->points.size();

  if (cloud_size <= 0)
  {
    PCL_WARN ("World model currently has no points. Skipping save procedure.\n");
    return;
  }
  else
  {
    PCL_INFO ("Saving current world to world.pcd with %d points.\n", cloud_size);
    pcl::io::savePCDFile<pcl::PointXYZI> ("world.pcd", *(cyclical_.getWorldModel ()->getWorld ()), true);
    return;
  }

}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl::gpu::kinfuLS::KinfuTracker::reset ()
{
  cout << "in reset function!" << std::endl;

  if (global_time_)
    PCL_WARN ("Reset\n");

  // dump current world to a pcd file
  /*
  if (global_time_)
  {
    PCL_INFO ("Saving current world to current_world.pcd\n");
    pcl::io::savePCDFile<pcl::PointXYZI> ("current_world.pcd", *(cyclical_.getWorldModel ()->getWorld ()), true);
    // clear world model
    cyclical_.getWorldModel ()->reset ();
  }
  */

  // clear world model
  cyclical_.getWorldModel ()->reset ();


  global_time_ = 0;
  rmats_.clear ();
  tvecs_.clear ();

  rmats_.push_back (init_Rcam_);
  tvecs_.push_back (init_tcam_);

  tsdf_volume_->reset ();

  // reset cyclical buffer as well
  cyclical_.resetBuffer (tsdf_volume_);

  if (color_volume_) // color integration mode is enabled
    color_volume_->reset ();

  // reset estimated pose
  last_estimated_rotation_= Eigen::Matrix3f::Identity ();
  last_estimated_translation_= Vector3f (volume_size_, volume_size_, volume_size_) * 0.5f - Vector3f (0, 0, volume_size_ / 2 * 1.2f);


  lost_=false;
  has_shifted_=false;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl::gpu::kinfuLS::KinfuTracker::allocateBufffers (int rows, int cols)
{
  depths_curr_R_.resize (LEVELS);
  vmaps_g_curr_R_.resize (LEVELS);
  nmaps_g_curr_R_.resize (LEVELS);

  vmaps_g_prev_R_.resize (LEVELS);
  nmaps_g_prev_R_.resize (LEVELS);
  vmaps_g_prev_L_.resize (LEVELS);
  nmaps_g_prev_L_.resize (LEVELS);

  vmaps_curr_R_.resize (LEVELS);
  nmaps_curr_R_.resize (LEVELS);

  vmaps_prev_R_.resize (LEVELS);
  nmaps_prev_R_.resize (LEVELS);

  coresps_R_.resize (LEVELS);

  for (int i = 0; i < LEVELS; ++i)
  {
    int pyr_rows = rows >> i;
    int pyr_cols = cols >> i;

    depths_curr_R_[i].create (pyr_rows, pyr_cols);

    vmaps_g_curr_R_[i].create (pyr_rows*3, pyr_cols);
    nmaps_g_curr_R_[i].create (pyr_rows*3, pyr_cols);

    vmaps_g_prev_R_[i].create (pyr_rows*3, pyr_cols);
    nmaps_g_prev_R_[i].create (pyr_rows*3, pyr_cols);
    vmaps_g_prev_L_[i].create (pyr_rows*3, pyr_cols);
    nmaps_g_prev_L_[i].create (pyr_rows*3, pyr_cols);

    vmaps_curr_R_[i].create (pyr_rows*3, pyr_cols);
    nmaps_curr_R_[i].create (pyr_rows*3, pyr_cols);

    vmaps_prev_R_[i].create (pyr_rows*3, pyr_cols);
    nmaps_prev_R_[i].create (pyr_rows*3, pyr_cols);

    coresps_R_[i].create (pyr_rows, pyr_cols);
  }
  depthRawScaled_.create (rows, cols);
  // see estimate tranform for the magic numbers
  int r = (int)ceil ( ((float)rows) / ESTIMATE_COMBINED_CUDA_GRID_Y );
  int c = (int)ceil ( ((float)cols) / ESTIMATE_COMBINED_CUDA_GRID_X );
  gbuf_.create (27, r * c);
  sumbuf_.create (27);
}

inline void
pcl::gpu::kinfuLS::KinfuTracker::convertTransforms (Matrix3frm& rotation_in_1, Matrix3frm& rotation_in_2, Vector3f& translation_in_1, Vector3f& translation_in_2, Mat33& rotation_out_1, Mat33& rotation_out_2, float3& translation_out_1, float3& translation_out_2)
{
  rotation_out_1 = device_cast<Mat33> (rotation_in_1);
  rotation_out_2 = device_cast<Mat33> (rotation_in_2);
  translation_out_1 = device_cast<float3>(translation_in_1);
  translation_out_2 = device_cast<float3>(translation_in_2);
}

inline void
pcl::gpu::kinfuLS::KinfuTracker::convertTransforms (Matrix3frm& rotation_in_1, Matrix3frm& rotation_in_2, Vector3f& translation_in, Mat33& rotation_out_1, Mat33& rotation_out_2, float3& translation_out)
{
  rotation_out_1 = device_cast<Mat33> (rotation_in_1);
  rotation_out_2 = device_cast<Mat33> (rotation_in_2);
  translation_out = device_cast<float3>(translation_in);
}

inline void
pcl::gpu::kinfuLS::KinfuTracker::convertTransforms (Matrix3frm& rotation_in, Vector3f& translation_in, Mat33& rotation_out, float3& translation_out)
{
  rotation_out = device_cast<Mat33> (rotation_in);
  translation_out = device_cast<float3>(translation_in);
}

inline void
pcl::gpu::kinfuLS::KinfuTracker::prepareMaps (const DepthMap& depth_raw, const Intr& cam_intrinsics)
{
  // blur raw map
  bilateralFilter (depth_raw, depths_curr_R_[0]);

  // optionally remove points that are farther than a threshold
  if (max_icp_distance_ > 0)
    truncateDepth(depths_curr_R_[0], max_icp_distance_);

  // downsample map for each pyramid level
  for (int i = 1; i < LEVELS; ++i)
    pyrDown (depths_curr_R_[i-1], depths_curr_R_[i]);

  // create vertex and normal maps for each pyramid level
  for (int i = 0; i < LEVELS; ++i)
  {
    createVMap (cam_intrinsics(i), depths_curr_R_[i], vmaps_curr_R_[i]);
    computeNormalsEigen (vmaps_curr_R_[i], nmaps_curr_R_[i]);
  }

}

inline void
pcl::gpu::kinfuLS::KinfuTracker::saveCurrentMaps()
{
  Matrix3frm rot_id = Eigen::Matrix3f::Identity ();
  Mat33 identity_rotation = device_cast<Mat33> (rot_id);

  // save vertex and normal maps for each level, keeping camera coordinates (i.e. no transform)
  for (int i = 0; i < LEVELS; ++i)
  {
   transformMaps (vmaps_curr_R_[i], nmaps_curr_R_[i],identity_rotation, make_float3(0,0,0), vmaps_prev_R_[i], nmaps_prev_R_[i]);
  }
}

inline bool
pcl::gpu::kinfuLS::KinfuTracker::performICP(const Intr& cam_intrinsics, Matrix3frm& previous_global_rotation, Vector3f& previous_global_translation, Matrix3frm& current_global_rotation , Vector3f& current_global_translation)
{

  if(disable_icp_)
  {
    lost_=true;
    return (false);
  }

  // Compute inverse rotation
  Matrix3frm previous_global_rotation_inv = previous_global_rotation.inverse ();  // Rprev.t();

 ///////////////////////////////////////////////
  // Convert pose to device type
  Mat33  device_cam_rot_local_prev_inv;
  float3 device_cam_trans_local_prev;
  convertTransforms(previous_global_rotation_inv, previous_global_translation, device_cam_rot_local_prev_inv, device_cam_trans_local_prev);
  device_cam_trans_local_prev -= getCyclicalBufferStructure ()->origin_metric; ;

  // Use temporary pose, so that we modify the current global pose only if ICP converged
  Matrix3frm resulting_rotation;
  Vector3f resulting_translation;

  // Initialize output pose to current pose
  current_global_rotation = previous_global_rotation;
  current_global_translation = previous_global_translation;

  ///////////////////////////////////////////////
  // Run ICP
  {
    //ScopeTime time("icp-all");
    for (int level_index = LEVELS-1; level_index>=0; --level_index)
    {
      int iter_num = icp_iterations_[level_index];

      // current vertex and normal maps
      MapArr& vmap_curr_R = vmaps_curr_R_[level_index];
      MapArr& nmap_curr_R = nmaps_curr_R_[level_index];

      // previous vertex and normal maps
      MapArr& vmap_g_prev_R = vmaps_g_prev_R_[level_index];
      MapArr& nmap_g_prev_R = nmaps_g_prev_R_[level_index];

      // We need to transform the maps from global to local coordinates
      Mat33&  rotation_id = device_cast<Mat33> (rmats_[0]); // Identity Rotation Matrix. Because we only need translation
      float3 cube_origin = (getCyclicalBufferStructure ())->origin_metric;
      cube_origin = -cube_origin;

      MapArr& vmap_temp_R = vmap_g_prev_R;
      MapArr& nmap_temp_R = nmap_g_prev_R;
      transformMaps (vmap_temp_R, nmap_temp_R, rotation_id, cube_origin, vmap_g_prev_R, nmap_g_prev_R);

      // run ICP for iter_num iterations (return false when lost)
      for (int iter = 0; iter < iter_num; ++iter)
      {
        //CONVERT POSES TO DEVICE TYPES
        // CURRENT LOCAL POSE
        Mat33 R_device_R_g_cam = device_cast<Mat33> (current_global_rotation); // We do not deal with changes in rotations
        float3 t_device_R_g_cam = device_cast<float3> (current_global_translation);
        t_device_R_g_cam -= getCyclicalBufferStructure ()->origin_metric;

        Eigen::Matrix<double, 6, 6, Eigen::RowMajor> A;
        Eigen::Matrix<double, 6, 1> b;

        // call the ICP function (see paper by Kok-Lim Low "Linear Least-squares Optimization for Point-to-Plane ICP Surface Registration")
        estimateCombined (R_device_R_g_cam, t_device_R_g_cam, vmap_curr_R, nmap_curr_R, device_cam_rot_local_prev_inv, device_cam_trans_local_prev, cam_intrinsics (level_index),
                          vmap_g_prev_R, nmap_g_prev_R, distThres_, angleThres_, gbuf_, sumbuf_, A.data (), b.data ());

        // checking nullspace
        double det = A.determinant ();

        if ( fabs (det) < 100000 /*1e-15*/ || pcl_isnan (det) ) //TODO find a threshold that makes ICP track well, but prevents it from generating wrong transforms
        {
          if (pcl_isnan (det)) cout << "qnan" << endl;
          if(lost_ == false)
            PCL_ERROR ("ICP LOST... PLEASE COME BACK TO THE LAST VALID POSE (green)\n");
          //reset (); //GUI will now show the user that ICP is lost. User needs to press "R" to reset the volume
          lost_ = true;
          return (false);
        }

        Eigen::Matrix<float, 6, 1> result = A.llt ().solve (b).cast<float>();
        float alpha = result (0);
        float beta  = result (1);
        float gamma = result (2);

        // deduce incremental rotation and translation from ICP's results
        Eigen::Matrix3f cam_rot_incremental = (Eigen::Matrix3f)AngleAxisf (gamma, Vector3f::UnitZ ()) * AngleAxisf (beta, Vector3f::UnitY ()) * AngleAxisf (alpha, Vector3f::UnitX ());
        Vector3f cam_trans_incremental = result.tail<3> ();

        //compose global pose
        current_global_translation = cam_rot_incremental * current_global_translation + cam_trans_incremental;
        current_global_rotation = cam_rot_incremental * current_global_rotation;
      }
    }
  }
  // ICP has converged
  lost_ = false;
  return (true);
}

inline bool
pcl::gpu::kinfuLS::KinfuTracker::performPairWiseICP(const Intr cam_intrinsics, Matrix3frm& resulting_rotation , Vector3f& resulting_translation)
{
  // we assume that both v and n maps are in the same coordinate space
  // initialize rotation and translation to respectively identity and zero
  Matrix3frm previous_rotation = Eigen::Matrix3f::Identity ();
  Matrix3frm previous_rotation_inv = previous_rotation.inverse ();
  Vector3f previous_translation = Vector3f(0,0,0);

 ///////////////////////////////////////////////
  // Convert pose to device type
  Mat33  device_cam_rot_prev_inv;
  float3 device_cam_trans_prev;
  convertTransforms(previous_rotation_inv, previous_translation, device_cam_rot_prev_inv, device_cam_trans_prev);

  // Initialize output pose to current pose (i.e. identity and zero translation)
  Matrix3frm current_rotation = previous_rotation;
  Vector3f current_translation = previous_translation;

  ///////////////////////////////////////////////
  // Run ICP
  {
    //ScopeTime time("icp-all");
    for (int level_index = LEVELS-1; level_index>=0; --level_index)
    {
      int iter_num = icp_iterations_[level_index];

      // current vertex and normal maps
      MapArr& vmap_curr_R = vmaps_curr_R_[level_index];
      MapArr& nmap_curr_R = nmaps_curr_R_[level_index];

      // previous vertex and normal maps
      MapArr& vmap_prev_R = vmaps_prev_R_[level_index];
      MapArr& nmap_prev_R = nmaps_prev_R_[level_index];

      // no need to transform maps from global to local since they are both in camera coordinates

      // run ICP for iter_num iterations (return false when lost)
      for (int iter = 0; iter < iter_num; ++iter)
      {
        //CONVERT POSES TO DEVICE TYPES
        // CURRENT LOCAL POSE
        Mat33  R_device_R_g_cam = device_cast<Mat33> (current_rotation);
        float3 t_device_R_g_cam = device_cast<float3> (current_translation);

        Eigen::Matrix<double, 6, 6, Eigen::RowMajor> A;
        Eigen::Matrix<double, 6, 1> b;

        // call the ICP function (see paper by Kok-Lim Low "Linear Least-squares Optimization for Point-to-Plane ICP Surface Registration")
        estimateCombined (R_device_R_g_cam, t_device_R_g_cam, vmap_curr_R, nmap_curr_R, device_cam_rot_prev_inv, device_cam_trans_prev, cam_intrinsics (level_index),
                          vmap_prev_R, nmap_prev_R, distThres_, angleThres_, gbuf_, sumbuf_, A.data (), b.data ());

        // checking nullspace
        double det = A.determinant ();

        if ( fabs (det) < 1e-15 || pcl_isnan (det) )
        {
          if (pcl_isnan (det)) cout << "qnan" << endl;

          PCL_WARN ("ICP PairWise LOST...\n");
          //reset ();
          return (false);
        }

        Eigen::Matrix<float, 6, 1> result = A.llt ().solve (b).cast<float>();
        float alpha = result (0);
        float beta  = result (1);
        float gamma = result (2);

        // deduce incremental rotation and translation from ICP's results
        Eigen::Matrix3f cam_rot_incremental = (Eigen::Matrix3f)AngleAxisf (gamma, Vector3f::UnitZ ()) * AngleAxisf (beta, Vector3f::UnitY ()) * AngleAxisf (alpha, Vector3f::UnitX ());
        Vector3f cam_trans_incremental = result.tail<3> ();

        //compose global pose
        current_translation = cam_rot_incremental * current_translation + cam_trans_incremental;
        current_rotation = cam_rot_incremental * current_rotation;
      }
    }
  }

  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // since raw depthmaps are quite noisy, we make sure the estimated transform is big enought to be taken into account
  float rnorm = rodrigues2(current_rotation).norm();
  float tnorm = (current_translation).norm();
  const float alpha = 1.f;
  bool integrate = (rnorm + alpha * tnorm)/2 >= integration_metric_threshold_ * 2.0f;

  if(integrate)
  {
    resulting_rotation = current_rotation;
    resulting_translation = current_translation;
  }
  else
  {
    resulting_rotation = Eigen::Matrix3f::Identity ();
    resulting_translation = Vector3f(0,0,0);
  }
  // ICP has converged
  return (true);
}

bool
pcl::gpu::kinfuLS::KinfuTracker::operator() (const DepthMap& depth_raw)
{
  // Intrisics of the camera
  Intr intr (fx_, fy_, cx_, cy_);

  // Physical volume size (meters)
  float3 device_volume_size = device_cast<const float3> (tsdf_volume_->getSize());

  // process the incoming raw depth map
  prepareMaps (depth_raw, intr);

  // sync GPU device
  pcl::device::kinfuLS::sync ();

  ///////////////////////////////////////////////////////////////////////////////////////////
  // Initialization at first frame
  if (global_time_ == 0) // this is the frist frame, the tsdf volume needs to be initialized
  {
    // Initial rotation
    Matrix3frm initial_cam_rot = rmats_[0]; //  [Ri|ti] - pos of camera
    Matrix3frm initial_cam_rot_inv = initial_cam_rot.inverse ();
    // Initial translation
    Vector3f   initial_cam_trans = tvecs_[0]; //  transform from camera to global coo space for (i-1)th camera pose

    // Convert pose to device types
    Mat33 device_initial_cam_rot, device_initial_cam_rot_inv;
    float3 device_initial_cam_trans;
    convertTransforms (initial_cam_rot, initial_cam_rot_inv, initial_cam_trans, device_initial_cam_rot, device_initial_cam_rot_inv, device_initial_cam_trans);

    // integrate current depth map into tsdf volume, from default initial pose.
    integrateTsdfVolume(depth_raw, intr, device_volume_size, device_initial_cam_rot_inv, device_initial_cam_trans, tsdf_volume_->getTsdfTruncDist(), tsdf_volume_->data(), getCyclicalBufferStructure (), depthRawScaled_);

    // transform vertex and normal maps for each pyramid level
    for (int i = 0; i < LEVELS; ++i)
      transformMaps (vmaps_curr_R_[i], nmaps_curr_R_[i], device_initial_cam_rot, device_initial_cam_trans, vmaps_g_prev_R_[i], nmaps_g_prev_R_[i]);

    if(perform_last_scan_)
      finished_ = true;

    ++global_time_;

    // save current vertex and normal maps
    saveCurrentMaps ();
    // return and wait for next frame

    // Right camera rotation and translation
    current_R_R_g_cam_ = initial_cam_rot;
    current_t_R_g_cam_ = initial_cam_trans;

    // Left camera rotations and translations
    current_R_L_g_cam_ = current_R_R_g_cam_ * relative_R_R_L_;
    current_t_L_g_cam_ = current_R_R_g_cam_ * relative_t_R_L_ + current_t_R_g_cam_;

    current_R_L_cam_g_ = current_R_L_g_cam_.inverse ();
    current_R_R_cam_g_ = current_R_R_g_cam_.inverse ();

    current_t_L_cam_g_ = -current_R_L_cam_g_ * current_t_L_g_cam_;
    current_t_R_cam_g_ = -current_R_R_cam_g_ * current_t_R_g_cam_;

    R_L_cam_g_.push_back(current_R_L_cam_g_);
    R_R_cam_g_.push_back(current_R_R_cam_g_);
    R_L_g_cam_.push_back(current_R_L_g_cam_);
    R_R_g_cam_.push_back(current_R_R_g_cam_);
    t_L_cam_g_.push_back(current_t_L_cam_g_);
    t_R_cam_g_.push_back(current_t_R_cam_g_);
    t_L_g_cam_.push_back(current_t_L_g_cam_);
    t_R_g_cam_.push_back(current_t_R_g_cam_);
    return (false);
  }

  Matrix3frm last_known_global_rotation;
  Vector3f   last_known_global_translation;
  if (use_external_poses_) {
    current_R_L_cam_g_ = R_L_cam_g_[global_time_];
    current_R_R_cam_g_ = R_R_cam_g_[global_time_];
    current_R_L_g_cam_ = R_L_g_cam_[global_time_];
    current_R_R_g_cam_ = R_R_g_cam_[global_time_];
    current_t_L_cam_g_ = t_L_cam_g_[global_time_];
    current_t_R_cam_g_ = t_R_cam_g_[global_time_];
    current_t_L_g_cam_ = t_L_g_cam_[global_time_];
    current_t_R_g_cam_ = t_R_g_cam_[global_time_];

    last_known_global_rotation = R_R_g_cam_[global_time_ - 1];
    last_known_global_translation = t_R_g_cam_[global_time_ - 1];

    last_estimated_translation_ = current_t_R_g_cam_;
    last_estimated_rotation_ = current_R_R_g_cam_;
  } else {
    ///////////////////////////////////////////////////////////////////////////////////////////
    // Iterative Closest Point
    // Get the last-known pose
    last_known_global_rotation = rmats_[global_time_ - 1];            // [Ri|ti] - pos of camera, i.e.
    last_known_global_translation = tvecs_[global_time_ - 1];          // transform from camera to global coo space for (i-1)th camera pose
    // Declare variables to host ICP results
    // Call ICP
    if(!performICP(intr, last_known_global_rotation, last_known_global_translation, current_R_R_g_cam_, current_t_R_g_cam_))
    {
      // ICP based on synthetic maps failed -> try to estimate the current camera pose based on previous and current raw maps
      Matrix3frm delta_rotation;
      Vector3f delta_translation;
      if(!performPairWiseICP(intr, delta_rotation, delta_translation))
      {
        // save current vertex and normal maps
        saveCurrentMaps ();
        return (false);
      }

      // Pose estimation succeeded -> update estimated pose
      last_estimated_translation_ = delta_rotation * last_estimated_translation_ + delta_translation;
      last_estimated_rotation_ = delta_rotation * last_estimated_rotation_;
      // save current vertex and normal maps
      saveCurrentMaps ();
      return (true);
    }
    else
    {
      // ICP based on synthetic maps succeeded
      // Save newly-computed pose
      rmats_.push_back (current_R_R_g_cam_);
      tvecs_.push_back (current_t_R_g_cam_);
      // Update last estimated pose to current pairwise ICP result
      last_estimated_translation_ = current_t_R_g_cam_;
      last_estimated_rotation_ = current_R_R_g_cam_;
    }

    // Left camera rotation and translation
    current_R_L_g_cam_ = current_R_R_g_cam_ * relative_R_R_L_;
    current_t_L_g_cam_ = current_R_R_g_cam_ * relative_t_R_L_ + current_t_R_g_cam_;

    float3 o = getCyclicalBufferStructure()->origin_metric;
    Vector3f origin(o.x, o.y, o.z);
    current_t_R_g_cam_ -= origin;
    current_t_L_g_cam_ -= origin;

    current_R_L_cam_g_ = current_R_L_g_cam_.inverse();
    current_R_R_cam_g_ = current_R_R_g_cam_.inverse();
    current_t_L_cam_g_ = -current_R_L_cam_g_ * current_t_L_g_cam_;
    current_t_R_cam_g_ = -current_R_R_cam_g_ * current_t_R_g_cam_;

    R_L_cam_g_.push_back(current_R_L_cam_g_);
    R_R_cam_g_.push_back(current_R_R_cam_g_);
    R_L_g_cam_.push_back(current_R_L_g_cam_);
    R_R_g_cam_.push_back(current_R_R_g_cam_);
    t_L_cam_g_.push_back(current_t_L_cam_g_);
    t_R_cam_g_.push_back(current_t_R_cam_g_);
    t_L_g_cam_.push_back(current_t_L_g_cam_);
    t_R_g_cam_.push_back(current_t_R_g_cam_);
  }

  ///////////////////////////////////////////////////////////////////////////////////////////
  // check if we need to shift
  has_shifted_ = cyclical_.checkForShift(tsdf_volume_, getCameraPose (), 0.6 * volume_size_, true, perform_last_scan_); // TODO make target distance from camera a param
  if(has_shifted_)
    PCL_WARN ("SHIFTING\n");

  ///////////////////////////////////////////////////////////////////////////////////////////
  // get the NEW local pose as device types
  Mat33  R_device_R_cam_g, R_device_R_g_cam;
  float3 t_device_R_g_cam;
  convertTransforms(current_R_R_cam_g_,
                    current_R_R_g_cam_,
                    current_t_R_g_cam_,
                    R_device_R_cam_g,
                    R_device_R_g_cam,
                    t_device_R_g_cam);
  //t_device_R_g_cam -= getCyclicalBufferStructure()->origin_metric;   // translation (local translation = global translation - origin of cube)

  Mat33  R_device_L_g_cam;
  float3 t_device_L_g_cam;
  convertTransforms(current_R_L_g_cam_,
                    current_t_L_g_cam_,
                    R_device_L_g_cam,
                    t_device_L_g_cam);
  //t_device_L_g_cam -= getCyclicalBufferStructure()->origin_metric;   // translation (local translation = global translation - origin of cube)

  ///////////////////////////////////////////////////////////////////////////////////////////
  // Integration check - We do not integrate volume if camera does not move far enought.
  {
    float rnorm = rodrigues2(current_R_R_g_cam_.inverse() * last_known_global_rotation).norm();
    float tnorm = (current_t_R_g_cam_ - last_known_global_translation).norm();
    const float alpha = 1.f;
    bool integrate = (rnorm + alpha * tnorm)/2 >= integration_metric_threshold_;
    ///////////////////////////////////////////////////////////////////////////////////////////
    // Volume integration
    if (integrate)
    {
      integrateTsdfVolume (depth_raw, intr, device_volume_size, R_device_R_cam_g, t_device_R_g_cam, tsdf_volume_->getTsdfTruncDist (), tsdf_volume_->data (), getCyclicalBufferStructure (), depthRawScaled_);
    }
  }

  ///////////////////////////////////////////////////////////////////////////////////////////
  // Ray casting
  {
    // generate synthetic vertex and normal maps from newly-found pose.
    raycast (intr, R_device_R_g_cam, t_device_R_g_cam, tsdf_volume_->getTsdfTruncDist (), device_volume_size, tsdf_volume_->data (), getCyclicalBufferStructure (), vmaps_g_prev_R_[0], nmaps_g_prev_R_[0]);

    // Ray casting for the left camera
    raycast (intr, R_device_L_g_cam, t_device_L_g_cam, tsdf_volume_->getTsdfTruncDist (), device_volume_size, tsdf_volume_->data (), getCyclicalBufferStructure (), vmaps_g_prev_L_[0], nmaps_g_prev_L_[0]);

    // POST-PROCESSING: We need to transform the newly raycasted maps into the global space.
    Mat33&  rotation_id = device_cast<Mat33> (rmats_[0]); /// Identity Rotation Matrix. Because we never rotate our volume
    float3 cube_origin = (getCyclicalBufferStructure ())->origin_metric;
    MapArr& vmap_temp_R = vmaps_g_prev_R_[0];
    MapArr& nmap_temp_R = nmaps_g_prev_R_[0];
    MapArr& vmap_temp_L = vmaps_g_prev_L_[0];
    MapArr& nmap_temp_L = nmaps_g_prev_L_[0];
    transformMaps (vmap_temp_R, nmap_temp_R, rotation_id, cube_origin, vmaps_g_prev_R_[0], nmaps_g_prev_R_[0]);
    transformMaps (vmap_temp_L, nmap_temp_L, rotation_id, cube_origin, vmaps_g_prev_L_[0], nmaps_g_prev_L_[0]);

    // create pyramid levels for vertex and normal maps
    for (int i = 1; i < LEVELS; ++i)
    {
      resizeVMap (vmaps_g_prev_R_[i-1], vmaps_g_prev_R_[i]);
      resizeNMap (nmaps_g_prev_R_[i-1], nmaps_g_prev_R_[i]);
      resizeVMap (vmaps_g_prev_L_[i-1], vmaps_g_prev_L_[i]);
      resizeNMap (nmaps_g_prev_L_[i-1], nmaps_g_prev_L_[i]);
    }
    pcl::device::kinfuLS::sync ();
  }

  if(has_shifted_ && perform_last_scan_)
    extractAndSaveWorld ();


  ++global_time_;
  // save current vertex and normal maps
  saveCurrentMaps ();
  return (true);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
Eigen::Affine3f
pcl::gpu::kinfuLS::KinfuTracker::getCameraPose (int time) const
{
  if (time > (int)rmats_.size () || time < 0)
    time = rmats_.size () - 1;

  Eigen::Affine3f aff;
  aff.linear () = rmats_[time];
  aff.translation () = tvecs_[time];
  return (aff);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
Eigen::Affine3f
pcl::gpu::kinfuLS::KinfuTracker::getLastEstimatedPose () const
{
  Eigen::Affine3f aff;
  aff.linear () = last_estimated_rotation_;
  aff.translation () = last_estimated_translation_;
  return (aff);
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
size_t
pcl::gpu::kinfuLS::KinfuTracker::getNumberOfPoses () const
{
  return rmats_.size();
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
const pcl::gpu::kinfuLS::TsdfVolume&
pcl::gpu::kinfuLS::KinfuTracker::volume() const
{
  return *tsdf_volume_;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
pcl::gpu::kinfuLS::TsdfVolume&
pcl::gpu::kinfuLS::KinfuTracker::volume()
{
  return *tsdf_volume_;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
const pcl::gpu::kinfuLS::ColorVolume&
pcl::gpu::kinfuLS::KinfuTracker::colorVolume() const
{
  return *color_volume_;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

pcl::gpu::kinfuLS::ColorVolume&
pcl::gpu::kinfuLS::KinfuTracker::colorVolume()
{
  return *color_volume_;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl::gpu::kinfuLS::KinfuTracker::getImage (View& view) const
{
  //Eigen::Vector3f light_source_pose = tsdf_volume_->getSize() * (-3.f);
  Eigen::Vector3f light_source_pose = tvecs_[tvecs_.size () - 1];

  LightSource light;
  light.number = 1;
  light.pos[0] = device_cast<const float3>(light_source_pose);

  view.create (rows_, cols_);
  generateImage (vmaps_g_prev_R_[0], nmaps_g_prev_R_[0], light, view);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl::gpu::kinfuLS::KinfuTracker::getLImage (View& view) const
{
  //Eigen::Vector3f light_source_pose = tsdf_volume_->getSize() * (-3.f);
  Eigen::Vector3f light_source_pose = tvecs_[tvecs_.size () - 1];

  LightSource light;
  light.number = 1;
  light.pos[0] = device_cast<const float3>(light_source_pose);

  view.create (rows_, cols_);
  generateImage (vmaps_g_prev_L_[0], nmaps_g_prev_L_[0], light, view);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl::gpu::kinfuLS::KinfuTracker::getRImage (View& view) const
{
  getImage(view);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl::gpu::kinfuLS::KinfuTracker::getAnaglyphImage (View& view) const
{
  //Eigen::Vector3f light_source_pose = tsdf_volume_->getSize() * (-3.f);
  Eigen::Vector3f light_source_pose = tvecs_[tvecs_.size () - 1];

  LightSource light;
  light.number = 1;
  light.pos[0] = device_cast<const float3>(light_source_pose);

  view.create (rows_, cols_);
  generateAnaglyphImage (vmaps_g_prev_L_[0], nmaps_g_prev_L_[0],
                         vmaps_g_prev_R_[0], nmaps_g_prev_R_[0],
                         light, view);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl::gpu::kinfuLS::KinfuTracker::getLastFrameCloud (DeviceArray2D<PointType>& cloud) const
{
  cloud.create (rows_, cols_);
  DeviceArray2D<float4>& c = (DeviceArray2D<float4>&)cloud;
  convert (vmaps_g_prev_R_[0], c);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl::gpu::kinfuLS::KinfuTracker::getLastFrameNormals (DeviceArray2D<NormalType>& normals) const
{
  normals.create (rows_, cols_);
  DeviceArray2D<float8>& n = (DeviceArray2D<float8>&)normals;
  convert (nmaps_g_prev_R_[0], n);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void
pcl::gpu::kinfuLS::KinfuTracker::initColorIntegration(int max_weight)
{
  color_volume_ = pcl::gpu::kinfuLS::ColorVolume::Ptr( new ColorVolume(*tsdf_volume_, max_weight) );
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
bool
pcl::gpu::kinfuLS::KinfuTracker::operator() (const DepthMap& depth, const View& colors)
{
  bool res = (*this)(depth);

  if (res && color_volume_)
  {
    const float3 device_volume_size = device_cast<const float3> (tsdf_volume_->getSize());
    Intr intr(fx_, fy_, cx_, cy_);

    Matrix3frm R_inv = rmats_.back().inverse();
    Vector3f   t     = tvecs_.back();

    Mat33&  device_Rcurr_inv = device_cast<Mat33> (R_inv);
    float3& device_tcurr = device_cast<float3> (t);

    updateColorVolume(intr, tsdf_volume_->getTsdfTruncDist(), device_Rcurr_inv, device_tcurr, vmaps_g_prev_R_[0],
        colors, device_volume_size, color_volume_->data(), color_volume_->getMaxWeight());
  }

  return res;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace pcl
{
  namespace gpu
  {
    namespace kinfuLS
    {
      PCL_EXPORTS void
      paint3DView(const KinfuTracker::View& rgb24, KinfuTracker::View& view, float colors_weight = 0.5f)
      {
        pcl::device::kinfuLS::paint3DView(rgb24, view, colors_weight);
      }

      PCL_EXPORTS void
      paint3DViewProj(const KinfuTracker::View& rgb24,
                      const pcl::device::kinfuLS::Mat33 R_cam_g,
                      const float3 t_cam_g,
                      float fx, float fy, float cx, float cy,
                      const KinfuTracker::MapArr vmaps,
                      KinfuTracker::View& view, float colors_weight = 0.5f)
      {
        pcl::device::kinfuLS::paint3DViewProj(rgb24,
                                              R_cam_g, t_cam_g,
                                              fx, fy, cx, cy,
                                              vmaps,
                                              view, colors_weight);
      }

      PCL_EXPORTS void
      paint3DViewProj(const KinfuTracker::View& rgb24,
                      const pcl::device::kinfuLS::Mat33 R_cam_g,
                      const float3 t_cam_g,
                      float fx, float fy, float cx, float cy,
                      const KinfuTracker::MapArr vmaps,
                      KinfuTracker::View& view, KinfuTracker::View& mask,
                      float colors_weight = 0.5f)
      {
        pcl::device::kinfuLS::paint3DViewProj(rgb24,
                                              R_cam_g, t_cam_g,
                                              fx, fy, cx, cy,
                                              vmaps,
                                              view, mask, colors_weight);
      }

      PCL_EXPORTS void
      paint3DViewProj(const KinfuTracker::View& rgb24,
                      const pcl::device::kinfuLS::Mat33 R_cam_g,
                      const float3 t_cam_g,
                      const pcl::device::kinfuLS::Mat33 R_view_img,
                      const float3 t_view_img,
                      float fx, float fy, float cx, float cy,
                      const KinfuTracker::MapArr vmaps,
                      KinfuTracker::View& view, float colors_weight = 0.5f)
      {
        pcl::device::kinfuLS::paint3DViewProj(rgb24,
                                              R_cam_g, t_cam_g,
                                              R_view_img, t_view_img,
                                              fx, fy, cx, cy,
                                              vmaps,
                                              view, colors_weight);
      }

      PCL_EXPORTS void
      paint3DViewProj(const KinfuTracker::View& rgb24,
                      const pcl::device::kinfuLS::Mat33 R_cam_g_R,
                      const float3 t_cam_g_R,
                      float fx, float fy, float cx, float cy,
                      const KinfuTracker::MapArr vmapsL,
                      const KinfuTracker::MapArr vmapsR,
                      KinfuTracker::View& view, float colors_weight = 0.5f)
      {
        pcl::device::kinfuLS::paint3DViewProj(rgb24,
                                              R_cam_g_R, t_cam_g_R,
                                              fx, fy, cx, cy,
                                              vmapsR, vmapsL,
                                              view, colors_weight);
      }

      PCL_EXPORTS void
      paint3DViewProj(const std::vector< KinfuTracker::View >& images,
                      const pcl::device::kinfuLS::Mat33 Rt_g_cam,
                      const float3 tt_g_cam,
                      const std::vector< pcl::device::kinfuLS::Mat33 > Rs_cam_g,
                      const std::vector< float3 > ts_cam_g,
                      const std::vector< pcl::device::kinfuLS::Mat33 > Rs_g_cam,
                      const std::vector< float3 > ts_g_cam,
                      float fx, float fy, float cx, float cy,
                      const KinfuTracker::MapArr vmaps,
                      KinfuTracker::View& view,
                      KinfuTracker::View& mask,
                      float colors_weight = 0.5f)
      {
        std::vector< PtrStep<uchar3> > _images;
        for (int index = 0; index < images.size(); index++) {
          _images.push_back(images.at(index));
        }
        pcl::device::kinfuLS::paint3DViewProj(_images,
                                              Rt_g_cam, tt_g_cam,
                                              Rs_cam_g, ts_cam_g,
                                              Rs_g_cam, ts_g_cam,
                                              fx, fy, cx, cy,
                                              vmaps,
                                              view, mask, colors_weight);
      }

      PCL_EXPORTS void
      mergePointNormal(const DeviceArray<PointXYZ>& cloud, const DeviceArray<Normal>& normals, DeviceArray<PointNormal>& output)
      {
        const size_t size = min(cloud.size(), normals.size());
        output.create(size);

        const DeviceArray<float4>& c = (const DeviceArray<float4>&)cloud;
        const DeviceArray<float8>& n = (const DeviceArray<float8>&)normals;
        const DeviceArray<float12>& o = (const DeviceArray<float12>&)output;
        pcl::device::kinfuLS::mergePointNormal(c, n, o);
      }

      Eigen::Vector3f rodrigues2(const Eigen::Matrix3f& matrix)
      {
        Eigen::JacobiSVD<Eigen::Matrix3f> svd(matrix, Eigen::ComputeFullV | Eigen::ComputeFullU);
        Eigen::Matrix3f R = svd.matrixU() * svd.matrixV().transpose();

        double rx = R(2, 1) - R(1, 2);
        double ry = R(0, 2) - R(2, 0);
        double rz = R(1, 0) - R(0, 1);

        double s = sqrt((rx*rx + ry*ry + rz*rz)*0.25);
        double c = (R.trace() - 1) * 0.5;
        c = c > 1. ? 1. : c < -1. ? -1. : c;

        double theta = acos(c);

        if( s < 1e-5 )
        {
          double t;

          if( c > 0 )
            rx = ry = rz = 0;
          else
          {
            t = (R(0, 0) + 1)*0.5;
            rx = sqrt( std::max(t, 0.0) );
            t = (R(1, 1) + 1)*0.5;
            ry = sqrt( std::max(t, 0.0) ) * (R(0, 1) < 0 ? -1.0 : 1.0);
            t = (R(2, 2) + 1)*0.5;
            rz = sqrt( std::max(t, 0.0) ) * (R(0, 2) < 0 ? -1.0 : 1.0);

            if( fabs(rx) < fabs(ry) && fabs(rx) < fabs(rz) && (R(1, 2) > 0) != (ry*rz > 0) )
              rz = -rz;
            theta /= sqrt(rx*rx + ry*ry + rz*rz);
            rx *= theta;
            ry *= theta;
            rz *= theta;
          }
        }
        else
        {
          double vth = 1/(2*s);
          vth *= theta;
          rx *= vth; ry *= vth; rz *= vth;
        }
        return Eigen::Vector3d(rx, ry, rz).cast<float>();
      }
    }
  }
}
