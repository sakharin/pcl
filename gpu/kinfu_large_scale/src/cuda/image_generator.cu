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

#include "device.hpp"
//#include <boost/graph/buffer_concepts.hpp>

namespace pcl
{
  namespace device
  {
    namespace kinfuLS
    {
      struct ImageGenerator
      {
        enum
        {
          CTA_SIZE_X = 32, CTA_SIZE_Y = 8
        };

        PtrStep<float> vmap;
        PtrStep<float> nmap;

        LightSource light;

        mutable PtrStepSz<uchar3> dst;

        __device__ __forceinline__ void
        operator () () const
        {
          int x = threadIdx.x + blockIdx.x * CTA_SIZE_X;
          int y = threadIdx.y + blockIdx.y * CTA_SIZE_Y;

          if (x >= dst.cols || y >= dst.rows)
            return;

          float3 v, n;
          v.x = vmap.ptr (y)[x];
          n.x = nmap.ptr (y)[x];

          uchar3 color = make_uchar3 (0, 0, 0);

          if (!isnan (v.x) && !isnan (n.x))
          {
            v.y = vmap.ptr (y + dst.rows)[x];
            v.z = vmap.ptr (y + 2 * dst.rows)[x];

            n.y = nmap.ptr (y + dst.rows)[x];
            n.z = nmap.ptr (y + 2 * dst.rows)[x];

            float weight = 1.f;

            for (int i = 0; i < light.number; ++i)
            {
              float3 vec = normalized (light.pos[i] - v);

              weight *= fabs (dot (vec, n));
            }

            int br = (int)(205 * weight) + 50;
            br = max (0, min (255, br));
            color = make_uchar3 (br, br, br);
          }
          dst.ptr (y)[x] = color;
        }
      };

      __global__ void
      generateImageKernel (const ImageGenerator ig) {
        ig ();
      }

      void
      generateImage (const MapArr& vmap, const MapArr& nmap,
          const LightSource& light,
          PtrStepSz<uchar3> dst)
      {
        ImageGenerator ig;
        ig.vmap = vmap;
        ig.nmap = nmap;
        ig.light = light;
        ig.dst = dst;

        dim3 block (ImageGenerator::CTA_SIZE_X, ImageGenerator::CTA_SIZE_Y);
        dim3 grid (divUp (dst.cols, block.x), divUp (dst.rows, block.y));

        generateImageKernel<<<grid, block>>>(ig);
        cudaSafeCall (cudaGetLastError ());
        cudaSafeCall (cudaDeviceSynchronize ());
      }
    }
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace pcl
{
  namespace device
  {
    namespace kinfuLS
    {
      struct AnaglyphImageGenerator
      {
        enum
        {
          CTA_SIZE_X = 32, CTA_SIZE_Y = 8
        };

        PtrStep<float> vmapL;
        PtrStep<float> nmapL;
        PtrStep<float> vmapR;
        PtrStep<float> nmapR;

        LightSource light;

        mutable PtrStepSz<uchar3> dst;

        __device__ __forceinline__ uchar3
        computeSurfaceColor (PtrStep<float> vmap, PtrStep<float> nmap,
            int x, int y) const {
          uchar3 color = make_uchar3(0, 0, 0);
          float3 v, n;
          v.x = vmap.ptr (y)[x];
          n.x = nmap.ptr (y)[x];
          if (!isnan (v.x) && !isnan (n.x)) {
            v.y = vmap.ptr (y + dst.rows    )[x];
            v.z = vmap.ptr (y + 2 * dst.rows)[x];

            n.y = nmap.ptr (y + dst.rows    )[x];
            n.z = nmap.ptr (y + 2 * dst.rows)[x];

            float weight = 1.f;
            for (int i = 0; i < light.number; ++i)
            {
              float3 vec = normalized (light.pos[i] - v);

              weight *= fabs (dot (vec, n));
            }

            int br = (int)(205 * weight) + 50;
            br = max (0, min (255, br));
			color.x = br;
			color.y = br;
			color.z = br;
          }
          return color;
        }

        __device__ __forceinline__ void
        operator () () const
        {
          int x = threadIdx.x + blockIdx.x * CTA_SIZE_X;
          int y = threadIdx.y + blockIdx.y * CTA_SIZE_Y;

          if (x >= dst.cols || y >= dst.rows)
            return;

          uchar3 brL = computeSurfaceColor(vmapL, nmapL, x, y);
          uchar3 brR = computeSurfaceColor(vmapR, nmapR, x, y);

          uchar3 color = make_uchar3 (brL.x, brR.y, brR.z);
          dst.ptr (y)[x] = color;
        }
      };

      __global__ void
      generateAnaglyphImageKernel (const AnaglyphImageGenerator aig) {
        aig ();
      }

      void
      generateAnaglyphImage (const MapArr& vmapL, const MapArr& nmapL,
          const MapArr& vmapR, const MapArr& nmapR,
          const LightSource& light,
          PtrStepSz<uchar3> dst)
      {
        AnaglyphImageGenerator aig;
        aig.vmapL = vmapL;
        aig.nmapL = nmapL;
        aig.vmapR = vmapR;
        aig.nmapR = nmapR;
        aig.light = light;
        aig.dst = dst;

        dim3 block (AnaglyphImageGenerator::CTA_SIZE_X, AnaglyphImageGenerator::CTA_SIZE_Y);
        dim3 grid (divUp (dst.cols, block.x), divUp (dst.rows, block.y));

        generateAnaglyphImageKernel<<<grid, block>>>(aig);
        cudaSafeCall (cudaGetLastError ());
        cudaSafeCall (cudaDeviceSynchronize ());
      }
    }
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
namespace pcl
{
  namespace device
  {
    namespace kinfuLS
    {
      __global__ void generateDepthKernel(const float3 R_inv_row3,
          const float3 t, const PtrStep<float> vmap,
          PtrStepSz<unsigned short> depth)
      {
        int x = threadIdx.x + blockIdx.x * blockDim.x;
        int y = threadIdx.y + blockIdx.y * blockDim.y;

        if (x < depth.cols && y < depth.rows)
        {
          unsigned short result = 0;

          float3 v_g;
          v_g.x = vmap.ptr (y)[x];
          if (!isnan (v_g.x))
          {
            v_g.y = vmap.ptr (y +     depth.rows)[x];
            v_g.z = vmap.ptr (y + 2 * depth.rows)[x];

            float v_z = dot(R_inv_row3, v_g - t);

            result = static_cast<unsigned short>(v_z * 1000);
          }
          depth.ptr(y)[x] = result;
        }
      }

      void
      generateDepth (const Mat33& R_inv, const float3& t, const MapArr& vmap,
          DepthMap& dst)
      {
        dim3 block(32, 8);
        dim3 grid(divUp(dst.cols(), block.x), divUp(dst.rows(), block.y));

        generateDepthKernel<<<grid, block>>>(R_inv.data[2], t, vmap, dst);
        cudaSafeCall (cudaGetLastError ());
        cudaSafeCall (cudaDeviceSynchronize ());
      }
    }
  }
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace pcl
{
  namespace device
  {
    namespace kinfuLS
    {
      __global__ void
      paint3DViewKernel(const PtrStep<uchar3> colors, PtrStepSz<uchar3> dst,
          float colors_weight)
      {
        int x = threadIdx.x + blockIdx.x * blockDim.x;
        int y = threadIdx.y + blockIdx.y * blockDim.y;

        if (x < dst.cols && y < dst.rows)
        {
          uchar3 value = dst.ptr(y)[x];
          uchar3 color = colors.ptr(y)[x];

          if (value.x != 0 || value.y != 0 || value.z != 0)
          {
            float cx = value.x * (1.f - colors_weight) + color.x * colors_weight;
            float cy = value.y * (1.f - colors_weight) + color.y * colors_weight;
            float cz = value.z * (1.f - colors_weight) + color.z * colors_weight;

            value.x = min(255, max(0, __float2int_rn(cx)));
            value.y = min(255, max(0, __float2int_rn(cy)));
            value.z = min(255, max(0, __float2int_rn(cz)));
          }

          dst.ptr(y)[x] = value;
        }
      }

      void
      paint3DView(const PtrStep<uchar3>& colors, PtrStepSz<uchar3> dst,
          float colors_weight)
      {
        dim3 block(32, 8);
        dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y));

        colors_weight = min(1.f, max(0.f, colors_weight));

        paint3DViewKernel<<<grid, block>>>(colors, dst, colors_weight);
        cudaSafeCall (cudaGetLastError ());
        cudaSafeCall (cudaDeviceSynchronize ());
      }
    }
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
namespace pcl
{
  namespace device
  {
    namespace kinfuLS
    {
      struct Paint3DProj {
        enum { CTA_SIZE_X = 32, CTA_SIZE_Y = 8 };

        PtrStep<uchar3> colors;

        Mat33 R_cam_g;
        float3 t_g_cam;
        float fx, fy, cx, cy;
        PtrStep<float> vmap;
        mutable PtrStepSz<uchar3> dst;
        float colors_weight;

        int vmap_cols, vmap_rows;

        __device__ __forceinline__ float3
        get_ray_next (int x, int y) const
        {
          float3 ray_next;
          ray_next.x = (x - cx) / fx;
          ray_next.y = (y - cy) / fy;
          ray_next.z = 1;
          return ray_next;
        }

        __device__ __forceinline__ float3
        XYZ2uv (float3 XYZ) const
        {
          float3 uv;
          uv.x = (fx * XYZ.x + cx * XYZ.z) / XYZ.z;
          uv.y = (fy * XYZ.y + cy * XYZ.z) / XYZ.z;
          uv.z = 1;
          return uv;
        }

        __device__ __forceinline__ void
        linearInterColori (uchar3 c1, uchar3 c2, int p1, int p2, float p,
            float3 &c) const {
          float s1, s2;
          s1 = (p2 - p) / (p2 - p1);
          s2 = (p - p1) / (p2 - p1);
          c.x = s1 * c1.x + s2 * c2.x;
          c.y = s1 * c1.y + s2 * c2.y;
          c.z = s1 * c1.z + s2 * c2.z;
        }

        __device__ __forceinline__ void
        linearInterColorf (float3 c1, float3 c2, int p1, int p2, float p,
            float3 &c) const {
          float s1, s2;
          s1 = (p2 - p) / (p2 - p1);
          s2 = (p - p1) / (p2 - p1);
          c.x = s1 * c1.x + s2 * c2.x;
          c.y = s1 * c1.y + s2 * c2.y;
          c.z = s1 * c1.z + s2 * c2.z;
        }

        __device__ __forceinline__ void
        projectPixel (float3 uv, PtrStep<uchar3> colors, int rows, int cols,
            uchar3& pixel) const {
          int x1 = floor(uv.x);
          int x2 = ceil(uv.x);
          int y1 = floor(uv.y);
          int y2 = ceil(uv.y);
          if (x1 >=0 && y1 >=0 && x2 < cols && y2 < rows) {
            uchar3 p1 = colors.ptr(y1)[x1];
            uchar3 p2 = colors.ptr(y1)[x2];
            uchar3 p3 = colors.ptr(y2)[x1];
            uchar3 p4 = colors.ptr(y2)[x2];
            float3 p12, p34, p1234;
            linearInterColori(p1, p2, x1, x2, uv.x, p12);
            linearInterColori(p3, p4, x1, x2, uv.x, p34);
            linearInterColorf(p12, p34, y1, y2, uv.y, p1234);
            pixel.x = min(255, max(0, __float2int_rn(p1234.x)));
            pixel.y = min(255, max(0, __float2int_rn(p1234.y)));
            pixel.z = min(255, max(0, __float2int_rn(p1234.z)));
          }
        }

        __device__ __forceinline__ void
        weightSumPixel(uchar3 pixel_1, uchar3 pixel_2, float weight,
            uchar3& pixel) const {
          pixel = make_uchar3(0, 0, 0);
          if (pixel_1.x != 0 || pixel_1.y != 0 || pixel_1.z != 0 ||
              pixel_2.x != 0 || pixel_2.y != 0 || pixel_2.z != 0) {
            float cx = pixel_1.x * weight + pixel_2.x * (1.f - weight);
            float cy = pixel_1.y * weight + pixel_2.y * (1.f - weight);
            float cz = pixel_1.z * weight + pixel_2.z * (1.f - weight);
            pixel.x = min(255, max(0, __float2int_rn(cx)));
            pixel.y = min(255, max(0, __float2int_rn(cy)));
            pixel.z = min(255, max(0, __float2int_rn(cz)));
          }
        }

        __device__ __forceinline__ void
        operator () () const
        {
          // Get vmap index
          int u = threadIdx.x + blockIdx.x * CTA_SIZE_X;
          int v = threadIdx.y + blockIdx.y * CTA_SIZE_Y;
          int x = threadIdx.x + blockIdx.x * blockDim.x;
          int y = threadIdx.y + blockIdx.y * blockDim.y;

          // Index must be in vmap
          if (u >= vmap_cols || v >= vmap_rows)
            return;

          float3 ray_end;
          ray_end.x = vmap.ptr (v                )[u];
          ray_end.y = vmap.ptr (v + vmap_rows    )[u];
          ray_end.z = vmap.ptr (v + 2 * vmap_rows)[u];

          uchar3 result_pixel = make_uchar3(0, 0, 0);
          if (!isnan (ray_end.x)) {
            float3 proj_ray = R_cam_g * ray_end + t_g_cam;
            float3 uv = XYZ2uv(proj_ray);

            uchar3 pixel;
            projectPixel(uv, colors, dst.rows, dst.cols, pixel);
            uchar3 value = dst.ptr(y)[x];
            weightSumPixel(pixel, value, colors_weight, result_pixel);
          }
          dst.ptr(y)[x] = result_pixel;
        }
      };

      __global__ void
      paint3DViewProjKernel (const Paint3DProj p3Dv) {
        p3Dv ();
      }

      void
      paint3DViewProj(const PtrStep<uchar3>& colors,
          const Mat33 R_cam_g, const float3 t_g_cam,
          float fx, float fy, float cx, float cy,
          const MapArr vmap,
          PtrStepSz<uchar3> dst, float colors_weight)
      {
        Paint3DProj p3Dv;
        p3Dv.colors = colors;
        p3Dv.R_cam_g = R_cam_g;
        p3Dv.t_g_cam = t_g_cam;
        p3Dv.fx = fx;
        p3Dv.fy = fy;
        p3Dv.cx = cx;
        p3Dv.cy = cy;
        p3Dv.vmap = vmap;
        p3Dv.dst = dst;
        p3Dv.colors_weight = min(1.f, max(0.f, colors_weight));

        p3Dv.vmap_cols = vmap.cols ();
        p3Dv.vmap_rows = vmap.rows () / 3;

        dim3 block(32, 8);
        dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y));

        paint3DViewProjKernel<<<grid, block>>>(p3Dv);
        cudaSafeCall (cudaGetLastError ());
        cudaSafeCall (cudaDeviceSynchronize ());
      }
    }
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
namespace pcl
{
  namespace device
  {
    namespace kinfuLS
    {
      struct Paint3DProjRelativeImage {
        enum { CTA_SIZE_X = 32, CTA_SIZE_Y = 8 };

        PtrStep<uchar3> colors;

        Mat33 R_cam_g;
        float3 t_g_cam;
        Mat33 R_view_img;
        float3 t_view_img;
        float fx, fy, cx, cy;
        PtrStep<float> vmap;
        mutable PtrStepSz<uchar3> dst;
        float colors_weight;

        int vmap_cols, vmap_rows;

        __device__ __forceinline__ float3
        get_ray_next (int x, int y) const
        {
          float3 ray_next;
          ray_next.x = (x - cx) / fx;
          ray_next.y = (y - cy) / fy;
          ray_next.z = 1;
          return ray_next;
        }

        __device__ __forceinline__ float3
        XYZ2uv (float3 XYZ) const
        {
          float3 uv;
          uv.x = (fx * XYZ.x + cx * XYZ.z) / XYZ.z;
          uv.y = (fy * XYZ.y + cy * XYZ.z) / XYZ.z;
          uv.z = 1;
          return uv;
        }

        __device__ __forceinline__ void
        linearInterColori (uchar3 c1, uchar3 c2, int p1, int p2, float p,
            float3 &c) const {
          float s1, s2;
          s1 = (p2 - p) / (p2 - p1);
          s2 = (p - p1) / (p2 - p1);
          c.x = s1 * c1.x + s2 * c2.x;
          c.y = s1 * c1.y + s2 * c2.y;
          c.z = s1 * c1.z + s2 * c2.z;
        }

        __device__ __forceinline__ void
        linearInterColorf (float3 c1, float3 c2, int p1, int p2, float p,
            float3 &c) const {
          float s1, s2;
          s1 = (p2 - p) / (p2 - p1);
          s2 = (p - p1) / (p2 - p1);
          c.x = s1 * c1.x + s2 * c2.x;
          c.y = s1 * c1.y + s2 * c2.y;
          c.z = s1 * c1.z + s2 * c2.z;
        }

        __device__ __forceinline__ void
        projectPixel (float3 uv, PtrStep<uchar3> colors, int rows, int cols,
            uchar3& pixel) const {
          int x1 = floor(uv.x);
          int x2 = ceil(uv.x);
          int y1 = floor(uv.y);
          int y2 = ceil(uv.y);
          if (x1 >=0 && y1 >=0 && x2 < cols && y2 < rows) {
            uchar3 p1 = colors.ptr(y1)[x1];
            uchar3 p2 = colors.ptr(y1)[x2];
            uchar3 p3 = colors.ptr(y2)[x1];
            uchar3 p4 = colors.ptr(y2)[x2];
            float3 p12, p34, p1234;
            linearInterColori(p1, p2, x1, x2, uv.x, p12);
            linearInterColori(p3, p4, x1, x2, uv.x, p34);
            linearInterColorf(p12, p34, y1, y2, uv.y, p1234);
            pixel.x = min(255, max(0, __float2int_rn(p1234.x)));
            pixel.y = min(255, max(0, __float2int_rn(p1234.y)));
            pixel.z = min(255, max(0, __float2int_rn(p1234.z)));
          }
        }

        __device__ __forceinline__ void
        weightSumPixel(uchar3 pixel_1, uchar3 pixel_2, float weight,
            uchar3& pixel) const {
          pixel = make_uchar3(0, 0, 0);
          if (pixel_1.x != 0 || pixel_1.y != 0 || pixel_1.z != 0 ||
              pixel_2.x != 0 || pixel_2.y != 0 || pixel_2.z != 0) {
            float cx = pixel_1.x * weight + pixel_2.x * (1.f - weight);
            float cy = pixel_1.y * weight + pixel_2.y * (1.f - weight);
            float cz = pixel_1.z * weight + pixel_2.z * (1.f - weight);
            pixel.x = min(255, max(0, __float2int_rn(cx)));
            pixel.y = min(255, max(0, __float2int_rn(cy)));
            pixel.z = min(255, max(0, __float2int_rn(cz)));
          }
        }

        __device__ __forceinline__ void
        operator () () const
        {
          // Get vmap index
          int u = threadIdx.x + blockIdx.x * CTA_SIZE_X;
          int v = threadIdx.y + blockIdx.y * CTA_SIZE_Y;
          int x = threadIdx.x + blockIdx.x * blockDim.x;
          int y = threadIdx.y + blockIdx.y * blockDim.y;

          // Index must be in vmap
          if (u >= vmap_cols || v >= vmap_rows)
            return;

          float3 ray_end;
          ray_end.x = vmap.ptr (v                )[u];
          ray_end.y = vmap.ptr (v + vmap_rows    )[u];
          ray_end.z = vmap.ptr (v + 2 * vmap_rows)[u];

          uchar3 result_pixel = make_uchar3(0, 0, 0);
          if (!isnan (ray_end.x)) {
            float3 proj_ray = R_cam_g * ray_end + t_g_cam;
            proj_ray = R_view_img * proj_ray + t_view_img;
            float3 uv = XYZ2uv(proj_ray);

            uchar3 pixel;
            projectPixel(uv, colors, dst.rows, dst.cols, pixel);
            uchar3 value = dst.ptr(y)[x];
            weightSumPixel(pixel, value, colors_weight, result_pixel);
          }
          dst.ptr(y)[x] = result_pixel;
        }
      };

      __global__ void
      paint3DViewProjRelativeImageKernel (const Paint3DProjRelativeImage p3Dvri) {
        p3Dvri ();
      }

      void
      paint3DViewProj(const PtrStep<uchar3>& colors,
          const Mat33 R_cam_g, const float3 t_g_cam,
          const Mat33 R_view_img, const float3 t_view_img,
          float fx, float fy, float cx, float cy,
          const MapArr vmap,
          PtrStepSz<uchar3> dst, float colors_weight)
      {
        Paint3DProjRelativeImage p3Dvri;
        p3Dvri.colors = colors;
        p3Dvri.R_cam_g = R_cam_g;
        p3Dvri.t_g_cam = t_g_cam;
        p3Dvri.R_view_img = R_view_img;
        p3Dvri.t_view_img = t_view_img;
        p3Dvri.fx = fx;
        p3Dvri.fy = fy;
        p3Dvri.cx = cx;
        p3Dvri.cy = cy;
        p3Dvri.vmap = vmap;
        p3Dvri.dst = dst;
        p3Dvri.colors_weight = min(1.f, max(0.f, colors_weight));

        p3Dvri.vmap_cols = vmap.cols ();
        p3Dvri.vmap_rows = vmap.rows () / 3;

        dim3 block(32, 8);
        dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y));

        paint3DViewProjRelativeImageKernel<<<grid, block>>>(p3Dvri);
        cudaSafeCall (cudaGetLastError ());
        cudaSafeCall (cudaDeviceSynchronize ());
      }
    }
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
namespace pcl
{
  namespace device
  {
    namespace kinfuLS
    {
      struct Paint3DProjRelativeImageAnaglyph {
        enum { CTA_SIZE_X = 32, CTA_SIZE_Y = 8 };

        PtrStep<uchar3> colors;

        Mat33 R_cam_g_L;
        float3 t_g_cam_L;
        Mat33 R_view_img;
        float3 t_view_img;
        Mat33 R_cam_g_R;
        float3 t_g_cam_R;
        float fx, fy, cx, cy;
        PtrStep<float> vmapL;
        PtrStep<float> vmapR;
        mutable PtrStepSz<uchar3> dst;
        float colors_weight;

        int vmap_cols, vmap_rows;

        __device__ __forceinline__ float3
        get_ray_next (int x, int y) const
        {
          float3 ray_next;
          ray_next.x = (x - cx) / fx;
          ray_next.y = (y - cy) / fy;
          ray_next.z = 1;
          return ray_next;
        }

        __device__ __forceinline__ float3
        XYZ2uv (float3 XYZ) const
        {
          float3 uv;
          uv.x = (fx * XYZ.x + cx * XYZ.z) / XYZ.z;
          uv.y = (fy * XYZ.y + cy * XYZ.z) / XYZ.z;
          uv.z = 1;
          return uv;
        }

        __device__ __forceinline__ void
        linearInterColori (uchar3 c1, uchar3 c2, int p1, int p2, float p,
            float3 &c) const {
          float s1, s2;
          s1 = (p2 - p) / (p2 - p1);
          s2 = (p - p1) / (p2 - p1);
          c.x = s1 * c1.x + s2 * c2.x;
          c.y = s1 * c1.y + s2 * c2.y;
          c.z = s1 * c1.z + s2 * c2.z;
        }

        __device__ __forceinline__ void
        linearInterColorf (float3 c1, float3 c2, int p1, int p2, float p,
            float3 &c) const {
          float s1, s2;
          s1 = (p2 - p) / (p2 - p1);
          s2 = (p - p1) / (p2 - p1);
          c.x = s1 * c1.x + s2 * c2.x;
          c.y = s1 * c1.y + s2 * c2.y;
          c.z = s1 * c1.z + s2 * c2.z;
        }

        __device__ __forceinline__ void
        projectPixel (float3 uv, PtrStep<uchar3> colors, int rows, int cols,
            uchar3& pixel) const {
          int x1 = floor(uv.x);
          int x2 = ceil(uv.x);
          int y1 = floor(uv.y);
          int y2 = ceil(uv.y);
          if (x1 >=0 && y1 >=0 && x2 < cols && y2 < rows) {
            uchar3 p1 = colors.ptr(y1)[x1];
            uchar3 p2 = colors.ptr(y1)[x2];
            uchar3 p3 = colors.ptr(y2)[x1];
            uchar3 p4 = colors.ptr(y2)[x2];
            float3 p12, p34, p1234;
            linearInterColori(p1, p2, x1, x2, uv.x, p12);
            linearInterColori(p3, p4, x1, x2, uv.x, p34);
            linearInterColorf(p12, p34, y1, y2, uv.y, p1234);
            pixel.x = min(255, max(0, __float2int_rn(p1234.x)));
            pixel.y = min(255, max(0, __float2int_rn(p1234.y)));
            pixel.z = min(255, max(0, __float2int_rn(p1234.z)));
          }
        }

        __device__ __forceinline__ void
        weightSumPixel(uchar3 pixel_1, uchar3 pixel_2, float weight,
            uchar3& pixel) const {
          pixel = make_uchar3(0, 0, 0);
          if (pixel_1.x != 0 || pixel_1.y != 0 || pixel_1.z != 0 ||
              pixel_2.x != 0 || pixel_2.y != 0 || pixel_2.z != 0) {
            float cx = pixel_1.x * weight + pixel_2.x * (1.f - weight);
            float cy = pixel_1.y * weight + pixel_2.y * (1.f - weight);
            float cz = pixel_1.z * weight + pixel_2.z * (1.f - weight);
            pixel.x = min(255, max(0, __float2int_rn(cx)));
            pixel.y = min(255, max(0, __float2int_rn(cy)));
            pixel.z = min(255, max(0, __float2int_rn(cz)));
          }
        }

        __device__ __forceinline__ void
        operator () () const
        {
          // Get vmap index
          int u = threadIdx.x + blockIdx.x * CTA_SIZE_X;
          int v = threadIdx.y + blockIdx.y * CTA_SIZE_Y;
          int x = threadIdx.x + blockIdx.x * blockDim.x;
          int y = threadIdx.y + blockIdx.y * blockDim.y;

          // Index must be in vmap
          if (u >= vmap_cols || v >= vmap_rows)
            return;

          float3 ray_end_L, ray_end_R;
          ray_end_L.x = vmapL.ptr (v                )[u];
          ray_end_L.y = vmapL.ptr (v + vmap_rows    )[u];
          ray_end_L.z = vmapL.ptr (v + 2 * vmap_rows)[u];
          ray_end_R.x = vmapR.ptr (v                )[u];
          ray_end_R.y = vmapR.ptr (v + vmap_rows    )[u];
          ray_end_R.z = vmapR.ptr (v + 2 * vmap_rows)[u];

          uchar3 result_pixel_L = make_uchar3(0, 0, 0);
          if (!isnan (ray_end_L.x)) {
            float3 proj_ray = R_cam_g_L * ray_end_L + t_g_cam_L;
            proj_ray = R_view_img * proj_ray + t_view_img;
            float3 uv = XYZ2uv(proj_ray);

            uchar3 pixel;
            projectPixel(uv, colors, dst.rows, dst.cols, pixel);
            uchar3 value = dst.ptr(y)[x];
            weightSumPixel(pixel, value, colors_weight, result_pixel_L);
          }

          uchar3 result_pixel_R = make_uchar3(0, 0, 0);
          if (!isnan (ray_end_R.x)) {
            float3 proj_ray = R_cam_g_R * ray_end_R + t_g_cam_R;
            float3 uv = XYZ2uv(proj_ray);

            uchar3 pixel;
            projectPixel(uv, colors, dst.rows, dst.cols, pixel);
            uchar3 value = dst.ptr(y)[x];
            weightSumPixel(pixel, value, colors_weight, result_pixel_R);
          }

          uchar3 result_pixel;
          result_pixel.x = result_pixel_L.x;
          result_pixel.y = result_pixel_R.y;
          result_pixel.z = result_pixel_R.z;
          dst.ptr(y)[x] = result_pixel;
        }
      };

      __global__ void
      paint3DViewProjRelativeImageAnaglyphKernel (const
          Paint3DProjRelativeImageAnaglyph p3Dvria) {
        p3Dvria ();
      }

      void
      paint3DViewProj(const PtrStep<uchar3>& colors,
          const Mat33 R_cam_g_L, const float3 t_g_cam_L,
          const Mat33 R_view_img, const float3 t_view_img,
          const Mat33 R_cam_g_R, const float3 t_g_cam_R,
          float fx, float fy, float cx, float cy,
          const MapArr vmapL,
          const MapArr vmapR,
          PtrStepSz<uchar3> dst, float colors_weight)
      {
        Paint3DProjRelativeImageAnaglyph p3Dvria;
        p3Dvria.colors = colors;
        p3Dvria.R_cam_g_L = R_cam_g_L;
        p3Dvria.t_g_cam_L = t_g_cam_L;
        p3Dvria.R_view_img = R_view_img;
        p3Dvria.t_view_img = t_view_img;
        p3Dvria.R_cam_g_R = R_cam_g_R;
        p3Dvria.t_g_cam_R = t_g_cam_R;
        p3Dvria.fx = fx;
        p3Dvria.fy = fy;
        p3Dvria.cx = cx;
        p3Dvria.cy = cy;
        p3Dvria.vmapL = vmapL;
        p3Dvria.vmapR = vmapR;
        p3Dvria.dst = dst;
        p3Dvria.colors_weight = min(1.f, max(0.f, colors_weight));

        p3Dvria.vmap_cols = vmapL.cols ();
        p3Dvria.vmap_rows = vmapL.rows () / 3;

        dim3 block(32, 8);
        dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y));

        paint3DViewProjRelativeImageAnaglyphKernel<<<grid, block>>>(p3Dvria);
        cudaSafeCall (cudaGetLastError ());
        cudaSafeCall (cudaDeviceSynchronize ());
      }
    }
  }
}
