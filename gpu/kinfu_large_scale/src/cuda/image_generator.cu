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
      generateImage (const MapArr& vmap, const MapArr& nmap, const LightSource& light,
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

        __device__ __forceinline__ void
        operator () () const
        {
          int x = threadIdx.x + blockIdx.x * CTA_SIZE_X;
          int y = threadIdx.y + blockIdx.y * CTA_SIZE_Y;

          if (x >= dst.cols || y >= dst.rows)
            return;

          float3 vL, nL, vR, nR;
          vL.x = vmapL.ptr (y)[x];
          nL.x = nmapL.ptr (y)[x];
          vR.x = vmapR.ptr (y)[x];
          nR.x = nmapR.ptr (y)[x];

          uchar3 color = make_uchar3 (0, 0, 0);
		  int brR = 0;
		  int brL = 0;

		  // Left
		  if (!isnan (vL.x) && !isnan (nL.x)) {
            vL.y = vmapL.ptr (y + dst.rows)[x];
            vL.z = vmapL.ptr (y + 2 * dst.rows)[x];

            nL.y = nmapL.ptr (y + dst.rows)[x];
            nL.z = nmapL.ptr (y + 2 * dst.rows)[x];

            float weightL = 1.f;
            for (int i = 0; i < light.number; ++i)
            {
              float3 vecL = normalized (light.pos[i] - vL);

              weightL *= fabs (dot (vecL, nL));
            }

            brL = (int)(205 * weightL) + 50;
            brL = max (0, min (255, brL));
		  }

		  // Right
          if (!isnan (vR.x) && !isnan (nR.x)) {
            vR.y = vmapR.ptr (y + dst.rows)[x];
            vR.z = vmapR.ptr (y + 2 * dst.rows)[x];

            nR.y = nmapR.ptr (y + dst.rows)[x];
            nR.z = nmapR.ptr (y + 2 * dst.rows)[x];

            float weightR = 1.f;

            for (int i = 0; i < light.number; ++i)
            {
              float3 vecR = normalized (light.pos[i] - vR);

              weightR *= fabs (dot (vecR, nR));
            }

            brR = (int)(205 * weightR) + 50;
            brR = max (0, min (255, brR));
          }

          color = make_uchar3 (brL, brR, brR);
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
      __global__ void generateDepthKernel(const float3 R_inv_row3, const float3 t, const PtrStep<float> vmap, PtrStepSz<unsigned short> depth)
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
      generateDepth (const Mat33& R_inv, const float3& t, const MapArr& vmap, DepthMap& dst)
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
      paint3DViewKernel(const PtrStep<uchar3> colors, PtrStepSz<uchar3> dst, float colors_weight)
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
      paint3DView(const PtrStep<uchar3>& colors, PtrStepSz<uchar3> dst, float colors_weight)
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
