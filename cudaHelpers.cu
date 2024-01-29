//
// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

#include "optixRayTracedCollision/cudaHelpers.h"
#include <optixRayTracedCollision/optixRayTracedCollision.h>
#include <helpers.h>
#include <device_launch_parameters.h>
#include "GeometryData.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/transform_reduce.h>
#include <thrust/device_ptr.h>
#include <__msvc_chrono.hpp>
#include <sutil/sutil.h>

using namespace std;
// Mennyi az impulzus.. írjuk bele az impusle buffer-be
// impulse feltöltve -> sebességek módosítása
// gömbökre indexelve tudjuk módosítani melyik gömb sebessége változik
// több sugár ugyanazt a szomszédot gömböt találja el 
//   -> impulzus számítása ahányszor sugár találja el és ez nem jó!
//   -> csak egy impulzus számolódjon
// atomic és minden nélkül csak egyenlõséggel írjuk fel -> csak egy tud érvényesülni
// Cuda kernel ami hozzáadja a sebességekhez a lendületeket, 3 pointer és hanyadik szál és megfelelû indexet
// hány szál és workgroup-ok , hány példányban

/*
extern "C" __global__ void impulseSumPerBall(GeometryData::impulseData* v, GeometryData::impulseData* v_r) {
    __shared__ GeometryData::impulseData partial_sum[103];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int ballIdx = threadIdx.y;

    if (tid >= threadIdx.y * 5000 && tid < 5000 * (threadIdx.y+1)) {
        partial_sum[threadIdx.x + ballIdx] = v[tid + ballIdx];
        __syncthreads();
        for (int s = 1; s < blockDim.x + ballIdx; s *= 2) {
            if (threadIdx.x % (2 * s) == 0) {
                partial_sum[threadIdx.x + ballIdx].impulse.x += partial_sum[threadIdx.x + s + ballIdx].impulse.x;
                partial_sum[threadIdx.x + ballIdx].impulse.y += partial_sum[threadIdx.x + s + ballIdx].impulse.y;
                partial_sum[threadIdx.x + ballIdx].impulse.z += partial_sum[threadIdx.x + s + ballIdx].impulse.z;

                partial_sum[threadIdx.x + ballIdx].pos.x += partial_sum[threadIdx.x + s + ballIdx].pos.x;
                partial_sum[threadIdx.x + ballIdx].pos.y += partial_sum[threadIdx.x + s + ballIdx].pos.y;
                partial_sum[threadIdx.x + ballIdx].pos.z += partial_sum[threadIdx.x + s + ballIdx].pos.z;
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            v_r[blockIdx.x + ballIdx].impulse.x = partial_sum[ballIdx].impulse.x;
            v_r[blockIdx.x + ballIdx].impulse.y = partial_sum[ballIdx].impulse.y;
            v_r[blockIdx.x + ballIdx].impulse.z = partial_sum[ballIdx].impulse.z;

            v_r[blockIdx.x + ballIdx].pos.x = partial_sum[ballIdx].pos.x;
            v_r[blockIdx.x + ballIdx].pos.y = partial_sum[ballIdx].pos.y;
            v_r[blockIdx.x + ballIdx].pos.z = partial_sum[ballIdx].pos.z;
        }
    }
}

extern "C" __global__ void impulseSumAll(GeometryData::impulseData* v, GeometryData::impulseData* v_r) {
    __shared__ GeometryData::impulseData partial_sum[103];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int ballIdx = threadIdx.y;


  // if (tid >= threadIdx.y * 3000 && tid < 3000 * (threadIdx.y + 1)) {
    partial_sum[threadIdx.x + ballIdx] = v[tid + ballIdx];
    __syncthreads();
    for (int s = 1; s < blockDim.x + ballIdx; s *= 2) {
        if (threadIdx.x % (2 * s) == 0) {
            partial_sum[threadIdx.x + ballIdx].impulse.x += partial_sum[threadIdx.x + s + ballIdx].impulse.x;
            partial_sum[threadIdx.x + ballIdx].impulse.y += partial_sum[threadIdx.x + s + ballIdx].impulse.y;
            partial_sum[threadIdx.x + ballIdx].impulse.z += partial_sum[threadIdx.x + s + ballIdx].impulse.z;

            partial_sum[threadIdx.x + ballIdx].pos.x += partial_sum[threadIdx.x + s + ballIdx].pos.x;
            partial_sum[threadIdx.x + ballIdx].pos.y += partial_sum[threadIdx.x + s + ballIdx].pos.y;
            partial_sum[threadIdx.x + ballIdx].pos.z += partial_sum[threadIdx.x + s + ballIdx].pos.z;
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        v_r[blockIdx.x + ballIdx].impulse.x = partial_sum[ballIdx].impulse.x;
        v_r[blockIdx.x + ballIdx].impulse.y = partial_sum[ballIdx].impulse.y;
        v_r[blockIdx.x + ballIdx].impulse.z = partial_sum[ballIdx].impulse.z;

        v_r[blockIdx.x + ballIdx].pos.x = partial_sum[ballIdx].pos.x;
        v_r[blockIdx.x + ballIdx].pos.y = partial_sum[ballIdx].pos.y;
        v_r[blockIdx.x + ballIdx].pos.z = partial_sum[ballIdx].pos.z;
    }
   // }
}

extern "C" __global__ void addImpulse(GeometryData::Sphere* sphere, GeometryData::impulseData* v_r) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < 3) {
        GeometryData::impulseData sum = v_r[tid];
        GeometryData::Sphere s = sphere[tid];
        s.vel -= sum.impulse;
        s.center -= sum.pos;
        sphere[tid] = s;
    }
}

extern "C" __global__ void collectIndexes(GeometryData::impulseData* v, int* v_r, int ballInd, int* ind) {
    __shared__ unsigned int c[2];
    __shared__ unsigned int ic;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid == ballInd * 5000) {
         ic = 0;
         c[0] = 0xffff;
         c[1] = 0xffff;
    }
    if (tid >= ballInd * 5000 && tid < 5000 * (ballInd + 1)) 
    {
        __syncthreads();
        if (v[0].ballIndex < 0xffff && c[ic] != v[0].ballIndex && ic < 2)
        {
                c[ic] = v[0].ballIndex;
                v_r[ic] = c[ic];
                ++ic;    
                *ind = ic;
            __syncthreads();
        } 
    }
}

struct ballImpulse {

    int ballIndex;
    ballImpulse(int ballIndex) : ballIndex(ballIndex) {}

    __host__ __device__
        GeometryData::impulseData operator()(const  GeometryData::impulseData& a, const  GeometryData::impulseData& b) const {
        GeometryData::impulseData calc;
        calc.impulse = make_float3(0);
        calc.pos = make_float3(0);
        calc.target_pos = make_float3(0);
        calc.target_vel = make_float3(0);

        if (a.ballIndex == ballIndex) {
            calc.target_vel = make_float3(a.target_vel.x, a.target_vel.y, a.target_vel.z);
            calc.target_pos = make_float3(a.target_pos.x, a.target_pos.y, a.target_pos.z);
        }

        calc.impulse += a.impulse + b.impulse;
        calc.pos += a.pos + b.pos;

        if (b.ballIndex == ballIndex) {
            calc.target_vel += make_float3(b.target_vel.x, b.target_vel.y, b.target_vel.z);
            calc.target_pos += make_float3(b.target_pos.x, b.target_pos.y, b.target_pos.z);
        }
        return calc;
    }
};
*/


struct indexing {

    __host__ __device__
        GeometryData::impulseData operator()(const GeometryData::impulseData& a, const GeometryData::impulseData& b) const {
        GeometryData::impulseData c;

        int ia = 0;
        int ib = 0;
        int ic = 0;

        while ((a.ballsInd[ia] < 0xffff || b.ballsInd[ib] < 0xffff) && ic < 32)
        {
            c.ballsInd[ic] = min(a.ballsInd[ia], b.ballsInd[ib]);
            if (a.ballsInd[ia] == c.ballsInd[ic] && ia < 32) ia++;
            if (b.ballsInd[ib] == c.ballsInd[ic] && ib < 32) ib++;
            
            ic++;
        }
        c.ballsInd[ic] = 0xffff;


        return c;
    }
};

extern "C" __host__ void sphereCollisionHandle(GeometryData::Sphere* s1, GeometryData::Sphere* s2) {

    float3 collisionNormal = s1->center - s2->center;
    float3 relativeVelocity = s1->vel - s2->vel;

    float3 unitCollisionNorm = normalize(collisionNormal);
    float distance = length(collisionNormal);

    float vDn = dot(relativeVelocity, collisionNormal);
    float nDn = dot(collisionNormal, collisionNormal);

    s1->vel -= (2.f * s2->mass / (s1->mass + s2->mass)) * (vDn / nDn) * collisionNormal;
    vDn = dot(-relativeVelocity, -collisionNormal);
    s2->vel -= (2.f * s1->mass / (s1->mass + s2->mass)) * (vDn / nDn) * -collisionNormal;

    //  s1->center += s1->vel * deltaTime;
    //  s2->center += s2->vel * deltaTime;

    s1->center += (s1->radius * 2.f - distance)/2  * unitCollisionNorm;
    s2->center -= (s2->radius * 2.f - distance)/2   * unitCollisionNorm;
}

extern "C" __host__ void
UpdateSphereImpulseWithRayCollision(GeometryData::impulseData* rays, GeometryData::Sphere* sphereList[OBJ_COUNT], int nBalls, int rayPerBalls)
{
    auto tstart = std::chrono::system_clock::now();
    thrust::device_vector<GeometryData::impulseData> device(rays, rays + (nBalls * rayPerBalls));

    GeometryData::impulseData indexData;
    indexData.ballsInd[0] = 0xffff;
    for (int b = 0; b < OBJ_COUNT; b++)
    {
        GeometryData::impulseData indexes = thrust::reduce(device.begin() + rayPerBalls * b, device.begin() + rayPerBalls * (b + 1), indexData, indexing());
        for (int i = 0; indexes.ballsInd[i] < 0xffff; i++) {
          //  std::cerr << "GPU FPS: "<< std::endl;
            sphereCollisionHandle(sphereList[b], sphereList[indexes.ballsInd[i]]);
             /*   std::cerr << i << " " << indexes.collisonResponse[0].impulse.y << std::endl;
                std::cerr << i << " " << indexes.collisonResponse[0].impulse.z << std::endl;
                sphereList[b]->vel -= indexData.collisonResponse[i].impulse;
                sphereList[indexes.ballsInd[i]]->vel -= indexData.collisonResponse[i].target_vel;

                sphereList[b]->center += indexData.collisonResponse[i].pos;
                sphereList[indexes.ballsInd[i]]->center -= indexData.collisonResponse[i].target_pos; */
        }
    }
    auto tnow = std::chrono::system_clock::now();
    std::chrono::duration<double> time = tnow - tstart;

  //  std::cerr << "GPU FPS: " << time.count() << std::endl;

    /* 
      thrust::host_vector<GeometryData::impulseData> h(nBalls * rayPerBalls);
        cudaMemcpy(h.data(), (void*)rays, sizeof(GeometryData::impulseData) * nBalls * rayPerBalls, cudaMemcpyDeviceToHost);

     //   std::cerr << "----" << std::endl;
        for (int i = 0; i < rayPerBalls; i++) {
            if(h[i].ballsInd[0] < 0xffff)
             //   std::cerr << i << " " << " index:" << h[i * rayPerBalls].ballsInd[0] <<  std::endl;
              std::cerr << i << " " << " x:" << (h[i].ballsInd[0]) << std::endl;
        //    std::cerr << i << " " << " y:" << (h[i * rayPerBalls].pos.y) << std::endl;
        //    std::cerr << i << " " << " z:" << (h[i * rayPerBalls].pos.z) << std::endl;
        }
 
 
  
        * O(logn) <=> (n*(n-1)), n=3 -> O(6)

     std::cerr << "----" << std::endl;
        for (int i = 0; i < 5000; i++)
             if (h[i].ballIndex == 1 || h[i].ballIndex == 2)
                 std::cerr << i << " " << h[i].ballIndex << std::endl;
        std::cerr << "----" << std::endl;



        GeometryData::impulseData ball_1 = thrust::reduce(device.begin(), device.end() - 10000, GeometryData::impulseData(), ball1());
        GeometryData::impulseData plus_1 = thrust::reduce(device.end() - 10000, device.end(), GeometryData::impulseData(), ball1());
        sphereList[0]->vel -= plus_1.target_vel;
        sphereList[0]->center += plus_1.target_pos;
        sphereList[0]->vel -= ball_1.impulse;
        sphereList[0]->center -= ball_1.pos;

        GeometryData::impulseData ball_2 = thrust::reduce(device.end() - 10000, device.end() - 5000, GeometryData::impulseData(), ball2());
        GeometryData::impulseData plus_2 = thrust::reduce(device.begin(), device.end(), GeometryData::impulseData(), ball2());
        sphereList[1]->vel -= plus_2.target_vel;
        sphereList[1]->center += plus_2.target_pos;
        sphereList[1]->vel -= ball_2.impulse;
        sphereList[1]->center -= ball_2.pos;

        GeometryData::impulseData ball_3 = thrust::reduce(device.end() - 5000, device.end(), GeometryData::impulseData(), ball3());
        GeometryData::impulseData plus_3 = thrust::reduce(device.begin(), device.end() - 5000, GeometryData::impulseData(), ball3());
        sphereList[2]->vel -= plus_3.target_vel;
        sphereList[2]->center += plus_3.target_pos;
        sphereList[2]->vel -= ball_3.impulse;
        sphereList[2]->center -= ball_3.pos;
        */
    /*
              int N = 15000;
              const int TB_SIZE = 150;
              dim3 threadsPerBlock(150);

              int GRID_SIZE = (int)ceil(N / TB_SIZE);
              for (int b = 0; b < 3; b++) {
                int* d_v_r;
                cudaMalloc(reinterpret_cast<void**>(& d_v_r), 3 * sizeof(int));
                int* ind;
                cudaMalloc(reinterpret_cast<void**>(&ind), sizeof(int));

                std::vector<int> h_v_r(2);
                int numOfIndex = 0;
                h_v_r.push_back(0xffff);
                h_v_r.push_back(0xffff);
                  collectIndexes <<< GRID_SIZE, threadsPerBlock >>> (rays, d_v_r, b, ind);
                  cudaMemcpy(h_v_r.data(), d_v_r, 2 * sizeof(int), cudaMemcpyDeviceToHost);
                  cudaMemcpy(&numOfIndex, ind, sizeof(int), cudaMemcpyDeviceToHost);
                  for (int j = 0; j < numOfIndex; j++)
                  {
                   //   if(h_v_r[j] == b)
                   //     std::cerr << j << " " << h_v_r[j] << std::endl;
                      if (h_v_r[j] < 0xffff && h_v_r[j] != b) {

                          sphereCollisionHandle(sphereList[b], sphereList[h_v_r[j]], dt);
                      }
                  }
              }


            thrust::host_vector<GeometryData::impulseData> h(15000);
            cudaMemcpy(h.data(), (void*)rays, sizeof(GeometryData::impulseData) * 15000, cudaMemcpyDeviceToHost);

            for (int b = 0; b < 3; b++)
            {
                int ia = 0;
                int ib = 0;
                int ic = 1;
                int calc[32];
                // ball indexek megtalálása reduce-cal
                for (int a = 0; a < 32; a++)
                    calc[a] = 0xffff;
                for (int i = b * 5000 + 1; i < 5000 * (b+1); i++)
                {
                    if ((h[i - 1].ballIndex < 0xfffff || h[i].ballIndex < 0xffff) && ic < 32) {
                        if (calc[ic - 1] != min(h[i - 1].ballIndex, h[i].ballIndex)) {
                            calc[ic - 1] = min(h[i - 1].ballIndex, h[i].ballIndex);
                            ic++;
                        }
                    }
                    if (ic == 3)
                        break;
                }
                for (int j = 0; j < ic-1; j++)
                {
                    if (calc[j] < 0xffff && calc[j] != b)
                        sphereCollisionHandle(sphereList[b], sphereList[calc[j]], dt);
                }
            }



         //   const int TB_SIZE = 100;

          //  int GRID_SIZE = (int)ceil(N / TB_SIZE);

          //  dim3 threadsPerBlock(100, 3);

            //impulseSumPerBall << <GRID_SIZE, threadsPerBlock >> > (rays, result_sum);
          //  impulseSumAll << <1, threadsPerBlock >> > (result_sum, result_sum);
         //   addImpulse <<< 1, 3 >>> (sphereList, result_sum);
        */
}
