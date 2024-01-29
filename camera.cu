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

#include <vector_types.h>
#include <optix_device.h>
#include "optixRayTracedCollision.h"
#include "random.h"
#include "helpers.h"
#include <cuda/helpers.h>
#include <optixWhitted/helpers.h>

extern "C" {
    __constant__  Params params;
}

extern "C" __global__ void __raygen__pinhole_camera()
{
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

     CameraData* camera = (CameraData*) optixGetSbtDataPointer();

    const unsigned int image_index = params.width * idx.y + idx.x;
    unsigned int       seed        = tea<16>( image_index, params.subframe_index );

    //589 824 Subpixel jitter: send the ray through a different position inside the pixel each time,
    // to provide antialiasing. The center of each pixel is at fraction (0.5,0.5)
    float2 subpixel_jitter = params.subframe_index == 0 ?
        make_float2(0.5f, 0.5f) : make_float2(rnd(seed), rnd(seed));

    float2 d = ((make_float2(idx.x, idx.y) + subpixel_jitter) / make_float2(params.width, params.height)) * 2.f - 1.f;
    float3 ray_origin = camera->eye;
    float3 ray_direction = normalize(d.x * camera->U + d.y * camera->V + camera->W);

    RadiancePRD prd;
    prd.importance = 1.f;
    prd.depth = 0;

    optixTrace(
        params.handle,
        ray_origin,
        ray_direction,
        params.scene_epsilon,
        1000.f,
        10.0f,
        OptixVisibilityMask(1),
        OPTIX_RAY_FLAG_NONE,
        RAY_TYPE_RADIANCE,
        RAY_TYPE_COUNT,
        RAY_TYPE_RADIANCE,
        float3_as_args(prd.result),
        reinterpret_cast<unsigned int&>(prd.importance),
        reinterpret_cast<unsigned int&>(prd.depth));

    float4 acc_val = params.accum_buffer[image_index];

    if (params.subframe_index > 0)
    {
        acc_val = lerp(acc_val, make_float4(prd.result, 0.f), 1.0f / static_cast<float>(params.subframe_index + 1));
    }
    else
    {
        acc_val = make_float4(prd.result, image_index);
    }

    params.frame_buffer[image_index] = make_color(acc_val);
    params.accum_buffer[image_index] = acc_val;
    
}

extern "C" __global__ void __raygen__sphere_motion()
{
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

     CameraData* camera = (CameraData*)optixGetSbtDataPointer();
    const unsigned int image_index = params.width_impulse * idx.y + idx.x + idx.z * dim.x * dim.y;

    float PI = 3.14;
    float theta = 2* PI * idx.x/20.f;
    float phi = PI/2 * idx.y/20.f;

    //sin/cos https://www.scratchapixel.com/lessons/mathematics-physics-for-computer-graphics/geometry/spherical-coordinates-and-trigonometric-functions

    float3 ray_direction = {
        cos(phi) * sin(theta),
        cos(theta),
        sin(phi) * sin(theta)
    };

    CollisionPRD prd;
    prd.result = GeometryData::impulseData();
 //   prd.result.impulse = { 0.f };
 //   prd.result.pos = { 0.f };
 //   prd.result.target_pos = { 0.f };
  //  prd.result.target_vel = { 0.f };
    prd.result.ballIndex = idx.z;

    for (int i = 0; i < 32; i++)
       prd.result.ballsInd[i] = 0xffff;

   // float3 ray_origin = camera->spheres.at(idx.z);
    float3 ray_origin = camera->pos[idx.z];
    optixTrace(
        params.handle,
        ray_origin,
        ray_direction,
        params.scene_epsilon,
        100.f,
        2.5f,
        OptixVisibilityMask(1),
        OPTIX_RAY_FLAG_NONE,
        RAY_TYPE_RADIANCE,
        RAY_TYPE_COUNT,
        RAY_TYPE_RADIANCE,
      //  float3_as_args(prd.result.pos),
      //  float3_as_args(prd.result.target_vel),
      //  float3_as_args(prd.result.target_pos),
        reinterpret_cast<unsigned int&>(prd.result.ballIndex),
        reinterpret_cast<unsigned int&>(prd.result.ballsInd)
        );
      
    GeometryData::impulseData acc_val = params.accum_buffer_motion[image_index];

    //   float4 acc = params.accum_buffer[image_index];
        acc_val = prd.result;
      //  acc_val.ballIndex = idx.z * 5000;
      //  acc_val.pos = camera->eye;

    //   acc_val.sourceIndex = image_index;
    //    acc = make_float4(prd.result.impulse, 0.f);
    // Impulse buffer (Ray traed collision)
    params.impulse_buffer[image_index] = acc_val;
    //  params.frame_buffer_motion[image_index] = acc_val.impulse;
    params.accum_buffer_motion[image_index] = acc_val;

    // Render buffer (small window with 100*150)
    //   params.frame_buffer[image_index] = make_color(acc);
    //   params.accum_buffer[image_index] =acc;

}