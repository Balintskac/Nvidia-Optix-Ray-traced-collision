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

#include <optix.h>

#include "optixRayTracedCollision.h"
#include "helpers.h"
#include <iostream>
#include <optixWhitted/helpers.h>

extern "C" {
__constant__ Params params;
}

extern "C" __global__ void __intersection__parallelogram()
{
    const Parallelogram* floor = reinterpret_cast<Parallelogram*>( optixGetSbtDataPointer() );

    const float3 ray_orig = optixGetWorldRayOrigin();
    const float3 ray_dir  = optixGetWorldRayDirection();
    const float ray_tmin = optixGetRayTmin(), ray_tmax = optixGetRayTmax();

    float3 n = make_float3( floor->plane );
    float dt = dot(ray_dir, n );
    float t = (floor->plane.w - dot(n, ray_orig))/dt;
    if( t > ray_tmin && t < ray_tmax )
    {
        float3 p = ray_orig + ray_dir * t;
        float3 vi = p - floor->anchor;
        float a1 = dot(floor->v1, vi);
        if(a1 >= 0 && a1 <= 1)
        {
            float a2 = dot(floor->v2, vi);
            if(a2 >= 0 && a2 <= 1)
            {
                optixReportIntersection(
                    t,
                    0,
                    float3_as_args(n),
                    __float_as_uint( a1 ), __float_as_uint( a2 )
                    );
            }
        }
    }
}

extern "C" __global__ void __intersection__sphere_motion()
{
    const GeometryData::Sphere* sbt_data = reinterpret_cast<GeometryData::Sphere*>(optixGetSbtDataPointer());
    const float3  ray_orig = optixGetWorldRayOrigin();
    const float3  ray_dir = optixGetWorldRayDirection();
    const float ray_tmin = optixGetRayTmin();

    float3 O = ray_orig - sbt_data->center;
    float  l = 1 / length(ray_dir);
    float3 D = ray_dir * l;

    float b = dot(O, D), sqr_b = b * b;
    float O_dot_O = dot(O, O);
    float radius1 = 1.0f;
    float sqr_radius1 = radius1 * radius1;
    float c = O_dot_O - sqr_radius1;
    float root = sqr_b - c;
    if (root > 0.0f) {
        float t1 = -b - sqrtf(root);
        float t2 = -b + sqrtf(root);
        // if (0.0f != length(O)) {
            if (t1 < t2 || t1 == t2) {
                optixReportIntersection(t1 * l, 0);
            }
            else
            {
                optixReportIntersection(t2 * l, 0);
            }
        //  }
        
    }
}

extern "C" __global__ void __intersection__floor_motion()
{
    const Parallelogram* floor = reinterpret_cast<Parallelogram*>(optixGetSbtDataPointer());

    const float3 ray_orig = optixGetWorldRayOrigin();
    const float3 ray_dir = optixGetWorldRayDirection();
    const float ray_tmin = optixGetRayTmin(), ray_tmax = optixGetRayTmax();

    float3 n = make_float3(floor->plane);
    float dt = dot(ray_dir, n);
    float t = (floor->plane.w - dot(n, ray_orig)) / dt;
    if (t > ray_tmin && t < ray_tmax)
    {
        float3 p = ray_orig + ray_dir * t;
        float3 vi = p - floor->anchor;
        float a1 = dot(floor->v1, vi);
        float a2 = dot(floor->v2, vi);

        optixReportIntersection(
            t,
            0,
            float3_as_args(n),
            __float_as_uint(a1), __float_as_uint(a2)
        );
    }
            
        
    
}