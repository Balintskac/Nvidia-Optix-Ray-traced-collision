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
#include <optixWhitted/helpers.h>

extern "C" {
__constant__ Params params;
}

static __device__ __inline__ RadiancePRD getRadiancePRD()
{
    RadiancePRD prd;
    prd.result.x = __uint_as_float( optixGetPayload_0() );
    prd.result.y = __uint_as_float( optixGetPayload_1() );
    prd.result.z = __uint_as_float( optixGetPayload_2() );
    prd.importance = __uint_as_float( optixGetPayload_3() );
    prd.depth = optixGetPayload_4();
    return prd;
}

static __device__ __inline__ CollisionPRD getCollisionPRD()
{
    CollisionPRD prd;
  /*  prd.result.impulse.x = __uint_as_float(optixGetPayload_0());
    prd.result.impulse.y = __uint_as_float( optixGetPayload_1() );
    prd.result.impulse.z = __uint_as_float( optixGetPayload_2() );

    prd.result.pos.x = __uint_as_float(optixGetPayload_3());
    prd.result.pos.y = __uint_as_float(optixGetPayload_4());
    prd.result.pos.z = __uint_as_float(optixGetPayload_5());

    prd.result.target_vel.x = __uint_as_float(optixGetPayload_6());
    prd.result.target_vel.y = __uint_as_float(optixGetPayload_7());
    prd.result.target_vel.z = __uint_as_float(optixGetPayload_8());

    prd.result.target_pos.x = __uint_as_float(optixGetPayload_9());
    prd.result.target_pos.y = __uint_as_float(optixGetPayload_10());
    prd.result.target_pos.z = __uint_as_float(optixGetPayload_11());
    */
    prd.result.ballIndex = optixGetPayload_0();
    *prd.result.ballsInd = optixGetPayload_1();
    return prd;
}

static __device__ __inline__ void setRadiancePRD( const RadiancePRD &prd )
{
    optixSetPayload_0( __float_as_uint(prd.result.x) );
    optixSetPayload_1( __float_as_uint(prd.result.y) );
    optixSetPayload_2( __float_as_uint(prd.result.z) );
    optixSetPayload_3( __float_as_uint(prd.importance) );
    optixSetPayload_4( prd.depth );
}

static __device__ __inline__ void setCollisionPRD(const CollisionPRD& prd)
{
  /*  optixSetPayload_0(__float_as_uint(prd.result.impulse.x));
    optixSetPayload_1(__float_as_uint(prd.result.impulse.y));
    optixSetPayload_2(__float_as_uint(prd.result.impulse.z));

    optixSetPayload_3(__float_as_uint(prd.result.pos.x));
    optixSetPayload_4(__float_as_uint(prd.result.pos.y));
    optixSetPayload_5(__float_as_uint(prd.result.pos.z));

    optixSetPayload_6(__float_as_uint(prd.result.target_vel.x));
    optixSetPayload_7(__float_as_uint(prd.result.target_vel.y));
    optixSetPayload_8(__float_as_uint(prd.result.target_vel.z));

    optixSetPayload_9(__float_as_uint(prd.result.target_pos.x));
    optixSetPayload_10(__float_as_uint(prd.result.target_pos.y));
    optixSetPayload_11(__float_as_uint(prd.result.target_pos.z));
    */
    optixSetPayload_0(prd.result.ballIndex);
    optixSetPayload_1(*prd.result.ballsInd);

}

static __device__ __inline__ void setOcclusionPRD( const OcclusionPRD &prd )
{
    optixSetPayload_0( __float_as_uint(prd.attenuation.x) );
    optixSetPayload_1( __float_as_uint(prd.attenuation.y) );
    optixSetPayload_2( __float_as_uint(prd.attenuation.z) );
}
 
/*
static __device__ __inline__ float3
traceRadianceRay(
    float3 origin,
    float3 direction,
    int depth,
    float importance)
{
    RadiancePRD prd;
    prd.depth = depth;
    prd.importance = importance;

    optixTrace(
        params.handle,
        origin,
        direction,
        params.scene_epsilon,
        1e16f,
        0.0f,
        OptixVisibilityMask( 1 ),
        OPTIX_RAY_FLAG_NONE,
        RAY_TYPE_RADIANCE,
        RAY_TYPE_COUNT,
        RAY_TYPE_RADIANCE,
        float3_as_args(prd.result),
        // Can't use __float_as_uint() because it returns rvalue but payload requires a lvalue
        reinterpret_cast<unsigned int&>(prd.importance),
        reinterpret_cast<unsigned int&>(prd.depth) );

    return prd.result;
}
*/
static
__device__ void phongShadowed()
{
    // this material is opaque, so it fully attenuates all shadow rays
    OcclusionPRD prd;
    prd.attenuation = make_float3(0.f);
    setOcclusionPRD(prd);
}

static
__device__ void phongShade(float3 p_Kd,
    float3 p_Ka,
    float3 p_Ks,
    float3 p_Kr,
    float  p_phong_exp,
    float3 p_normal,
    int index)
{
    const float3 ray_orig = optixGetWorldRayOrigin();
    const float3 ray_dir = optixGetWorldRayDirection();
    const float  ray_t = optixGetRayTmax();

    RadiancePRD prd = getRadiancePRD();

    float3 hit_point = ray_orig + ray_t * ray_dir;

    // ambient contribution
    float3 result = p_Ka * params.ambient_light_color;

    // compute direct lighting
    BasicLight light = params.light;
    float Ldist = length(light.pos - hit_point);
    float3 L = normalize(light.pos - hit_point);
    float nDl = dot(p_normal, L);

    // cast shadow ray
    float3 light_attenuation = make_float3(static_cast<float>(nDl > 0.0f));
    if (nDl > 0.0f)
    {
        OcclusionPRD shadow_prd;
        shadow_prd.attenuation = make_float3(1.0f);

        optixTrace(
            params.handle,
            hit_point,
            L,
            0.01f,
            Ldist,
            0.0f,
            OptixVisibilityMask(1),
            OPTIX_RAY_FLAG_NONE,
            RAY_TYPE_OCCLUSION,
            RAY_TYPE_COUNT,
            RAY_TYPE_OCCLUSION,
            float3_as_args(shadow_prd.attenuation));

        light_attenuation = shadow_prd.attenuation;
    }
    // If not completely shadowed, light the hit point
    if (fmaxf(light_attenuation) > 0.0f)
    {
        float3 Lc = light.color * light_attenuation;

        result += p_Kd * nDl * Lc;

        float3 H = normalize(L - ray_dir);
        float nDh = dot(p_normal, H);
        if (nDh > 0)
        {
            float power = pow(nDh, p_phong_exp);
            result += p_Ks * power * Lc;
        }
    }
    prd.result = result;
    // Azok a gömbök vannak benne, akik gömb mellett vannak (színezzük)
    /*  for (size_t i = 0; i < 1000; i++) {
        if (params.accum_buffer_motion[i].ballsInd[0] == index)
            prd.result = make_float3(1.0f, 1.0f, 0);
    }
    if (index == 0)
        prd.result = make_float3(1.0f, 0.0f, 0);
  
    for (size_t i = 0; i < 10000; i++) {
        if (params.accum_buffer_motion[i].ballsInd[0] == index)
            prd.result = make_float3(0, 1.0f, 0);
    }
  */

    

 // pass the color back

    setRadiancePRD(prd);
  }


extern "C" __global__ void __closesthit__metal_radiance()
{
    const HitGroupData* sbt_data = (HitGroupData*) optixGetSbtDataPointer();
    const Phong &phong = sbt_data->shading.metal;
    const int& ballIndex = sbt_data->intersectIndex.index;
    float3 object_normal = make_float3(
        __uint_as_float( optixGetAttribute_0() ),
        __uint_as_float( optixGetAttribute_1() ),
        __uint_as_float( optixGetAttribute_2() ));

    float3 world_normal = normalize( optixTransformNormalFromObjectToWorldSpace( object_normal ) );
    float3 ffnormal = faceforward( world_normal, -optixGetWorldRayDirection(), world_normal );
    phongShade( phong.Kd, phong.Ka, phong.Ks, phong.Kr, phong.phong_exp, ffnormal, ballIndex);
}

extern "C" __global__ void __closesthit__full_occlusion()
{
    phongShadowed();
}

extern "C" __global__ void sphereCollisionHandle(GeometryData::Sphere s1, GeometryData::Sphere& s2, GeometryData::impulseData& prd)
{

    float3 collisionNormal = s1.center - s2.center;
    float3 relativeVelocity = s1.vel - s2.vel;

    float3 unitCollisionNorm = normalize(collisionNormal);
    float distance = length(collisionNormal);

    float vDn = dot(relativeVelocity, collisionNormal);
    float nDn = dot(collisionNormal, collisionNormal);
 /*
    prd.impulse = s1.center;
    vDn = dot(-relativeVelocity, -collisionNormal);
    prd.target_vel = (1.5f * s1.mass / (s1.mass + s2.mass)) * (vDn / nDn) * -collisionNormal;

    prd.pos = (s1.radius * 2.f - distance) / 2 * unitCollisionNorm;
    prd.target_pos = (s2.radius * 2.f - distance) / 2 * unitCollisionNorm;

   
        s1.vel -= (1.5f * s2.mass / (s1.mass + s2.mass)) * (vDn / nDn) * collisionNormal;
    vDn = dot(-relativeVelocity, -collisionNormal);
    s1.vel -= (1.5f * s1.mass / (s1.mass + s2.mass)) * (vDn / nDn) * -collisionNormal;

    s1.center += (s1.radius * 2.f - distance) / 2 * unitCollisionNorm;
    s2.center -= (s2.radius * 2.f - distance) / 2 * unitCollisionNorm;

    prd.impulse = s1.vel;
    prd.pos = s1.center;
    prd.target_pos = s2.center;
    prd.target_vel = s2.vel;
    */

}

extern "C" __global__ void __closesthit__sphere_collision()
{
    HitGroupData* sbt_data = (HitGroupData*)optixGetSbtDataPointer();


    const int& index = sbt_data->sphereInd.index;

    CollisionPRD prd = getCollisionPRD();
 //   prd.result.impulse = make_float3(0.f);
  //  prd.result.pos = make_float3(0.f);
//    prd.result.target_vel = make_float3(0.f);
   // prd.result.target_pos = make_float3(0.f);
    int originBallIndex = prd.result.ballIndex;
    for (int i = 0; i < 32; i++) 
        prd.result.ballsInd[i] = 0xffff;

    if (index != originBallIndex) {
        const float3 s1 = optixGetWorldRayOrigin();
        const float3& s2 = sbt_data->geometry.sphere.center;
     //   GeometryData::Sphere s1 = sbt_data->intersects.sphereList[originBallIndex];
       // GeometryData::Sphere s2 = sbt_data->intersects.sphereList[index];
        float3 collisionNormal = s2 - s1;
        float distance = length(collisionNormal);

        if (distance <= 2.0f)
        {
            prd.result.ballsInd[0] = index;
          //  prd.result.pos.x = distance;
           /*
            float3 relativeVelocity = s1.vel - s2.vel;

            float3 unitCollisionNorm = normalize(collisionNormal);
            float distance = length(collisionNormal);

            float vDn = dot(relativeVelocity, collisionNormal);
            float nDn = dot(collisionNormal, collisionNormal);

            prd.result.impulse = (1.5f * s2.mass / (s1.mass + s2.mass)) * (vDn / nDn) * collisionNormal;
            vDn = dot(-relativeVelocity, -collisionNormal);
            prd.result.target_vel = (1.5f * s1.mass / (s1.mass + s2.mass)) * (vDn / nDn) * -collisionNormal;

            prd.result.pos = (s1.radius * 2.f - distance) / 2 * unitCollisionNorm;
            prd.result.target_pos = (s2.radius * 2.f - distance) / 2 * unitCollisionNorm;
            */
          //  spherecollisionhandle(sbt_data->intersects.sphereList[originBallIndex], sbt_data->intersects.sphereList[index], prd.result);
        }
    }
   // prd.result.pos = s;
  //  prd.result.pos = sbt_data->geometry.pos;
     prd.result.ballIndex = 1000 * (originBallIndex +1);
    setCollisionPRD(prd);
}

/*
extern "C" __global__ void __closesthit__floor_collision()
{
    const HitGroupData* sbt_data = (HitGroupData*)optixGetSbtDataPointer();

    const float3& sphere = sbt_data->geometry.velocity;
    const Parallelogram& floor = sbt_data->floor_impulse.floor;
    const float3& sphere_vel = sbt_data->rayTracedCollision.velocity;
    float3 object_normal = make_float3(
        __uint_as_float(optixGetAttribute_0()),
        __uint_as_float(optixGetAttribute_1()),
        __uint_as_float(optixGetAttribute_2()));

    // intersection vectors
    const float3 n = normalize(make_float3(floor.plane)); // normalize( optixTransformNormalFromObjectToWorldSpace( object_normal) ); // normal
    const float3 ray_orig = optixGetWorldRayOrigin();
    const float3 ray_dir  = optixGetWorldRayDirection();                 // incident direction
    const float  ray_t    = optixGetRayTmax();

    float3 hit_point = ray_orig + ray_t * ray_dir;

    CollisionPRD prd = getCollisionPRD();
    prd.result.impulse = make_float3(0);
    prd.result.ballsInd[0] = 0xffff;
    prd.result.ballsInd[1] = 0xffff;

    float3 collisionNormal = make_float3(floor.plane.x - sphere.x,
        floor.plane.y - sphere.y,
        floor.plane.z - sphere.z);

    float distance = sqrt(dot(collisionNormal, collisionNormal));
    float deltaVel = 0.0001f;


        if (sphere.y <= 0.0f && floor.floor_index == 0) {
         //  prd.result.impulse = (2.0f * n * dot(n, sphere_vel)) * deltaVel;
        }

       if (sphere.z >= -4.9f && floor.floor_index == 1)  {
        //   prd.result.impulse = (2.0f * n * dot(n, sphere_vel)) * deltaVel;
        }  


       if (sphere.x <= 5.f && floor.floor_index == 4) {
       //    prd.result.impulse = (2.0f * n * dot(n, sphere_vel)) * deltaVel;
       }

   //  setCollisionPRD(prd);
}
*/

extern "C" __global__ void __miss__constant_bg()
{
    const MissData* sbt_data = (MissData*) optixGetSbtDataPointer();
    RadiancePRD prd = getRadiancePRD();
    prd.result = sbt_data->bg_color;
    setRadiancePRD(prd);
}

extern "C" __global__ void __miss__constant_bg_impulse()
{
    const MissData* sbt_data = (MissData*) optixGetSbtDataPointer();
    CollisionPRD prd = getCollisionPRD();
    for (int i = 0; i < 32; i++)
        prd.result.ballsInd[i] = 0xffff;
    setCollisionPRD(prd);
}
