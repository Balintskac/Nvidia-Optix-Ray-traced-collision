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

#include <glad/glad.h> // Needs to be included before gl_interop

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

#include <sampleConfig.h>

#include <sutil/Camera.h>
#include <sutil/Trackball.h>
#include <sutil/CUDAOutputBuffer.h>
#include <sutil/Exception.h>
#include <sutil/GLDisplay.h>
#include <sutil/Matrix.h>
#include <sutil/sutil.h>
#include <sutil/vec_math.h>

#include <GLFW/glfw3.h>
#include <iomanip>
#include <cstring>
#include <array>

#include "optixRayTracedCollision.h"
#include "cudaHelpers.h"
#include <imgui/imgui.h>
#include "imgui/imgui_impl_glfw.h"
#include "imgui/imgui_impl_opengl3.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

using std::vector;

//------------------------------------------------------------------------------
//
// Globals
//
//------------------------------------------------------------------------------

bool              resize_dirty  = false;
bool              minimized     = false;
bool              changedx = false;
bool              changedy = false;
bool              changedz = false;

// Camera state
bool              camera_changed = true;
sutil::Camera     camera;
sutil::Trackball  trackball;

// Mouse state
int32_t           mouse_button = -1;

const int         max_trace = 12;

//------------------------------------------------------------------------------
//
// Local types
// TODO: some of these should move to sutil or optix util header
//
//------------------------------------------------------------------------------

template <typename T>
struct Record
{
    __align__( OPTIX_SBT_RECORD_ALIGNMENT)

    char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

typedef Record<CameraData>      RayGenRecord;
typedef Record<MissData>        MissRecord;
typedef Record<HitGroupData>    HitGroupRecord;
typedef Record<RayGenMotionData>  RayGenMotionRecord;

GeometryData::Sphere* sphereList[OBJ_COUNT];

struct RayTracedState
{
    size_t                      output_buffer_size = 0;
    CUdeviceptr                 d_temp_buffer_gas = 0;
    CUdeviceptr                 d_temp_buffer = 0;
    OptixAccelBufferSizes       gas_buffer_sizes;
    size_t                      temp_buffer_size = 0;

    OptixDeviceContext          context                   = 0;
    OptixDeviceContext          context_implse = 0;
    OptixTraversableHandle      gas_handle                = {};
    CUdeviceptr                 output_buffer       = {};
    CUdeviceptr                 output_impulse_buffer = {};
    CUdeviceptr                 d_ias_output_buffer = 0;
    CUdeviceptr                 d_static_gas_output_buffer;

    OptixModule                 geometry_module           = 0;
    OptixModule                 camera_module             = 0;
    OptixModule                 shading_module            = 0;
    OptixModule                 sphere_module             = 0;
    OptixModule                 sphere_module_impulse     = 0;
    OptixModule                 raygen_module             = 0;
    OptixModule                 motion_module             = 0;

    OptixProgramGroup           raygen_prog_group         = 0;
    OptixProgramGroup           radiance_miss_prog_group  = 0;
    OptixProgramGroup           occlusion_miss_prog_group = 0;
    OptixProgramGroup           radiance_glass_sphere_prog_group  = 0;
    OptixProgramGroup           occlusion_glass_sphere_prog_group = 0;
    OptixProgramGroup           radiance_metal_sphere_prog_group  = 0;
    OptixProgramGroup           occlusion_metal_sphere_prog_group = 0;
    OptixProgramGroup           radiance_floor_prog_group         = 0;
    OptixProgramGroup           occlusion_floor_prog_group        = 0;

    OptixProgramGroup           raygen_motion_prog_group = 0;

    OptixProgramGroup           intersect_motion_prog_group = 0;
    OptixProgramGroup           intersect_floor_motion_prog_group = 0;
    OptixProgramGroup           closeshit_floor_motion_prog_group = 0;
    OptixProgramGroup           closeshit_motion_prog_group = 0;
    OptixProgramGroup           radiance_motion_miss_prog_group = 0;

    OptixPipeline               pipeline                  = 0;
    OptixPipelineCompileOptions pipeline_compile_options  = {};

    OptixPipeline               pipeline_motion = 0;
    OptixPipelineCompileOptions pipeline_motion_compile_options = {};

    CUstream                    stream                    = 0;
    Params                      params;
    Params*                     d_params                  = nullptr;

    CUstream                    stream_motion = 0;
    Params                      params_motion;
    Params*                     d_params_motion = nullptr;

    OptixShaderBindingTable     sbt                       = {};
    OptixShaderBindingTable     sbt_motion = {};
    float                       time = 0.f;
    int is_ray_traced_pyhsics = false;
    bool isWallVisible = false;
    float                       deltaTime = 0.f;
    HitGroupRecord*            hitgroup_records;
    HitGroupRecord*            hitgroup_records_impulse;

    float3 accel = { 0,-0.000981,0 };

    OptixBuildInput aabb_input = {};
    CUdeviceptr    d_aabb;
    RayGenRecord* rg_sbt;
    RayGenRecord* rg_sbt_motion;
    RayGenRecord* rg_sbt_cam;
    OptixAabb* aabb;
    uint32_t* aabb_input_flags;
    CameraData* camData;
    
};

//------------------------------------------------------------------------------
//
//  Geometry and Camera data
//
//------------------------------------------------------------------------------

/*
const Parallelogram g_floor(
    make_float3( 10.0f, 0.0f, 0.0f ),    // v1
    make_float3( 0.0f, 0.0f, 12.f ),    // v2
    make_float3( -5.075f, -1.f, -5.507f ),  // anchor
    0
    );

const Parallelogram g_floor_right(
    make_float3(10.0f, 0.0f, 0.0f),    // v1
    make_float3(0.0f, 12.0f, 0.0f),    // v2
    make_float3(-5.075f, -1.f, 6.4f), // anchor,
    1
);

const Parallelogram g_floor_left(
    make_float3(10.0f, 0.0f, 0.0f),    // v1
    make_float3(0.0f, 12.f, 0.0f),    // v2
    make_float3(-5.075f, -1.f, -5.5f),
    2// anchor
);

const Parallelogram g_floor_side(
    make_float3(0.0f, 0.0f, 12.0f),    // v1
    make_float3(0.0f, 12.0f, 0.0f),    // v2
    make_float3(4.9075f, -1.f, -5.5f),
    3// anchor
);

const Parallelogram g_floor_cross (
    make_float3(0.0f, 0.0f, 12.0f),    // v1
    make_float3(0.0f, 12.0f, 0.0f),    // v2
    make_float3(-5.075f, -1.f, -5.5f),
    4// anchor
);

const Parallelogram g_floor_top (
    make_float3(10.0f, 0.0f, 0.0f),    // v1
    make_float3(0.0f, 0.0f, 12.f),    // v2
    make_float3(-5.075f, 11.f, -5.507f),  // anchor
    5// anchor
);
*/
const BasicLight g_light = {
    make_float3( 60.0f, 40.0f, 0.0f ),   // pos
    make_float3( 1.0f, 1.0f, 1.0f )      // color
};


//------------------------------------------------------------------------------
//
// GLFW callbacks
//
//------------------------------------------------------------------------------

static void mouseButtonCallback( GLFWwindow* window, int button, int action, int mods )
{
    double xpos, ypos;
    glfwGetCursorPos( window, &xpos, &ypos );

    if( action == GLFW_PRESS )
    {
        mouse_button = button;
        trackball.startTracking(static_cast<int>( xpos ), static_cast<int>( ypos ));
    }
    else
    {
        mouse_button = -1;
    }
}


static void cursorPosCallback( GLFWwindow* window, double xpos, double ypos )
{
    Params* params = static_cast<Params*>( glfwGetWindowUserPointer( window ) );

    if( mouse_button == GLFW_MOUSE_BUTTON_LEFT )
    {
        trackball.setViewMode( sutil::Trackball::LookAtFixed );
        trackball.updateTracking( static_cast<int>( xpos ), static_cast<int>( ypos ), params->width, params->height );
        camera_changed = true;
    }
    else if( mouse_button == GLFW_MOUSE_BUTTON_RIGHT )
    {
        trackball.setViewMode( sutil::Trackball::EyeFixed );
        trackball.updateTracking( static_cast<int>( xpos ), static_cast<int>( ypos ), params->width, params->height );
        camera_changed = true;
    }
    camera_changed = true;
}


static void windowSizeCallback( GLFWwindow* window, int32_t res_x, int32_t res_y )
{
    // Keep rendering at the current resolution when the window is minimized.
    if( minimized )
        return;

    // Output dimensions must be at least 1 in both x and y.
    sutil::ensureMinimumSize( res_x, res_y );

    Params* params = static_cast<Params*>( glfwGetWindowUserPointer( window ) );
    params->width  = res_x;
    params->height = res_y;
    camera_changed = true;
    resize_dirty   = true;
}


static void windowIconifyCallback( GLFWwindow* window, int32_t iconified )
{
    minimized = ( iconified > 0 );
}


static void keyCallback( GLFWwindow* window, int32_t key, int32_t /*scancode*/, int32_t action, int32_t /*mods*/ )
{
    if( action == GLFW_PRESS )
    {
        if( key == GLFW_KEY_Q ||
            key == GLFW_KEY_ESCAPE )
        {
            glfwSetWindowShouldClose( window, true );
        }
    }
    else if( key == GLFW_KEY_G )
    {
        // toggle UI draw
    }
}


static void scrollCallback( GLFWwindow* window, double xscroll, double yscroll )
{
    if(trackball.wheelEvent((int)yscroll))
        camera_changed = true;
}

//------------------------------------------------------------------------------
//
//  Helper functions
//
//------------------------------------------------------------------------------

void printUsageAndExit( const char* argv0 )
{
    std::cerr << "Usage  : " << argv0 << " [options]\n";
    std::cerr << "Options: --file | -f <filename>      File for image output\n";
    std::cerr << "         --no-gl-interop             Disable GL interop for display\n";
    std::cerr << "         --dim=<width>x<height>      Set image dimensions; defaults to 768x768\n";
    std::cerr << "         --help | -h                 Print this usage message\n";
    exit( 0 );
}

void initLaunchParams( RayTracedState& state )
{
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>( &state.params.accum_buffer ),
        state.params.width*state.params.height*sizeof(float4)
    ) );

    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&state.params.accum_buffer_motion),
        state.params.height_impulse * state.params.width_impulse * sizeof(float3)
    ));

   /* CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&state.params.impulse_buffer),
       10 * 10 * sizeof(float4)
    )); 

    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&state.params.sphere),
        sizeof(float3) * 3
    ));*/
  //  state.hitgroup_records = (HitGroupRecord*)malloc(OBJ_COUNT * 2);
   // state.hitgroup_records_impulse = (HitGroupRecord*)malloc(OBJ_COUNT * 2);

    state.params.impulse_buffer = nullptr;
    state.params.frame_buffer = nullptr; // Will be set when output buffer is mapped
    state.params.frame_buffer_motion = nullptr; // Will be set when output buffer is mapped
    state.params.subframe_index = 0u;
    state.params.light = g_light;
    state.params.ambient_light_color = make_float3( 0.4f, 0.4f, 0.4f );
    state.params.max_depth = max_trace;
    state.params.scene_epsilon = 1.e-4f;

    CUDA_CHECK( cudaStreamCreate( &state.stream ) );
    CUDA_CHECK(cudaStreamCreate(&state.stream_motion));
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &state.d_params ), sizeof( Params ) ) );

    state.params.handle = state.gas_handle;
}

inline OptixAabb sphere_bound( float3 center, float radius )
{
    float3 m_min = center - radius;
    float3 m_max = center + radius;

    return {
        m_min.x, m_min.y, m_min.z,
        m_max.x, m_max.y, m_max.z
    };
}

inline OptixAabb parallelogram_bound( float3 v1, float3 v2, float3 anchor )
{
    // v1 and v2 are scaled by 1./length^2.  Rescale back to normal for the bounds computation.
    const float3 tv1  = v1 / dot( v1, v1 );
    const float3 tv2  = v2 / dot( v2, v2 );
    const float3 p00  = anchor;
    const float3 p01  = anchor + tv1;
    const float3 p10  = anchor + tv2;
    const float3 p11  = anchor + tv1 + tv2;

    float3 m_min = fminf( fminf( p00, p01 ), fminf( p10, p11 ));
    float3 m_max = fmaxf( fmaxf( p00, p01 ), fmaxf( p10, p11 ));
    return {
        m_min.x, m_min.y, m_min.z,
        m_max.x, m_max.y, m_max.z
    };
}

static void buildGas(
    RayTracedState& state,
    OptixAccelBuildOptions& accel_options,
    OptixBuildInput& build_input, 
    OptixTraversableHandle& gas_handle,
    CUdeviceptr& d_gas_output_buffer
)
{
    OptixAccelBufferSizes gas_buffer_sizes;

    OPTIX_CHECK(optixAccelComputeMemoryUsage(
        state.context,
        &accel_options,
        &build_input,
        1,
        &gas_buffer_sizes));


    state.temp_buffer_size = gas_buffer_sizes.tempSizeInBytes;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.d_temp_buffer), gas_buffer_sizes.tempSizeInBytes));

    // non-compacted output and size of compacted GAS
    CUdeviceptr d_buffer_temp_output_gas_and_compacted_size;
    size_t compactedSizeOffset = roundUp<size_t>(gas_buffer_sizes.outputSizeInBytes, 8ull);
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&d_buffer_temp_output_gas_and_compacted_size),
        compactedSizeOffset + 8
    ));

    OptixAccelEmitDesc emitProperty = {};
    emitProperty.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emitProperty.result = (CUdeviceptr)((char*)d_buffer_temp_output_gas_and_compacted_size + compactedSizeOffset);

    OPTIX_CHECK(optixAccelBuild(
        state.context,
        state.stream,
        &accel_options,
        &build_input,
        1,
        state.d_temp_buffer,
        gas_buffer_sizes.tempSizeInBytes,
        d_buffer_temp_output_gas_and_compacted_size,
        gas_buffer_sizes.outputSizeInBytes,
        &gas_handle,
        &emitProperty,
        1));

    size_t compacted_gas_size;
    CUDA_CHECK(cudaMemcpy(&compacted_gas_size, (void*)emitProperty.result, sizeof(size_t), cudaMemcpyDeviceToHost));

    if (compacted_gas_size < gas_buffer_sizes.outputSizeInBytes)
    {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_gas_output_buffer), compacted_gas_size));

        // use handle as input and output
        OPTIX_CHECK(optixAccelCompact(state.context, 0, gas_handle, d_gas_output_buffer, compacted_gas_size, &gas_handle));


        CUDA_CHECK(cudaFree((void*)d_buffer_temp_output_gas_and_compacted_size));
        state.output_buffer_size = compacted_gas_size;
    }
    else
    {
        d_gas_output_buffer = d_buffer_temp_output_gas_and_compacted_size;
        state.output_buffer_size = gas_buffer_sizes.outputSizeInBytes;
    }

    // allocate enough temporary update space for updating the deforming GAS, exploding GAS and IAS.
    size_t maxUpdateTempSize = std::max(gas_buffer_sizes.tempUpdateSizeInBytes, gas_buffer_sizes.tempUpdateSizeInBytes);
    if (state.temp_buffer_size < maxUpdateTempSize)
    {
        CUDA_CHECK(cudaFree((void*)state.d_temp_buffer));
        state.temp_buffer_size = maxUpdateTempSize;
        CUDA_CHECK(cudaMalloc((void**)&state.d_temp_buffer, state.temp_buffer_size));
    }
    state.params.handle = state.gas_handle;
}

void createGeometry(RayTracedState& state)
{
    //
    // Build Custom Primitives
    //  

    state.aabb = new OptixAabb[OBJ_COUNT];
    state.aabb_input_flags = new uint32_t[OBJ_COUNT];
    uint32_t sbt_index[OBJ_COUNT];
    for (uint32_t i = 0; i < OBJ_COUNT; i++) {
        state.aabb[i] = sphere_bound(sphereList[i]->center, sphereList[i]->radius);
        state.aabb_input_flags[i] = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;
           sbt_index[i] = i;
    }   


    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.d_aabb
        ), OBJ_COUNT * sizeof(OptixAabb)));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(state.d_aabb),
        state.aabb,
        OBJ_COUNT * sizeof(OptixAabb),
        cudaMemcpyHostToDevice
    ));

    CUdeviceptr    d_sbt_index;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_sbt_index), sizeof(sbt_index)));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(d_sbt_index),
        &sbt_index,
        sizeof(sbt_index),
        cudaMemcpyHostToDevice));

    state.aabb_input.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
    state.aabb_input.customPrimitiveArray.aabbBuffers = &state.d_aabb;
    state.aabb_input.customPrimitiveArray.flags = state.aabb_input_flags;
    state.aabb_input.customPrimitiveArray.numSbtRecords = OBJ_COUNT;
    state.aabb_input.customPrimitiveArray.numPrimitives = OBJ_COUNT;
    state.aabb_input.customPrimitiveArray.sbtIndexOffsetBuffer = d_sbt_index;
    state.aabb_input.customPrimitiveArray.sbtIndexOffsetSizeInBytes = sizeof(uint32_t);


    OptixAccelBuildOptions accel_options = {
          OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_ALLOW_UPDATE,  // buildFlags
        OPTIX_BUILD_OPERATION_BUILD,        // operation
    };

    buildGas(
        state,
        accel_options,
        state.aabb_input,
        state.gas_handle,
        state.output_buffer);

    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_sbt_index)));
}

void buildGasUpdate(RayTracedState& state)
{
  //  uint32_t* sbt_index = (uint32_t*)malloc(sizeof(uint32_t) * OBJ_COUNT);
    for (int i = 0; i < OBJ_COUNT; i++) {
        state.aabb[i] = sphere_bound(sphereList[i]->center, sphereList[i]->radius);
        state.aabb_input_flags[i] = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;
      //  sbt_index[i] = i;
    }

    /* for (size_t i = 0; i < 10; i++)
     {
         float3 center = { rand() * 2.0 + 0,rand() * 2.0 + 0,rand() * 2.0 + 0 };
         aabb[i] = sphere_bound(center, state.g_sphere.radius);
         state.aabb[i] = GeometryData::Sphere{ center, state.g_sphere.radius };
     }*/


    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(state.d_aabb),
        state.aabb,
        OBJ_COUNT * sizeof(OptixAabb),
        cudaMemcpyHostToDevice
    ));

    state.aabb_input.customPrimitiveArray.aabbBuffers = &state.d_aabb;
    state.aabb_input.customPrimitiveArray.flags = state.aabb_input_flags;

    OptixAccelBuildOptions accel_options = {
       OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_ALLOW_UPDATE,  // buildFlags
        OPTIX_BUILD_OPERATION_UPDATE,       // operation
    };

    OPTIX_CHECK(optixAccelBuild(
        state.context,
        state.stream,
        &accel_options,
        &state.aabb_input,
        1,
        state.d_temp_buffer,
        state.temp_buffer_size,
        state.output_buffer,
        state.output_buffer_size,
        &state.gas_handle,
        nullptr,
        0));
}

void createModules( RayTracedState &state )
{
    OptixModuleCompileOptions module_compile_options = {};
#if !defined( NDEBUG )
    module_compile_options.optLevel   = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
    module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
#endif

    char log[2048];
    size_t sizeof_log = sizeof(log);
    {
        size_t      inputSize = 0;
        const char* input     = sutil::getInputData( OPTIX_SAMPLE_NAME, OPTIX_SAMPLE_DIR, "geometry.cu", inputSize );
        OPTIX_CHECK_LOG( optixModuleCreateFromPTX(
            state.context,
            &module_compile_options,
            &state.pipeline_compile_options,
            input,
            inputSize,
            log,
            &sizeof_log,
            &state.geometry_module ) );
    }

    {
        size_t      inputSize = 0;
        const char* input     = sutil::getInputData( OPTIX_SAMPLE_NAME, OPTIX_SAMPLE_DIR, "camera.cu", inputSize );
        OPTIX_CHECK_LOG( optixModuleCreateFromPTX(
            state.context,
            &module_compile_options,
            &state.pipeline_compile_options,
            input,
            inputSize,
            log,
            &sizeof_log,
            &state.camera_module ) );
    }

    {
        size_t      inputSize = 0;
        const char* input     = sutil::getInputData( OPTIX_SAMPLE_NAME, OPTIX_SAMPLE_DIR, "shading.cu", inputSize );
        OPTIX_CHECK_LOG( optixModuleCreateFromPTX(
            state.context,
            &module_compile_options,
            &state.pipeline_compile_options,
            input,
            inputSize,
            log,
            &sizeof_log,
            &state.shading_module ) );
    }

    {
        size_t      inputSize = 0;
        const char* input     = sutil::getInputData( nullptr, nullptr, "sphere.cu", inputSize );
        OPTIX_CHECK_LOG( optixModuleCreateFromPTX(
            state.context,
            &module_compile_options,
            &state.pipeline_compile_options,
            input,
            inputSize,
            log,
            &sizeof_log,
            &state.sphere_module ) );
    }
}

void createMotionModules(RayTracedState& state)
{
    OptixModuleCompileOptions module_compile_options = {};
#if !defined( NDEBUG )
    module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
    module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
#endif

    char log[2048];
    size_t sizeof_log = sizeof(log);
    {
        size_t      inputSize = 0;
        const char* input = sutil::getInputData(OPTIX_SAMPLE_NAME, OPTIX_SAMPLE_DIR, "camera.cu", inputSize);
        OPTIX_CHECK_LOG(optixModuleCreateFromPTX(
            state.context_implse,
            &module_compile_options,
            &state.pipeline_motion_compile_options,
            input,
            inputSize,
            log,
            &sizeof_log,
            &state.motion_module));
    }

    {
        size_t      inputSize = 0;
        const char* input = sutil::getInputData(OPTIX_SAMPLE_NAME, OPTIX_SAMPLE_DIR, "shading.cu", inputSize);
        OPTIX_CHECK_LOG(optixModuleCreateFromPTX(
            state.context_implse,
            &module_compile_options,
            &state.pipeline_motion_compile_options,
            input,
            inputSize,
            log,
            &sizeof_log,
            &state.shading_module));
    }

    {
        size_t      inputSize = 0;
        const char* input = sutil::getInputData(OPTIX_SAMPLE_NAME, OPTIX_SAMPLE_DIR, "geometry.cu", inputSize);
        OPTIX_CHECK_LOG(optixModuleCreateFromPTX(
            state.context_implse,
            &module_compile_options,
            &state.pipeline_motion_compile_options,
            input,
            inputSize,
            log,
            &sizeof_log,
            &state.geometry_module));
    }

    {
        size_t      inputSize = 0;
        const char* input = sutil::getInputData(nullptr, nullptr, "sphere.cu", inputSize);
        OPTIX_CHECK_LOG(optixModuleCreateFromPTX(
            state.context_implse,
            &module_compile_options,
            &state.pipeline_motion_compile_options,
            input,
            inputSize,
            log,
            &sizeof_log,
            &state.sphere_module_impulse));
    }
}

static void createCameraProgram( RayTracedState &state, std::vector<OptixProgramGroup> &program_groups )
{
    OptixProgramGroup           cam_prog_group;
    OptixProgramGroupOptions    cam_prog_group_options = {};
    OptixProgramGroupDesc       cam_prog_group_desc = {};
    cam_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    cam_prog_group_desc.raygen.module = state.camera_module;
    cam_prog_group_desc.raygen.entryFunctionName = "__raygen__pinhole_camera";

    char    log[2048];
    size_t  sizeof_log = sizeof( log );
    OPTIX_CHECK_LOG( optixProgramGroupCreate(
        state.context,
        &cam_prog_group_desc,
        1,
        &cam_prog_group_options,
        log,
        &sizeof_log,
        &cam_prog_group ) );

    program_groups.push_back(cam_prog_group);
    state.raygen_prog_group = cam_prog_group;
} 

static void createMetalSphereProgram( RayTracedState &state, std::vector<OptixProgramGroup> &program_groups )
{
    char    log[2048];
    size_t  sizeof_log = sizeof(log);

    OptixProgramGroup           radiance_sphere_prog_group;
    OptixProgramGroupOptions    radiance_sphere_prog_group_options = {};
    OptixProgramGroupDesc       radiance_sphere_prog_group_desc = {};
    radiance_sphere_prog_group_desc.kind   = OPTIX_PROGRAM_GROUP_KIND_HITGROUP,
        radiance_sphere_prog_group_desc.hitgroup.moduleIS           = state.sphere_module;
    radiance_sphere_prog_group_desc.hitgroup.entryFunctionNameIS    = "__intersection__sphere";
    radiance_sphere_prog_group_desc.hitgroup.moduleCH               = state.shading_module;
    radiance_sphere_prog_group_desc.hitgroup.entryFunctionNameCH    = "__closesthit__metal_radiance";
    radiance_sphere_prog_group_desc.hitgroup.moduleAH               = nullptr;
    radiance_sphere_prog_group_desc.hitgroup.entryFunctionNameAH    = nullptr;

    OPTIX_CHECK_LOG( optixProgramGroupCreate(
        state.context,
        &radiance_sphere_prog_group_desc,
        1,
        &radiance_sphere_prog_group_options,
        log,
        &sizeof_log,
        &radiance_sphere_prog_group ) );

    program_groups.push_back(radiance_sphere_prog_group);
    state.radiance_metal_sphere_prog_group = radiance_sphere_prog_group;

    OptixProgramGroup           occlusion_sphere_prog_group;
    OptixProgramGroupOptions    occlusion_sphere_prog_group_options = {};
    OptixProgramGroupDesc       occlusion_sphere_prog_group_desc = {};
    occlusion_sphere_prog_group_desc.kind   = OPTIX_PROGRAM_GROUP_KIND_HITGROUP,
        occlusion_sphere_prog_group_desc.hitgroup.moduleIS           = state.sphere_module;
    occlusion_sphere_prog_group_desc.hitgroup.entryFunctionNameIS    = "__intersection__sphere";
    occlusion_sphere_prog_group_desc.hitgroup.moduleCH               = state.shading_module;
    occlusion_sphere_prog_group_desc.hitgroup.entryFunctionNameCH    = "__closesthit__full_occlusion";
    occlusion_sphere_prog_group_desc.hitgroup.moduleAH               = nullptr;
    occlusion_sphere_prog_group_desc.hitgroup.entryFunctionNameAH    = nullptr;

    OPTIX_CHECK_LOG( optixProgramGroupCreate(
        state.context,
        &occlusion_sphere_prog_group_desc,
        1,
        &occlusion_sphere_prog_group_options,
        log,
        &sizeof_log,
        &occlusion_sphere_prog_group ) );

    program_groups.push_back(occlusion_sphere_prog_group);
    state.occlusion_metal_sphere_prog_group = occlusion_sphere_prog_group;
}

static void createFloorProgram( RayTracedState &state, std::vector<OptixProgramGroup> &program_groups )
{
    OptixProgramGroup           radiance_floor_prog_group;
    OptixProgramGroupOptions    radiance_floor_prog_group_options = {};
    OptixProgramGroupDesc       radiance_floor_prog_group_desc = {};
    radiance_floor_prog_group_desc.kind   = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    radiance_floor_prog_group_desc.hitgroup.moduleIS               = state.geometry_module;
    radiance_floor_prog_group_desc.hitgroup.entryFunctionNameIS    = "__intersection__parallelogram";
    radiance_floor_prog_group_desc.hitgroup.moduleCH               = state.shading_module;
    radiance_floor_prog_group_desc.hitgroup.entryFunctionNameCH    = "__closesthit__checker_radiance";
    radiance_floor_prog_group_desc.hitgroup.moduleAH               = nullptr;
    radiance_floor_prog_group_desc.hitgroup.entryFunctionNameAH    = nullptr;

    char    log[2048];
    size_t  sizeof_log = sizeof( log );
    OPTIX_CHECK_LOG( optixProgramGroupCreate(
        state.context,
        &radiance_floor_prog_group_desc,
        1,
        &radiance_floor_prog_group_options,
        log,
        &sizeof_log,
        &radiance_floor_prog_group ) );

    program_groups.push_back(radiance_floor_prog_group);
    state.radiance_floor_prog_group = radiance_floor_prog_group;

    OptixProgramGroup           occlusion_floor_prog_group;
    OptixProgramGroupOptions    occlusion_floor_prog_group_options = {};
    OptixProgramGroupDesc       occlusion_floor_prog_group_desc = {};
    occlusion_floor_prog_group_desc.kind   = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    occlusion_floor_prog_group_desc.hitgroup.moduleIS               = state.geometry_module;
    occlusion_floor_prog_group_desc.hitgroup.entryFunctionNameIS    = "__intersection__parallelogram";
    occlusion_floor_prog_group_desc.hitgroup.moduleCH               = state.shading_module;
    occlusion_floor_prog_group_desc.hitgroup.entryFunctionNameCH    = "__closesthit__full_occlusion";
    occlusion_floor_prog_group_desc.hitgroup.moduleAH               = nullptr;
    occlusion_floor_prog_group_desc.hitgroup.entryFunctionNameAH    = nullptr;

    OPTIX_CHECK_LOG( optixProgramGroupCreate(
        state.context,
        &occlusion_floor_prog_group_desc,
        1,
        &occlusion_floor_prog_group_options,
        log,
        &sizeof_log,
        &occlusion_floor_prog_group ) );

    program_groups.push_back(occlusion_floor_prog_group);
    state.occlusion_floor_prog_group = occlusion_floor_prog_group;
}

static void createMissProgram( RayTracedState &state, std::vector<OptixProgramGroup> &program_groups )
{
    OptixProgramGroupOptions    miss_prog_group_options = {};
    OptixProgramGroupDesc       miss_prog_group_desc = {};
    miss_prog_group_desc.kind   = OPTIX_PROGRAM_GROUP_KIND_MISS;
    miss_prog_group_desc.miss.module             = state.shading_module;
    miss_prog_group_desc.miss.entryFunctionName  = "__miss__constant_bg";

    char    log[2048];
    size_t  sizeof_log = sizeof( log );
    OPTIX_CHECK_LOG( optixProgramGroupCreate(
        state.context,
        &miss_prog_group_desc,
        1,
        &miss_prog_group_options,
        log,
        &sizeof_log,
        &state.radiance_miss_prog_group ) );

    program_groups.push_back(state.radiance_miss_prog_group);

    miss_prog_group_desc.miss = {
        nullptr,    // module
        nullptr     // entryFunctionName
    };
    OPTIX_CHECK_LOG( optixProgramGroupCreate(
        state.context,
        &miss_prog_group_desc,
        1,
        &miss_prog_group_options,
        log,
        &sizeof_log,
        &state.occlusion_miss_prog_group ) );

    program_groups.push_back(state.occlusion_miss_prog_group);
}

static void createMotionMissProgram(RayTracedState& state, std::vector<OptixProgramGroup>& program_groups)
{
    OptixProgramGroupOptions    miss_prog_group_options = {};
    OptixProgramGroupDesc       miss_prog_group_desc = {};
    miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    miss_prog_group_desc.miss.module = state.shading_module;
    miss_prog_group_desc.miss.entryFunctionName = "__miss__constant_bg_impulse";

    char    log[2048];
    size_t  sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG(optixProgramGroupCreate(
        state.context_implse,
        &miss_prog_group_desc,
        1,
        &miss_prog_group_options,
        log,
        &sizeof_log,
        &state.radiance_motion_miss_prog_group));

    program_groups.push_back(state.radiance_motion_miss_prog_group);
}

static void createMotionRayGenProgram(RayTracedState& state, std::vector<OptixProgramGroup>& program_groups)
{
    OptixProgramGroup           cam_prog_group;
    OptixProgramGroupOptions    cam_prog_group_options = {};
    OptixProgramGroupDesc       cam_prog_group_desc = {};
    cam_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    cam_prog_group_desc.raygen.module = state.motion_module;
    cam_prog_group_desc.raygen.entryFunctionName = "__raygen__sphere_motion";

    char    log[2048];
    size_t  sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG(optixProgramGroupCreate(
        state.context_implse,
        &cam_prog_group_desc,
        1,
        &cam_prog_group_options,
        log,
        &sizeof_log,
        &cam_prog_group));

    program_groups.push_back(cam_prog_group);
    state.raygen_motion_prog_group = cam_prog_group;
}

static void createMotionSphereProgram(RayTracedState& state, std::vector<OptixProgramGroup>& program_groups)
{
    char    log[2048];
    size_t  sizeof_log = sizeof(log);

    OptixProgramGroup       raygen_prog_group = {};
    OptixProgramGroupOptions    raygen_prog_group_options = {};
    OptixProgramGroupDesc       raygen_prog_group_desc = {};
    raygen_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    raygen_prog_group_desc.hitgroup.moduleIS = state.sphere_module_impulse;
    raygen_prog_group_desc.hitgroup.entryFunctionNameIS = "__intersection__sphere";
    raygen_prog_group_desc.hitgroup.moduleCH = state.shading_module;
    raygen_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__sphere_collision";
    raygen_prog_group_desc.hitgroup.moduleAH = nullptr;
    raygen_prog_group_desc.hitgroup.entryFunctionNameAH = nullptr;
    OPTIX_CHECK_LOG(optixProgramGroupCreate(
        state.context_implse,
        &raygen_prog_group_desc,
        1,
        &raygen_prog_group_options,
        log,
        &sizeof_log,
        &raygen_prog_group
    ));

    program_groups.push_back(raygen_prog_group);
    state.closeshit_motion_prog_group= raygen_prog_group;

    OptixProgramGroup       occlusion_prog_group = {};
    OptixProgramGroupOptions    occlusion_prog_group_options = {};
    OptixProgramGroupDesc       occlusion_sphere_prog_group_desc = {};
    occlusion_sphere_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    occlusion_sphere_prog_group_desc.hitgroup.moduleIS = state.sphere_module_impulse;
    occlusion_sphere_prog_group_desc.hitgroup.entryFunctionNameIS = "__intersection__sphere";
    occlusion_sphere_prog_group_desc.hitgroup.moduleCH = state.shading_module;
    occlusion_sphere_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__sphere_collision";
    occlusion_sphere_prog_group_desc.hitgroup.moduleAH = nullptr;
    occlusion_sphere_prog_group_desc.hitgroup.entryFunctionNameAH = nullptr;
    OPTIX_CHECK_LOG(optixProgramGroupCreate(
        state.context_implse,
        &occlusion_sphere_prog_group_desc,
        1,
        &occlusion_prog_group_options,
        log,
        &sizeof_log,
        &occlusion_prog_group
    ));

    program_groups.push_back(occlusion_prog_group);
    state.intersect_motion_prog_group = occlusion_prog_group;
}

static void createMotionFloorProgram(RayTracedState& state, std::vector<OptixProgramGroup>& program_groups)
{
    OptixProgramGroup           radiance_floor_prog_group;
    OptixProgramGroupOptions    radiance_floor_prog_group_options = {};
    OptixProgramGroupDesc       radiance_floor_prog_group_desc = {};
    radiance_floor_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    radiance_floor_prog_group_desc.hitgroup.moduleIS = state.geometry_module;
    radiance_floor_prog_group_desc.hitgroup.entryFunctionNameIS = "__intersection__floor_motion";
    radiance_floor_prog_group_desc.hitgroup.moduleCH = state.shading_module;
    radiance_floor_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__floor_collision";
    radiance_floor_prog_group_desc.hitgroup.moduleAH = nullptr;
    radiance_floor_prog_group_desc.hitgroup.entryFunctionNameAH = nullptr;

    char    log[2048];
    size_t  sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG(optixProgramGroupCreate(
        state.context_implse,
        &radiance_floor_prog_group_desc,
        1,
        &radiance_floor_prog_group_options,
        log,
        &sizeof_log,
        &radiance_floor_prog_group));

    program_groups.push_back(radiance_floor_prog_group);
    state.closeshit_floor_motion_prog_group = radiance_floor_prog_group;

    OptixProgramGroup           occlusion_floor_prog_group;
    OptixProgramGroupOptions    occlusion_floor_prog_group_options = {};
    OptixProgramGroupDesc       occlusion_floor_prog_group_desc = {};
    occlusion_floor_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    occlusion_floor_prog_group_desc.hitgroup.moduleIS = state.geometry_module;
    occlusion_floor_prog_group_desc.hitgroup.entryFunctionNameIS = "__intersection__floor_motion";
    occlusion_floor_prog_group_desc.hitgroup.moduleAH = nullptr;
    occlusion_floor_prog_group_desc.hitgroup.entryFunctionNameAH = nullptr;

    OPTIX_CHECK_LOG(optixProgramGroupCreate(
        state.context_implse,
        &occlusion_floor_prog_group_desc,
        1,
        &occlusion_floor_prog_group_options,
        log,
        &sizeof_log,
        &occlusion_floor_prog_group));

    program_groups.push_back(occlusion_floor_prog_group);
    state.intersect_floor_motion_prog_group = occlusion_floor_prog_group;
}

static void context_log_cb( unsigned int level, const char* tag, const char* message, void* /*cbdata */)
{
    std::cerr << "[" << std::setw( 2 ) << level << "][" << std::setw( 12 ) << tag << "]: "
              << message << "\n";
}

void createPipeline( RayTracedState &state )
{
    std::vector<OptixProgramGroup> program_groups;

    state.pipeline_compile_options = {
        false,                                                  // usesMotionBlur
        OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS,          // traversableGraphFlags
        7,    /* RadiancePRD uses 5 payloads */                 // numPayloadValues
        5,    /* Parallelogram intersection uses 5 attrs */     // numAttributeValues
        OPTIX_EXCEPTION_FLAG_NONE,                              // exceptionFlags
        "params"                                                // pipelineLaunchParamsVariableName
    };

    // Prepare program groups
    createModules( state );
    createCameraProgram( state, program_groups );
    createMetalSphereProgram( state, program_groups );
    createMissProgram( state, program_groups );

    // Link program groups to pipeline
    OptixPipelineLinkOptions pipeline_link_options = {
        max_trace,                          // maxTraceDepth
        OPTIX_COMPILE_DEBUG_LEVEL_FULL      // debugLevel
    };
    char    log[2048];
    size_t  sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG( optixPipelineCreate(
        state.context,
        &state.pipeline_compile_options,
        &pipeline_link_options,
        program_groups.data(),
        static_cast<unsigned int>( program_groups.size() ),
        log,
        &sizeof_log,
        &state.pipeline ) );

    OptixStackSizes stack_sizes = {};
    for( auto& prog_group : program_groups )
    {
        OPTIX_CHECK( optixUtilAccumulateStackSizes( prog_group, &stack_sizes ) );
    }

    uint32_t direct_callable_stack_size_from_traversal;
    uint32_t direct_callable_stack_size_from_state;
    uint32_t continuation_stack_size;
    OPTIX_CHECK( optixUtilComputeStackSizes( &stack_sizes, max_trace,
                                             0,  // maxCCDepth
                                             0,  // maxDCDepth
                                             &direct_callable_stack_size_from_traversal,
                                             &direct_callable_stack_size_from_state, &continuation_stack_size ) );
    OPTIX_CHECK( optixPipelineSetStackSize( state.pipeline, direct_callable_stack_size_from_traversal,
                                            direct_callable_stack_size_from_state, continuation_stack_size,
                                            1  // maxTraversableDepth
                                            ) );
}

void createMotionPipeline(RayTracedState& state)
{
    std::vector<OptixProgramGroup> program_groups;

    state.pipeline_motion_compile_options = {
        false,                                                  // usesMotionBlur
        OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS,          // traversableGraphFlags
        7,    /* RadiancePRD uses 5 payloads */                 // numPayloadValues
        5,    /* Parallelogram intersection uses 5 attrs */     // numAttributeValues
        OPTIX_EXCEPTION_FLAG_NONE,                              // exceptionFlags
        "params"                                                // pipelineLaunchParamsVariableName
    };

    // Prepare program groups
    createMotionModules(state);
    createMotionRayGenProgram(state, program_groups);
    createMotionSphereProgram(state, program_groups);
   // createMotionFloorProgram(state, program_groups);
    createMotionMissProgram(state, program_groups);

     // Link program groups to pipeline
    OptixPipelineLinkOptions pipeline_link_options = {
        max_trace,                          // maxTraceDepth
        OPTIX_COMPILE_DEBUG_LEVEL_FULL      // debugLevel
    };
    char    log[2048];
    size_t  sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG(optixPipelineCreate(
        state.context_implse,
        &state.pipeline_motion_compile_options,
        &pipeline_link_options,
        program_groups.data(),
        static_cast<unsigned int>(program_groups.size()),
        log,
        &sizeof_log,
        &state.pipeline_motion));

    OptixStackSizes stack_sizes = {};
    for (auto& prog_group : program_groups)
    {
        OPTIX_CHECK(optixUtilAccumulateStackSizes(prog_group, &stack_sizes));
    }

    uint32_t direct_callable_stack_size_from_traversal;
    uint32_t direct_callable_stack_size_from_state;
    uint32_t continuation_stack_size;
    OPTIX_CHECK(optixUtilComputeStackSizes(&stack_sizes, max_trace,
        0,  // maxCCDepth
        0,  // maxDCDepth
        &direct_callable_stack_size_from_traversal,
        &direct_callable_stack_size_from_state, &continuation_stack_size));
    OPTIX_CHECK(optixPipelineSetStackSize(state.pipeline_motion, direct_callable_stack_size_from_traversal,
        direct_callable_stack_size_from_state, continuation_stack_size,
        1  // maxTraversableDepth
    ));
}

void syncCameraDataToSbt( RayTracedState &state, const CameraData& camData )
{
    {
        optixSbtRecordPackHeader(state.raygen_prog_group, &state.rg_sbt_cam[0]);
        state.rg_sbt_cam[0].data = camData;

        CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void*>(state.sbt.raygenRecord),
            state.rg_sbt_cam,
            sizeof(RayGenRecord),
            cudaMemcpyHostToDevice
        ));
    }
}

void createSBT( RayTracedState &state )
{
    // Raygen program record
    {  
        state.camData = (CameraData*)malloc(sizeof(CameraData));
        state.rg_sbt_cam = (RayGenRecord*)malloc(sizeof(RayGenRecord));
        CUdeviceptr d_raygen_record;
        size_t sizeof_raygen_record = sizeof(RayGenRecord);
        CUDA_CHECK( cudaMalloc(
            reinterpret_cast<void**>( &d_raygen_record ),
            sizeof_raygen_record ) );
        /*
        RayGenRecord rg_sbt;

        camera.setAspectRatio(static_cast<float>(state.params.width) / static_cast<float>(state.params.height));
        CameraData* camData = (CameraData*)malloc(sizeof(CameraData));
        camData->eye = camera.eye();
        camera.UVWFrame(camData->U, camData->V, camData->W);

        optixSbtRecordPackHeader(state.raygen_prog_group, &rg_sbt);
        rg_sbt.data = *camData;

        CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void*>(&d_raygen_record),
            &rg_sbt,
            sizeof(RayGenRecord),
            cudaMemcpyHostToDevice
        ));
        */
        state.sbt.raygenRecord = d_raygen_record;
    }

    // Miss program record
    {
        CUdeviceptr d_miss_record;
        size_t sizeof_miss_record = sizeof( MissRecord );
        CUDA_CHECK( cudaMalloc(
            reinterpret_cast<void**>( &d_miss_record ),
            sizeof_miss_record*RAY_TYPE_COUNT ) );

        MissRecord ms_sbt[RAY_TYPE_COUNT];
        optixSbtRecordPackHeader( state.radiance_miss_prog_group, &ms_sbt[0] );
        optixSbtRecordPackHeader( state.occlusion_miss_prog_group, &ms_sbt[1] );
        ms_sbt[1].data = ms_sbt[0].data = { 0.34f, 0.55f, 0.85f };

        CUDA_CHECK( cudaMemcpy(
            reinterpret_cast<void*>( d_miss_record ),
            ms_sbt,
            sizeof_miss_record*RAY_TYPE_COUNT,
            cudaMemcpyHostToDevice
        ) );

        state.sbt.missRecordBase          = d_miss_record;
        state.sbt.missRecordCount         = RAY_TYPE_COUNT;
        state.sbt.missRecordStrideInBytes = static_cast<uint32_t>( sizeof_miss_record );
    }

    // Hitgroup program record
    {
        const int count_records = 2 * OBJ_COUNT;
        state.hitgroup_records = (HitGroupRecord*)malloc( sizeof(HitGroupRecord) * count_records);

        // Note: Fill SBT record array the same order like AS is built.
        int sbt_idx = 0;

        for (int i = 0; i < OBJ_COUNT; i++) {
            OPTIX_CHECK(optixSbtRecordPackHeader(
                state.radiance_metal_sphere_prog_group,
                &state.hitgroup_records[sbt_idx]));
            state.hitgroup_records[sbt_idx].data.geometry.sphere = *sphereList[i];
            state.hitgroup_records[sbt_idx].data.shading.metal = {
                { 0.2f, 0.5f, 0.5f },   // Ka
                { 0.2f, 0.7f, 0.8f },   // Kd
                { 0.9f, 0.9f, 0.9f },   // Ks
                { 0.5f, 0.5f, 0.5f },   // Kr
                64,                     // phong_exp
            };
            state.hitgroup_records[sbt_idx].data.intersectIndex.index = i;
            sbt_idx++;

            OPTIX_CHECK(optixSbtRecordPackHeader(
                state.occlusion_metal_sphere_prog_group,
                &state.hitgroup_records[sbt_idx]));
            state.hitgroup_records[sbt_idx].data.geometry.sphere = *sphereList[i];
            sbt_idx++;
        }

        CUdeviceptr d_hitgroup_records;
        size_t      sizeof_hitgroup_record = sizeof( HitGroupRecord );
        CUDA_CHECK( cudaMalloc(
            reinterpret_cast<void**>( &d_hitgroup_records ),
            sizeof_hitgroup_record * count_records
        ) );

        CUDA_CHECK( cudaMemcpy(
            reinterpret_cast<void*>( d_hitgroup_records ),
            state.hitgroup_records,
            sizeof_hitgroup_record * count_records,
            cudaMemcpyHostToDevice
        ) );

        state.sbt.hitgroupRecordBase            = d_hitgroup_records;
        state.sbt.hitgroupRecordCount           = count_records;
        state.sbt.hitgroupRecordStrideInBytes   = static_cast<uint32_t>( sizeof_hitgroup_record );
     //   delete[] hitgroup_records;
     //   free(hitgroup_records);
    }
}

void updateSTB(RayTracedState& state)
{
    // Hitgroup program record
    {
        const size_t count_records = 2 * OBJ_COUNT;

        // Note: Fill SBT record array the same order like AS is built.
        int sbt_idx = 0;

        for (int i = 0; i < OBJ_COUNT; i++) {
            OPTIX_CHECK(optixSbtRecordPackHeader(
                state.radiance_metal_sphere_prog_group,
                &state.hitgroup_records[sbt_idx]));
            state.hitgroup_records[sbt_idx].data.geometry.sphere = *sphereList[i];
            state.hitgroup_records[sbt_idx].data.shading.metal = {
                { 0.2f, 0.5f, 0.5f },   // Ka
                { 0.2f, 0.7f, 0.8f },   // Kd
                { 0.9f, 0.9f, 0.9f },   // Ks
                { 0.5f, 0.5f, 0.5f },   // Kr
                64,                     // phong_exp
            };
            state.hitgroup_records[sbt_idx].data.intersectIndex.index = i;
            sbt_idx++;

            OPTIX_CHECK(optixSbtRecordPackHeader(
                state.occlusion_metal_sphere_prog_group,
                &state.hitgroup_records[sbt_idx]));
            state.hitgroup_records[sbt_idx].data.geometry.sphere = *sphereList[i];
            sbt_idx++;
        }

        size_t sizeof_hitgroup_record = sizeof(HitGroupRecord);

        CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void*>(state.sbt.hitgroupRecordBase),
            state.hitgroup_records,
            sizeof_hitgroup_record * count_records,
            cudaMemcpyHostToDevice
        ));
    }
}

void createMotionSBT(RayTracedState& state)
{
    // Raygen program record
    {
        CUdeviceptr d_raygen_record;
        size_t sizeof_raygen_record = sizeof(RayGenRecord);
        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void**>(&d_raygen_record),
            sizeof_raygen_record));


        state.rg_sbt = (RayGenRecord*)malloc(sizeof(RayGenRecord));
        for (int i = 0; i < OBJ_COUNT; i++) {
            OPTIX_CHECK(optixSbtRecordPackHeader(
                state.raygen_motion_prog_group,
                &state.rg_sbt[0]));
            state.rg_sbt->data.pos[i] = sphereList[i]->center;
        }
        

        CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void*>(d_raygen_record),
            state.rg_sbt,
            sizeof(RayGenRecord),
            cudaMemcpyHostToDevice
        ));

        state.sbt_motion.raygenRecord = d_raygen_record;
    }

    // Hitgroup program record
    {
        const size_t count_records = 2 * OBJ_COUNT;
        state.hitgroup_records_impulse = (HitGroupRecord*)malloc(sizeof(HitGroupRecord) * count_records);

        int sbt_idx = 0;

        for (int i = 0; i < OBJ_COUNT; i++) {
            OPTIX_CHECK(optixSbtRecordPackHeader(
                state.closeshit_motion_prog_group,
                &state.hitgroup_records_impulse[sbt_idx]));
            state.hitgroup_records_impulse[sbt_idx].data.sphereInd.index = i;
            state.hitgroup_records_impulse[sbt_idx].data.geometry.sphere = *sphereList[i];
            /*     for (int j = 0; j < OBJ_COUNT; j++) {
                  state.hitgroup_records_impulse[sbt_idx].data.intersects.sphereList[j] = *sphereList[j];
              } */
            sbt_idx++;
            OPTIX_CHECK(optixSbtRecordPackHeader(
                state.intersect_motion_prog_group,
                &state.hitgroup_records_impulse[sbt_idx]));
            state.hitgroup_records_impulse[sbt_idx].data.geometry.sphere = *sphereList[i];
            sbt_idx++;
        }

        CUdeviceptr d_hitgroup_records;
        size_t      sizeof_hitgroup_record = sizeof(HitGroupRecord);
        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void**>(&d_hitgroup_records),
            sizeof_hitgroup_record * count_records
        ));

        CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void*>(d_hitgroup_records),
            state.hitgroup_records_impulse,
            sizeof_hitgroup_record * count_records,
            cudaMemcpyHostToDevice
        ));

        state.sbt_motion.hitgroupRecordBase = d_hitgroup_records;
        state.sbt_motion.hitgroupRecordCount = count_records;
        state.sbt_motion.hitgroupRecordStrideInBytes = static_cast<uint32_t>(sizeof_hitgroup_record);
    }

    // Miss program record
    {
        CUdeviceptr d_miss_record;
        size_t sizeof_miss_record = sizeof(MissRecord);
        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void**>(&d_miss_record),
            sizeof_miss_record));

        MissRecord ms_sbt[1];
        optixSbtRecordPackHeader(state.radiance_motion_miss_prog_group, &ms_sbt[0]);
        ms_sbt[0].data = { 0.0f, 0.0f, 0.0f };

        CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void*>(d_miss_record),
            ms_sbt,
            sizeof_miss_record,
            cudaMemcpyHostToDevice
        ));

        state.sbt_motion.missRecordBase = d_miss_record;
        state.sbt_motion.missRecordCount = 1;
        state.sbt_motion.missRecordStrideInBytes = static_cast<uint32_t>(sizeof_miss_record);
    }
}

void updateImpulseSBT(RayTracedState& state)
{
    // Raygen program record
    {

    //    for (int i = 0; i < OBJ_COUNT; i++)
       //     rg_sbt.data.sphereList[i] = *sphereList[i];

        for (int i = 0; i < OBJ_COUNT; i++) {
            OPTIX_CHECK(optixSbtRecordPackHeader(
                state.raygen_motion_prog_group,
                &state.rg_sbt[0]));
            state.rg_sbt->data.pos[i] = sphereList[i]->center;
        }

        CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void*>(state.sbt_motion.raygenRecord),
            state.rg_sbt,
            sizeof(RayGenRecord),
            cudaMemcpyHostToDevice
        ));
    }

    // Hitgroup program record
    {
        const size_t count_records = 2 * OBJ_COUNT;
        int sbt_idx = 0;

        for (int i = 0; i < OBJ_COUNT; i++) {
            OPTIX_CHECK(optixSbtRecordPackHeader(
                state.closeshit_motion_prog_group,
                &state.hitgroup_records_impulse[sbt_idx]));
            state.hitgroup_records_impulse[sbt_idx].data.sphereInd.index = i;
            state.hitgroup_records_impulse[sbt_idx].data.geometry.sphere = *sphereList[i];
       /*     for (int j = 0; j < OBJ_COUNT; j++) {
                state.hitgroup_records_impulse[sbt_idx].data.intersects.sphereList[j] = *sphereList[j];
            } */
            sbt_idx++;
            OPTIX_CHECK(optixSbtRecordPackHeader(
                state.intersect_motion_prog_group,
                &state.hitgroup_records_impulse[sbt_idx]));
            state.hitgroup_records_impulse[sbt_idx].data.geometry.sphere = *sphereList[i];
            sbt_idx++;
        }

        CUdeviceptr d_hitgroup_records;
        size_t      sizeof_hitgroup_record = sizeof(HitGroupRecord);

        CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void*>(state.sbt_motion.hitgroupRecordBase),
            state.hitgroup_records_impulse,
            sizeof_hitgroup_record * count_records,
            cudaMemcpyHostToDevice
        ));
    }
}

void createContext( RayTracedState& state )
{
    // Initialize CUDA
    CUDA_CHECK( cudaFree( 0 ) );

    OptixDeviceContext context;
    CUcontext          cuCtx = 0;  // zero means take the current context
    OPTIX_CHECK( optixInit() );
    OptixDeviceContextOptions options = {};
    options.logCallbackFunction       = &context_log_cb;
    options.logCallbackLevel          = 4;
    OPTIX_CHECK( optixDeviceContextCreate( cuCtx, &options, &context ) );

    state.context = context;
}

void createContextImpulse(RayTracedState& state)
{
    // Initialize CUDA
    CUDA_CHECK(cudaFree(0));

    OptixDeviceContext context;
    CUcontext          cuCtx = 0;  // zero means take the current context
    OPTIX_CHECK(optixInit());
    OptixDeviceContextOptions options = {};
    options.logCallbackFunction = &context_log_cb;
    options.logCallbackLevel = 4;
    OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &options, &context));

    state.context_implse = context;
}

void initCameraState()
{
    camera.setEye( make_float3( -4.0f, 0.0f, -15.0f ) );
    camera.setLookat( make_float3( 4.0f, 2.3f, -4.0f ) );
    camera.setUp( make_float3( 0.0f, 1.0f, 0.0f ) );
    camera.setFovY( 60.0f );
    camera_changed = true;

    trackball.setCamera( &camera );
    trackball.setMoveSpeed( 10.0f );
    trackball.setReferenceFrame( make_float3( 1.0f, 0.0f, 0.0f ), make_float3( 0.0f, 0.0f, 1.0f ), make_float3( 0.0f, 1.0f, 0.0f ) );
    trackball.setGimbalLock(true);
}

void handleCameraUpdate( RayTracedState &state )
{
    if( !camera_changed )
        return;
    //camera_changed = false;

    camera.setAspectRatio( static_cast<float>( state.params.width ) / static_cast<float>( state.params.height ) );
    state.camData->eye = camera.eye();
    camera.UVWFrame(state.camData->U, state.camData->V, state.camData->W );
    syncCameraDataToSbt(state, *state.camData);
}

void handleResize( sutil::CUDAOutputBuffer<uchar4>& output_buffer, Params& params )
{
    if( !resize_dirty )
        return;
    resize_dirty = false;

    output_buffer.resize( params.width, params.height );

    // Realloc accumulation buffer
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( params.accum_buffer ) ) );
    CUDA_CHECK( cudaMalloc(
        reinterpret_cast<void**>( &params.accum_buffer ),
        params.width*params.height*sizeof(float4)
    ) );

    // Realloc accumulation buffer
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(params.accum_buffer_motion)));
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&params.accum_buffer_motion),
        params.width * params.height * sizeof(float4)
    ));
}

void updateState( sutil::CUDAOutputBuffer<uchar4>& output_buffer, RayTracedState &state )
{
    // Update params on device
    if( camera_changed || resize_dirty )
        state.params.subframe_index = 0;

    handleCameraUpdate( state );
    handleResize( output_buffer, state.params );
}

void launchSubframe( sutil::CUDAOutputBuffer<uchar4>& output_buffer, RayTracedState& state )
{

    // Launch
    uchar4* result_buffer_data = output_buffer.map();
    state.params.frame_buffer = result_buffer_data;

    CUDA_CHECK( cudaMemcpyAsync( reinterpret_cast<void*>( state.d_params ),
                                 &state.params,
                                 sizeof( Params ),
                                 cudaMemcpyHostToDevice,
                                 state.stream
    ) );

    OPTIX_CHECK( optixLaunch(
        state.pipeline,
        state.stream,
        reinterpret_cast<CUdeviceptr>( state.d_params ),
        sizeof( Params ),
        &state.sbt,
        state.params.width,  // launch width
        state.params.height, // launch height
        1                    // launch depth
    ) );

    output_buffer.unmap();
    CUDA_SYNC_CHECK();
}

void launchSubframeMotion(sutil::CUDAOutputBuffer< GeometryData::impulseData>& output_buffer, RayTracedState& state)
{
    // Launch
    GeometryData::impulseData* result_buffer_data = output_buffer.map();
    state.params.impulse_buffer = result_buffer_data;
   // state.params.frame_buffer_motion = &result_buffer_data->impulse;

    CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(state.d_params),
        &state.params,
        sizeof(Params),
        cudaMemcpyHostToDevice,
        state.stream_motion
    ));

    OPTIX_CHECK(optixLaunch(
        state.pipeline_motion,
        state.stream_motion,
        reinterpret_cast<CUdeviceptr>(state.d_params),
        sizeof(Params),
        &state.sbt_motion,
        state.params.width_impulse, // ray launch number
        1, //    state.params.height_impulse / OBJ_COUNT, // ray launch number
        OBJ_COUNT // balls number z
    ));
    output_buffer.unmap();
    CUDA_SYNC_CHECK();
}

void displaySubframe(
    sutil::CUDAOutputBuffer<uchar4>&  output_buffer,
    sutil::GLDisplay&                 gl_display,
    GLFWwindow*                       window )
{
    // Display
    int framebuf_res_x = 0;   // The display's resolution (could be HDPI res)
    int framebuf_res_y = 0;   //
    glfwGetFramebufferSize( window, &framebuf_res_x, &framebuf_res_y );
    gl_display.display(
        output_buffer.width(),
        output_buffer.height(),
        framebuf_res_x,
        framebuf_res_y,
        output_buffer.getPBO()
    );
}

void displaySubframeImpulse(
    sutil::CUDAOutputBuffer<GeometryData::impulseData>& output_buffer,
    sutil::GLDisplay& gl_display,
    GLFWwindow* window)
{
    // Display
    int framebuf_res_x = 0;   // The display's resolution (could be HDPI res)
    int framebuf_res_y = 0;   //
    glfwGetFramebufferSize(window, &framebuf_res_x, &framebuf_res_y);
    gl_display.display(
        output_buffer.width(),
        output_buffer.height(),
        framebuf_res_x,
        framebuf_res_y,
        output_buffer.getPBO()
    );
}

void cleanupState( RayTracedState& state, GLFWwindow* window)
{
    sutil::cleanupUI(window);
    ImGui_ImplOpenGL3_Shutdown();
    ImGui::DestroyContext();

    OPTIX_CHECK( optixPipelineDestroy     ( state.pipeline                ) );
    OPTIX_CHECK(optixPipelineDestroy(state.pipeline_motion));
    OPTIX_CHECK( optixProgramGroupDestroy ( state.raygen_prog_group       ) );
    OPTIX_CHECK(optixProgramGroupDestroy  ( state.raygen_motion_prog_group) );
    OPTIX_CHECK( optixProgramGroupDestroy ( state.radiance_metal_sphere_prog_group ) );
    OPTIX_CHECK( optixProgramGroupDestroy ( state.occlusion_metal_sphere_prog_group ) );
    OPTIX_CHECK( optixProgramGroupDestroy ( state.radiance_miss_prog_group         ) );
    OPTIX_CHECK( optixProgramGroupDestroy ( state.radiance_floor_prog_group        ) );
    OPTIX_CHECK( optixProgramGroupDestroy ( state.occlusion_floor_prog_group       ) );
    OPTIX_CHECK(optixProgramGroupDestroy(state.intersect_motion_prog_group));
    OPTIX_CHECK(optixProgramGroupDestroy(state.closeshit_motion_prog_group));
    OPTIX_CHECK(optixProgramGroupDestroy(state.closeshit_floor_motion_prog_group));
    OPTIX_CHECK(optixProgramGroupDestroy(state.intersect_floor_motion_prog_group));
    OPTIX_CHECK(optixProgramGroupDestroy(state.radiance_motion_miss_prog_group));
    OPTIX_CHECK( optixModuleDestroy       ( state.shading_module          ) );
    OPTIX_CHECK( optixModuleDestroy       ( state.geometry_module         ) );
    OPTIX_CHECK( optixModuleDestroy       ( state.camera_module           ) );
    OPTIX_CHECK( optixModuleDestroy       ( state.sphere_module           ) );
    OPTIX_CHECK(optixModuleDestroy(state.sphere_module_impulse));
    OPTIX_CHECK(optixModuleDestroy(state.motion_module));
    OPTIX_CHECK( optixDeviceContextDestroy( state.context                 ) );
    OPTIX_CHECK(optixDeviceContextDestroy(state.context_implse));

    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.params.accum_buffer_motion)));
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.sbt.raygenRecord       ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.sbt.missRecordBase     ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.sbt.hitgroupRecordBase ) ) );
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.sbt_motion.raygenRecord)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.sbt_motion.missRecordBase)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.sbt_motion.hitgroupRecordBase)));
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.output_buffer    ) ) );
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.output_buffer_size)));
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.params.accum_buffer    ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.d_params               ) ) );
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.d_static_gas_output_buffer)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.d_ias_output_buffer)));
    free(state.hitgroup_records);
    free(state.hitgroup_records_impulse);
    CUDA_CHECK(cudaFree((void*)state.d_aabb));
    free(state.rg_sbt);
    free(state.rg_sbt_cam);
    delete state.aabb;
    delete state.aabb_input_flags;
    free(state.camData);
  //  free(state.hitgroup_records);
  //  free(state.hitgroup_records);
    //delete state.params.sphere;
}

void sphereCollisionHandle(GeometryData::Sphere* s1, GeometryData::Sphere* s2, float deltaTime) 
{

    float3 collisionNormal = s1->center - s2->center;
    float3 relativeVelocity = s1->vel - s2->vel;

    float3 unitCollisionNorm = normalize(collisionNormal);
    float distance =length(collisionNormal);

    float vDn = dot(relativeVelocity, collisionNormal);
    float nDn = dot(collisionNormal, collisionNormal);

    s1->vel -= (1.5f * s2->mass / (s1->mass + s2->mass)) * (vDn / nDn) * collisionNormal;
    vDn = dot(-relativeVelocity, -collisionNormal);
    s2->vel -= (1.5f * s1->mass / (s1->mass + s2->mass)) * (vDn / nDn) * -collisionNormal;
  
  //  s1->center += s1->vel * deltaTime;
  //  s2->center += s2->vel * deltaTime;

    s1->center += (s1->radius * 2.f - distance) /2 * unitCollisionNorm;
    s2->center -= (s2->radius * 2.f - distance)/ 2 * unitCollisionNorm;
}

void updatePhysics(RayTracedState& state)
{
    float size = 20;
    //https://openstax.org/books/physics/pages/9-2-mechanical-energy-and-conservation-of-energy
    auto tstart = std::chrono::system_clock::now();

    for (size_t i = 0; i < OBJ_COUNT; ++i) {

        sphereList[i]->vel += state.accel;

        sphereList[i]->center += sphereList[i]->vel;

        if (sphereList[i]->center.y <= 0) {
            sphereList[i]->center.y = 0;
            sphereList[i]->vel.y *= -0.95f;
        }


        if (sphereList[i]->center.y >= 200) {
            sphereList[i]->center.y = 70;
            sphereList[i]->vel.y *= -0.95f;
        }

        if (sphereList[i]->center.x <= -size) {
            sphereList[i]->center.x = -size;

            sphereList[i]->vel.x *= -0.95f;

        }

        if (sphereList[i]->center.x >= size) {
            sphereList[i]->center.x = size;
            sphereList[i]->vel.x *= -0.95f;
        }

        if (sphereList[i]->center.z >= size) {
            sphereList[i]->center.z = size;

            sphereList[i]->vel.z *= -0.95f;

        }

        if (sphereList[i]->center.z <= -size) {
            sphereList[i]->center.z = -size;
            sphereList[i]->vel.z *= -0.95f;
        }
        if (!state.is_ray_traced_pyhsics) {
            for (size_t j = 0; j < OBJ_COUNT; j++) {
                float3 collisionNormal = make_float3(sphereList[j]->center.x - sphereList[i]->center.x,
                    sphereList[j]->center.y - sphereList[i]->center.y,
                    sphereList[j]->center.z - sphereList[i]->center.z);

                float distance = sqrt(collisionNormal.x * collisionNormal.x + collisionNormal.y * collisionNormal.y + collisionNormal.z * collisionNormal.z);
                if (sphereList[i]->radius + sphereList[j]->radius >= distance && i != j)
                {
                    sphereCollisionHandle(sphereList[i], sphereList[j], state.time);
                }
            }
        }
    }

    auto tnow = std::chrono::system_clock::now();
    std::chrono::duration<double> time = tnow - tstart;

   // std::cerr << "CPU FPS: " << time.count() << std::endl;
}

void createStates(RayTracedState& state) 
{
    initCameraState();

    //
    // Set up OptiX state
    //
    createContext(state);
    createGeometry(state);
    createPipeline(state);
    createSBT(state);

    // Set up sphere collision ray tracing pipeline
   /*
    */
    createContextImpulse(state);
    createMotionPipeline(state);
    createMotionSBT(state);
    initLaunchParams(state);
}

void setupCallBacks(RayTracedState& state, GLFWwindow* window) {
    glfwSetMouseButtonCallback(window, mouseButtonCallback);
    glfwSetCursorPosCallback(window, cursorPosCallback);
    glfwSetWindowSizeCallback(window, windowSizeCallback);
    glfwSetWindowIconifyCallback(window, windowIconifyCallback);
    glfwSetKeyCallback(window, keyCallback);
    glfwSetScrollCallback(window, scrollCallback);
    glfwSetWindowUserPointer(window, &state.params);
}

void updateRenderStates(RayTracedState& state) {
    updatePhysics(state);
    buildGasUpdate(state);
    updateSTB(state);
    updateImpulseSBT(state);
}

void setupGUI(GLFWwindow* window) {
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init();
    ImGui::StyleColorsDark();
}

void calculateRayTracedCollisions(RayTracedState& state)
{
   /*  int N = 15000;

    size_t bytes_float3 = N * sizeof(GeometryData::impulseData);
    GeometryData::Sphere* d_spheres;
    GeometryData::impulseData* d_v_r;
   // cudaMalloc(&d_v_r, bytes_float3);
   // cudaMalloc(&d_spheres, 3 * sizeof(GeometryData::Sphere));

    const int numElements = 3;
    const int numBytes = numElements * sizeof(GeometryData::Sphere);

   
       vector <std::shared_ptr<GeometryData::Sphere>> h_v{
            std::make_shared<GeometryData::Sphere>(state.g_sphere),
            std::make_shared<GeometryData::Sphere>(state.g_sphere2),
             std::make_shared<GeometryData::Sphere>(state.g_sphere3)
        };
    */

     // Host data
  //  vector<GeometryData::impulseData> h_v_r(N);

    // Copy to device
  //  cudaMemcpy(d_spheres, h_v.data(), numBytes, cudaMemcpyHostToDevice);

 //   UpdateSphereImpulseWithRayCollision(nullptr, state.params.impulse_buffer, sphereList, state.time);

    // Copy to host;
  //  cudaMemcpy(h_v.data(), d_spheres, numBytes, cudaMemcpyDeviceToHost);
 //   cudaMemcpy(h_v_r.data(), d_v_r, bytes_float3, cudaMemcpyDeviceToHost);

   // for (size_t i = 0; i < 0; i++) {

      //  if (h_v_r[i].impulse.x > 0.0f || h_v_r[i].impulse.y > 0.0f || h_v_r[i].impulse.z > 0.0f) {

        //  std::cerr << i << " " << h_v_r[i].impulse.x << std::endl;
         //   sphereList[i]->vel -= h_v_r[i].impulse;
           // sphereList[i]->center -= h_v_r[i].pos;
         //   sphereList[i]->center.x += 0.005f;
          //  sphereList[i]->center.z += 0.005f;
          //  sphereList[i]->center -= h_v[i]. * 0.05f;
    //    }

   // }
}

void renderGUI(RayTracedState& state, GLFWwindow* window, GuiElements* gui)
{
    ImGui::Begin("Change balls propeerty");
    ImGui::SliderFloat3("balls position", gui->position1, -10.0f, 10.0f);

    if (ImGui::Button("Change pos")) {
        sphereList[0]->center.x = gui->position1[0];
        sphereList[0]->center.y = gui->position1[1];
        sphereList[0]->center.z = gui->position1[2];
    }

    ImGui::SliderFloat3("balls velocity", gui->velocity1, 0.0f, 5.0f);
    if (ImGui::Button("Change vel")) {
        for (int i = 0; i < OBJ_COUNT; i++) {
            sphereList[i]->vel.x = gui->velocity1[0];
            sphereList[i]->vel.y = gui->velocity1[1];
            sphereList[i]->vel.z = gui->velocity1[2];
        }
    }

    /*
    ImGui::SliderInt("Ball number", &gui->ballSize, 3, 100);
    if (ImGui::Button("Update number")) {
        state.params.height_impulse = gui->ballSize;

            for (int i = 0; i < gui->ballSize; i++) {
        sphereList[i] = new GeometryData::Sphere({ 
            { -3.0f, 9.0f, -2.5f }, // center
            1.0f,                  // radius
            { 0.0f, 0.0f, 0.0f }, // velocity
            1.0 }); // mass
    }
    }
    */

  //    if (ImGui::Button("Collision type (normal/ray traced)")) {
  //        state.isWallVisible = !state.isWallVisible;
   //    }

  //    ImGui::Text("Physics collision with ray tracing: " + (state.isWallVisible == true)?  "ON" : "OFF");
      ImGui::RadioButton("CPU physics", &state.is_ray_traced_pyhsics, false);
      ImGui::SameLine();
      ImGui::RadioButton("GPU physics", &state.is_ray_traced_pyhsics, true);
      ImGui::SameLine();

      if (state.is_ray_traced_pyhsics) {
          state.isWallVisible == true;
      }
      else {
          state.isWallVisible == false;
      }
      ImGui::NewLine();
      /*
      if (ImGui::BeginMenuBar())
      {
          if (ImGui::BeginMenu("Menu"))
          {
              ImGui::MenuItem("Main menu bar", NULL);
          }
      }
      */
  //  if (ImGui::Button("Add new ball")) {
        // TODO
  //  }

    ImGui::End();
    ImGui::Render();
    int display_w, display_h;
    glfwGetFramebufferSize(window, &display_w, &display_h);
    glViewport(0, 0, display_w, display_h);

    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void calculateTime(std::chrono::duration<double>& time, std::chrono::steady_clock::time_point& t0) 
{
    auto t1 = std::chrono::steady_clock::now();
    time += t1 - t0;
    t0 = t1;
}

int main()
{
    RayTracedState state;
    GuiElements gui{};
    state.params.width = 1000;
    state.params.height = 1000;
    state.params.width_impulse = 100; //65 000 000
    state.params.height_impulse = OBJ_COUNT;
    state.time = 0.f;
   // state.is_ray_traced_pyhsics = true;
    sutil::CUDAOutputBufferType output_buffer_type = sutil::CUDAOutputBufferType::GL_INTEROP;

    for (int i = 0; i < OBJ_COUNT; i++) {
        sphereList[i] = new GeometryData::Sphere({ 
            { -3.0f, 9.0f, -2.5f }, // center
            1.0f,                  // radius
            { 0.0f, 0.0f, 0.0f }, // velocity
            1.0 }); // mass
        sphereList[i]->vel += make_float3(0.05f);
    }


 
    gui.position1[0] = 0.f;
    gui.position1[1] = 0.f;
    gui.position1[2] = 0.f;

    gui.velocity1[0] = 0.f;
    gui.velocity1[1] = 0.f;
    gui.velocity1[2] = 0.f;
    try
    {
        createStates(state);

        GLFWwindow* window = sutil::initUI("optixRayTracedCollision", state.params.width, state.params.height);

        setupCallBacks(state, window);

        // output_buffer needs to be destroyed before cleanupUI is called
        sutil::CUDAOutputBuffer<uchar4> output_buffer(
            output_buffer_type,
            state.params.width,
            state.params.height
        );

        sutil::CUDAOutputBuffer<GeometryData::impulseData> output_buffer_impulse(
            output_buffer_type,
            state.params.width_impulse,
            state.params.height_impulse
        );
        
        output_buffer.setStream(state.stream);
        output_buffer_impulse.setStream(state.stream_motion);

        sutil::GLDisplay gl_display;
        sutil::GLDisplay gl_display2;
        std::chrono::duration<double> state_update_time(0.0);
        std::chrono::duration<double> render_time(0.0);
        std::chrono::duration<double> display_time(0.0);

        auto tstart = std::chrono::system_clock::now();
        auto lastTime = std::chrono::system_clock::now();

       setupGUI(window);

        //
        // Render loop
        //
        do
        {
            auto t0 = std::chrono::steady_clock::now();
            updateState(output_buffer, state);
             
            auto tnow = std::chrono::system_clock::now();
            std::chrono::duration<double> time = tnow - tstart;

            state.time = (float)time.count();
            ImGui_ImplOpenGL3_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();
            glfwPollEvents();
            
            updateRenderStates(state);

            calculateTime(state_update_time, t0);
            displaySubframe(output_buffer, gl_display, window);


            if (state.is_ray_traced_pyhsics) {
                launchSubframeMotion(output_buffer_impulse, state);
                UpdateSphereImpulseWithRayCollision(state.params.impulse_buffer, sphereList, state.params.height_impulse, state.params.width_impulse);
              //  displaySubframeImpulse(output_buffer_impulse, gl_display, window);
            }    
       //   else {

            launchSubframe(output_buffer, state);       

        //   }
            calculateTime(render_time, t0);


            auto t1 = std::chrono::steady_clock::now();
            display_time += t1 - t0;

            sutil::displayStats(state_update_time, render_time, display_time);
            renderGUI(state, window, &gui);

            glfwSwapBuffers(window);     
            ++state.params.subframe_index;
            lastTime = tnow;

            std::chrono::duration<double> deltaTime = tnow - lastTime;
            state.deltaTime = (float)deltaTime.count();
          //  state.params.deltaTime = state.deltaTime;
        } while (!glfwWindowShouldClose(window));
       
        delete sphereList;
        cleanupState(state, window);
    }
    catch (std::exception& e)
    {
        std::cerr << "Caught exception: " << e.what() << "\n";
        return 1;
    }
    return 0;
}