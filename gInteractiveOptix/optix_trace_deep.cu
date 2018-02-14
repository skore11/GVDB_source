
/*
 * Copyright (c) 2008 - 2009 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and proprietary
 * rights in and to this software, related documentation and any modifications thereto.
 * Any use, reproduction, disclosure or distribution of this software and related
 * documentation without an express license agreement from NVIDIA Corporation is strictly
 * prohibited.
 *
 * TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*
 * AND NVIDIA AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
 * INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE.  IN NO EVENT SHALL NVIDIA OR ITS SUPPLIERS BE LIABLE FOR ANY
 * SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
 * LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
 * BUSINESS INFORMATION, OR ANY OTHER PECUNIARY LOSS) ARISING OUT OF THE USE OF OR
 * INABILITY TO USE THIS SOFTWARE, EVEN IF NVIDIA HAS BEEN ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGES
 */

#include "optix_extra_math.cuh"

#define ANY_RAY			0
#define	SHADOW_RAY		1
#define VOLUME_RAY		2
#define MESH_RAY		3

rtDeclareVariable(float3,       light_pos, , );

rtDeclareVariable(rtObject,     top_object, , );
rtDeclareVariable(float,        scene_epsilon, , );
rtDeclareVariable(unsigned int, shadow_enable, , );
rtDeclareVariable(unsigned int, mirror_enable, , );
rtDeclareVariable(unsigned int, cone_enable, , );
rtDeclareVariable(int,          max_depth, , );

rtDeclareVariable(float3,		shading_normal,		attribute shading_normal, ); 
rtDeclareVariable(float3,		front_hit_point,	attribute front_hit_point, );
rtDeclareVariable(float3,		back_hit_point,		attribute back_hit_point, );
rtDeclareVariable(float4,		deep_color,			attribute deep_color, );
rtDeclareVariable(int,			obj_type,			attribute obj_type, );

rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(float, t_hit, rtIntersectionDistance, );
rtDeclareVariable(uint2,        launch_index, rtLaunchIndex, );
rtDeclareVariable(unsigned int, sample, , );

rtBuffer<unsigned int, 2>       rnd_seeds;

struct PerRayData_radiance
{
	float3	result;
	float	length; 
	float	alpha;
	int		depth;
	int		rtype;
};

rtDeclareVariable(PerRayData_radiance, prd_radiance, rtPayload, );

// -----------------------------------------------------------------------------

static __device__ __inline__ float3 TraceRay (float3 origin, float3 direction, int depth, int rtype, float& length )
{
  optix::Ray ray = optix::make_Ray( origin, direction, 0, 0.0f, RT_DEFAULT_MAX );
  PerRayData_radiance prd;
  prd.length = 0.f;
  prd.depth = depth; 
  prd.rtype = rtype;
  rtTrace( top_object, ray, prd );  
  length = prd.length;
  return prd.result;
}

float3 __device__ __inline__ jitter_sample ( const uint2& index )
{	 
    volatile unsigned int seed  = rnd_seeds[ index ]; // volatile workaround for cuda 2.0 bug
    unsigned int new_seed  = seed;
    float uu = rnd( new_seed )-0.5f;
    float vv = rnd( new_seed )-0.5f;
	float ww = rnd( new_seed )-0.5f;
    rnd_seeds[ index ] = new_seed;	
    return make_float3(uu,vv,ww);
}

RT_PROGRAM void trace_deep ()
{
	// Volumetric material

	// We arrive here after vol_deep has already traced into the gvdb volume.
	// - deep_color is the accumulated color along the volume ray
	// - front_hit_point is the start point of the volume
	// - back_hit_point is the ending point of the volume
	
	// Blending with polygonal objects is achieved by stochastically 
	// tracing a MESH_RAY from a random point inside the volume toward the background.
	float rlen;	
	float3 jit = jitter_sample ( make_uint2( (launch_index.x + sample) % blockDim.x, (launch_index.y + sample) % blockDim.y) );	
	float3 pos = front_hit_point + (jit.x+0.5f) * (back_hit_point - front_hit_point);
	float3 bgclr = TraceRay ( pos, ray.direction, 1, MESH_RAY, rlen );	

	// Result is blending of background color and the volume color (deep_color)	
	prd_radiance.result = lerp3 ( bgclr, fxyz(deep_color), deep_color.w );
	prd_radiance.length = length ( back_hit_point - ray.origin );
	prd_radiance.alpha = deep_color.w;

	// prd_radiance.result = fhp/200.0;			-- debugging
}

// -----------------------------------------------------------------------------

//
// Attenuates shadow rays for shadowing transparent objects
//
RT_PROGRAM void trace_shadow ()
{
	// rtype is SHADOW_RAY
	prd_radiance.alpha = 0; //deep_color.w;
}
