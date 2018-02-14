
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

//rtTextureSampler<float4, 2>		envmap;

rtDeclareVariable(float3,        cam_pos, , );
rtDeclareVariable(float3,        cam_U, , );
rtDeclareVariable(float3,        cam_V, , );
rtDeclareVariable(float3,        cam_W, , );

rtDeclareVariable(uint2,		launch_index, rtLaunchIndex, );
rtDeclareVariable(uint2,		launch_dim,   rtLaunchDim, );
rtBuffer<unsigned int, 2>       rnd_seeds;

rtDeclareVariable(optix::Ray,	ray, rtCurrentRay, );

struct PerRayData_radiance
{
	float3	result;
	float	length; 
	float	alpha;
	int		depth;
	int		rtype;
};

rtDeclareVariable(PerRayData_radiance, prd_radiance, rtPayload, );
rtDeclareVariable(PerRayData_radiance, prd_shadow, rtPayload, );

// -----------------------------------------------------------------------------

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

RT_PROGRAM void miss()
{
	float2 d = make_float2(launch_index) / make_float2(launch_dim); // - 0.5f;  

	float3 clr = make_float3 ( 0.0, fabs(ray.direction.y)*0.15, fabs(ray.direction.y)*0.2 );
	clr += jitter_sample ( launch_index ) * 0.05;    

	prd_radiance.result = clr;
}
