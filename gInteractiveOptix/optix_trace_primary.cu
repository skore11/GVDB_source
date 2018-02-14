
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
#include "texture_fetch_functions.h"			// from OptiX SDK

struct PerRayData_radiance
{
	float3	result;
	float	length; 
	float	alpha;
	int		depth;
	int		rtype;
};

rtDeclareVariable(float3,        cam_pos, , );
rtDeclareVariable(float3,        cam_U, , );
rtDeclareVariable(float3,        cam_V, , );
rtDeclareVariable(float3,        cam_W, , );
rtDeclareVariable(float3,        bad_color, , );
rtDeclareVariable(float,         scene_epsilon, , );
rtDeclareVariable(unsigned int,  sample, , );
rtBuffer<float3, 2>              output_buffer;
rtDeclareVariable(rtObject,      top_object, , );
rtBuffer<unsigned int, 2>        rnd_seeds;

rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
rtDeclareVariable(uint2, launch_dim,   rtLaunchDim, );

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

RT_PROGRAM void trace_primary ()
{
  float2 d = make_float2(launch_index) / make_float2(launch_dim) - 0.5f;  
  float pixsize = length ( cam_U ) / launch_dim.x;	
  float3 ray_direction;
  float3 result;

  PerRayData_radiance prd;
  prd.length = 0.f;
  prd.depth = 0;
  prd.rtype = 0;	// ANY_RAY

  int initial_samples = 1;
  
  if ( sample <= initial_samples ) {
	  result = make_float3(0,0,0);	  
	  for (int n=0; n < initial_samples; n++ ) {
		  ray_direction = normalize (d.x*cam_U + d.y*cam_V + cam_W + jitter_sample ( launch_index )*make_float3(pixsize,pixsize,pixsize) );
		  optix::Ray ray = optix::make_Ray( cam_pos, ray_direction, 0, 0.0f, RT_DEFAULT_MAX);
		  rtTrace( top_object, ray, prd );
		  result += prd.result;
	  }
	  prd.result = result / float(initial_samples);
  } else {	  
	  ray_direction = normalize (d.x*cam_U + d.y*cam_V + cam_W + jitter_sample ( launch_index )*make_float3(pixsize,pixsize,pixsize) );
	  optix::Ray ray = optix::make_Ray( cam_pos, ray_direction, 0, 0.0f, RT_DEFAULT_MAX);
	  rtTrace( top_object, ray, prd );
	  prd.result = (output_buffer[launch_index]*(sample-1) + prd.result) / float(sample);
  }

  output_buffer[launch_index] = prd.result;
}

RT_PROGRAM void exception()
{
  const unsigned int code = rtGetExceptionCode();
  //rtPrintf( "Exception 0x%X at launch index (%d,%d)\n", code, launch_index.x, launch_index.y );
  printf( "Exception 0x%X at launch index (%d,%d)\n", code, launch_index.x, launch_index.y );
  output_buffer[launch_index] = bad_color;
}
