/*
  FLUIDS v.3 - SPH Fluid Simulator for CPU and GPU
  Copyright (C) 2012-2013. Rama Hoetzlein, http://fluids3.com

  Attribute-ZLib license (* See additional part 4)

  This software is provided 'as-is', without any express or implied
  warranty. In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software.
  2. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.
  3. This notice may not be removed or altered from any source distribution.
  4. Any published work based on this code must include public acknowledgement
     of the origin. This includes following when applicable:
	   - Journal/Paper publications. Credited by reference to work in text & citation.
	   - Public presentations. Credited in at least one slide.
	   - Distributed Games/Apps. Credited as single line in game or app credit page.	 
	 Retaining this additional license term is required in derivative works.
	 Acknowledgement may be provided as:
	   Publication version:  
	      2012-2013, Hoetzlein, Rama C. Fluids v.3 - A Large-Scale, Open Source
	 	  Fluid Simulator. Published online at: http://fluids3.com
	   Single line (slides or app credits):
	      GPU Fluids: Rama C. Hoetzlein (Fluids v3 2013)

 Notes on Clause 4:
  The intent of this clause is public attribution for this contribution, not code use restriction. 
  Both commerical and open source projects may redistribute and reuse without code release.
  However, clause #1 of ZLib indicates that "you must not claim that you wrote the original software". 
  Clause #4 makes this more specific by requiring public acknowledgement to be extended to 
  derivative licenses. 

*/

#ifndef DEF_KERN_CUDA
	#define DEF_KERN_CUDA

	#include <curand.h>
	#include <curand_kernel.h>
	#include <stdio.h>
	#include <math.h>

	#define CUDA_KERNEL
	#include "fluid.h"

	#define EPSILON				0.00001f
	#define GRID_UCHAR			0xFF
	#define GRID_UNDEF			4294967295
	#define TOTAL_THREADS		1000000
	#define BLOCK_THREADS		256
	#define MAX_NBR				80		
	#define FCOLORA(r,g,b,a)	( (uint((a)*255.0f)<<24) | (uint((b)*255.0f)<<16) | (uint((g)*255.0f)<<8) | uint((r)*255.0f) )

	typedef unsigned int		uint;
	typedef unsigned short int	ushort;
	typedef unsigned char		uchar;
	
	extern "C" {
		__global__ void insertParticles ( int pnum );		
		__global__ void countingSortFull ( int pnum );		
		__global__ void computeQuery ( int pnum );	
		__global__ void computePressure ( int pnum );		
		__global__ void computeForce ( int pnum );	
		__global__ void advanceParticles ( float time, float dt, float ss, int numPnts );
		__global__ void emitParticles ( float frame, int emit, int numPnts );
		__global__ void randomInit ( int seed, int numPnts );
		__global__ void sampleParticles ( float* brick, uint3 res, float3 bmin, float3 bmax, int numPnts, float scalar );	
		__global__ void prefixFixup ( uint *input, uint *aux, int len);
		__global__ void prefixSum ( uint* input, uint* output, uint* aux, int len, int zeroff );
		__global__ void countActiveCells ( int pnum );		
	}

#endif
