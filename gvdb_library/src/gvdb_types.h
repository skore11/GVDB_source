//--------------------------------------------------------------------------------
//
// File:   gvdb_types.h
// 
// NVIDIA GVDB Sparse Volumes
// Copyright (c) 2015, NVIDIA. All rights reserved.
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of the <organization> nor the
//    names of its contributors may be used to endorse or promote products
//    derived from this software without specific prior written permission.
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Last Updated: Rama Hoetzlein, rhoetzlein@nvidia.com, 8/17/2016
//
//----------------------------------------------------------------------------------

#ifndef GVDB_TYPES
	#define GVDB_TYPES

	#include <stdint.h>
        #include <cstdarg>

		#if !defined ( GVDB_STATIC )
		#if defined ( GVDB_EXPORTS )				// inside DLL
			#if defined(_WIN32) || defined(__CYGWIN__)
				#define GVDB_API		__declspec(dllexport)
			#else
				#define GVDB_API		__attribute__((visibility("default")))
			#endif
		#else										// outside DLL
			#if defined(_WIN32) || defined(__CYGWIN__)
				#define GVDB_API		__declspec(dllimport)
			#else
				#define GVDB_API		__attribute__((dllimport))
			#endif
		#endif          
	#else
		#define GVDB_API
	#endif
	#ifdef _WIN32
		#define ALIGN(x)			__declspec(align(x))	 
	#else
		 #define ALIGN(x)    __attribute__((aligned(x)))
	#endif

	typedef uint32_t			uint32;
	typedef uint64_t			uint64;
	typedef int16_t				sint16;
	typedef int32_t				sint32;
	typedef int64_t				sint64;
	typedef unsigned char		byte;
	typedef unsigned char		uchar;
	typedef signed char			schar;		
	typedef uint16_t			ushort;	
	typedef uint32_t			uint;
	typedef uint64_t			ulong;	
	typedef int64_t				slong;

	#define NOHIT			1.0e10f
	#define	ID_UNDEFB		0xFF			// 1 byte
	#define	ID_UNDEFS		0xFFFF			// 2 byte
	#define	ID_UNDEFL		0xFFFFFFFF		// 4 byte
	#define CHAN_UNDEF		255

	#define DEGtoRAD		(3.141592f/180.0f)

	#define CLRVAL			uint
	#define COLOR(r,g,b)	( (uint((r)*255.0f)<<24) | (uint((g)*255.0f)<<16) | (uint((b)*255.0f)<<8) )
	#define COLORA(r,g,b,a)	( (uint((a)*255.0f)<<24) | (uint((b)*255.0f)<<16) | (uint((g)*255.0f)<<8) | uint((r)*255.0f) )
	#define ALPH(c)			(float((c>>24) & 0xFF)/255.0)
	#define BLUE(c)			(float((c>>16) & 0xFF)/255.0)
	#define GRN(c)			(float((c>>8)  & 0xFF)/255.0)
	#define RED(c)			(float( c      & 0xFF)/255.0)
	#ifndef INT64_C
		#define INT64_C(c) (c ## LL)
		#define UINT64_C(c) (c ## ULL)
	#endif

	#define T_UCHAR			0
	#define T_UCHAR3		1
	#define T_UCHAR4		2
	#define	T_FLOAT			3
	#define	T_FLOAT3		4
	#define T_FLOAT4		5
	#define T_INT			6
	#define T_INT3			7
	#define T_INT4			8

	#undef min
	#undef max

	// forward references
	struct cudaGraphicsResource;
	struct cudaArray;

	// these will be eliminated in the final API
	#define ENGINE_GLPREVIEW	0	// Engine
	#define ENGINE_CUDA			1	
	#define ENGINE_OPTIX		2	

	#define SHADE_VOXEL		0		// Shading
	#define SHADE_SECTION2D	1	
	#define SHADE_SECTION3D	2
	#define SHADE_EMPTYSKIP	3
	#define SHADE_TRILINEAR	4
	#define SHADE_TRICUBIC	5
	#define SHADE_LEVELSET	6
	#define SHADE_VOLUME	7
	#define SHADE_OFF		100

	// gprintf
	extern void GVDB_API gprintf(const char * fmt, ...);
	extern void GVDB_API gprintfLevel(int level, const char * fmt, ...);
	extern void GVDB_API gprintf2(va_list &vlist, const char * fmt, int level);
	extern void GVDB_API gprintSetLevel(int l);
	extern int  GVDB_API gprintGetLevel();
	extern void GVDB_API gprintSetLogging(bool b);
	extern void GVDB_API gerror();			

	#define  LOGLEVEL_INFO 0
	#define  LOGLEVEL_WARNING 1
	#define  LOGLEVEL_ERROR 2
	#define  LOGLEVEL_FATAL 3
	#define  LOGLEVEL_OK 7
	#define  LOGI(...)  { gprintfLevel(0, __VA_ARGS__); }
	#define  LOGW(...)  { gprintfLevel(1, __VA_ARGS__); }
	#define  LOGE(...)  { gprintfLevel(2, __FILE__"("S__LINE__"): "__VA_ARGS__); }
	#define  LOGOK(...)  { gprintfLevel(7, __VA_ARGS__); }

#endif
