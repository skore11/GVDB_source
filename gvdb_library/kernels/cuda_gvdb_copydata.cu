
//----------------------------------------------
// File: cuda_gvdb_copydata.cu
//
// GVDB Data Transfers
// - CopyData		3D volume into sub-volume
// - CopyDataZYX	3D volume into sub-volume with ZYX swizzle
// - RetreiveData	3D sub-volume into cuda buffer
// - CopyTexToBuf	2D texture into cuda buffer
// - CopyBufToTex	cuda buffer into 2D texture
//-----------------------------------------------

#include "cuda_math.cuh"

texture<uchar, 3, cudaReadModeElementType>		volTexInC;
texture<float, 3, cudaReadModeElementType>		volTexInF;
surface<void, 3>								volTexOut;

// Zero memory of 3D volume
extern "C" __global__ void kernelFillTex ( int3 res, int dsize )
{
	uint3 t = blockIdx * make_uint3(blockDim.x, blockDim.y, blockDim.z) + threadIdx;	
	if ( t.x >= res.x || t.y >= res.y || t.z >= res.z ) return;

	surf3Dwrite ( 0, volTexOut, t.x*dsize, t.y, t.z );
}

// Copy 3D texture into sub-volume of another 3D texture (char)
extern "C" __global__ void kernelCopyTexC ( int3 offs, int3 res )
{
	uint3 t = blockIdx * make_uint3(blockDim.x, blockDim.y, blockDim.z) + threadIdx;	
	if ( t.x >= res.x || t.y >= res.y || t.z >= res.z ) return;
	uchar val = tex3D ( volTexInC, t.x, t.y, t.z );
	surf3Dwrite ( val, volTexOut, (t.x+offs.x)*sizeof(char), (t.y+offs.y), (t.z+offs.z) );
}

// Copy 3D texture into sub-volume of another 3D texture (float)
extern "C" __global__ void kernelCopyTexF ( int3 offs, int3 res )
{
	uint3 t = blockIdx * make_uint3(blockDim.x, blockDim.y, blockDim.z) + threadIdx;	
	if ( t.x >= res.x || t.y >= res.y || t.z >= res.z ) return;	
	float val = tex3D ( volTexInF, t.x, t.y, t.z );
	surf3Dwrite ( val, volTexOut, (t.x+offs.x)*sizeof(float), (t.y+offs.y), (t.z+offs.z) );
}

// Copy linear memory as 3D volume into sub-volume of a 3D texture
extern "C" __global__ void kernelCopyBufToTexC ( int3 offs, int3 res, uchar* inbuf)
{
	uint3 t = blockIdx * make_uint3(blockDim.x, blockDim.y, blockDim.z) + threadIdx;	
	if ( t.x >= res.x || t.y >= res.y || t.z >= res.z ) return;	
	uchar val = inbuf[ (t.z*res.y + t.y)*res.x + t.x ];
	surf3Dwrite ( val, volTexOut, (t.x+offs.x)*sizeof(uchar), (t.y+offs.y), (t.z+offs.z) );
}
// Copy linear memory as 3D volume into sub-volume of a 3D texture
extern "C" __global__ void kernelCopyBufToTexF ( int3 offs, int3 res, float* inbuf)
{
	uint3 t = blockIdx * make_uint3(blockDim.x, blockDim.y, blockDim.z) + threadIdx;	
	if ( t.x >= res.x || t.y >= res.y || t.z >= res.z ) return;	
	float val = inbuf[ (t.z*res.y + t.y)*res.x + t.x ];	
	surf3Dwrite ( val, volTexOut, (t.x+offs.x)*sizeof(float), (t.y+offs.y), (t.z+offs.z) );
}

// Copy 3D texture into sub-volume of another 3D texture with ZYX swizzle (float)
extern "C" __global__ void kernelCopyTexZYX (  int3 offs, int3 res )
{
	uint3 t = blockIdx * make_uint3(blockDim.x, blockDim.y, blockDim.z) + threadIdx;	
	if ( t.x >= res.x || t.y >= res.y || t.z >= res.z ) return;
	float val = tex3D ( volTexInF, t.z, t.y, t.x );
	surf3Dwrite ( val, volTexOut, (t.x+offs.x)*sizeof(float), (t.y+offs.y), (t.z+offs.z) );
}

// Retrieve 3D texture into linear memory (float)
extern "C" __global__ void kernelRetrieveTexXYZ ( int3 offs, int3 src_res, int3 res, float* buf )
{
	uint3 t = blockIdx * make_uint3(blockDim.x, blockDim.y, blockDim.z) + threadIdx;	
	if ( t.x >= src_res.x || t.y >= src_res.y || t.z >= src_res.z ) return;
	float val = tex3D ( volTexInF, t.x+offs.x, t.y+offs.y, t.z+offs.z );
	buf[ (t.x*res.y + t.y)*res.x + t.z ] = val;
}

// Copy 2D slice of 3D texture into 2D linear buffer
extern "C" __global__ void kernelSliceTexToBuf ( int slice, int3 res, float* outbuf  )
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;	
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if ( x >= res.x || y >= res.y ) return;
	float val = tex3D ( volTexInF, x, y, slice );
	outbuf[ y*res.x + x ] = val;
}

// Copy 2D linear buffer into the 2D slice of a 3D texture
extern "C" __global__ void kernelSliceBufToTex ( int slice, int3 res, float* inbuf  )
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;	
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if ( x >= res.x || y >= res.y ) return;
	float val = inbuf[ y*res.x + x ];
	surf3Dwrite ( val, volTexOut, x*sizeof(float), y, slice );
}


