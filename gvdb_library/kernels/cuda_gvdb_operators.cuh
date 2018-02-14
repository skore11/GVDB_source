


//----------------------------------------------
// File: cuda_gvdb_operators.cu
//
// GVDB Operators
// - CopyApron		- update apron voxels
// - OpSmooth		- smooth operation
// - OpNoise		- noise operation
//-----------------------------------------------

#define CMAX(x,y,z)  (imax3(x,y,z)-imin3(x,y,z))

#define COLORA(r,g,b,a)	 make_uchar4(r*255.0f, g*255.0f, b*255.0f, a*255.0f)

extern "C" __global__ void gvdbUpdateApronF ( int axis, int3 res, uchar chan, int blkres )
{
	// Recreate the compressed axis
	uint3 vox = blockIdx * make_uint3(blockDim.x, blockDim.y, blockDim.z) + threadIdx;	
	switch ( axis ) {														
	case 0:		vox.x = blockIdx.x * blkres + gvdb.apron_table[threadIdx.x];	break;
	case 1:		vox.y = blockIdx.y * blkres + gvdb.apron_table[threadIdx.y];	break;
	case 2:		vox.z = blockIdx.z * blkres + gvdb.apron_table[threadIdx.z];	break;
	};
	if ( vox.x >= res.x || vox.y >= res.y || vox.z >= res.z ) return;

	float3 wpos = getAtlasToWorld ( vox );
	
	float3 offs, vmin, vdel; uint64 nid;
	VDBNode* node = getNodeAtPoint ( wpos, &offs, &vmin, &vdel, &nid );		// Evaluate at world position
	offs += (wpos-vmin)/vdel;

	float v = (node==0x0) ? 0.0 : tex3D<float> ( volIn[chan], offs.x, offs.y, offs.z );		// Sample at world point

	surf3Dwrite ( v, volOut[chan], vox.x*sizeof(float), vox.y, vox.z );	// Write to apron voxel
}

extern "C" __global__ void gvdbUpdateApronC4 ( int axis, int3 res, uchar chan, int blkres )
{
	// Recreate the compressed axis
	uint3 vox = blockIdx * make_uint3(blockDim.x, blockDim.y, blockDim.z) + threadIdx;	
	switch ( axis ) {														
	case 0:		vox.x = blockIdx.x * blkres + gvdb.apron_table[threadIdx.x];	break;
	case 1:		vox.y = blockIdx.y * blkres + gvdb.apron_table[threadIdx.y];	break;
	case 2:		vox.z = blockIdx.z * blkres + gvdb.apron_table[threadIdx.z];	break;
	};
	if ( vox.x >= res.x || vox.y >= res.y || vox.z >= res.z ) return;

	float3 wpos = getAtlasToWorld ( vox );

	float3 offs, vmin, vdel; uint64 nid;	
	VDBNode* node = getNodeAtPoint ( wpos, &offs, &vmin, &vdel, &nid );		// Evaluate at world position	
	offs += (wpos-vmin) / vdel;

	uchar4 v = (node==0x0) ? make_uchar4(0,0,0,0) : tex3D<uchar4> ( volIn[chan], offs.x, offs.y, offs.z );	// Sample at world point
		
	surf3Dwrite ( v, volOut[chan], vox.x*sizeof(uchar4), vox.y, vox.z );	// Write to apron voxel
}

#define GVDB_COPY_SMEM_F																	\
	uint3 vox, ndx;																			\
	__shared__ float  svox[10][10][10]; 													\
	ndx = threadIdx + make_uint3(1,1,1);													\
	vox = blockIdx * make_uint3(blockDim.x, blockDim.y, blockDim.z) + ndx;					\
	if ( vox.x >= res.x || vox.y >= res.y || vox.z >= res.z ) return;						\
	svox[ndx.x][ndx.y][ndx.z] = tex3D<float> ( volIn[chan], vox.x, vox.y, vox.z );			\
	if ( ndx.x==1 ) {																		\
		svox[0][ndx.y][ndx.z] = tex3D<float> ( volIn[chan], vox.x-1, vox.y, vox.z );		\
		svox[9][ndx.y][ndx.z] = tex3D<float> ( volIn[chan], vox.x+8, vox.y, vox.z );		\
	}																						\
	if ( ndx.y==1 ) {																		\
		svox[ndx.x][0][ndx.z] = tex3D<float> ( volIn[chan], vox.x, vox.y-1, vox.z );		\
		svox[ndx.x][9][ndx.z] = tex3D<float> ( volIn[chan], vox.x, vox.y+8, vox.z );		\
	}																						\
	if ( ndx.z==1 ) {																		\
		svox[ndx.x][ndx.y][0] = tex3D<float> ( volIn[chan], vox.x, vox.y, vox.z-1 );		\
		svox[ndx.x][ndx.y][9] = tex3D<float> ( volIn[chan], vox.x, vox.y, vox.z+8 );		\
	}																						\
	__syncthreads ();

#define GVDB_COPY_SMEM_UC4																	\
	uint3 vox, ndx;																			\
	__shared__ uchar4 svox[10][10][10]; 													\
	ndx = threadIdx + make_uint3(1,1,1);													\
	vox = blockIdx * make_uint3(blockDim.x, blockDim.y, blockDim.z) + ndx;					\
	if ( vox.x >= res.x || vox.y >= res.y || vox.z >= res.z ) return;						\
	svox[ndx.x][ndx.y][ndx.z] = tex3D<uchar4> ( volIn[chan], vox.x, vox.y, vox.z );			\
	if ( ndx.x==1 ) {																		\
		svox[0][ndx.y][ndx.z] = tex3D<uchar4> ( volIn[chan], vox.x-1, vox.y, vox.z );		\
		svox[9][ndx.y][ndx.z] = tex3D<uchar4> ( volIn[chan], vox.x+8, vox.y, vox.z );		\
	}																						\
	if ( ndx.y==1 ) {																		\
		svox[ndx.x][0][ndx.z] = tex3D<uchar4> ( volIn[chan], vox.x, vox.y-1, vox.z );		\
		svox[ndx.x][9][ndx.z] = tex3D<uchar4> ( volIn[chan], vox.x, vox.y+8, vox.z );		\
	}																						\
	if ( ndx.z==1 ) {																		\
		svox[ndx.x][ndx.y][0] = tex3D<uchar4> ( volIn[chan], vox.x, vox.y, vox.z-1 );		\
		svox[ndx.x][ndx.y][9]  = tex3D<uchar4> ( volIn[chan], vox.x, vox.y, vox.z+1 );		\
	}																						\
	__syncthreads ();


extern "C" __global__ void gvdbOpGrow ( int3 res, uchar chan, float p1, float p2, float p3 )
{
	GVDB_COPY_SMEM_F
	
	/*float nl;	
	float3 n;
	n.x = 0.5 * (svox[ndx.x-1][ndx.y][ndx.z] - svox[ndx.x+1][ndx.y][ndx.z]);
	n.y = 0.5 * (svox[ndx.x][ndx.y-1][ndx.z] - svox[ndx.x][ndx.y+1][ndx.z]);
	n.z = 0.5 * (svox[ndx.x][ndx.y][ndx.z-1] - svox[ndx.x][ndx.y][ndx.z+1]);
	nl = sqrt(n.x*n.x+n.y*n.y+n.z*n.z);	
	float v = svox[ndx.x][ndx.y][ndx.z];
	if ( nl > 0.2 ) v += amt;*/ 

	float v = svox[ndx.x][ndx.y][ndx.z];
	if ( v != 0.0) v += p1 * 10.0; //*0.1;
	if ( v < 0.01) v = 0.0;
	
	surf3Dwrite ( v, volOut[chan], vox.x*sizeof(float), vox.y, vox.z );		
}

extern "C" __global__ void gvdbOpCut ( int3 res, uchar chan, float p1, float p2, float p3 )
{
	GVDB_COPY_SMEM_F			

	// Determine block and index position	
	float3 wpos = getAtlasToWorld ( vox );	

	float v = svox[ndx.x][ndx.y][ndx.z];
	if ( wpos.x < 50 && v > 0 && v < 2 ) {
		v = 0.02;
		surf3Dwrite ( v, volOut[chan], vox.x*sizeof(float), vox.y, vox.z );		
	}	
}

//-- smooth
	/*float v = 6.0*svox[ndx.x][ndx.y][ndx.z];
	v += svox[ndx.x-1][ndx.y][ndx.z];
	v += svox[ndx.x+1][ndx.y][ndx.z];
	v += svox[ndx.x][ndx.y-1][ndx.z];
	v += svox[ndx.x][ndx.y+1][ndx.z];
	v += svox[ndx.x][ndx.y][ndx.z-1];
	v += svox[ndx.x][ndx.y][ndx.z+1];
	v /= 12.0;*/


extern "C" __global__ void gvdbOpFillF  ( int3 res, uchar chan, float p1, float p2, float p3 )
{
	uint3 vox = blockIdx * make_uint3(blockDim.x, blockDim.y, blockDim.z) + threadIdx + make_uint3(1,1,1);
	if ( vox.x >= res.x|| vox.y >= res.y || vox.z >= res.z ) return;	

	surf3Dwrite ( p1, volOut[chan], vox.x*sizeof(float), vox.y, vox.z );
}
extern "C" __global__ void gvdbOpFillC4 ( int3 res, uchar chan, float p1, float p2, float p3 )
{
	uint3 vox = blockIdx * make_uint3(blockDim.x, blockDim.y, blockDim.z) + threadIdx + make_uint3(1,1,1);
	if ( vox.x >= res.x || vox.y >= res.y || vox.z >= res.z ) return;	

	surf3Dwrite ( make_uchar4(p1*255,p2*255,p3*255,255), volOut[chan], vox.x*sizeof(uchar4), vox.y, vox.z );
}

extern "C" __global__ void gvdbOpSmooth ( int3 res, uchar chan, float p1, float p2, float p3 )
{
	GVDB_COPY_SMEM_F

	//-- smooth
	float v = p1 * svox[ndx.x][ndx.y][ndx.z];
	v += svox[ndx.x-1][ndx.y][ndx.z];
	v += svox[ndx.x+1][ndx.y][ndx.z];
	v += svox[ndx.x][ndx.y-1][ndx.z];
	v += svox[ndx.x][ndx.y+1][ndx.z];
	v += svox[ndx.x][ndx.y][ndx.z-1];
	v += svox[ndx.x][ndx.y][ndx.z+1];
	v = v / (p1 + 6.0) + p2;

	surf3Dwrite ( v, volOut[chan], vox.x*sizeof(float), vox.y, vox.z );
}

extern "C" __global__ void gvdbOpClrExpand ( int3 res, uchar chan, float p1, float p2, float p3 )
{
	GVDB_COPY_SMEM_UC4

	int3 c, cs;
	int cp;
	c = make_int3( svox[ndx.x][ndx.y][ndx.z] );   cs =  c*p1;
	c = make_int3( svox[ndx.x-1][ndx.y][ndx.z] ); cs += c*p2;	
	c = make_int3( svox[ndx.x+1][ndx.y][ndx.z] ); cs += c*p2;
	c = make_int3( svox[ndx.x][ndx.y-1][ndx.z] ); cs += c*p2;
	c = make_int3( svox[ndx.x][ndx.y+1][ndx.z] ); cs += c*p2;
	c = make_int3( svox[ndx.x][ndx.y][ndx.z-1] ); cs += c*p2;
	c = make_int3( svox[ndx.x][ndx.y][ndx.z+1] ); cs += c*p2;
	cp = max(cs.x, max(cs.y,cs.z) );
	cs = (cp > 255) ? make_int3(cs.x*255/cp, cs.y*255/cp, cs.z*255/cp) : cs;

	surf3Dwrite ( make_uchar4(cs.x, cs.y, cs.z, 1), volOut[chan], vox.x*sizeof(uchar4), vox.y, vox.z );
}

extern "C" __global__ void gvdbOpNoise ( int3 res, uchar chan, float p1, float p2, float p3 )
{
	GVDB_COPY_SMEM_F

	//-- noise
	float v = svox[ndx.x][ndx.y][ndx.z];
	if ( v > 0.01 ) v += random(make_float3(vox.x,vox.y,vox.z)) * p1;

	surf3Dwrite ( v, volOut[chan], vox.x*sizeof(float), vox.y, vox.z );

}


/*__device__ bool implicit_func ( int res, uint3 vox )
{
	// Determine world position 	
	int3 i = make_int3( vox / (uint) gvdb.brick_res );
	int3 p = make_int3( vox % (uint) gvdb.brick_res ) - make_int3(gvdb.atlas_apron);			
	VDBAtlasNode* an = getAtlasNodeFromIndex ( i );		
	float3 wpos = ( an->mPos + p + make_float3(0.5,0.5,0.5) ) * gvdb.voxelsize;
	float3 near = make_float3(int((wpos.x+1.0)/2.0)*2.0, int((wpos.y+1.0)/2.0)*2, int((wpos.z+1.0)/2.0)*2 );
	float d = length ( wpos - near );

	return d > ((wpos.y/100.0)*0.75+0.75);

	return (((vox.x-2) % 20 == 0) || ((vox.y-2) % 20 == 0) || ((vox.z-2) % 20 == 0) );
}

	float v = svox[ndx.x][ndx.y][ndx.z];
	if ( v == 0 ) {
		int nbrs = 0;
		nbrs += (fabs(svox[ndx.x-1][ndx.y][ndx.z])==2);
		nbrs += (fabs(svox[ndx.x+1][ndx.y][ndx.z])==2);
		nbrs += (fabs(svox[ndx.x][ndx.y-1][ndx.z])==2);
		nbrs += (fabs(svox[ndx.x][ndx.y+1][ndx.z])==2);
		nbrs += (fabs(svox[ndx.x][ndx.y][ndx.z-1])==2);		
		nbrs += (fabs(svox[ndx.x][ndx.y][ndx.z+1])==2);		
		if ( nbrs > 0 ) {
			v = implicit_func(res, vox) ? 2 : -2;					
			surf3Dwrite ( v, volTexOut, vox.x*sizeof(float), vox.y, vox.z );		
		}
		if ( vox.x == 15 && vox.y == 15 && vox.z == 15 ) {
			v = 2;
			surf3Dwrite ( v, volTexOut, vox.x*sizeof(float), vox.y, vox.z );		
		}
	}*/

