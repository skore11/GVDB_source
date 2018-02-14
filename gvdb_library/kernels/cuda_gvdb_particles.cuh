
//----------------------------------------------
// File: cuda_gvdb_particles.cu
//
// GVDB Points
// - ClearNodeCounts	- clear brick particle counts
// - InsertPoints		- insert points into bricks
// - SplatPoints		- splat points into bricks
//-----------------------------------------------

extern "C" __global__ void gvdbInsertPoints ( int num_pnts, char* ppos, int pos_off, int pos_stride, int* pnode, int* poff, int* gcnt, float3 ptrans )
{
	uint i = blockIdx.x * blockDim.x + threadIdx.x;
	if ( i >= num_pnts ) return;

	float3 wpos = (*(float3*) (ppos + i*pos_stride + pos_off)); // NOTE: +ptrans is below. Allows check for wpos.z==NOHIT 

	if ( wpos.z == NOHIT ) { pnode[i] = ID_UNDEFL; return; }		// If position invalid, return. 
	float3 offs, vmin, vdel;										// Get GVDB node at the particle point
	uint64 nid;
	VDBNode* node = getNodeAtPoint ( wpos + ptrans, &offs, &vmin, &vdel, &nid );	
	if ( node == 0x0 ) { pnode[i] = ID_UNDEFL; return; }			// If no brick at location, return.	

	__syncthreads();

	pnode[i] = nid;													// Place point in brick
	poff[i] = atomicAdd ( &gcnt[nid], (uint) 1 );					// Increment brick pcount, and retrieve this point index at the same time
}

inline __device__ float distFunc ( float3 a, float bx, float by, float bz, float r )
{
	bx -= a.x; by -= a.y; bz -= a.z;	
	return (r - sqrt(bx*bx+by*by+bz*bz)) / r;
}

extern "C" __global__ void gvdbSplatPoints ( int num_pnts,  float radius, float amp, char* ppos, int pos_off, int pos_stride, char* pclr, int clr_off, int clr_stride, int* pnode, float3 ptrans, bool expand, uint* colorBuf)
{
	uint i = blockIdx.x * blockDim.x + threadIdx.x;
	if ( i >= num_pnts ) return;

	// Get brick ID	
	uint  nid = pnode[i];
	if ( nid == ID_UNDEFL ) return;
	
	// Get particle position in brick	
	float3 wpos = (*(float3*) (ppos + i*pos_stride + pos_off)) + ptrans;	
	float3 vmin;
	VDBNode* node = getNode ( 0, pnode[i], &vmin );			// Get node	
	float3 p = (wpos-vmin)/gvdb.vdel[0];
	float3 pi = make_float3(int(p.x), int(p.y), int(p.z));

	// -- should be ok that pi.x,pi.y,pi.z = 0 
	if ( pi.x <= -1 || pi.y <= -1 || pi.z <= -1 || pi.x >= gvdb.res[0] || pi.y >= gvdb.res[0] || pi.z >= gvdb.res[0] ) 
		return;

	uint3 q = make_uint3(pi.x,pi.y,pi.z) + make_uint3( node->mValue );	
	float v = max( tex3D<float>( volIn[0], q.x,q.y,q.z ), amp * distFunc(p, pi.x, pi.y,pi.z,radius) );
	surf3Dwrite ( v, volOut[0], q.x*sizeof(float), q.y, q.z );

	if ( expand ) {
		float w[6];
		w[0] = tex3D<float> (volIn[0], q.x-1,q.y,q.z) + distFunc(p, pi.x-1, pi.y, pi.z, radius);
		w[1] = tex3D<float> (volIn[0], q.x+1,q.y,q.z) + distFunc(p, pi.x+1, pi.y, pi.z, radius); 
		w[2] = tex3D<float> (volIn[0], q.x,q.y-1,q.z) + distFunc(p, pi.x, pi.y-1, pi.z, radius);
		w[3] = tex3D<float> (volIn[0], q.x,q.y+1,q.z) + distFunc(p, pi.x, pi.y+1, pi.z, radius); 
		w[4] = tex3D<float> (volIn[0], q.x,q.y,q.z-1) + distFunc(p, pi.x, pi.y, pi.z-1, radius); 
		w[5] = tex3D<float> (volIn[0], q.x,q.y,q.z+1) + distFunc(p, pi.x, pi.y, pi.z+1, radius);	
		surf3Dwrite ( w[0], volOut[0], (q.x-1)*sizeof(float), q.y, q.z );
		surf3Dwrite ( w[1], volOut[0], (q.x+1)*sizeof(float), q.y, q.z );
		surf3Dwrite ( w[2], volOut[0], q.x*sizeof(float), (q.y-1), q.z );
		surf3Dwrite ( w[3], volOut[0], q.x*sizeof(float), (q.y+1), q.z );
		surf3Dwrite ( w[4], volOut[0], q.x*sizeof(float), q.y, (q.z-1) );
		surf3Dwrite ( w[5], volOut[0], q.x*sizeof(float), q.y, (q.z+1) );
	}

	if ( pclr != 0 ) {
		uchar4 wclr = *(uchar4*) (pclr + i*clr_stride + clr_off );

		if ( colorBuf != 0 ) {
			
			// Increment index
			uint brickres = gvdb.res[0];
			uint vid = (brickres * brickres * brickres * nid) + (brickres * brickres * (uint)pi.z) + (brickres * (uint)pi.y) + (uint)pi.x;
			uint colorIdx = vid * 4;
		
			// Store in color in the colorbuf
			atomicAdd(&colorBuf[colorIdx + 0], 1);
			atomicAdd(&colorBuf[colorIdx + 1], wclr.x);
			atomicAdd(&colorBuf[colorIdx + 2], wclr.y);
			atomicAdd(&colorBuf[colorIdx + 3], wclr.z);
		}
		else {
			surf3Dwrite(wclr, volOut[1], q.x*sizeof(uchar4), q.y, q.z);
		}
	}
}

extern "C" __global__ void gvdbSplatPointsAvgCol(int num_voxels, uint* colorBuf)
{
  uint vid = blockIdx.x * blockDim.x + threadIdx.x;
  if (vid >= num_voxels) return;

  uint colorIdx = vid * 4;
  uint count = colorBuf[colorIdx + 0];
  if (count > 0)
  {
    // Average color dividing by count
    uint colx = colorBuf[colorIdx + 1] / count;
    uint coly = colorBuf[colorIdx + 2] / count;
    uint colz = colorBuf[colorIdx + 3] / count;
    uchar4 pclr = make_uchar4(colx, coly, colz, 255);

    // Get node
    uint brickres = gvdb.res[0];
    uint nid = vid / (brickres * brickres * brickres);
    float3 vmin;
    VDBNode* node = getNode(0, nid, &vmin);

    // Get local 3d indices
    uint3 pi;
    pi.x = vid % (brickres);
    pi.y = vid % (brickres * brickres) / (brickres);
    pi.z = vid % (brickres * brickres * brickres) / (brickres * brickres);
    
    // Get global atlas index
    uint3 q = make_uint3(pi.x, pi.y, pi.z) + make_uint3(node->mValue);
    
    surf3Dwrite(pclr, volOut[1], q.x*sizeof(uchar4), q.y, q.z);
  }
}

	


#define SCAN_BLOCKSIZE		512

extern "C" __global__ void prefixFixup ( uint *input, uint *aux, int len) 
{
    unsigned int t = threadIdx.x;
	unsigned int start = t + 2 * blockIdx.x * SCAN_BLOCKSIZE; 	
	if (start < len)					input[start] += aux[blockIdx.x] ;
	if (start + SCAN_BLOCKSIZE < len)   input[start + SCAN_BLOCKSIZE] += aux[blockIdx.x];
}

extern "C" __global__ void prefixSum ( uint* input, uint* output, uint* aux, int len, int zeroff )
{
    __shared__ uint scan_array[SCAN_BLOCKSIZE << 1];    
	unsigned int t1 = threadIdx.x + 2 * blockIdx.x * SCAN_BLOCKSIZE;
	unsigned int t2 = t1 + SCAN_BLOCKSIZE;
    
	// Pre-load into shared memory
    scan_array[threadIdx.x] = (t1<len) ? input[t1] : 0.0f;
	scan_array[threadIdx.x + SCAN_BLOCKSIZE] = (t2<len) ? input[t2] : 0.0f;
    __syncthreads();

    // Reduction
    int stride;
    for (stride = 1; stride <= SCAN_BLOCKSIZE; stride <<= 1) {
       int index = (threadIdx.x + 1) * stride * 2 - 1;
       if (index < 2 * SCAN_BLOCKSIZE)
          scan_array[index] += scan_array[index - stride];
       __syncthreads();
    }

    // Post reduction
    for (stride = SCAN_BLOCKSIZE >> 1; stride > 0; stride >>= 1) {
       int index = (threadIdx.x + 1) * stride * 2 - 1;
       if (index + stride < 2 * SCAN_BLOCKSIZE)
          scan_array[index + stride] += scan_array[index];
       __syncthreads();
    }
	__syncthreads();
	
	// Output values & aux
	if (t1+zeroff < len)	output[t1+zeroff] = scan_array[threadIdx.x];
	if (t2+zeroff < len)	output[t2+zeroff] = (threadIdx.x==SCAN_BLOCKSIZE-1 && zeroff) ? 0 : scan_array[threadIdx.x + SCAN_BLOCKSIZE];	
	if ( threadIdx.x == 0 ) {
		if ( zeroff ) output[0] = 0;
		if (aux) aux[blockIdx.x] = scan_array[2 * SCAN_BLOCKSIZE - 1];				
	}    	
}

extern "C" __global__ void gvdbInsertTriangles ( float bdiv, int bmax, int* bcnt, int vcnt, int ecnt, float3* vbuf, int* ebuf )
{
	uint n = blockIdx.x * blockDim.x + threadIdx.x;
	if ( n >= ecnt ) return;

	// get transformed triangle
	float3 v0, v1, v2;
	int3 f = make_int3( ebuf[n*3], ebuf[n*3+1], ebuf[n*3+2] );
	v0 = vbuf[f.x << 1]; v0 = mul4x ( v0, cxform );
	v1 = vbuf[f.y << 1]; v1 = mul4x ( v1, cxform );
	v2 = vbuf[f.z << 1]; v2 = mul4x ( v2, cxform );

	// compute bounds on y-axis	
	float p0, p1;
	fminmax3( v0.y, v1.y, v2.y, p0, p1 );
	p0 = int(p0/bdiv);	p1 = int(p1/bdiv);							// y-min and y-max bins
	
	// scan bins covered by triangle	
	for (int y=p0; y <= p1; y++) {
		atomicAdd ( &bcnt[y], (uint) 1 );							// histogram bin counts
	}	
}

// Sort triangles
// Give a list of bins and known offsets (prefixes), and a list of vertices and faces,
// performs a deep copy of triangles into bins, where some may be duplicated.
// This may be used generically by others kernel that need a bin-sorted mesh.
// Input: 
//   bdiv, bmax - bins division and maximum number
//   bcnt       - number of triangles in each bin
//   boff       - starting offset of each bin in triangle buffer
//   tricnt     - total number of triangles when sorted into bins
//   tbuf       - triangle buffer: list of bins and their triangles (can be more than vcnt due to overlaps)
//   vcnt, vbuf - vertex buffer (VBO) and number of verts
//   ecnt, ebuf - element buffer and number of faces
extern "C" __global__ void gvdbSortTriangles ( float bdiv, int bmax, int* bcnt, int* boff, int tricnt, float3* tbuf,
													int vcnt, int ecnt, float3* vbuf, int* ebuf )
{
	uint n = blockIdx.x * blockDim.x + threadIdx.x;
	if ( n >= ecnt ) return;

	// get transformed triangle
	float3 v0, v1, v2;
	int3 f = make_int3( ebuf[n*3], ebuf[n*3+1], ebuf[n*3+2] );
	v0 = vbuf[f.x << 1]; v0 = mul4x ( v0, cxform );
	v1 = vbuf[f.y << 1]; v1 = mul4x ( v1, cxform );
	v2 = vbuf[f.z << 1]; v2 = mul4x ( v2, cxform );

	// compute bounds on y-axis	
	float p0, p1;
	fminmax3( v0.y, v1.y, v2.y, p0, p1 );
	p0 = int(p0/bdiv);	p1 = int(p1/bdiv);							// y-min and y-max bins
	
	// scan bins covered by triangle	
	int bndx;
	for (int y=p0; y <= p1; y++) {
		bndx = atomicAdd ( &bcnt[y], (uint) 1 );		// get bin index (and histogram bin counts)
		bndx += boff[y];								// get offset into triangle buffer (tbuf)		
		tbuf[ bndx*3   ] = v0;							// deep copy transformed vertices of face
		tbuf[ bndx*3+1 ] = v1;
		tbuf[ bndx*3+2 ] = v2;
	}	
}