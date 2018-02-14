
//----------------------------------------------
// File: cuda_gvdb_dda.cuh
//
// DDA header
// - Transfer function 
// - DDA stepping macros
//-----------------------------------------------

// Tranfser function
inline __device__ float4 transfer ( float v )
{
	return TRANSFER_FUNC [ int(min( 1.0, max( 0.0,  (v - gvdb.thresh.y) / (gvdb.thresh.z - gvdb.thresh.y) ) ) * 16300.0f) ];
}

// Prepare DDA - This macro sets up DDA variables to start stepping sequence
#define PREPARE_DDA {											\
	p = ( pos + t.x*dir - vmin) / gvdb.vdel[lev];				\
	tDel = fabs3 ( gvdb.vdel[lev] / dir );						\
	tSide	= (( floor3(p) - p + 0.5)*pStep+0.5) * tDel + t.x;	\
	p = floor3(p);												\
}
#define PREPARE_DDA_LEAF {										\
	p = ( pos + t.x*dir - vmin) / gvdb.vdel[0];					\
	tDel = fabs3 ( gvdb.vdel[0] / dir );						\
	tSide	= (( floor3(p) - p + 0.5)*pStep+0.5) * tDel;		\
	p = floor3(p);												\
}

// Next DDA - This macro computes the next time step from DDA
#define NEXT_DDA {														\
	mask.x = float ( (tSide.x < tSide.y) & (tSide.x <= tSide.z) );		\
	mask.y = float ( (tSide.y < tSide.z) & (tSide.y <= tSide.x) );		\
	mask.z = float ( (tSide.z < tSide.x) & (tSide.z <= tSide.y) );		\
	t.y = mask.x ? tSide.x : (mask.y ? tSide.y : tSide.z);				\
}					
// Step DDA - This macro advances the DDA to next point
#define STEP_DDA {						\
	t.x = t.y;							\
	tSide	+= mask * tDel;				\
	p		+= mask * pStep;			\
}
