
//----------------------------------------------
// File: cuda_gvdb_scene.cuh
//
// CUDA Scene
// - Scene shading consts
// - Scene structure
// - Scene rays 
//-----------------------------------------------

typedef unsigned char		uchar;
typedef unsigned int		uint;
typedef unsigned short		ushort;
typedef unsigned long long	uint64;
__constant__ float			NOHIT = 1.0e10f;
__constant__ uchar4			BLACK = {0,0,0,0};
#define ALIGN(x)			__align__(x)

// Scene shading consts
#define SHADE_VOXEL		0	
#define SHADE_SECTION2D	1	
#define SHADE_SECTION3D	2
#define SHADE_EMPTYSKIP	3
#define SHADE_TRILINEAR	4
#define SHADE_TRICUBIC	5
#define SHADE_LEVELSET	6
#define SHADE_VOLUME	7

// GVDB Scene Info
struct ALIGN(16) ScnInfo {
	int			width;
	int			height;
	float		camnear;
	float		camfar;
	float3		campos;
	float3		cams;
	float3		camu;
	float3		camv;	
	float4 		camivprow0;
	float4 		camivprow1;
	float4 		camivprow2;
	float4 		camivprow3;
	float3		light_pos;		
	float3		slice_pnt;
	float3		slice_norm;
	float		bias;
	float		shadow_amt;
	char		shading;
	char		filtering;	
	int			frame;
	int			samples;	
	float3		extinct;
	float3		steps;
	float3		cutoff;
	char*		outbuf;
  	char*   	dbuf;
	float4		backclr;
};

struct ALIGN(16) ScnRay {
	float3		hit;
	float3		normal;
	float3		orig;
	float3		dir;	
	uint		clr;
	uint		pnode;
	uint		pndx;
};

#ifdef CUDA_PATHWAY
	__constant__ ScnInfo		scn;					// Scene info

	#define SCN_SHADE			scn.shading
	#define SCN_EXTINCT			scn.extinct.x
	#define SCN_ALBEDO			scn.extinct.y
	#define SCN_PSTEP			scn.steps.x
	#define SCN_SSTEP			scn.steps.y
	#define SCN_FSTEP			scn.steps.z
	#define SCN_MINVAL			scn.cutoff.x
	#define SCN_ALPHACUT		scn.cutoff.y
	#define SCN_SHADOWAMT		scn.shadow_amt
	#define SCN_DBUF			(float*) scn.dbuf
	#define SCN_WIDTH			scn.width
	#define SCN_HEIGHT			scn.height
	#define SCN_BACKCLR			scn.backclr
	#define TRANSFER_FUNC		gvdb.transfer
#endif
#ifdef OPTIX_PATHWAY
	rtBuffer<float4>			scn_transfer_func;		// Scene info
	rtDeclareVariable(uint,		scn_shading, , );
	rtDeclareVariable(float,	scn_shadowamt, , );
	rtDeclareVariable(float4,	scn_backclr, , );
	rtDeclareVariable(float3,	scn_extinct, , );
	rtDeclareVariable(float3,	scn_steps, , );
	rtDeclareVariable(float3,	scn_cutoff, , );
	rtDeclareVariable(int,		scn_width, , );
	rtDeclareVariable(int,		scn_height, , );
	rtBuffer<float>				scn_dbuf;

	#define SCN_DBUF			0x0
    #define SCN_WIDTH			scn_width
    #define SCN_HEIGHT			scn_height
	#define SCN_SHADE			scn_shading		
	#define SCN_EXTINCT			scn_extinct.x
	#define SCN_ALBEDO			scn_extinct.y
	#define SCN_PSTEP			scn_steps.x
	#define SCN_SSTEP			scn_steps.y
	#define SCN_FSTEP			scn_steps.z
	#define SCN_MINVAL			scn_cutoff.x
	#define SCN_ALPHACUT		scn_cutoff.y
	#define SCN_SHADOWAMT		scn_shadowamt
	#define SCN_BACKCLR			scn_backclr
	#define TRANSFER_FUNC		scn_transfer_func
#endif