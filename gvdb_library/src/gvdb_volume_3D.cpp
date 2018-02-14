
#include "gvdb_volume_3D.h"
#include "gvdb_render.h"
#include "app_perf.h"

using namespace nvdb;

int Volume3D::mVFBO[2] = {-1, -1};
int Volume3D::mVCLEAR = -1;

Volume3D::Volume3D ( Scene* scn )
{
	mPool = new Allocator;	
	mScene = scn;
}

Volume3D::~Volume3D ()
{
	Clear ();
}

void Volume3D::Resize ( char typ, Vector3DI res, Matrix4F* xform, bool bGL )
{
	mVoxRes = res;				

	mPool->AtlasReleaseAll ();
	mPool->TextureCreate ( 0, typ, res, true, bGL );
}

void Volume3D::SetDomain ( Vector3DF vmin, Vector3DF vmax )
{
	mObjMin = vmin;
	mObjMax = vmax;
}

void Volume3D::Clear ()
{
	mPool->AtlasReleaseAll ();
}

#define max3(a,b,c)		( (a>b) ? ((a>c) ? a : c) : ((b>c) ? b : c) )

void Volume3D::Empty ()
{
	mPool->AtlasFill ( 0 );
}

void Volume3D::CommitFromCPU ( float* src )
{
	mPool->AtlasCommitFromCPU ( 0, (uchar*) src );
}

void Volume3D::RetrieveGL ( char* dest )
{
	#ifdef BUILD_OPENGL
		mPool->AtlasRetrieveGL ( 0, dest );
	#endif 
}

void Volume3D::PrepareRasterGL ( bool start )
{
	#ifdef BUILD_OPENGL
		if ( start ) {
			// Enable opengl state once
			glColorMask (GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
			glDepthMask (GL_FALSE);
			glDisable ( GL_DEPTH_TEST );
			glDisable ( GL_TEXTURE_2D );
			glEnable ( GL_TEXTURE_3D );	

			// Set rasterize program once
			glUseProgram ( GLS_VOXELIZE );
			checkGL ( "glUseProgram(VOX) (PrepareRaster)" );

			// Set raster sampling to major axis						
			int smax = max3(mVoxRes.x, mVoxRes.y, mVoxRes.z);
			glViewport(0, 0, smax, smax );

			// Bind texture
			int glid = mPool->getAtlasGLID ( 0 );
			glActiveTexture ( GL_TEXTURE0 );
			glBindTexture ( GL_TEXTURE_3D, glid );
			checkGL ( "glBindTexture (RasterizeFast)" );

			if ( mVFBO[0] == -1 ) glGenFramebuffers(1, (GLuint*) &mVFBO);
			glBindFramebuffer(GL_FRAMEBUFFER, mVFBO[0] );
			glFramebufferTexture( GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, glid, 0);
				
			switch ( mPool->getAtlas(0).type ) {
			case T_UCHAR:	glBindImageTexture( 0, glid, 0, GL_TRUE, 0, GL_WRITE_ONLY, GL_R8 );		break;
			case T_FLOAT:	glBindImageTexture( 0, glid, 0, GL_TRUE, 0, GL_WRITE_ONLY, GL_R32F );	break;
			};
			checkGL ( "glBindImageTexture (RasterizeFast)" );	

			// Indicate res of major axis
			glProgramUniform1i ( GLS_VOXELIZE, getScene()->getParam(GLS_VOXELIZE, USAMPLES), smax );

		} else {
			// Restore state
			glUseProgram ( 0 );
			glEnable (GL_DEPTH_TEST);
			glFramebufferTexture (GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, 0, 0);
			glBindFramebuffer (GL_FRAMEBUFFER, 0);		
			glDepthMask ( GL_TRUE);
			glColorMask ( GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE); 			
			checkGL ( "glUseProgram(0) (PrepareRaster)" );			
		}
	#endif 
}

void Volume3D::PolyToVoxelsFastGL ( Vector3DF vmin, Vector3DF vmax, Matrix4F* model )
{
	mObjMin = vmin;
	mObjMax = vmax;

	#ifdef BUILD_OPENGL

		// Clear texture. 		
		Empty ();												// using cuda kernel						
		cudaCheck ( cuCtxSynchronize(), "sync(empty)" );		// must sync for opengl to use		
		
		//glClearColor ( 0, 0, 0, 0);			// using FBO, see PrepareRasterGL setup
		//glClear(GL_COLOR_BUFFER_BIT);	

		// Setup transform matrix
		Matrix4F mw;
		mw.Translate ( -mObjMin.x, -mObjMin.y, -mObjMin.z );
		mw *= (*model);
		mw *= Vector3DF( 2.0/(mObjMax.x-mObjMin.x), 2.0/(mObjMax.y-mObjMin.y), 2.0/(mObjMax.z-mObjMin.z) );		
		renderSetUW ( getScene(), GLS_VOXELIZE, &mw, mVoxRes );

		// Rasterize		
		renderSceneGL ( getScene(), GLS_VOXELIZE, false );
		checkGL ( "renderSceneGL (RasterizeFast)" );

		glFinish ();

	#endif	
}


void Volume3D::PolyToVoxelsGL ( uchar chan, Model* model, Matrix4F* xform )
{
	#ifdef BUILD_OPENGL

		// Full rasterize
		// ** Note ** This is slow if called repeatedly, 
		// since it prepares all the necessary opengl/shader state.
		// See PrepareRaster and RasterizeFast above when using for staging.

		// Configure model
		model->ComputeBounds ( *xform, 0.05 );
		mObjMin = model->objMin; mObjMax = model->objMax;
		mVoxMin = mObjMin;	mVoxMin /= mVoxsize;
		mVoxMax = mObjMax;  mVoxMax /= mVoxsize;
		mVoxRes = mVoxMax;	mVoxRes -= mVoxMin;

		// Create atlas if none exists
		mPool->AtlasReleaseAll ();
		mPool->AtlasCreate ( chan, T_FLOAT, mVoxRes, 1, 0, 0, true, true );		

		if ( mVFBO[0] == -1 ) glGenFramebuffers(1, (GLuint*) &mVFBO);

		// Bind frame buffer to 3D texture to clear it
		int glid = mPool->getAtlasGLID ( chan );
		glBindFramebuffer(GL_FRAMEBUFFER, mVFBO[0] );
		glFramebufferTexture (GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, glid, 0);
		glClearColor ( 0, 0, 0, 0);
		glClear(GL_COLOR_BUFFER_BIT);	
		glFramebufferTexture (GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, 0, 0);
		glBindFramebuffer (GL_FRAMEBUFFER, 0);		

		// Set raster sampling to major axis
		int s = max3( mVoxRes.x, mVoxRes.y, mVoxRes.z );
		glViewport(0, 0, s , s );

		// Not using ROP to write to FB maks out all color/depth
		glColorMask (GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
		glDepthMask (GL_FALSE);
		glDisable (GL_DEPTH_TEST);
	
		// Bind 3D texture for write. Layered is set to true for 3D texture
		glEnable ( GL_TEXTURE_3D );
		glActiveTexture ( GL_TEXTURE0 );
		glBindTexture ( GL_TEXTURE_3D, glid );
		glBindImageTexture( 0, glid, 0, GL_TRUE, 0, GL_WRITE_ONLY, GL_R32F );
    
		glUseProgram ( GLS_VOXELIZE );

		glProgramUniform1i ( GLS_VOXELIZE, getScene()->getParam(GLS_VOXELIZE, USAMPLES), s );	// indicate res of major axis
    
		// Send model orientation, scaled to fit in volume
		Matrix4F mw;
		mw.Translate ( -mObjMin.x, -mObjMin.y, -mObjMin.z );
		mw *= (*xform);
		mw *= Vector3DF( 2.0/(mObjMax.x-mObjMin.x), 2.0/(mObjMax.y-mObjMin.y), 2.0/(mObjMax.z-mObjMin.z) );		
		renderSetUW ( getScene(), GLS_VOXELIZE, &mw, mVoxRes );		// this sets uTexRes in shader

		renderSceneGL ( getScene(), GLS_VOXELIZE, false );

		glUseProgram ( 0 );	
	
		// restore screen raster 
		glBindImageTexture (0, 0, 0, GL_TRUE, 0, GL_WRITE_ONLY, GL_R32F );
		glEnable (GL_DEPTH_TEST);
		glDepthMask ( GL_TRUE);
		glColorMask ( GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE); 	
		glViewport ( 0, 0, getScene()->mXres, getScene()->mYres );

	#endif
}

void Volume3D::getMemory ( float& voxels, float& overhead, float& effective )
{
	// all measurements in MB
	voxels = float(mVoxRes.x*mVoxRes.y*mVoxRes.z*4.0) / (1024.0*1024.0);
	overhead = (float) 0.0;
	effective = float(mVoxRes.x*mVoxRes.y*mVoxRes.z*4.0) / (1024.0*1024.0);
}
