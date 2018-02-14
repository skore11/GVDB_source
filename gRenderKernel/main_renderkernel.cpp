
/*
-------------------------------------------------------
NVIDIA GVDB Sparse Volumes

Sample: gRenderKernel

Description: This sample shows how to write a custom raytracing
kernel in CUDA to render VBX sparse volume data. 
The user-defined raytracer is written in render_custom.cu and 
compiled as a PTX module, which is specified to GVDB using the 
GVDB::SetModule function. Once specified, the GVDB::RenderKernel 
function accepts the CUfunction for the custom raytracer.

Last Update: Rama Hoetzlein, rhoetzlein@nvidia.com. 7/15/2016
-------------------------------------------------------
*/

#include "gvdb.h"
#include "file_png.h"

#include <stdlib.h>
#include <stdio.h>

#include <cuda.h>

CUmodule		cuCustom;
CUfunction		cuRaycastKernel;

bool cudaCheck ( CUresult status, char* msg )
{
	if ( status != CUDA_SUCCESS ) {
		const char* stat = "";
		cuGetErrorString ( status, &stat );
		printf ( "CUDA ERROR: %s (in %s)\n", stat, msg  );	
		exit(-1);
		return false;
	} 
	return true;
}

int main (int argc, char* argv)
{
	int w = 1024, h = 768;

	VolumeGVDB gvdb;

	// Initialize GVDB
	printf ( "Starting GVDB.\n" );	
	int devid = -1;
	gvdb.SetCudaDevice ( devid );
	gvdb.Initialize ();				
	gvdb.AddPath ( std::string(ASSET_PATH) );

	// Load VBX
	char scnpath[1024];
	if ( !gvdb.FindFile ( "explosion.vbx", scnpath ) ) {
		printf ( "Cannot find vbx files.\n" );
		exit(-1);
	}
	printf ( "Loading VBX. %s\n", scnpath );
	gvdb.LoadVBX ( scnpath );	
	
	gvdb.getScene()->SetVolumeRange ( 0.01f, 1.0f, 0.0f );

	// Create Camera and Light
	Camera3D* cam = new Camera3D;						
	cam->setOrbit ( Vector3DF(20,30,0), Vector3DF(125,160,125), 800, 1.0 );		
	gvdb.getScene()->SetCamera( cam );	
	gvdb.getScene()->SetRes ( w, h );
	
	Light* lgt = new Light;	
	lgt->setOrbit ( Vector3DF(50,65,0), Vector3DF(125,140,125), 200, 1.0 );
	gvdb.getScene()->SetLight ( 0, lgt );		
	
	// Add render buffer 
	printf ( "Creating screen buffer. %d x %d\n", w, h );
	gvdb.AddRenderBuf ( 0, w, h, 4 );					

	// Load custom module and kernel
	printf ( "Loading module: render_custom.ptx\n");
	size_t sz;
	cudaCheck ( cuModuleLoad ( &cuCustom, "render_custom.ptx" ), "cuModuleLoad (render_custom)" );
	cudaCheck ( cuModuleGetFunction ( &cuRaycastKernel, cuCustom, "raycast_kernel" ), "cuModuleGetFunction (raycast_kernel)" );	

	// Set GVDB to custom module 
	gvdb.SetModule ( cuCustom );

	// Render with user-defined kernel
	printf ( "Render custom kernel.\n" );
	gvdb.getScene()->SetSteps ( 0.5, 16, 0.5 );
	gvdb.getScene()->SetVolumeRange ( 0.1, 0.0, 1.0 );
	gvdb.RenderKernel ( 0, cuRaycastKernel, SHADE_TRILINEAR, 0, 0, 1, 1, 1 );	

	// Read render buffer
	unsigned char* buf = (unsigned char*) malloc ( w*h*4 );
	gvdb.ReadRenderBuf ( 0, buf );						

	// Save as png
	printf ( "Saving img_rendkernel.png\n");
	save_png ( "img_rendkernel.png", buf, w, h, 4 );				

	free ( buf );
	delete cam;
	delete lgt;

	printf ( "Done.\n" );	
	#ifdef _WIN32
   	  getchar();
	#endif

 	return 1;
}
