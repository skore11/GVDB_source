//--------------------------------------------------------------------------------
//
// Sample:		gRenderToFile
// 
// NVIDIA GVDB Sparse Volumes
// Copyright (c) 2015, NVIDIA. All rights reserved.
//
// Description: This sample demonstrates basic loading and rendering 
// a VBX volume file to a png image. Using CUDA for rendering, 
// the sample runs as a console mode app, and does not need a graphics api. 
// Verbose output from GVDB is enabled/disables with the SetVerbose function. 
// Output prints the details of the GVDB data structure for the input data.
// 
// Created: Rama Hoetzlein, rhoetzlein@nvidia.com. 7/15/2016
//
//---------------------------------------------------------------------------------

#include "gvdb.h"
using namespace nvdb;

#include <stdlib.h>
#include <stdio.h>

#include "file_png.h"		// sample utils

int main (int argc, char* argv)
{
	int w = 1024, h = 768;

	VolumeGVDB gvdb;

	// Initialize GVDB
	printf ( "Starting GVDB.\n" );	
	int devid = -1;	
	gvdb.SetVerbose ( true );		// enable/disable console output from gvdb
	gvdb.SetCudaDevice ( devid );
	gvdb.Initialize ();			
	gvdb.AddPath ( std::string(ASSET_PATH) );

	// Load VBX
	char scnpath[1024];		
	if ( !gvdb.FindFile ( "explosion.vbx", scnpath ) ) {
		printf ( "Cannot find vbx file.\n" );
		exit (-1);
	}
	printf ( "Loading VBX. %s\n", scnpath );
	gvdb.LoadVBX ( scnpath );							// Load VBX

	// Set volume params
	gvdb.getScene()->SetSteps ( 0.5, 16, 0 );			// Set raycasting steps
	gvdb.getScene()->SetExtinct ( -1.0f, 1.5f, 0.0f );		// Set volume extinction
	gvdb.getScene()->SetVolumeRange ( 0.1f, 1.0f, 0.0f );		// Set volume value range
	gvdb.getScene()->SetCutoff ( 0.005f, 0.01f, 0.0f );
	gvdb.getScene()->LinearTransferFunc ( 0.00f, 0.25f, Vector4DF(1,1,0,0.05), Vector4DF(1,1,0,0.03) );
	gvdb.getScene()->LinearTransferFunc ( 0.25f, 0.50f, Vector4DF(1,1,1,0.03), Vector4DF(1,0,0,0.02) );
	gvdb.getScene()->LinearTransferFunc ( 0.50f, 0.75f, Vector4DF(1,0,0,0.02), Vector4DF(1,.5,0,0.01) );
	gvdb.getScene()->LinearTransferFunc ( 0.75f, 1.00f, Vector4DF(1,.5,0,0.01), Vector4DF(0,0,0,0.005) );
	gvdb.getScene()->SetBackgroundClr ( 0.1, 0.2, 0.4, 1.0 );
	gvdb.CommitTransferFunc ();

	Camera3D* cam = new Camera3D;						// Create Camera 
	cam->setFov ( 50.0 );
	cam->setOrbit ( Vector3DF(20,30,0), Vector3DF(125,160,125), 500, 1.0 );	
	gvdb.getScene()->SetCamera( cam );	
	gvdb.getScene()->SetRes ( w, h );
	
	Light* lgt = new Light;								// Create Light
	lgt->setOrbit ( Vector3DF(299,57.3,0), Vector3DF(132,-20,50), 200, 1.0 );
	gvdb.getScene()->SetLight ( 0, lgt );		
	
	printf ( "Creating screen buffer. %d x %d\n", w, h );
	gvdb.AddRenderBuf ( 0, w, h, 4 );					// Add render buffer 

	gvdb.TimerStart ();
	gvdb.Render ( 0, SHADE_VOLUME, 0, 0, 1, 1, 1 );			// Render as volume
	float rtime = gvdb.TimerStop();
	printf ( "Render volume. %6.3f ms\n", rtime );

	printf ( "Writing img_rendfile.png\n" );
	unsigned char* buf = (unsigned char*) malloc ( w*h*4 );
	gvdb.ReadRenderBuf ( 0, buf );						// Read render buffer

	save_png ( "img_rendfile.png", buf, w, h, 4 );				// Save as png

	free ( buf );
	delete cam;
	delete lgt;

	printf ( "Done.\n" );	
	#ifdef _WIN32
  	  getchar();
	#endif

	  return 1;
}
