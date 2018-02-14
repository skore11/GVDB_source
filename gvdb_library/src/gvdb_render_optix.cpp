

#include "gvdb_render.h"
#include "gvdb_scene.h"
using namespace nvdb;

#ifdef BUILD_OPTIX

	#include <optix.h>
	#include <optixu/optixu.h>
	#include <optixu/optixpp_namespace.h>
	using namespace optix;

	optix::Context	g_OptixContext;
	Group			g_OptixMainGroup;
	TextureSampler	g_OptixVolSampler;
	Program			g_OptixVolIntersectSurfProg;
	Program			g_OptixVolIntersectLevelSetProg;
	Program			g_OptixVolIntersectDeepProg;
	Program			g_OptixVolBBoxProg;
	Program			g_OptixMeshIntersectProg;
	Program			g_OptixMeshBBoxProg;
	std::vector< Transform >		g_OptixVolumes;
	std::vector< Transform >		g_OptixModels;
	std::vector< optix::Material >	g_OptixMats;
	int				g_OptixTex;

	int OPTIX_PROGRAM = -1;	// FTIZB Shader Program
	int OPTIX_VIEW;		
	int OPTIX_PROJ;		
	int OPTIX_MODEL;		
	int OPTIX_LIGHTPOS;	
	int OPTIX_LIGHTTARGET;
	int OPTIX_CLRAMB;
	int OPTIX_CLRDIFF;
	int OPTIX_CLRSPEC;

	optix::Program CreateProgramOptix ( std::string name, std::string prog_func )
	{
		optix::Program program;

		gprintf  ( "  Loading: %s, %s\n", name.c_str(), prog_func.c_str() );
		try { 
			program = g_OptixContext->createProgramFromPTXFile ( name, prog_func );
		} catch (Exception e) {
			gprintf  ( "  OPTIX ERROR: %s \n", g_OptixContext->getErrorString( e.getErrorCode() ).c_str() );
			gerror ();		
		}
		return program;
	}

	Buffer CreateOutputOptix ( RTformat format, unsigned int width, unsigned int height )
	{
		Buffer buffer;

		#ifdef BUILD_OPENGL
			GLuint vbo = 0;
			glGenBuffers (1, &vbo );
			glBindBuffer ( GL_ARRAY_BUFFER, vbo );
			size_t element_size;
			g_OptixContext->checkError( rtuGetSizeForRTformat(format, &element_size));
			glBufferData(GL_ARRAY_BUFFER, element_size * width * height, 0, GL_STREAM_DRAW);
			glBindBuffer(GL_ARRAY_BUFFER, 0);

			buffer = g_OptixContext->createBufferFromGLBO(RT_BUFFER_OUTPUT, vbo);
			buffer->setFormat(format);
			buffer->setSize( width, height );
		#endif

		return buffer;
	}

	void CreateMaterialOptix ( Material material, std::string filename, std::string ch_name, std::string ah_name )
	{
		std::string ptx_file = filename + ".ptx";
		Program ch_program = CreateProgramOptix ( ptx_file, ch_name );
		Program ah_program = CreateProgramOptix ( ptx_file, ah_name );
		material->setClosestHitProgram ( 0, ch_program );
		material->setAnyHitProgram ( 1, ah_program );
	}


	void renderAddShaderOptix ( Scene* scene, char* vertname, char* fragname )
	{
		OPTIX_PROGRAM = scene->AddShader ( vertname, fragname );
		scene->AddParam ( OPTIX_PROGRAM, UVIEW, "uView" );
		scene->AddParam ( OPTIX_PROGRAM, UPROJ, "uProj" );
		scene->AddParam ( OPTIX_PROGRAM, UMODEL, "uModel" );
		scene->AddParam ( OPTIX_PROGRAM, ULIGHTPOS, "uLightPos" );
		//scene->AddParam ( OPTIX_PROGRAM, UCLRAMB, "uClrAmb" );
		//scene->AddParam ( OPTIX_PROGRAM, UCLRDIFF, "uClrDiff" );
		//scene->AddParam ( OPTIX_PROGRAM, UCLRSPEC, "uClrSpec" );

		scene->AddParam ( OPTIX_PROGRAM, UOVERTEX, "uOverTex" );
		scene->AddParam ( OPTIX_PROGRAM, UOVERSIZE, "uOverSize" );
	}

	// Get CUDA device ordinal of given OptiX device
	int renderGetOptixCUDADevice ()
	{
	  std::vector<int> devices = g_OptixContext->getEnabledDevices();
	  unsigned int numOptixDevices = static_cast<unsigned int>( devices.size() );
	  int ordinal;  
	  g_OptixContext->getDeviceAttribute( devices[0], RT_DEVICE_ATTRIBUTE_CUDA_DEVICE_ORDINAL, sizeof(ordinal), &ordinal );
	  return ordinal;  
	}

	void renderAssignSamplerOptix ( int glid )
	{
		if ( g_OptixVolSampler != 0x0 )
			g_OptixVolSampler->destroy();

		gprintf ( "GOT HERE!\n" );
		// Create volume texture sampler for OpenGL interop
		g_OptixVolSampler = g_OptixContext->createTextureSamplerFromGLImage( glid, RT_TARGET_GL_TEXTURE_3D );
		g_OptixVolSampler->setWrapMode( 0, RT_WRAP_CLAMP_TO_EDGE );
		g_OptixVolSampler->setWrapMode( 1, RT_WRAP_CLAMP_TO_EDGE );
		g_OptixVolSampler->setWrapMode( 2, RT_WRAP_CLAMP_TO_EDGE );
		g_OptixVolSampler->setIndexingMode( RT_TEXTURE_INDEX_NORMALIED_COORDINATES );
		g_OptixVolSampler->setReadMode( RT_TEXTURE_READ_ELEMENT_TYPE );	
		g_OptixVolSampler->setFilteringModes( RT_FILTER_LINEAR, RT_FILTER_LINEAR, RT_FILTER_LINEAR );
	
		g_OptixContext[ "volTexIn" ]->setTextureSampler( g_OptixVolSampler );
	}

	void renderAssignGVDBOptix ( void* ctx, std::string name, int vdb_sz, void* vdb_dat )
	{
		optix::Context optix_ctx = *(optix::Context*) (ctx);

		optix_ctx[ name ]->setUserData ( vdb_sz, vdb_dat );
	}

	void renderInitializeOptix ( int w, int h )
	{
		// Create OptiX context
		g_OptixContext = Context::create();
		g_OptixContext->setEntryPointCount ( 1 );
		g_OptixContext->setRayTypeCount( 2 );
		g_OptixContext->setStackSize( 2400 );

		g_OptixContext["scene_epsilon"]->setFloat( 1.0e-6f );	
		g_OptixContext["max_depth"]->setInt( 1 );		

		// Create Output buffer
		Variable outbuf = g_OptixContext["output_buffer"];
		Buffer buffer = CreateOutputOptix( RT_FORMAT_FLOAT3, w, h );
		outbuf->set ( buffer );

		// Camera ray gen and exception program  
		g_OptixContext->setRayGenerationProgram( 0, CreateProgramOptix( "optix_trace_primary.ptx", "trace_primary" ) );
		g_OptixContext->setExceptionProgram(     0, CreateProgramOptix( "optix_trace_primary.ptx", "exception" ) );

		// Used by both exception programs
		g_OptixContext["bad_color"]->setFloat( 0.0f, 1.0f, 1.0f );

		// Assign miss program
		g_OptixContext->setMissProgram( 0, CreateProgramOptix( "optix_trace_miss.ptx", "miss" ) );
		g_OptixContext["background_light"]->setFloat( 1.0f, 1.0f, 1.0f );
		g_OptixContext["background_dark"]->setFloat( 0.3f, 0.3f, 0.3f );

		// Align background's up direction with camera's look direction
		float3 bg_up;  bg_up.x=0; bg_up.y=1; bg_up.z=0;
		g_OptixContext["up"]->setFloat( bg_up.x, bg_up.y, bg_up.z );

		// Random seed buffer
		Buffer rnd_seeds = g_OptixContext->createBuffer( RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL, RT_FORMAT_UNSIGNED_INT, w, h );
		unsigned int* seeds = (unsigned int*) rnd_seeds->map();
		for ( int i=0; i < w*h; i++ ) {
			seeds[i] = rand()*0xffffL / RAND_MAX;
		}
		rnd_seeds->unmap();
		g_OptixContext["rnd_seeds"]->set( rnd_seeds );

		// Initialize mesh intersection program
		g_OptixMeshIntersectProg =	CreateProgramOptix ( "optix_mesh_intersect.ptx", "mesh_intersect" );
		g_OptixMeshBBoxProg	=		CreateProgramOptix ( "optix_mesh_intersect.ptx", "mesh_bounds" );

		// Initialize volume intersection program
		g_OptixVolIntersectSurfProg =	CreateProgramOptix ( "optix_vol_intersect.ptx", "vol_intersect" );
		g_OptixVolIntersectLevelSetProg = CreateProgramOptix ( "optix_vol_intersect.ptx", "vol_levelset" );
		g_OptixVolIntersectDeepProg =	CreateProgramOptix ( "optix_vol_intersect.ptx", "vol_deep" );
		g_OptixVolBBoxProg	=		CreateProgramOptix ( "optix_vol_intersect.ptx", "vol_bounds" );

		if (g_OptixMeshIntersectProg==0)	{ gprintf  ( "Error: Unable to load mesh_intersect program.\n" ); gerror (); }
		if (g_OptixMeshBBoxProg==0)			{ gprintf  ( "Error: Unable to load mesh_bounds program.\n" ); gerror (); }	
		if (g_OptixVolIntersectSurfProg==0)		{ gprintf  ( "Error: Unable to load vol_intersect program.\n" ); gerror (); }
		if (g_OptixVolIntersectLevelSetProg==0)		{ gprintf  ( "Error: Unable to load vol_levelset program.\n" ); gerror (); }
		if (g_OptixVolIntersectDeepProg==0)		{ gprintf  ( "Error: Unable to load vol_deep program.\n" ); gerror (); }
		if (g_OptixVolBBoxProg==0)			{ gprintf  ( "Error: Unable to load vol_bounds program.\n" ); gerror (); }

		// Create main group (no geometry yet)
		g_OptixMainGroup = g_OptixContext->createGroup ();
		g_OptixMainGroup->setChildCount ( 0 );
		g_OptixMainGroup->setAcceleration( g_OptixContext->createAcceleration("Bvh","Bvh") );
	
		g_OptixContext["top_object"]->set( g_OptixMainGroup );

		// Create an output texture for OpenGL
		#ifdef BUILD_OPENGL
			glGenTextures( 1, (GLuint*) &g_OptixTex );
			glBindTexture( GL_TEXTURE_2D, g_OptixTex );	
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);		// Change these to GL_LINEAR for super- or sub-sampling
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);			
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);	// GL_CLAMP_TO_EDGE for linear filtering, not relevant for nearest.
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
			glBindTexture( GL_TEXTURE_2D, 0);
		#endif
	}

	void renderValidateOptix ()
	{
		try {
			g_OptixContext->validate ();
		} catch (const Exception& e) {		
			std::string msg = g_OptixContext->getErrorString ( e.getErrorCode() );		
			gprintf  ( "OPTIX ERROR:\n%s\n", msg.c_str() );
			gerror ();		
		}
		try {
			g_OptixContext->compile ();
		} catch (const Exception& e) {		
			std::string msg = g_OptixContext->getErrorString ( e.getErrorCode() );		
			gprintf  ( "OPTIX ERROR:\n%s\n", msg.c_str() );
			gerror ();		
		}
	}

	void renderClearOptix ()
	{
		for (int n=0; n < g_OptixModels.size(); n++ ) {
			optix::GeometryGroup	geomgroup	= (optix::GeometryGroup) g_OptixModels[n]->getChild<optix::GeometryGroup> ();
			optix::GeometryInstance geominst	= (optix::GeometryInstance) geomgroup->getChild(0);
			optix::Geometry			geom		= (optix::Geometry) geominst->getGeometry ();
			geom->destroy();
			geominst->destroy();
			geomgroup->destroy();
			g_OptixModels[n]->destroy();
		}
		if ( g_OptixModels.size() > 0 ) g_OptixModels.clear ();

		for (int n=0; n < g_OptixVolumes.size(); n++ ) {
			optix::GeometryGroup	geomgroup	= (optix::GeometryGroup) g_OptixVolumes[n]->getChild<optix::GeometryGroup> ();
			optix::GeometryInstance geominst	= (optix::GeometryInstance) geomgroup->getChild(0);
			optix::Geometry			geom		= (optix::Geometry) geominst->getGeometry ();
			geom->destroy();
			geominst->destroy();
			geomgroup->destroy();
			g_OptixVolumes[n]->destroy();
		}
		if ( g_OptixVolumes.size() > 0 ) g_OptixVolumes.clear ();

		for (int n=0; n < g_OptixMats.size(); n++ )
			g_OptixMats[n]->destroy();

		if ( g_OptixMats.size() > 0 ) g_OptixMats.clear ();	
	}


	int renderAddMaterialOptix ( Scene* scene, std::string cast_prog, std::string shadow_prog, std::string name )
	{
		// Create Optix material
		optix::Material omat = g_OptixContext->createMaterial();
		int oid = g_OptixMats.size();

		// Add material to scene 
		int mid = scene->AddMaterial ();
		scene->SetMaterialParam ( mid, 0, Vector3DF(oid, 0, 0) );	// Link to optix material id

		CreateMaterialOptix ( omat, name, cast_prog, shadow_prog );

		omat["importance_cutoff"  ]->setFloat( 0.01f );
		omat["cutoff_color"       ]->setFloat( 0.1f, 0.1f, 0.1f );
		omat["reflection_maxdepth"]->setInt( 1 );  
		omat["reflection_color"   ]->setFloat( 0.2f, 0.2f, 0.2f );  
		omat["shadow_attenuation"]->setFloat( 1.0f, 1.0f, 1.0f );
		
		g_OptixMats.push_back ( omat );	

		return oid;
	}

	void renderAddVolumeOptix ( Vector3DF vmin, Vector3DF vmax, int mat_id, Matrix4F& xform, bool deep, bool lset )
	{
		int id = g_OptixModels.size() + g_OptixVolumes.size();

		int num_bricks = 1;

		//------------------ Per-brick
		// Brick buffer	
		Buffer bbuffer = g_OptixContext->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, num_bricks*2 );
		float3* bbuffer_data = static_cast<float3*>( bbuffer->map() );

		//Buffer mbuffer = g_OptixContext->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_INT, num_bricks );
		//unsigned int* mbuffer_data = static_cast<unsigned int*>( mbuffer->map() );

		// Copy brick data		
		bbuffer_data[0] = * (float3*) &vmin;		// Cast Vector3DF to float3. Assumes both are 3x floats
		bbuffer_data[1] = * (float3*) &vmax;
	
		//mbuffer_data[0] = mat_id;

		// Model definition
		//    Transform 
		//        |
		//   GeometryGroup -- Acceleration 
		//        |
		//  GeometryInstance
		//        |
		//     Geometry -- Intersect Prog/BBox Prog

		// Geometry node
		optix::Geometry geom;
		geom = g_OptixContext->createGeometry ();
		geom->setPrimitiveCount ( 1 );	
		if ( deep ) {
			geom->setIntersectionProgram ( g_OptixVolIntersectDeepProg );
		} else {
			if ( lset ) 
				geom->setIntersectionProgram ( g_OptixVolIntersectLevelSetProg );
			else
				geom->setIntersectionProgram ( g_OptixVolIntersectSurfProg );
		}
		geom->setBoundingBoxProgram ( g_OptixVolBBoxProg );
	
		geom[ "brick_buffer" ]->setBuffer( bbuffer );
		geom[ "mat_id"]->setUint( 0 );
		//geom[ "mat_bufffer" ]->setBuffer( mbuffer );
    
		// Unmap buffers
		bbuffer->unmap();		
		//mbuffer->unmap();

		// Geometry Instance node
		Material mat;
		mat = g_OptixMats[ mat_id ];
		GeometryInstance geominst = g_OptixContext->createGeometryInstance ( geom, &mat, &mat+1 );		// <-- geom is specified as child here

		// Geometry Group node
		GeometryGroup geomgroup;
		geomgroup = g_OptixContext->createGeometryGroup ();		

		const char* Builder = "Sbvh";
		const char* Traverser = "Bvh";
		const char* Refine = "0";
	//	const char* Refit = "1";
		optix::Acceleration acceleration = g_OptixContext->createAcceleration( Builder, Traverser );
	//	acceleration->setProperty( "refit", Refit );
		acceleration->setProperty( "refine", Refine );
		if ( Builder   == std::string("Sbvh") || Builder == std::string("TriangleKdTree") || Traverser == std::string( "KdTree" )) {
		  acceleration->setProperty( "vertex_buffer_name", "vertex_buffer" );
		  acceleration->setProperty( "index_buffer_name", "vindex_buffer" );
		}
		acceleration->markDirty();
		geomgroup->setAcceleration( acceleration );	
		geomgroup->setChildCount ( 1 );
		geomgroup->setChild( 0, geominst );

		// Transform node
		Transform tform = g_OptixContext->createTransform ();
		tform->setMatrix ( true, xform.GetDataF(), 0x0 );	
		tform->setChild ( geomgroup );
	
		// Add model root (Transform) to the Main Group
		g_OptixMainGroup->setChildCount ( id+1 );
		g_OptixMainGroup->setChild ( id, tform );

		g_OptixVolumes.push_back ( tform );
	}

	void renderAddModelOptix ( Model* model, int mat_id, Matrix4F& xform )
	{
		int id = g_OptixModels.size() + g_OptixVolumes.size();
	
		int num_vertices = model->vertCount;
		int num_triangles = model->elemCount;
		int num_normals = num_vertices;
	
		//------------------ Per-vertex
		// Vertex buffer
		Buffer vbuffer = g_OptixContext->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, num_vertices );
		float3* vbuffer_data = static_cast<float3*>( vbuffer->map() );

		// Normal buffer
		Buffer nbuffer = g_OptixContext->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, num_normals );
		float3* nbuffer_data = static_cast<float3*>( nbuffer->map() );

		// Texcoord buffer
		Buffer tbuffer = g_OptixContext->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_FLOAT2, num_vertices );
		float2* tbuffer_data = static_cast<float2*>( tbuffer->map() );

		//------------------ Per-triangle
		// Vertex index buffer
		Buffer vindex_buffer = g_OptixContext->createBuffer ( RT_BUFFER_INPUT, RT_FORMAT_INT3, num_triangles );
		int3* vindex_data = static_cast<int3*>( vindex_buffer->map() );

		// Normal index buffer
		Buffer nindex_buffer = g_OptixContext->createBuffer ( RT_BUFFER_INPUT, RT_FORMAT_INT3, num_triangles );
		int3* nindex_data = static_cast<int3*>( nindex_buffer->map() );

		// Material id buffer 
		Buffer mindex_buffer = g_OptixContext->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_INT, num_triangles );
		unsigned int* mindex_data = static_cast<unsigned int*>( mindex_buffer->map() );

		// Copy vertex data
		float2 tc;
		tc.x = 0; tc.y = 0;
		char* vdat = (char*) model->vertBuffer;
		float3* v3;
		Vector4DF vert;
		for (int i=0; i < num_vertices; i++ ) {
			v3 = (float3*) (vdat + model->vertOffset);
			vbuffer_data[i] = *v3;
			v3 = (float3*) (vdat + model->normOffset);
			nbuffer_data[i] = *v3;
			tbuffer_data[i] = tc;
			vdat += model->vertStride;
		}	

		// Copy element data (indices)
		for (int i=0; i < num_triangles; i++ ) {
			int3 tri_verts;		// vertices in trangle
			tri_verts.x = model->elemBuffer[ i*3   ];
			tri_verts.y = model->elemBuffer[ i*3+1 ];
			tri_verts.z = model->elemBuffer[ i*3+2 ];
			vindex_data [ i ] = tri_verts;
			nindex_data [ i ] = tri_verts;
			mindex_data [ i ] = 0;
		}

		// Model definition
		//    Transform 
		//        |
		//   GeometryGroup -- Acceleration 
		//        |
		//  GeometryInstance
		//        |
		//     Geometry -- Intersect Prog/BBox Prog

		// Geometry node
		optix::Geometry geom;
		geom = g_OptixContext->createGeometry ();
		geom->setPrimitiveCount ( num_triangles );
		geom->setIntersectionProgram ( g_OptixMeshIntersectProg );
		geom->setBoundingBoxProgram ( g_OptixMeshBBoxProg );
	
		geom[ "vertex_buffer" ]->setBuffer( vbuffer );			// num verts
		geom[ "normal_buffer" ]->setBuffer( nbuffer );	
		geom[ "texcoord_buffer" ]->setBuffer( tbuffer );	

		geom[ "vindex_buffer" ]->setBuffer( vindex_buffer );	// num tris
		geom[ "nindex_buffer" ]->setBuffer( nindex_buffer );
		geom[ "tindex_buffer" ]->setBuffer( nindex_buffer );
		geom[ "mindex_buffer" ]->setBuffer( mindex_buffer );

		// Unmap buffers
		vbuffer->unmap();	
		nbuffer->unmap();
		tbuffer->unmap();
		vindex_buffer->unmap();
		nindex_buffer->unmap();
		//tindex_buffer->unmap();
		mindex_buffer->unmap();

		// Geometry Instance node
		Material mat;
		mat = g_OptixMats[ mat_id ];
		GeometryInstance geominst = g_OptixContext->createGeometryInstance ( geom, &mat, &mat+1 );		// <-- geom is specified as child here

		// Geometry Group node
		GeometryGroup geomgroup;
		geomgroup = g_OptixContext->createGeometryGroup ();		

		const char* Builder = "Sbvh";
		const char* Traverser = "Bvh";
		const char* Refine = "0";
		optix::Acceleration acceleration = g_OptixContext->createAcceleration( Builder, Traverser );
		acceleration->setProperty( "refine", Refine );
		if ( Builder   == std::string("Sbvh") || Builder == std::string("TriangleKdTree") || Traverser == std::string( "KdTree" )) {
		  acceleration->setProperty( "vertex_buffer_name", "vertex_buffer" );
		  acceleration->setProperty( "index_buffer_name", "vindex_buffer" );
		}
		acceleration->markDirty();
		geomgroup->setAcceleration( acceleration );	
		geomgroup->setChildCount ( 1 );
		geomgroup->setChild( 0, geominst );

		// Transform node
		Transform tform = g_OptixContext->createTransform ();
		tform->setMatrix ( true, xform.GetDataF(), 0x0 );	
		tform->setChild ( geomgroup );
	
		// Add model root (Transform) to the Main Group
		g_OptixMainGroup->setChildCount ( id+1 );
		g_OptixMainGroup->setChild ( id, tform );

		g_OptixModels.push_back ( tform );
	}

	Buffer getOptixBuffer ()
	{
		return g_OptixContext["output_buffer"]->getBuffer();
	}

	#define RAD2DEG  57.2957795131
	#define DEG2RAD  0.01745329251

	float renderOptix ( Scene* scene, int rend, int frame, int sample, int spp, int msz, void* mdat )
	{
		// Set camera params for Optix
		Camera3D* cam = scene->getCamera();
		Vector3DF eye = cam->inverseRay ( 0, 0, cam->getNear() ); 
		Vector3DF U   = cam->inverseRay ( 1, 0, cam->getNear() ); U -= eye;
		Vector3DF V   = cam->inverseRay ( 0, 1, cam->getNear() ); V -= eye;

		g_OptixContext["cam_pos"]->setFloat ( cam->getPos().x, cam->getPos().y, cam->getPos().z );
		g_OptixContext["U"]->setFloat ( U.x, U.y, U.z );
		g_OptixContext["V"]->setFloat ( V.x, V.y, V.z );
		g_OptixContext["W"]->setFloat ( -cam->getW().x, -cam->getW().y, -cam->getW().z );
		g_OptixContext["frame_number"]->setUint ( frame );	
		g_OptixContext["sample"]->setUint ( sample );
		g_OptixContext["mat"]->setUserData ( msz, mdat );

		// Set light params for Optix
		Light* light = scene->getLight();		
		g_OptixContext["light_pos"]->setFloat ( light->getPos().x, light->getPos().y, light->getPos().z );

		// Get buffer size
		Buffer buffer = getOptixBuffer();
		RTsize bw, bh;	
		buffer->getSize ( bw, bh );	

		// Launch Optix render
		// entry point 0 - pinhole camera
		try {
			g_OptixContext->launch ( 0, (int) bw, (int) bh );
		} catch (const Exception& e) {		
			std::string msg = g_OptixContext->getErrorString ( e.getErrorCode() );		
			gprintf  ( "OPTIX ERROR:\n%s\n", msg.c_str() );
			gerror ();		
		}

		// Transfer output to OpenGL texture
		#ifdef BUILD_OPENGL
			glBindTexture( GL_TEXTURE_2D, g_OptixTex );	

			int vboid = buffer->getGLBOId ();
			glBindBuffer ( GL_PIXEL_UNPACK_BUFFER, vboid );		// Bind to the optix buffer
		
			RTsize elementSize = buffer->getElementSize();
			if      ((elementSize % 8) == 0) glPixelStorei(GL_UNPACK_ALIGNMENT, 8);
			else if ((elementSize % 4) == 0) glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
			else if ((elementSize % 2) == 0) glPixelStorei(GL_UNPACK_ALIGNMENT, 2);
			else                             glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

			// Copy the OptiX results into a GL texture
			//  (device-to-device transfer using bound gpu buffer)
			RTformat buffer_format = buffer->getFormat();
			switch (buffer_format) {
			case RT_FORMAT_UNSIGNED_BYTE4:	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8,			bw, bh, 0, GL_BGRA, GL_UNSIGNED_BYTE, 0 );	break;
			case RT_FORMAT_FLOAT4:			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F_ARB,		bw, bh, 0, GL_RGBA, GL_FLOAT, 0);	break;
			case RT_FORMAT_FLOAT3:			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F_ARB,		bw, bh, 0, GL_RGB, GL_FLOAT, 0);		break;
			case RT_FORMAT_FLOAT:			glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE32F_ARB, bw, bh, 0, GL_LUMINANCE, GL_FLOAT, 0);	break;	
			}
			glBindBuffer( GL_PIXEL_UNPACK_BUFFER, 0 );	
			glBindTexture( GL_TEXTURE_2D, 0);
		#endif

		//-- Debugging: Pass in a known buffer
		/*char* pix = (char*) malloc ( bw*bh*4 );
		for (int n=0; n < bw*bh*4; n+=4 ) {
			pix[n+0] = rand()*255/RAND_MAX;		// b
			pix[n+1] = 0;		// g
			pix[n+2] = 0;		// r
			pix[n+3] = 255;		// a
		}
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8,			bw, bh, 0, GL_BGRA, GL_UNSIGNED_BYTE, pix );	
		free ( pix );   */

		return 1;
	}

	int renderGetOptixGLID ()
	{
		return g_OptixTex;
	}

	void renderSetupOptixGL ( Scene* scene, int prog )
	{
		#ifdef BUILD_OPENGL
			// OpenGL specific code to bind the 
			// optix GL texture to the GLSL shader
			Buffer buffer = getOptixBuffer();
			RTsize bw, bh;	
			buffer->getSize ( bw, bh );	
			int sz[2] = { bw, bh };
			glProgramUniform2iv( prog, scene->getParam(prog, UOVERSIZE), 1, sz );     // Set value for "renderSize" uniform in the shader
	
			glProgramUniform1i ( prog, scene->getParam(prog, UOVERTEX), 0 );
			glActiveTexture ( GL_TEXTURE0 );
			glBindTexture ( GL_TEXTURE_2D, g_OptixTex );
		#endif
	}

#endif
