

// GVDB library
#include "gvdb.h"			
using namespace nvdb;

// Sample utils
#include "main.h"			// window system 
#include "nv_gui.h"			// gui system
#include <GL/glew.h>
#include <algorithm>

VolumeGVDB	gvdb;

class Sample : public NVPWindow {
public:
	virtual bool init();
	virtual void display();
	virtual void reshape(int w, int h);
	virtual void motion(int x, int y, int dx, int dy);
	virtual void keyboardchar(unsigned char key, int mods, int x, int y);
	virtual void mouse (NVPWindow::MouseButton button, NVPWindow::ButtonAction state, int mods, int x, int y);
	
	void		draw_topology ();	// draw gvdb topology
	void		draw_rays ();		// draw deposition rays
	void		simulate ();		// simulation material deposition
	void		start_guis ( int w, int h );

	int			gl_screen_tex;
	int			mouse_down;
	int			m_numrays;			// number of deposition rays
	DataPtr		m_rays;				// bundle of rays
	float		m_time;				// simulation time	
	bool		m_show_topo;
	bool		m_show_rays;
	bool		m_simulate;
	int			m_shade_style;
	int			m_wand_style;
};

#define WAND_ROTATE		0
#define WAND_SWEEP		1
#define WAND_WAVE		2

void handle_gui ( int gui, float val )
{
	if ( gui==3 ) {				// If shading gui has changed..
		if ( val==3 ) {			// Set cross section style, orange border 
			gvdb.getScene()->LinearTransferFunc ( 0.0f, 0.2f,  Vector4DF(0,0,0,0.0), Vector4DF(1,0.5,0,0.5) );	
			gvdb.getScene()->LinearTransferFunc ( 0.2f, 0.3f,   Vector4DF(1,0.5,0,0.5), Vector4DF(0,0,0,0.0) );	
			gvdb.getScene()->LinearTransferFunc ( 0.3f, 1.0f,   Vector4DF(0,0,0,0.0), Vector4DF(0,0,0,0.0) );	
			gvdb.CommitTransferFunc (); 
		} else {				// Or set volumetric style, x-ray white
			gvdb.getScene()->LinearTransferFunc ( 0.00f, 0.1f,  Vector4DF(0,0,0,0), Vector4DF(0,0,0, 0) );
			gvdb.getScene()->LinearTransferFunc ( 0.1f, 0.25f,  Vector4DF(0,0,0,0), Vector4DF(1,1,1, 0.8) );
			gvdb.getScene()->LinearTransferFunc ( 0.25f, 0.5f,  Vector4DF(1,1,1,0.8), Vector4DF(0,0,0,0) );
			gvdb.getScene()->LinearTransferFunc ( 0.5f, 1.0f,   Vector4DF(0,0,0,0.0), Vector4DF(0,0,0,0) );
			gvdb.CommitTransferFunc ();		
		}
	}
}

void Sample::start_guis (int w, int h)
{
	setview2D (w, h);
	guiSetCallback ( handle_gui );	
	addGui (  10, h-30, 130, 20, "Simulate", GUI_CHECK, GUI_BOOL, &m_simulate, 0, 1 );
	addGui ( 150, h-30, 130, 20, "Topology", GUI_CHECK, GUI_BOOL, &m_show_topo, 0, 1 );	
	addGui ( 300, h-30, 130, 20, "Rays",	 GUI_CHECK, GUI_BOOL, &m_show_rays, 0, 1 );
	addGui ( 450, h-30, 130, 20, "Shading",  GUI_COMBO, GUI_INT,  &m_shade_style, 0, 5 );
		addItem ( "Off" );
		addItem ( "Voxel" );
		addItem ( "Surface" );
		addItem ( "Section" );
		addItem ( "Volume" );
	addGui ( 600, h-30, 130, 20, "Wand Style",  GUI_COMBO, GUI_INT,  &m_wand_style, 0, 5 );
		addItem ( "Rotate" );
		addItem ( "Sweep" );
		addItem ( "Wave" );
}

bool Sample::init() 
{
	int w = getWidth(), h = getHeight();			// window width & height
	mouse_down = -1;
	gl_screen_tex = -1;
	m_time = 0;
	m_simulate = true;
	m_show_topo = false;
	m_show_rays = true;
	m_shade_style = 2;
	m_wand_style = WAND_ROTATE;
	srand ( 6572 );

	init2D ( "arial" );
	setview2D ( w, h );

	// Initialize GVDB
	int devid = -1;
	gvdb.SetVerbose ( true );
	gvdb.SetProfile ( false );
	gvdb.SetCudaDevice ( devid );
	gvdb.Initialize ();
	gvdb.AddPath ( std::string(ASSET_PATH) );
	gvdb.StartRasterGL ();

	// Load polygons
	// This loads an obj file into scene memory on cpu.
	char scnpath[1024];		
	nvprintf ( "Loading polygon model.\n" );
	if ( !gvdb.FindFile ( "metal.obj", scnpath ) ) {
		nvprintf ( "Cannot find obj file.\n" );
		nverror();
	}
	gvdb.getScene()->AddModel ( scnpath, 1.0, 0, 0, 0 );
	gvdb.CommitGeometry( 0 );					// Send the polygons to GPU as OpenGL VBO

	// Set volume params
	gvdb.getScene()->SetSteps ( 0.5, 16, 0.1 );				// Set raycasting steps
	gvdb.getScene()->SetExtinct ( -1.0f, 1.2f, 0.0f );		// Set volume extinction
	gvdb.getScene()->SetVolumeRange ( 0.3f, 0.0f, 1.0f );	// Set volume value range
	gvdb.getScene()->SetCutoff ( 0.005f, 0.01f, 0.0f );
	gvdb.getScene()->SetBackgroundClr ( 0.1, 0.2, 0.4, 1.0 );
	gvdb.getScene()->LinearTransferFunc ( 0.0f, 0.2f,  Vector4DF(0,0,0,0.0), Vector4DF(1,1,1,0.1) );	
	gvdb.getScene()->LinearTransferFunc ( 0.2f, 0.3f,   Vector4DF(1,1,1,0.05), Vector4DF(1,1,1,0.05) );	
	gvdb.getScene()->LinearTransferFunc ( 0.3f, 1.0f,   Vector4DF(1,1,1,0.05), Vector4DF(0,0,0,0.0) );	
	gvdb.CommitTransferFunc (); 

	// Configure a new GVDB volume
	gvdb.Configure ( 3, 3, 3, 3, 5, Vector3DF(1,1,1), 1 );	

	// Atlas memory expansion will be supported in the Fall 2016 release, 
	// allowing the number of bricks to change dynamically. 
	// For this GVDB Beta, the last argument to AddChannel specifies the
	// maximum number of bricks. Keep this as low as possible for performance reasons.
	// AddChanell ( channel_id, channel_type, apron, max bricks )
	
	// Create two channels (density & color)
	gvdb.AddChannel ( 0, T_FLOAT, 1, 2000 );	// 2000 = estimated max bricks
	gvdb.AddChannel ( 1, T_UCHAR4, 1, 2000 );
	gvdb.SetColorChannel ( 1 );					// Let GVDB know channel 1 can be used for color

	// Create Camera 
	Camera3D* cam = new Camera3D;						
	cam->setFov ( 50.0 );
	cam->setOrbit ( Vector3DF(45,40,0), Vector3DF(50,55,50), 200, 1.0 );	
	gvdb.getScene()->SetCamera( cam );
	
	// Create Light
	Light* lgt = new Light;								
	lgt->setOrbit ( Vector3DF(80,50,0), Vector3DF(50,50,50), 60, 1.0 );
	gvdb.getScene()->SetLight ( 0, lgt );	

	// Add render buffer
	nvprintf ( "Creating screen buffer. %d x %d\n", w, h );
	gvdb.AddRenderBuf ( 0, w, h, 4 );	

	// Create rays
	m_numrays = 1000;
	gvdb.AllocData ( m_rays, m_numrays, sizeof(ScnRay) );

	// Rasterize the polygonal part to voxels
	Matrix4F xform;	
	float part_size = 25.0;						// Part size is set to 100 mm height.
	xform.SRT ( Vector3DF(1,0,0), Vector3DF(0,1,0), Vector3DF(0,0,1), Vector3DF(50,50,50), part_size );
	Vector3DF voxelsize ( 0.10, 0.10, 0.10 );	// Voxel = 1/20th mm = 50 microns
	int res = part_size / voxelsize.y;			// Resolution = Part size / Voxelsize = 100/0.05 = 2000 voxel res
	gvdb.SetVoxelSize ( voxelsize.x, voxelsize.y, voxelsize.z );
	Model* m = gvdb.getScene()->getModel(0);

	gvdb.PolyToVoxelsGL ( 0, m, &xform );		// polygons to voxels

	// Fill color channel	
	gvdb.FillChannel ( 1, Vector4DF(0.7,0.7,0.7,1) );
	gvdb.RunEffect ( FUNC_SMOOTH, 0, 2, Vector3DF(4,0,0), true );
	gvdb.UpdateApron ();

	// Create opengl texture for display
	glViewport ( 0, 0, w, h );
	createScreenQuadGL ( &gl_screen_tex, w, h );

	start_guis ( w, h );

	return true; 
}

void Sample::reshape (int w, int h)
{
	// Resize the opengl screen texture
	createScreenQuadGL ( &gl_screen_tex, w, h );

	// Resize the GVDB render buffers
	gvdb.ResizeRenderBuf ( 0, w, h, 4 );

	postRedisplay();
}

void Sample::keyboardchar(unsigned char key, int mods, int x, int y)
{
	switch ( key ) {
	case '1': case ' ': m_simulate = !m_simulate; break;
	case '2':  m_show_topo = !m_show_topo; break;	
	case '3':  m_show_rays = !m_show_rays; break;
	case '4':  m_shade_style = ( m_shade_style==4 ) ? 0 : m_shade_style+1; handle_gui(3, m_shade_style); break;	
	case '5':  m_wand_style = ( m_wand_style==2 ) ? 0 : m_wand_style+1;   break;	
	};
}


void Sample::simulate()
{
	m_time += 1.0;	

	// Metal Deposition simulation
	Vector3DF dir, rnd;
	ScnRay* ray = gvdb.getRayPtr( 0, m_rays );	
	float x, y, st, lt;
	int ndiv = sqrt( (float) m_numrays);
	Vector3DF effect;

	switch ( m_wand_style ) {
	case WAND_ROTATE:	effect.Set( 1.0, 0.0, 0.0 );	break;
	case WAND_SWEEP:	effect.Set( 0.0, 50.0, 0.0 );	break;
	case WAND_WAVE:		effect.Set( 0.0, 0.0, 0.5 );	break;
	};

	Matrix4F rot;
	rot.RotateY ( effect.x * m_time );		// wand rotation

	Vector3DF clr;							// color variation
	clr.x = sin( 2.0*m_time * DEGtoRAD)*0.5 + 0.5;
	clr.y = sin( 1.0*m_time * DEGtoRAD)*0.5 + 0.5;
	clr.z = sin( 0.5*m_time * DEGtoRAD)*0.5 + 0.5;

	st = sin( m_time * DEGtoRAD);			// sin time
	lt = (int(m_time) % 100)/100.0 - 0.5;		// linear time

	// Initial ray origins and directions
	for (int n=0; n < m_numrays; n++ ) {
		rnd.Random ( -1, 1, -1, 1, -1, 1);
		
		// set ray origin
		x = float(n % ndiv)/ndiv - 0.5 + rnd.x*0.1;		// random variation in rays
		y = float(n / ndiv)/ndiv - 0.5 + rnd.y*0.1;
		x = std::max<float>(-0.5, std::min<float>(0.5, x));
		y = std::max<float>(-0.5, std::min<float>(0.5, y));
		ray->orig.Set ( x*25.0, 0, y*0.5 + lt*effect.y );	// wand sweeping
		ray->orig *= rot;									// rotating wand over time
		ray->orig += Vector3DF( 50, 80, 50 );				// position of wand
		
		// set ray direction
		dir.Random ( -0.1, 0.1, 0, 0.0, 0, 0 );		
		dir += Vector3DF ( x, -0.5, y*0.04 + st*effect.z);	// wand angle (direction of rays)
		dir.Normalize ();		
		ray->dir = dir;
		ray->dir *= rot;									// rotate direction also
		
		// set ray color
		ray->clr = COLORA( clr.x, clr.y, clr.z, 0.2 );
		ray++;
	}

	// Transfer rays to the GPU	
	gvdb.CommitData ( m_rays );

	// Trace rays
	// Returns the hit position and normal of each ray
	gvdb.Raytrace ( m_rays, SHADE_TRILINEAR, 0, -0.0001);

	// Insert Points into the GVDBb grid
	// This identifies a grid cell for each point. The SetPointStruct function accepts an arbitrary 
	// structure, which must contain a vec3f position input, and uint node offset and index outputs.
	// We can use the ray hit points as input directly from the ScnRay data structure.
	DataPtr pntpos, pntclr; 
	gvdb.SetDataGPU ( pntpos, m_numrays, m_rays.gpu, 0, sizeof(ScnRay) );
	gvdb.SetDataGPU ( pntclr, m_numrays, m_rays.gpu, 48, sizeof(ScnRay) );
	gvdb.SetPoints ( pntpos, pntclr );

	gvdb.InsertPoints ( m_numrays, Vector3DF(0,0,0) );
		
	// Splat points
	// This adds voxels to the volume at the point locations
	gvdb.SplatPoints ( m_numrays, 2.0, 1.0, Vector3DF(0,0,0) );
	gvdb.UpdateApron ();

	// Smooth the volume
	// A smoothing effect simulates gradual erosion
	if ( int(m_time) % 20 == 0 ) {
		gvdb.RunEffect ( FUNC_SMOOTH, 0, 1, Vector3DF(4,0,0), true ); 	
	    gvdb.RunEffect ( FUNC_CLR_EXPAND, 1, 1, Vector3DF(2,1,0), true ); 		
	}
	
}

void Sample::draw_rays ()
{
	// Retrieve ray results from GPU
	gvdb.RetrieveData ( m_rays );

	// Draw rays
	Camera3D* cam = gvdb.getScene()->getCamera();
	start3D ( gvdb.getScene()->getCamera() );	
	Vector3DF hit, p1, p2;
	Vector4DF clr;	
	float w;	
	for (int n=0; n < m_numrays; n++ ) {
		ScnRay* ray = gvdb.getRayPtr ( n, m_rays );			
		clr.Set ( ray->clr );
		if (ray->hit.z != NOHIT ) {
			p2 = ray->hit;
			p1 = ray->orig; //dir; p1 *= -5.0; p1 += p2;
			drawLine3D ( p1.x, p1.y, p1.z, p2.x, p2.y, p2.z, clr.x, clr.y, clr.z, clr.w );
		}
	}
	end3D();
}

void Sample::draw_topology ()
{
	Vector3DF clrs[10];
	clrs[0] = Vector3DF(0,0,1);			// blue
	clrs[1] = Vector3DF(0,1,0);			// green
	clrs[2] = Vector3DF(1,0,0);			// red
	clrs[3] = Vector3DF(1,1,0);			// yellow
	clrs[4] = Vector3DF(1,0,1);			// purple
	clrs[5] = Vector3DF(0,1,1);			// aqua
	clrs[6] = Vector3DF(1,0.5,0);		// orange
	clrs[7] = Vector3DF(0,0.5,1);		// green-blue
	clrs[8] = Vector3DF(0.7,0.7,0.7);	// grey

	Camera3D* cam = gvdb.getScene()->getCamera();		
	
	start3D ( gvdb.getScene()->getCamera() );		// start 3D drawing
	
	Vector3DF bmin, bmax;
	Node* node;
	for (int lev=0; lev < 5; lev++ ) {				// draw all levels
		int node_cnt = gvdb.getNumNodes(lev);				
		for (int n=0; n < node_cnt; n++) {			// draw all nodes at this level
			node = gvdb.getNodeAtLevel ( n, lev );
			bmin = gvdb.getWorldMin ( node );		// get node bounding box
			bmax = gvdb.getWorldMax ( node );		// draw node as a box
			drawBox3D ( bmin.x, bmin.y, bmin.z, bmax.x, bmax.y, bmax.z, clrs[lev].x, clrs[lev].y, clrs[lev].z, 1 );			
		}		
	}
	end3D();										// end 3D drawing
}


// Primary display loop
void Sample::display() 
{
	clearScreenGL ();

	if ( m_simulate ) simulate();						// Simulation step

	gvdb.getScene()->SetCrossSection ( Vector3DF(50,50,50), Vector3DF(-1,0,0) );

	int sh;
	switch ( m_shade_style ) {
	case 0: sh = SHADE_OFF;			break;
	case 1: sh = SHADE_VOXEL;		break;
	case 2: sh = SHADE_TRILINEAR;	break;
	case 3: sh = SHADE_SECTION3D;	break;
	case 4: sh = SHADE_VOLUME;		break;
	};
	gvdb.Render ( 0, sh, 0, 0, 1, 1, 0.6 );			// Render volume to internal cuda buffer

	gvdb.ReadRenderTexGL ( 0, gl_screen_tex );		// Copy internal buffer into opengl texture

	renderScreenQuadGL ( gl_screen_tex );			// Render screen-space quad with texture 

	if ( m_show_rays && m_simulate ) draw_rays ();	// Draw deposition rays with OpenGL in 3D

	if ( m_show_topo ) draw_topology ();			// Draw GVDB topology
	
	draw3D ();										// Render the 3D drawing groups

	drawGui (0);									// Render the GUI

	draw2D ();

	postRedisplay();								// Post redisplay since simulation is continuous
}

void Sample::motion(int x, int y, int dx, int dy) 
{
	// Get camera for GVDB Scene
	Camera3D* cam = gvdb.getScene()->getCamera();	
	Light* lgt = gvdb.getScene()->getLight();
	bool shift = (getMods() & NVPWindow::KMOD_SHIFT);		// Shift-key to modify light

	switch ( mouse_down ) {	
	case NVPWindow::MOUSE_BUTTON_LEFT: {
		// Adjust orbit angles
		Vector3DF angs = (shift ? lgt->getAng() : cam->getAng() );
		angs.x += dx*0.2;
		angs.y -= dy*0.2;		
		if ( shift )	lgt->setOrbit ( angs, lgt->getToPos(), lgt->getOrbitDist(), lgt->getDolly() );				
		else			cam->setOrbit ( angs, cam->getToPos(), cam->getOrbitDist(), cam->getDolly() );				
		postRedisplay();	// Update display
		} break;
	
	case NVPWindow::MOUSE_BUTTON_MIDDLE: {
		// Adjust target pos		
		cam->moveRelative ( float(dx) * cam->getOrbitDist()/1000, float(-dy) * cam->getOrbitDist()/1000, 0 );	
		postRedisplay();	// Update display
		} break;
	
	case NVPWindow::MOUSE_BUTTON_RIGHT: {	
		// Adjust dist
		float dist = (shift ? lgt->getOrbitDist() : cam->getOrbitDist());
		dist -= dy;
		if ( shift )	lgt->setOrbit ( lgt->getAng(), lgt->getToPos(), dist, cam->getDolly() );
		else			cam->setOrbit ( cam->getAng(), cam->getToPos(), dist, cam->getDolly() );		
		postRedisplay();	// Update display
		} break;
	}
}

void Sample::mouse ( NVPWindow::MouseButton button, NVPWindow::ButtonAction state, int mods, int x, int y)
{
	if ( guiHandler ( button, state, x, y ) ) return;

	// Track when we are in a mouse drag
	mouse_down = (state == NVPWindow::BUTTON_PRESS) ? button : -1;	
}

int sample_main ( int argc, const char** argv ) 
{
	Sample sample_obj;
	return sample_obj.run ( "GVDB Sparse Volumes - gSprayDeposit", argc, argv, 1024, 768, 4, 5 );
}

void sample_print( int argc, char const *argv)
{
}

