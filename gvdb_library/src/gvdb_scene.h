//--------------------------------------------------------------------------------
//
// File:   gvdb_scene.h
// 
// NVIDIA GVDB Sparse Volumes
// Copyright (c) 2015, NVIDIA. All rights reserved.
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA nor the
//    names of its contributors may be used to endorse or promote products
//    derived from this software without specific prior written permission.
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Last Updated: Rama Hoetzlein, rhoetzlein@nvidia.com, 8/17/2016
//
//----------------------------------------------------------------------------------


#ifndef DEF_SCENE_H
	#define DEF_SCENE_H


	#include <vector>
	#include "gvdb_model.h"	
	#include "gvdb_camera.h"	
	#include "gvdb_allocator.h"

	#pragma warning (disable : 4251 )

	#define MAX_PATHS	32

	class CallbackParser;

	namespace nvdb {

	struct ParamList {
		int p[128];
	};
	struct Key {
		char			obj;		// object type
		int				objid;		// object ID
		unsigned long	varid;		// variable ID
		float			f1, f2;		// frame start/end
		Vector3DF		val1, val2;	// value start/end		
	};
	struct Mat {
		int				id;		
		Vector3DF		mAmb, mDiff, mSpec;		
		Vector3DF		mParam[64];
	};	

	class GVDB_API Scene {
	public:
		Scene();
		~Scene();
		static Scene*			gScene;
		static CallbackParser*	gParse;

		void		Clear ();
		void		LoadFile ( std::string filename );				
		void		AddPath ( std::string path );
		bool		FindFile ( std::string fname, char* path );
		Camera3D*	SetCamera ( Camera3D* cam );
		Light*		SetLight ( int n, Light* light );
		int			AddModel ( std::string filename, float scale, float tx, float ty, float tz );
		int			AddVolume ( std::string filename, Vector3DI res, char vtype, float scale=1.0 );
		int			AddGround ( float hgt, float scale=1.0 );
		void		SetAspect ( int w, int h );
		
		// Shaders
		int			AddShader ( char* vertfile, char* fragfile );
		int			AddShader ( char* vertfile, char* fragfile, char* geomfile );
		int			AddParam ( int prog, int id, char* name );
		int			AddAttrib ( int prog, int id, char* name );			

		// Materials
		int			AddMaterial ();
		void		SetMaterial ( int model, Vector4DF amb, Vector4DF diff, Vector4DF spec );		
		void		SetMaterialParam ( int id, int p, Vector3DF val );
		void		SetOverrideMaterial ( Vector4DF amb, Vector4DF diff, Vector4DF spec );
		void		SetVolumeRange ( float viso, float vmin, float vmax );

		// Animation
		void		AddKey ( std::string obj, std::string var, int f1, int f2, Vector3DF val1, Vector3DF val2 );
		bool		DoAnimation ( int frame );
		void		UpdateValue (  char obj, int objid, long varid, Vector3DF val );
		void		RecordKeypoint ( int w, int h );
		int			getFrameSamples ()	{ return mFrameSamples; }
		void		setFrameSamples ( int n )	{ mFrameSamples = n; }
		
		// Transfer functions
		void		AddTransfer ( std::string filestr, Vector3DF tvec );
		void		LinearTransferFunc ( float t0, float t1, Vector4DF a, Vector4DF b );
		void		UpdateTransferFromImg ();

		int			getNumModels ()		{ return (int) mModels.size(); }
		Model*		getModel ( int n )	{ return mModels[n]; }
		Camera3D*	getCamera ()		{ return mCamera; }
		Light*		getLight ()			{ return mLights[0]; }
		int			getSlot ( int prog );
		int			getParam ( int prog, int id )	{ return mParams[ mProgToSlot[prog] ].p[id]; }
		bool		useOverride ()		{ return clrOverride; }
		Vector4DF	getShadowParams ()	{ return mShadowParams; }		
		Vector3DI	getFrameRange ()	{ return mVFrames; }
		int			getNumLights()		{ return (int) mLights.size(); }
		
		// Loading scenes
		void		Load ( char *filename, float windowAspect );
		static void	LoadPath ();		
		static void	LoadModel ();
		static void	LoadVolume ();
		static void	LoadGround ();
		static void LoadCamera ();
		static void LoadLight ();
		static void LoadAnimation ();
		static void LoadShadow ();		
		static void VolumeThresh ();
		static void VolumeTransfer ();
		static void VolumeClip ();

		void SetRes ( int x, int y )	{ mXres=x; mYres=y; SetAspect(x,y); }
		Vector3DI getRes ()				{ return Vector3DI(mXres,mYres,0); }
		Vector4DF* getTransferFunc ()	{ return mTransferFunc; }

		Vector4DF getBackClr()			{ return mBackgroundClr; }
		Vector3DF getExtinct()			{ return mExtinct; }
		Vector3DF getCutoff()			{ return mCutoff; }
		Vector3DF getSteps()			{ return mSteps; }
		Vector3DF getSectionPnt()		{ return mSectionPnt; }
		Vector3DF getSectionNorm()		{ return mSectionNorm; }
		void SetExtinct ( float a, float b, float c )	{ mExtinct.Set(a,b,c); }
		void SetSteps ( float a, float b, float c )		{ mSteps.Set(a,b,c); }
		void SetCutoff ( float a, float b, float c )	{ mCutoff.Set(a,b,c); }
		void SetBackgroundClr ( float r, float g, float b, float a )	{ mBackgroundClr.Set(r,g,b,a); }
		void SetCrossSection ( Vector3DF pos, Vector3DF norm )	{ mSectionPnt = pos; mSectionNorm = norm; }		

	public:
		int						mXres, mYres;
		Camera3D*				mCamera;				
		std::vector<Model*>		mModels;
		std::vector<Light*>		mLights;
		std::vector<int>		mShaders;
		std::vector<ParamList>	mParams;
		std::vector<Key>		mKeys;
		std::vector<Mat>		mMaterials;
		int						mFrameSamples;

		bool					clrOverride;
		Vector4DF				clrAmb, clrDiff, clrSpec;	

		char*					mSearchPaths[MAX_PATHS];
		int						mNumPaths;

		int						mProgToSlot[ 512 ];
		
		// Animation recording
		std::string				mOutFile;
		int						mOutFrame;
		Camera3D*				mOutCam;
		Light*					mOutLight;	
		std::string				mOutModel;

		// Shadow parameters (independent of method used)
		Vector4DF				mShadowParams;

		// Volume import settings
		Vector3DF				mVClipMin, mVClipMax;
		Vector3DF				mVThreshold, mVLeaf;
		Vector3DF				mVFrames;
		std::string				mVName;

		// Transfer function				
		Vector3DF				mTransferVec;			// x=alpha, y=gain
		std::string				mTransferName;			// transfer function filename
		//nvImg					mTransferImg;			// transfer function image
		Vector4DF*				mTransferFunc;

		// Volume settings
		Vector3DF				mExtinct;
		Vector3DF				mSteps;
		Vector3DF				mCutoff;
		Vector4DF				mBackgroundClr;

		// Cross sections
		Vector3DF				mSectionPnt;
		Vector3DF				mSectionNorm;

		std::string		mLastShader;
	};

	}

#endif
