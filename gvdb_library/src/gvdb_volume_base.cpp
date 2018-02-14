
#include "gvdb_volume_base.h"
#include "gvdb_render.h"
#include "gvdb_allocator.h"
#include "gvdb_scene.h"

using namespace nvdb;

void VolumeBase::getDimensions ( Vector3DF& objmin, Vector3DF& objmax, Vector3DF& voxmin, Vector3DF& voxmax, Vector3DF& voxsize, Vector3DF& voxres )
{
	objmin = mObjMin;
	objmax = mObjMax;
	voxmin = mVoxMin;
	voxmax = mVoxMax;
	voxsize = mVoxsize * 1000;
	voxres = mVoxRes;
}

void VolumeBase::getTiming ( float& render_time )
{
	render_time = mRenderTime.x;
}

void VolumeBase::CommitGeometry ( int model_id )
{
	#ifdef BUILD_OPENGL

		Model* m = mScene->getModel( model_id );

		// Create VAO
		if ( m->vertArrayID == -1 )  glGenVertexArrays ( 1, (GLuint*) &m->vertArrayID );
		glBindVertexArray ( m->vertArrayID );

		// Update Vertex VBO
		if ( m->vertBufferID == -1 ) glGenBuffers( 1, (GLuint*) &m->vertBufferID );	
		//glBindBuffer ( GL_ARRAY_BUFFER, vertBufferID );	
		//glBufferData ( GL_ARRAY_BUFFER, vertCount * vertStride, vertBuffer, GL_STATIC_DRAW );
	
		glNamedBufferDataEXT( m->vertBufferID, m->vertCount * m->vertStride, m->vertBuffer, GL_STATIC_DRAW );
		glEnableVertexAttribArray ( 0 );
		glBindVertexBuffer ( 0, m->vertBufferID, 0, m->vertStride );
		glVertexAttribFormat ( 0, m->vertComponents, GL_FLOAT, false, m->vertOffset );
		glVertexAttribBinding ( 0, 0 );
		glEnableVertexAttribArray ( 1 );
		glVertexAttribFormat ( 1, m->normComponents, GL_FLOAT, false, m->normOffset );
		glVertexAttribBinding ( 1, 0 );
	
		// Update Element VBO
		if ( m->elemBufferID == -1 ) glGenBuffers( 1, (GLuint*) &m->elemBufferID );
		//glBindBuffer ( GL_ELEMENT_ARRAY_BUFFER, elemBufferID );
		//glBufferData( GL_ELEMENT_ARRAY_BUFFER, elemCount * elemStride, elemBuffer, GL_STATIC_DRAW );	
		glNamedBufferDataEXT( m->elemBufferID, m->elemCount * m->elemStride, m->elemBuffer, GL_STATIC_DRAW );	

		glBindVertexArray ( 0 );
	
		glBindBuffer ( GL_ARRAY_BUFFER, 0 );
		glBindBuffer ( GL_ELEMENT_ARRAY_BUFFER, 0 );

	#endif
}