//----------------------------------------------------------------------------------
//
// File:   load_tga.h
// 
// Copyright (c) 2013 NVIDIA Corporation. All rights reserved.
// TO  THE MAXIMUM  EXTENT PERMITTED  BY APPLICABLE  LAW, THIS SOFTWARE  IS PROVIDED
// *AS IS*  AND NVIDIA AND  ITS SUPPLIERS DISCLAIM  ALL WARRANTIES,  EITHER  EXPRESS
// OR IMPLIED, INCLUDING, BUT NOT LIMITED  TO, IMPLIED WARRANTIES OF MERCHANTABILITY
// AND FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL  NVIDIA OR ITS SUPPLIERS
// BE  LIABLE  FOR  ANY  SPECIAL,  INCIDENTAL,  INDIRECT,  OR  CONSEQUENTIAL DAMAGES
// WHATSOEVER (INCLUDING, WITHOUT LIMITATION,  DAMAGES FOR LOSS OF BUSINESS PROFITS,
// BUSINESS INTERRUPTION, LOSS OF BUSINESS INFORMATION, OR ANY OTHER PECUNIARY LOSS)
// ARISING OUT OF THE  USE OF OR INABILITY  TO USE THIS SOFTWARE, EVEN IF NVIDIA HAS
// BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.
//
// Last updated: Rama Hoetzlein, rhoetzlein@nvidia.com, 8/17/2016
//
//----------------------------------------------------------------------------------

#ifndef LOAD_TGA

	#include <stdio.h>

	class TGA {
	public:
		enum TGAFormat
		{
			RGB = 0x1907,
			RGBA = 0x1908,
			ALPHA = 0x1906,
			UNKNOWN = -1
		};

		enum TGAError
		{
			TGA_NO_ERROR = 1,   // No error
			TGA_FILE_NOT_FOUND, // File was not found 
			TGA_BAD_IMAGE_TYPE, // Color mapped image or image is not uncompressed
			TGA_BAD_DIMENSION,  // Dimension is not a power of 2 
			TGA_BAD_BITS,       // Image bits is not 8, 24 or 32 
			TGA_BAD_DATA        // Image data could not be loaded 
		};

		TGA(void) : 
			m_texFormat(TGA::UNKNOWN),
			m_nImageWidth(0),
			m_nImageHeight(0),
			m_nImageBits(0),
			m_nImageData(NULL) {}

		~TGA(void);

		TGA::TGAError load( const char *name );
		TGA::TGAError saveFromExternalData( const char *name, int w, int h, TGAFormat fmt, const unsigned char *externalImage );

		TGAFormat       m_texFormat;
		int             m_nImageWidth;
		int             m_nImageHeight;
		int             m_nImageBits;
		unsigned char * m_nImageData;
    
	private:

		int returnError(FILE *s, int error);
		unsigned char *getRGBA(FILE *s, int size);
		unsigned char *getRGB(FILE *s, int size);
		unsigned char *getGray(FILE *s, int size);
		void           writeRGBA(FILE *s, const unsigned char *externalImage, int size);
		void           writeRGB(FILE *s, const unsigned char *externalImage, int size);
		void           writeGrayAsRGB(FILE *s, const unsigned char *externalImage, int size);
		void           writeGray(FILE *s, const unsigned char *externalImage, int size);
	};

#endif