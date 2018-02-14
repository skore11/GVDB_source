//----------------------------------------------------------------------------------
//
// File:   string_helper.h
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
// Last Updated: Rama Hoetzlein, rhoetzlein@nvidia.com, 8/17/2016
//
//----------------------------------------------------------------------------------

#ifndef DEF_STRING_HELPER
	#define DEF_STRING_HELPER

	#include <string>
	#include <vector>

	std::string strFilebase ( std::string str );	// basename of a file (minus ext)
	std::string strFilepath ( std::string str );	// path of a file

	int strToI (std::string s);
	float strToF (std::string s);
	std::string strParse ( std::string& str, std::string lsep, std::string rsep );
	bool strGet ( std::string str, std::string& result, std::string lsep, std::string rsep );	
	std::string strSplit ( std::string& str, std::string sep );
	bool strSub ( std::string str, int first, int cnt, std::string cmp );
	std::string strReplace ( std::string str, std::string delim, std::string ins );
	std::string strLTrim ( std::string str );
	std::string strRTrim ( std::string str );
	std::string strTrim ( std::string str );
	std::string strLeft ( std::string str, int n );
	std::string strRight ( std::string str, int n );
	int strExtract ( std::string& str, std::vector<std::string>& list );
	unsigned long strToID ( std::string str );
	float strToNum ( std::string str );	
	void strToVec3 ( std::string str, float* vec );

	// File helpers
	unsigned long getFileSize ( char* fname );
	unsigned long getFilePos ( FILE* fp );
	bool getFileLocation ( char* filename, char* outpath, char** searchPaths, int numPaths );

#endif
