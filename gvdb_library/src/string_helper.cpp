
#include "string_helper.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>

int strToI (std::string s) {
	//return ::atoi ( s.c_str() );
	std::istringstream str_stream ( s ); 
	int x; 
	if (str_stream >> x) return x;		// this is the correct way to convert std::string to int, do not use atoi
	return 0;
};
float strToF (std::string s) {
	//return ::atof ( s.c_str() );
	std::istringstream str_stream ( s ); 
	float x; 
	if (str_stream >> x) return x;		// this is the correct way to convert std::string to float, do not use atof
	return 0;
};

std::string strFilebase ( std::string str )
{
	size_t pos = str.find_last_of ( '.' );
	if ( pos != std::string::npos ) 
		return str.substr ( 0, pos );
	return str;
}
std::string strFilepath ( std::string str )
{
	size_t pos = str.find_last_of ( '\\' );
	if ( pos != std::string::npos ) 
		return str.substr ( 0, pos+1 );
	return str;
}

std::string strParse ( std::string& str, std::string lsep, std::string rsep )
{
	std::string result;
	size_t lfound, rfound;

	lfound = str.find_first_of ( lsep );
	if ( lfound != std::string::npos) {
		rfound = str.find_first_of ( rsep, lfound+1 );
		if ( rfound != std::string::npos ) {
			result = str.substr ( lfound+1, rfound-lfound-1 );					// return string strickly between lsep and rsep
			str = str.substr ( 0, lfound ) + str.substr ( rfound+1 );
			return result;
		} 
	}
	return "";
}
bool strGet ( std::string str, std::string& result, std::string lsep, std::string rsep )
{	
	size_t lfound, rfound;

	lfound = str.find_first_of ( lsep );
	if ( lfound != std::string::npos) {
		rfound = str.find_first_of ( rsep, lfound+1 );
		if ( rfound != std::string::npos ) {
			result = str.substr ( lfound+1, rfound-lfound-1 );					// return string strickly between lsep and rsep			
			return true;
		} 
	}
	return false;
}

std::string strSplit ( std::string& str, std::string sep )
{
	std::string result;
	size_t f1, f2;

	f1 = str.find_first_not_of ( sep );
	if ( f1 == std::string::npos ) f1 = 0;
	f2 = str.find_first_of ( sep, f1 );
	if ( f2 != std::string::npos) {
		result = str.substr ( f1, f2-f1 );
		str = str.substr ( f2+1 );		
	} else {
		result = str;		
		str = "";
	}
	return result;
}


std::string strReplace ( std::string str, std::string delim, std::string ins )
{
	size_t found = str.find_first_of ( delim );
	while ( found != std::string::npos ) {
		str = str.substr ( 0, found ) + ins + str.substr ( found+1 );
		found = str.find_first_of ( delim );
	}
	return str;
}

bool strSub ( std::string str, int first, int cnt, std::string cmp )
{
	if ( str.substr ( first, cnt ).compare ( cmp ) == 0 ) return true;
	return false;
}


// Return 4-byte long int whose bytes
// match the first 4 ASCII chars of string given.
unsigned long strToID ( std::string str )
{
	str = str + "    ";	
	return (static_cast<unsigned long>(str.at(0)) << 24) | 
		   (static_cast<unsigned long>(str.at(1)) << 16) |
		   (static_cast<unsigned long>(str.at(2)) << 8) |
		   (static_cast<unsigned long>(str.at(3)) );
}

float strToNum ( std::string str )
{
	return (float) atof ( str.c_str() );
}
void strToVec3 ( std::string str, float* vec )
{
	str = strTrim( str );
	std::string arg = strSplit ( str, " " );
	vec[0] = strToNum ( arg );
	if  (str == "") return;
	arg = strSplit ( str, " " );
	vec[1] = strToNum ( arg );
	vec[2] = strToNum ( str );	
}

// trim from start
#include <algorithm>
#include <cctype>
/*std::string strLTrim(std::string str) {
        str.erase(str.begin(), std::find_if(str.begin(), str.end(), std::not1(std::ptr_fun<int, int>(std::isspace))));
        return str;
}

// trim from end
std::string strRTrim(std::string str) {
        str.erase(std::find_if(str.rbegin(), str.rend(), std::not1(std::ptr_fun<int, int>(std::isspace))).base(), str.end());
        return str;
} */

// trim from both ends
std::string strTrim(std::string str)
{
	size_t lft = str.find_first_not_of ( " \r\n" );
	size_t rgt = str.find_last_not_of ( " \r\n" );
	if ( lft == std::string::npos || rgt == std::string::npos ) return "";
	return str.substr ( lft, rgt-lft+1 );
}

std::string strLeft ( std::string str, int n )
{
	return str.substr ( 0, n );
}
std::string strRight ( std::string str, int n )
{
	if ( str.length() < n ) return "";
	return str.substr ( str.length()-n, n );
}
int strExtract ( std::string& str, std::vector<std::string>& list )
{
	size_t found ;
	for (int n=0; n < list.size(); n++) {
		found = str.find ( list[n] );
		if ( found != std::string::npos ) {
			str = str.substr ( 0, found ) + str.substr ( found + list[n].length() );
			return n;
		}
	}
	return -1;
}

unsigned long getFileSize ( char* fname )
{
	FILE* fp = fopen ( fname, "rb" );
	fseek ( fp, 0, SEEK_END );
	return ftell ( fp );
}
unsigned long getFilePos ( FILE* fp )
{
	return ftell ( fp );
}

bool getFileLocation ( char* filename, char* outpath, char** searchPaths, int numPaths )
{
	bool found = false;
	FILE* fp = fopen( filename, "rb" );
	if (fp) {
		found = true;
		strcpy ( outpath, filename );		
	} else {
		for (int i=0; i < numPaths; i++) {
			if (!searchPaths) break;        // If no valid search path list exit now
			if (!searchPaths[i]) continue;  // If any search path in the list is NULL, contine.  Other entries may be valid			
			sprintf ( outpath, "%s%s", searchPaths[i], filename );
			fp = fopen( outpath, "rb" );
			if (fp)	{ found = true;	break; }
		}		
	}
	if ( found ) fclose ( fp );
	return found;
}

