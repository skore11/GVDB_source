
# Try to find OpenVDB project dll/so and headers
#

# outputs
unset(OPENVDB_DLL CACHE)
unset(OPENVDB_LIB CACHE)
unset(OPENVDB_FOUND CACHE)
unset(OPENVDB_INCLUDE_DIR CACHE)

# search path, can be overridden by user
set ( OPENVDB_LOCATION "${PROJECT_SOURCE_DIR}/../shared_openvdb" CACHE PATH "Path to shared_openvdb library" )

macro ( folder_list result curdir substring )
  FILE(GLOB children RELATIVE ${curdir} ${curdir}/*${substring}*)
  SET(dirlist "")
  foreach ( child ${children})
    IF(IS_DIRECTORY ${curdir}/${child})
        LIST(APPEND dirlist ${child})
    ENDIF()
  ENDFOREACH()
  SET(${result} ${dirlist})
ENDMACRO()

macro(_find_version_path targetVersion targetPath rootName searchList platform )
  unset ( targetVersion )
  unset ( targetPath )
  SET ( bestver "0.0.0" )
  SET ( bestpath "" )
  foreach ( basedir ${searchList} )
    folder_list ( dirList ${basedir} ${platform} )	
	  foreach ( checkdir ${dirList} ) 	 
	    string ( REGEX MATCH "${rootName}.([0-9]+).([0-9]+).([0-9]+)(.*)$" result "${checkdir}" )
	    if ( "${result}" STREQUAL "${checkdir}" )
	       # found a path with versioning 
	       SET ( ver "${CMAKE_MATCH_1}.${CMAKE_MATCH_2}.${CMAKE_MATCH_3}" )
	       if ( ver GREATER bestver )
	  	    SET ( bestver ${ver} )
	  		SET ( bestpath "${basedir}/${checkdir}" )
	  	 endif ()
	    endif()	  
	  endforeach ()		
  endforeach ()  
  SET ( ${targetVersion} "${bestver}" )
  SET ( ${targetPath} "${bestpath}" )
endmacro()

macro(_find_files targetVar incDir dllName dllName64 folder)
  unset ( fileList )
  if(ARCH STREQUAL "x86")
      file(GLOB fileList "${${incDir}}/../${folder}${dllName}")
      list(LENGTH fileList NUMLIST)
      if(NUMLIST EQUAL 0)
        file(GLOB fileList "${${incDir}}/${folder}${dllName}")
      endif()
  else()      
      file(GLOB fileList "${${incDir}}/../${folder}${dllName64}")
      list(LENGTH fileList NUMLIST)
      if(NUMLIST EQUAL 0)
	    file(GLOB fileList "${${incDir}}/${folder}${dllName64}")
		list(LENGTH fileList NUMLIST)
		message ( STATUS "locate: ${${incDir}}/${folder}${dllName64}, found: ${NUMLIST}" )		
      endif()
  endif()  
  list(LENGTH fileList NUMLIST)
  if(NUMLIST EQUAL 0)
    message(FATAL_ERROR "MISSING: unable to find ${targetVar} files (${folder}${dllName}, ${folder}${dllName64})" )    
  else()
    list(APPEND ${targetVar} ${fileList} )  
  endif()  

  # message ( "File list: ${${targetVar}}" )		#-- debugging
endmacro()

 # Locate OpenVDB by version
set ( SEARCH_PATHS
  ${OPENVDB_LOCATION}
  ${PROJECT_SOURCE_DIR}/shared_openvdb
  ${PROJECT_SOURCE_DIR}/../shared_openvdb
  ${PROJECT_SOURCE_DIR}/../../shared_openvdb
  $ENV{OPENVDB_LOCATION}  
)
if (WIN32) 
  _find_version_path ( OPENVDB_VERSION OPENVDB_ROOT_DIR "OpenVDB" "${SEARCH_PATHS}" "win64" )
endif()
if (UNIX)
  _find_version_path ( OPENVDB_VERSION OPENVDB_ROOT_DIR "OpenVDB" "${SEARCH_PATHS}" "linux64" )
endif()
message ( STATUS "OpenVDB version: ${OPENVDB_VERSION}")

if (NOT OPENVDB_ROOT_DIR )
  # Locate by version failed. Handle user override for OPENVDB_LOCATION.
  find_path( OPENVDB_INCLUDE_DIR openvdb.h ${OPENVDB_LOCATION}/include )
  if ( OPENVDB_INCLUDE_DIR )
    set (OPENVDB_ROOT_DIR ${OPENVDB_INCLUDE_DIR}/../ )
  endif()
endif()


if (OPENVDB_ROOT_DIR)

  if (WIN32) 	 
    #-------- Locate DLL
	_find_files( OPENVDB_DLL OPENVDB_ROOT_DIR "lib/Blosc.dll" "lib64/Blosc.dll" "")   
	_find_files( OPENVDB_DLL OPENVDB_ROOT_DIR "lib/Half.dll" "lib64/Half.dll" "")    
	_find_files( OPENVDB_DLL OPENVDB_ROOT_DIR "lib/Iex.dll" "lib64/Iex.dll" "")    
	_find_files( OPENVDB_DLL OPENVDB_ROOT_DIR "lib/IexMath.dll" "lib64/IexMath.dll" "")    
	_find_files( OPENVDB_DLL OPENVDB_ROOT_DIR "lib/IlmThread.dll" "lib64/IlmThread.dll" "")    
	_find_files( OPENVDB_DLL OPENVDB_ROOT_DIR "lib/Imath.dll" "lib64/Imath.dll" "")    	
	_find_files( OPENVDB_DLL OPENVDB_ROOT_DIR "lib/tbb.dll" "lib64/tbb.dll" "")  
	_find_files( OPENVDB_DLL OPENVDB_ROOT_DIR "lib/tbb_debug.dll" "lib64/tbb_debug.dll" "")  

	#-------- Locate LIBS
    _find_files( OPENVDB_LIB_DEBUG OPENVDB_ROOT_DIR "lib/openvdb_d.lib" "lib64/openvdb_d.lib" "")    
	_find_files( OPENVDB_LIB_RELEASE OPENVDB_ROOT_DIR "lib/openvdb.lib" "lib64/openvdb.lib" "")        
	
  endif(WIN32)

  if (UNIX)
    _find_files( OPENVDB_LIB OPENVDB_ROOT_DIR "lib/libopenvdb.so" "lib64/libopenvdb.so" "" )        
	set(OPENVDB_DLL ${OPENVDB_LIB})
  endif(UNIX)

  #-------- Locate HEADERS
  _find_files( OPENVDB_HEADERS OPENVDB_ROOT_DIR "openvdb.h" "openvdb.h" "include/Openvdb/" )

  if(OPENVDB_DLL)
	  set( OPENVDB_FOUND "YES" )      
  else()
    message(STATUS "setting OPENVDB_DLL to ${OPENVDB_DLL}" )
  endif(OPENVDB_DLL)
else(OPENVDB_ROOT_DIR)

  message(STATUS "--> WARNING: OPENVDB not found. Some samples requiring OpenVDB may not run.
        Set USE_OPENVDB option above and set OPENVDB_LOCATION to full path of library." )

endif(OPENVDB_ROOT_DIR)

include(FindPackageHandleStandardArgs)

SET(OPENVDB_DLL ${OPENVDB_DLL} CACHE PATH "path")
SET(OPENVDB_LIB_DEBUG ${OPENVDB_LIB_DEBUG} CACHE PATH "path")
SET(OPENVDB_LIB_RELEASE ${OPENVDB_LIB_RELEASE} CACHE PATH "path")
SET(OPENVDB_INCLUDE_DIR "${OPENVDB_ROOT_DIR}/include" CACHE PATH "path")
get_filename_component ( LIB_PATH "${OPENVDB_LIB_RELEASE}" DIRECTORY )
SET(OPENVDB_LIB_DIR ${LIB_PATH} CACHE PATH "path")

mark_as_advanced( OPENVDB_FOUND )

