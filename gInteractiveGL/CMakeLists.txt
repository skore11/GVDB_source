cmake_minimum_required(VERSION 2.8)
set(PROJNAME gInteractiveGL)
Project(${PROJNAME})
Message(STATUS "-------------------------------")
Message(STATUS "Processing Project ${PROJNAME}:")

####################################################################################
# Bootstrap
#
set( BASE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} )
find_path ( HELPERS "Helpers.cmake" HINTS 
    ${CMAKE_MODULE_PATH}
    ${BASE_DIRECTORY}/sample_utils
    ${BASE_DIRECTORY}/../sample_utils
    ${BASE_DIRECTORY}/../source/sample_utils
    ${BASE_DIRECTORY}/../../source/sample_utils
)
if ( ${HELPERS} STREQUAL "HELPERS-NOTFOUND" )
    set ( CMAKE_MODULE_PATH "***FULL PATH TO***/gvdb/sample_utils/" CACHE PATH "Full path to gvdb/sample_utils/" )
    message ( FATAL_ERROR "\n
    Please set the CMAKE_MODULE_PATH 
    to the full path of for /gvdb/sample_utils/ above
    and configure again." )
endif()
get_filename_component ( CMAKE_MODULE_PATH ${HELPERS} REALPATH )
set ( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} CACHE PATH "Full path to gvdb/sample_utils/" )
set( EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_BINARY_DIR} CACHE PATH "Executable path" )
if ( NOT DEFINED ASSET_PATH ) 
  get_filename_component ( _assets "${CMAKE_MODULE_PATH}/../shared_assets" REALPATH )
  set ( ASSET_PATH ${_assets} CACHE PATH "Full path to gvdb/shared_assets/" )
endif()

include( ${CMAKE_MODULE_PATH}/Helpers.cmake )     # Cross-Platform functions

#####################################################################################
# Sample requirements

set ( REQUIRE_PNG "1" )
set ( REQUIRE_TGA "1" )
set ( REQUIRE_GLEW "1" )
set ( REQUIRE_MAIN "1" )
set ( REQUIRE_NVGUI "1" )

#####################################################################################
# Find GVDB
#
find_package(GVDB)

if ( GVDB_FOUND )
    message( STATUS "--> Using package GVDB (inc: ${GVDB_INCLUDE_DIR}) ")    
    include_directories( ${GVDB_INCLUDE_DIR} )
    add_definitions(-DUSE_GVDB)
    if(WIN32)
      LIST(APPEND LIBRARIES_OPTIMIZED ${GVDB_LIB_DIR}/${GVDB_LIB} )
      LIST(APPEND LIBRARIES_DEBUG ${GVDB_LIB_DIR}/${GVDB_LIB} )
    endif(WIN32)
    LIST(APPEND PACKAGE_SOURCE_FILES ${GVDB_INCLUDE_DIR}/${GVDB_HEADERS} )  	
    source_group(GVDB FILES ${GVDB_INCLUDE_DIR}/${GVDB_HEADERS} ) 
 else()
    message( FATAL_ERROR "--> Unable to find GVDB") 
 endif()

####################################################################################
# Find CUDA
#
find_package(CUDA)

if ( CUDA_FOUND )
    message( STATUS "--> Using package CUDA (ver ${CUDA_VERSION})") 
    add_definitions(-DUSE_CUDA)	
    include_directories(${CUDA_TOOLKIT_INCLUDE})	
	LIST(APPEND LIBRARIES_OPTIMIZED ${CUDA_CUDA_LIBRARY} )
	LIST(APPEND LIBRARIES_DEBUG ${CUDA_CUDA_LIBRARY} )
	LIST(APPEND PACKAGE_SOURCE_FILES ${CUDA_TOOLKIT_INCLUDE} )    
	source_group(CUDA FILES ${CUDA_TOOLKIT_INCLUDE} ) 
else()
   message ( FATAL_ERROR "---> Unable to find package CUDA")
endif()

####################################################################################
# Compile PTX Files
#
file(GLOB CUDA_FILES RELATIVE "${BASE_DIRECTORY}" *.cu *.cuh)
message ( STATUS "Build CUDA kernels: ${CUDA_FILES}" )

_COMPILEPTX ( SOURCES ${CUDA_FILES} TARGET_PATH ${EXECUTABLE_OUTPUT_PATH} GENERATED PTX_FILES OPTIONS -arch=compute_30 -code=sm_30 --ptxas-options=-v -O3 --use_fast_math --maxrregcount=128 "-I${GVDB_INCLUDE_DIR},${CMAKE_CURRENT_SOURCE_DIR}" )

####################################################################################
# Find Sample Utils
#
find_package(Utils)

#####################################################################################
# Source files for this project
#
file(GLOB SOURCE_FILES *.cpp *.hpp *.inl *.h *.c)

#####################################################################################
# Executable
#
unset ( ALL_SOURCE_FILES )
list( APPEND ALL_SOURCE_FILES ${SOURCE_FILES} )
list( APPEND ALL_SOURCE_FILES ${COMMON_SOURCE_FILES} )
list( APPEND ALL_SOURCE_FILES ${PACKAGE_SOURCE_FILES} )
list( APPEND ALL_SOURCE_FILES ${UTIL_SOURCE_FILES} )
list( APPEND ALL_SOURCE_FILES ${CUDA_FILES} )

if( WIN32 )
  LIST(APPEND LIBRARIES_OPTIMIZED "opengl32.lib" )
  LIST(APPEND LIBRARIES_DEBUG "opengl32.lib" )
else()
  find_library(NVTOOLSEXT nvToolsExt HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
  find_library(CUDART cudart HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
  find_library(GVDBL gvdb HINTS ${GVDB_LIB_DIR})
  set(libdeps GL GLEW X11 cuda ${GVDBL} ${NVTOOLSEXT} ${CUDART})
  LIST(APPEND LIBRARIES_OPTIMIZED ${libdeps})
  LIST(APPEND LIBRARIES_DEBUG ${libdeps})
endif()

include_directories ("${CMAKE_CURRENT_SOURCE_DIR}")    
add_definitions(-DGVDB_IMPORTS -DGLEW_STATIC -DGLEW_NO_GLU)  
add_definitions(-DASSET_PATH="${ASSET_PATH}/") 
add_executable (${PROJNAME} ${ALL_SOURCE_FILES} ${CUDA_FILES} ${PTX_SOURCES} )
set_property ( TARGET ${PROJNAME} APPEND PROPERTY DEPENDS ${PTX_SOURCES} )

if ( MSVC  )
    set_target_properties( ${PROJNAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${EXECUTABLE_OUTPUT_PATH} )
    set_target_properties( ${PROJNAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY_DEBUG ${EXECUTABLE_OUTPUT_PATH} )
    set_target_properties( ${PROJNAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY_RELEASE ${EXECUTABLE_OUTPUT_PATH} )    	
	source_group(CUDA FILES ${CUDA_FILES})
    source_group(PTX FILES  ${PTX_FILES})
endif ()

#####################################################################################
# Install Binaries
#
_INSTALL ( FILES ${GVDB_GLSL} SOURCE ${GVDB_SHARE_DIR} DESTINATION ${EXECUTABLE_OUTPUT_PATH} )
_INSTALL ( FILES ${GVDB_PTX} SOURCE ${GVDB_SHARE_DIR} DESTINATION ${EXECUTABLE_OUTPUT_PATH} )
_INSTALL ( FILES ${GVDB_DLL} SOURCE ${GVDB_LIB_DIR} DESTINATION ${EXECUTABLE_OUTPUT_PATH} )
_INSTALL ( FILES ${GVDB_EXTRA} SOURCE ${GVDB_LIB_DIR} DESTINATION ${EXECUTABLE_OUTPUT_PATH} )

#####################################################################################
# Install PTX to share location
#  
_INSTALL_PTX ( FILES ${PTX_FILES} DESTINATION ${EXECUTABLE_OUTPUT_PATH} )


#####################################################################################
# Library dependencies
#
set_property(GLOBAL PROPERTY DEBUG_CONFIGURATIONS Debug) 

foreach (loop_var IN ITEMS ${LIBRARIES_OPTIMIZED} )   
   target_link_libraries ( ${PROJNAME} optimized ${loop_var} )
endforeach()

foreach (loop_var IN ITEMS ${LIBRARIES_DEBUG} )
   target_link_libraries ( ${PROJNAME} debug ${loop_var} )
endforeach()

message ( STATUS "CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}" )
message ( STATUS "CMAKE_CURRENT_BINARY_DIR: ${CMAKE_CURRENT_BINARY_DIR}" )
message ( STATUS "EXECUTABLE_OUTPUT_PATH: ${EXECUTABLE_OUTPUT_PATH}" )


