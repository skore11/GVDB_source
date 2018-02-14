
NVIDIA GVDB Sparse Volumes SDK
Beta Release Program

OVERVIEW
============
NVIDIA GVDB is a new library and SDK for compute, simulation and rendering of sparse volumetric data. Details on the GVDB technology can be found at: http://developer.nvidia.com/gvdb
This Beta Release program shares GVDB and several samples with invited NVIDIA partners.


RELEASE NOTES
=============

12/6/2016, GVDB 1.0 Beta Release 3
- Improved build system for Windows and Linux
- Rendering using central rayCast function
- Function pointers for render specialization
- Fixes to trilinear surface rendering
- Added support for depth buffer integration (with GL)
- Improvements to color channels

10/25/2016, GVDB 1.0 Beta Release 2
- GVDB Library build on Windows & Linux 
- GVDB Samples build on Windows & Linux
- Linux install instructions (See below)
- Render to multiple render bufs (see g3DPrint)
- g3DPrint sample made interactive
- Dynamic atlas reallocation (see gFluidSim)
- Improved Insert/SplatPoints interface
- gImportVDB not yet updated

9/29/2016, GVDB 1.0 Beta Release 1
- Windows samples
- No linux build


REQUIREMENTS
============

  NVIDIA Kepler generation or later GPU
  Windows 7, 8, 10 64-bit
  Microsoft Visual Studio 2010 or 2012
  CUDA Toolkit 7.5 or higher
  CMake-GUI 2.7 or later
  OptiX 3.9.0 or later (InteractivOptix sample only, download from NVIDIA)
  OpenVBD for Windows (ImportVBD code sample only, available online as win_openvdb)

GVDB is released as a library with samples. 
The library and each sample is built separately, using cmake.


WHAT'S IN THE PACKAGE?
======================
	
   - GVDB API Library
   - Code Samples
	See the included GVDB_Samples_Description.pdf for detailed sample descriptions.
	- gRenderToFile     - Renders a sparse volume to a file using GVDB
	- gRenderToKernel   - Renders a sparse volume using a custom user-kernel
	- gInteractiveGL    - Interactive rendering of a volume using GVDB and OpenGL
	- gInteractiveOptiX - Interactive rendering of a volume and a polygonal model, with poly-to-poly and poly-to-voxel interactions.
	- g3DPrint          - Demonstrates generating cross section slices for 3D printing from a polygonal model
	- gSprayDeposit     - Demostrates simulated spray deposition onto a 3D part
	- gFluidSim         - Demostrates a dynamic simulation with surface rendering by GVDB
	- gImportVDB        - Loads and renders a sample OpenVDB file into GVDB
   - GVDB VBX File Specfication
   - GVDB Sample Descriptions


SAMPLES USAGE
=============
All interactive samples use the following user input interface
   Camera rotation -> move mouse
   Change orientation -> left mouse click
   Zoom -> right mouse click
   Panning -> hold middle button 
A few samples have on-screen GUIs with features that can be toggled by clicking on them.


BETA RELEASE NOTES
==================
   - The GVDB Beta implements GVDB API version 1.0
   - We may make changes to the GVDB API between Beta and Final releases.
   - We may make changes to the GVDB Samples between Beta and Final releases.
   - Package and samples run on Microsoft Visual Studio 2012 and 2015 for the Beta. A package that supports Linux-based projects will be available for 2016 release.
   - Dynamic topology is supported in this Beta release (see gFluidSim sample, ActivateSpace and FinishTopology)
   - Multiple channels are supported in this Beta release (see gFluidSim sample, AddChannel function)
   - Altas/pool expansion will be present in the 2016 release, but is not included in this Beta. (see arguments to the AddChannel function)
   - Additional optimizations may appear in future releases
   - Code is provided as-is. If you see any issues, please help us improve the package by using this submission form.
   - Future releases may include samples for export to OpenVDB, import of RAW data, direct simulation on GVDB grids, construction of GVDB from arbitrary functions, ray-guided out-of-core rendering, and potentially many others. Your input and feedback is welcome.


WINDOWS - QUICK INSTALLATION
============================

Instructions:

1. Unpackage GVDB and samples
    a. Unzip the GVDB SDK package to \gvdb\source

2. Install dependencies
    a. Install cmake-gui 2.7 or later
    b. Install CUDA Toolkit 7.5
    c. Install OptiX 3.9.0 or later (for gInteractiveOptix sample)

3. Build gvdb_library
    a. Run cmake-gui.
        Where is source code: /gvdb/source/gvdb_library
        Where to build bins:  /gvdb/build/gvdb_library
    b. Click Configure to prepare gvdb_library
    c. Click Generate
    d. Open /gvdb/build/gvdb_library/gvdb_library.sln in VS2010/2013
    e. Build the solution in Debug or Release mode.
       * For whichever mode, you must build later samples with same build type.
    f. The gvdb_library must be built prior to running cmake for any sample.

4. Build any sample, e.g. g3DPrint
    a. Run cmake-gui.
        Where is source code: /gvdb/source/g3DPrint
        Where to build bins:  /gvdb/build/g3DPrint
    b. Click Configure to prepare g3DPrint
    c. You should see that cmake locates the GVDB Library paths automatically
       * Specify any paths that cmake indicated are needed       
    d. Click Generate
    e. Open /gvdb/build/g3DPrint/g3DPrint.sln in VS2010/2013
    f. Build the solution

5. Run the sample!
    a. Select g3DPrint as the start up project.
    b. Click run/debug        

LINUX - QUICK INSTALLATION
==========================

Instructions: 


1. Pre-requisites
    a. Install CMake
          sudo apt-get install cmake-qt-gui
    b. Install the CUDA Toolkit 7.5 or later
          sudo ./cuda_7.5.18_linux.run
          * Must be done first, before you install NVIDIA drivers
    c. Install the NVIDIA R367 drivers or later 
          * These can be downloaded from the NVIDIA website
    d. Remove the symoblic libGL, which may incorrectly point to the libGL mesa driver.
          sudo rm -rf /usr/lib/x86_64-linux-gnu/libGL.so
    e. Link the libGL to the NVIDIA driver
          sudo ln -s /usr/lib/nvidia-367/libGL.so /usr/lib/x86_64-linux-gnu/libGL.so

2. Install OptiX [optional, for gInteractiveOptiX sample only]
      * OptiX is distributed as a .sh file, which extracts to the current dir.
      * Create a directory for optix in /usr/lib and move the package there before extracting.
      $ sudo mkdir /usr/lib/optix
      $ sudo mv NVIDIA-OptiX-SDK-4.0.1-linux64.sh /usr/lib/optix
      $ cd /usr/lib/optix
      $ sudo ./NVIDIA-OptiX-SDK-4.0.1.-linux64.sh

3. Set LD_LIBRARY_PATH in bashrc
     a. Open .bashrc. For example: $ emacs ~/.bashrc
     b. Add the following at the end:
           export LD_LIBRARY_PATH=/usr/local/gvdb/lib:/usr/lib/optix/lib64
           * The first path should be the location of libgvdb.so (once installed)
           * The second path should be the location of optix.so
     c. Source the bash (re-run it)
          $ source ~/.bashrc

4. Build the GVDB Library
     a. Unpackage the source tar.gz file
     b. mkdir ~/codes/build/gvdb_library   # make a folder for the build
     c. cmake-gui                          # run cmake-gui with the following settings:
         i.  source: ~/codes/source/gvdb_library
         ii. build:  ~/codes/build/gvdb_library
         iii. Click Configure, and the nGenerate
         iv. cd ~/codes/build/gvdb_library
         v.  sudo make  
         vi. sudo make install             # default install is to /usr/local/gvdb

5. Build a specific Sample
     a. Unpackage the source tar.gz file
     b. mkdir ~/codes/build/g3DPrint       # make a folder for the build
     c. cmake-gui                          # run cmake-gui with the following settings:
         i.  source: ~/codes/source/g3DPrint
         ii. build:  ~/codes/build/g3DPrint
         iii. Click Configure, and the nGenerate
              * Note: If GVDB is not found, set the GVDB_ROOT_DIR to /usr/local/gvdb
                or your preferred gvdb install location from step 4. 
         iv. cd ~/codes/build/g3DPrint
         v.  make
         vi. make install                  # remember to do 'make install' to get all files
     d. Run the sample!
          ./g3DPrint


End-User License Agreement 
==========================
Please refer to "Beta Software License Agreement.pdf"

