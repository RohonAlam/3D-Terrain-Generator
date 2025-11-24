##################################################
README — Building the "3D Terrain Generator" Application
##################################################
This guide explains how to set up the environment, install dependencies, 
and build the 3D Terrain Print application using PyInstaller.

###############################
@1. Create a Virtual Environment
###############################
In your project folder, run:
python -m venv venv

###############################################
@2.Activate the Virtual Environment (PowerShell)
###############################################
Run:
.\venv\Scripts\Activate.ps1

If you see this error:
“File Activate.ps1 cannot be loaded because running scripts is disabled...”

Fix it temporarily for this session:
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

Then activate again:
.\venv\Scripts\Activate.ps1

When activation succeeds, your prompt will look like:
(venv) PS D:\FINAL_AM_PROJECT>

######################
@3.Install Dependencies
#######################
Option A — From requirements.txt (recommended)

pip install -r requirements.txt

Option B — Install manually

pip install pyinstaller pyqt5 pyqtwebengine pyvista pyvistaqt vtk rasterio numpy matplotlib folium requests earthengine-api

##############################################
@4.(Optional) Create requirements.txt Manually
##############################################
If you want to maintain dependencies in a file:

Create the file:
requirements.txt

Paste the following lines into it:

	pyqt5
	pyqtwebengine
	pyvista
	pyvistaqt
	vtk
	rasterio
	numpy
	matplotlib
	folium
	requests
	earthengine-api
Save the file, then install:

pip install -r requirements.txt

#####################################
@5.Build the Application with PyInstaller
#####################################
Run:
pyinstaller terrainapp.spec

(Copy the terrainapp.spec from the zip folder )

This will generate a dist/ folder containing your built application.

Run the Application

After the build completes, run:


dist\TerrainApp\TerrainApp.exe

#################
@6.Troubleshooting
#################
If you encounter missing-module or DLL errors during runtime, rebuild with:

pyinstaller --clean terrainapp.spec
Run the EXE from a console (Command Prompt or PowerShell) to see detailed error messages.

You’re Done!

Note :- put App_icon in the same folder as all other source files

You now have a built executable of the 3D Terrain Print application ready to distribute or run on Windows.
