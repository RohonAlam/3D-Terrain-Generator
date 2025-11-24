# terrainapp.spec
# PyInstaller spec for 3D Terrain Generator (PyQt5 + QtWebEngine + pyvista/vtk + rasterio)
# Edit main_py if your main script is different.

import os
from PyInstaller.utils.hooks import collect_submodules, collect_data_files, copy_metadata
from PyInstaller.building.build_main import Analysis, PYZ, EXE, COLLECT

main_py = 'main.py'    # <-- change if needed
app_name = '3D Terrain Generator'

# ---------------------------
# Start with empty lists (important: define before using +=)
# ---------------------------
hiddenimports = []
datas = []
binaries = []

# ---------------------------
# Hidden imports: dynamic packages PyInstaller may miss
# ---------------------------
hiddenimports += [
    'PyQt5.QtWebEngineWidgets',
    'PyQt5.QtWebChannel',
    'pyvista', 'pyvistaqt',
    'vtkmodules', 'vtkmodules.vtkCommonCore',
    'requests',
    'rasterio', 'numpy', 'matplotlib'
]

# Collect submodules for heavy dynamic packages
hiddenimports += collect_submodules('pyvista')
hiddenimports += collect_submodules('vtk')
hiddenimports += collect_submodules('numpy')
hiddenimports += collect_submodules('matplotlib')
# rasterio dynamic submodules
hiddenimports += collect_submodules('rasterio')

# ---------------------------
# Data files to include
# ---------------------------
# Include your terrain_files folder (if present)
proj_root = os.path.abspath('.')
terrain_files_dir = os.path.join(proj_root, 'terrain_files')
if os.path.isdir(terrain_files_dir):
    for root, _, files in os.walk(terrain_files_dir):
        for f in files:
            src = os.path.join(root, f)
            rel_dir = os.path.relpath(root, proj_root)
            datas.append((src, rel_dir))

# Include map.html if present
map_html = os.path.join(proj_root, 'map.html')
if os.path.exists(map_html):
    datas.append((map_html, '.'))

# Include app icon (ensure app_icon.ico is in project root)
app_icon_path = os.path.join(proj_root, 'app_icon.ico')
if os.path.exists(app_icon_path):
    datas.append((app_icon_path, '.'))

# Include PyQt5/Qt resources (helps ensure QtWebEngine files are present)
try:
    datas += collect_data_files('PyQt5', subdir='Qt', include_py_files=True)
except Exception:
    pass

# Include package metadata (optional)
try:
    datas += copy_metadata('pyqt5')
except Exception:
    pass

# rasterio package data
try:
    datas += collect_data_files('rasterio')
except Exception:
    pass

# ---------------------------
# binaries: include compiled libs (rasterio .pyd/.dll and Qt binaries if found)
# ---------------------------
# Attempt to include rasterio compiled binaries from its package dir
try:
    import rasterio
    rasterio_pkg_dir = os.path.dirname(rasterio.__file__)
    for fname in os.listdir(rasterio_pkg_dir):
        if fname.lower().endswith(('.pyd', '.dll', '.so')):
            binaries.append((os.path.join(rasterio_pkg_dir, fname), os.path.join('rasterio')))
except Exception:
    # rasterio not installed in build env — OK, spec will still evaluate
    pass

# Attempt to include Qt WebEngineProcess (common cause of WebEngine failures)
try:
    import PyQt5
    pyqt5_path = os.path.dirname(PyQt5.__file__)
    qt_bin = os.path.join(pyqt5_path, 'Qt', 'bin')
    qweb_proc = os.path.join(qt_bin, 'QtWebEngineProcess.exe')
    if os.path.exists(qweb_proc):
        binaries.append((qweb_proc, os.path.join('Qt','bin')))
    # Include other Qt .dlls from Qt/bin if present (optional)
    for f in os.listdir(qt_bin) if os.path.isdir(qt_bin) else []:
        if f.lower().endswith(('.dll', '.exe', '.pyd')):
            binaries.append((os.path.join(qt_bin, f), os.path.join('Qt','bin')))
except Exception:
    pass

# ---------------------------
# Analysis / build
# ---------------------------
a = Analysis(
    [main_py],
    pathex=[proj_root],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=app_name,
    icon='app_icon.ico',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False  # set True while debugging to see tracebacks in a console
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name=app_name
)
