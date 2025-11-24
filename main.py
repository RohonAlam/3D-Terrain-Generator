#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3D Terrain Generator
"""

import sys
import os

# Prefer desktop OpenGL and optionally disable some VTK threading (set before importing VTK/Qt)
os.environ.setdefault('QT_OPENGL', 'desktop')
os.environ.setdefault('VTK_DISABLE_THREADS', '1')

import json
import time
import logging
import traceback
import re
from typing import List

# silence VTK textual output (must run before vtk modules are imported)
try:
    from vtkmodules.vtkCommonCore import vtkOutputWindow
    class _QuietOutputWindow(vtkOutputWindow):
        def DisplayText(self, text): return
        def DisplayErrorText(self, text): return
        def DisplayWarningText(self, text): return
    try:
        vtkOutputWindow.SetInstance(_QuietOutputWindow())
    except Exception:
        pass
except Exception:
    pass

# PyVista / 3D preview (do not set OFF_SCREEN globally here)
try:
    import pyvista as pv
    from pyvistaqt import QtInteractor
    PV_AVAILABLE = True

    # Reduce pyvista/vtk logger verbosity in Python logging
    try:
        logging.getLogger('pyvista').setLevel(logging.WARNING)
    except Exception:
        pass
    try:
        logging.getLogger('vtk').setLevel(logging.WARNING)
    except Exception:
        pass

    # If available, stop pyvista from redirecting error output to a file/stream
    try:
        pv.set_error_output_file(None)
    except Exception:
        pass
except Exception:
    PV_AVAILABLE = False

import requests

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QMessageBox,
    QSplitter, QListWidget, QLineEdit, QLabel, QDoubleSpinBox, QSplashScreen,
    QProgressDialog, QFileDialog, QHBoxLayout, QComboBox, QAction, QMenu, QInputDialog
)
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import QUrl, Qt, QObject, pyqtSlot, pyqtSignal, QThread, QSettings
from PyQt5.QtWebChannel import QWebChannel
from PyQt5.QtGui import QPixmap, QSurfaceFormat, QOpenGLContext

# Raster and image processing
try:
    import rasterio
    import numpy as np
    import matplotlib.pyplot as plt
    RASTER_AVAILABLE = True
except Exception:
    RASTER_AVAILABLE = False

from PyQt5.QtWidgets import QAction, QMessageBox
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtCore import QUrl


# ----------------- Helpers -----------------
def resource_path(relative_path: str) -> str:
    """Get absolute path (works with PyInstaller)."""
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath('.')
    return os.path.join(base_path, relative_path)

def next_filename(folder: str, prefix: str, ext: str) -> str:
    """Generate next available filename with incremental index and return full path."""
    os.makedirs(folder, exist_ok=True)
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+){re.escape(ext)}$")
    numbers = []
    for f in os.listdir(folder):
        m = pattern.match(f)
        if m:
            try:
                numbers.append(int(m.group(1)))
            except Exception:
                pass
    n = max(numbers) + 1 if numbers else 1
    return f"{prefix}_{n}{ext}"

def make_square(bounds: List[List[float]]) -> List[List[float]]:
    """
    Convert bounds [[south, west], [north, east]] (lat, lon) into a square bbox anchored at SW.
    Returns [[sw_lat, sw_lon], [ne_lat, ne_lon]].
    """
    (sw_lat, sw_lon), (ne_lat, ne_lon) = bounds
    lat_span = ne_lat - sw_lat
    lon_span = ne_lon - sw_lon
    side = max(lat_span, lon_span)
    return [[sw_lat, sw_lon], [sw_lat + side, sw_lon + side]]

# ----------------- Logging & settings -----------------
LOG_FOLDER = resource_path("terrain_logs")
os.makedirs(LOG_FOLDER, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_FOLDER, 'terrainapp.log')),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('TerrainApp')

SETTINGS = QSettings('TerrainAppAuthor', 'TerrainApp')

# ----------------- JS Bridge -----------------
class JSBridge(QObject):
    rectangleDrawn = pyqtSignal(str)

    @pyqtSlot(str)
    def sendRectangle(self, geojson_str):
        self.rectangleDrawn.emit(geojson_str)

# ----------------- Splash -----------------
class MapSplash(QSplashScreen):
    def __init__(self, text="Loading map..."):
        pixmap = QPixmap(600, 400)
        pixmap.fill(Qt.white)
        super().__init__(pixmap)
        self.showMessage(text, Qt.AlignCenter | Qt.AlignBottom, Qt.black)
        self.setWindowFlag(Qt.WindowStaysOnTopHint)
        self.setWindowFlag(Qt.FramelessWindowHint)
        self.setEnabled(False)

# ----------------- Map Loader -----------------
class MapLoader(QThread):
    map_ready = pyqtSignal(str, str)
    error = pyqtSignal(str)

    def __init__(self, center=(27.99,86.93), zoom_start=12, map_filename="map.html"):
        super().__init__()
        self.center = center
        self.zoom_start = zoom_start
        self.map_filename = map_filename

    def run(self):
        try:
            from folium import Map
            from folium.plugins import Draw
            import json
            import os

            # ---- Writable AppData folder ----
            appdata = os.path.join(os.path.expanduser("~"), "AppData", "Local", "TerrainApp", "internal")
            os.makedirs(appdata, exist_ok=True)
            out_path = os.path.join(appdata, self.map_filename)

            # Build the folium map
            m = Map(location=self.center, zoom_start=self.zoom_start)
            draw = Draw(export=True,
                        draw_options={'polyline': False, 'polygon': False, 'circle': False,
                                      'marker': False, 'circlemarker': False, 'rectangle': True})
            draw.add_to(m)

            map_html = m.get_root().render()
            map_var = m.get_name()

            # Inject JS for WebChannel link
            extra_js = f"""
<script src="qrc:///qtwebchannel/qwebchannel.js"></script>
<script>
document.addEventListener("DOMContentLoaded", function(){{
    new QWebChannel(qt.webChannelTransport, function(channel){{
        window.bridge = channel.objects.bridge;
        {map_var}.on('draw:created', function(e){{
            window.bridge.sendRectangle(JSON.stringify(e.layer.toGeoJSON()));
        }});
    }});
}});
</script>
"""
            final_html = map_html + extra_js

            # ---- Save HTML into AppData ----
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(final_html)

            self.map_ready.emit(out_path, map_var)

        except Exception as e:
            self.error.emit(str(e))

# ----------------- DEM Download Worker -----------------
class DEMDownloadWorker(QObject):
    finished=pyqtSignal(str)
    error=pyqtSignal(str)
    progress=pyqtSignal(int)
    def __init__(self, drawn_geojson, retries=3, backoff=2):
        super().__init__()
        self.drawn_geojson=drawn_geojson
        self._cancelled=False
        self.retries=retries; self.backoff=backoff
    def cancel(self): self._cancelled=True
    def run(self):
        try:
            # Attempt Earth Engine initialization (best-effort)
            try:
                import ee
                sa_path = resource_path(os.path.join("terrain_files","service_account.json"))
                credentials=None
                if os.path.exists(sa_path):
                    with open(sa_path,"r",encoding="utf-8") as f:
                        data=json.load(f)
                        email=data.get("client_email") or data.get("clientId")
                        if email:
                            credentials = ee.ServiceAccountCredentials(email, sa_path)
                try:
                    ee.Initialize(credentials) if credentials else ee.Initialize()
                except Exception:
                    ee.Authenticate(); ee.Initialize()
            except Exception as e:
                logger.warning("Earth Engine not available: %s", e)

            coords=self.drawn_geojson['geometry']['coordinates'][0]
            lons=[c[0] for c in coords]; lats=[c[1] for c in coords]
            sw, ne = make_square([[min(lats),min(lons)],[max(lats),max(lons)]])
            square_coords = [[sw[1],sw[0]],[sw[1],ne[0]],[ne[1],ne[0]],[ne[1],sw[0]],[sw[1],sw[0]]]
            square_geojson = {"type":"FeatureCollection","features":[{"type":"Feature","properties":{},"geometry":{"type":"Polygon","coordinates":[square_coords]}}]}
            url=None
            try:
                import ee
                geom = ee.Geometry(square_geojson['features'][0]['geometry'])
                dem = ee.Image('USGS/SRTMGL1_003').clip(geom)
                params = {'scale':30,'region':geom.getInfo()['coordinates'],'format':'GEO_TIFF','fileFormat':'GeoTIFF'}
                url = dem.getDownloadURL(params)
            except Exception as e:
                logger.exception("Failed to get EE download URL")
                self.error.emit("Failed to prepare DEM download: "+str(e)); return

            folder = resource_path(os.path.join("terrain_files","dem")); os.makedirs(folder, exist_ok=True)
            dem_file = os.path.join(folder, next_filename(folder,"dem",".tif"))
            attempt=0
            while attempt < self.retries and not self._cancelled:
                try:
                    with requests.get(url, stream=True, timeout=30) as r:
                        r.raise_for_status()
                        total = int(r.headers.get('Content-Length',0) or 0)
                        downloaded=0
                        with open(dem_file,"wb") as f:
                            for chunk in r.iter_content(chunk_size=8192):
                                if self._cancelled:
                                    f.close()
                                    try: os.remove(dem_file)
                                    except: pass
                                    self.error.emit("Download cancelled"); return
                                if chunk:
                                    f.write(chunk); downloaded+=len(chunk)
                                    if total:
                                        pct=int(downloaded/total*100); self.progress.emit(pct)
                    self.finished.emit(dem_file); return
                except Exception as e:
                    attempt+=1
                    logger.warning("Download attempt %d failed: %s", attempt, e)
                    if attempt >= self.retries:
                        logger.exception("All download attempts failed"); self.error.emit("DEM download failed: "+str(e)); return
                    time.sleep(self.backoff**attempt)
        except Exception as e:
            logger.exception("DEM worker crashed"); self.error.emit("DEM worker error: "+str(e))

# ----------------- MeshWorker (background - headless) -----------------
class MeshWorker(QObject):
    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    progress = pyqtSignal(int)

    def __init__(self,
                 dem_file,
                 vertical_scale,
                 base_size,
                 base_height,
                 stl_folder,
                 export_format='stl',
                 smooth=False,
                 smooth_iters=5,
                 decimate=0.0,
                 overlap_epsilon=1e-3):
        super().__init__()
        self.dem_file = dem_file
        self.vertical_scale = float(vertical_scale)
        self.base_size = float(base_size)
        self.base_height = float(base_height)
        self.stl_folder = stl_folder
        self.export_format = (export_format or 'stl').lower()
        self.smooth = bool(smooth)
        self.smooth_iters = int(smooth_iters)
        self.decimate = float(decimate)
        self.overlap_epsilon = float(overlap_epsilon)
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            import rasterio
            import numpy as np
            import pyvista as pv
            try:
                # Enable off-screen rendering only in this worker thread
                pv.OFF_SCREEN = True
            except Exception:
                pass

            # Read DEM
            with rasterio.open(self.dem_file) as src:
                elevation = src.read(1).astype(np.float32)
                nodata = src.nodata

            self.progress.emit(5)

            # Handle nodata
            if nodata is not None:
                mask = (elevation == nodata)
            else:
                mask = np.isnan(elevation)
            if mask.any():
                valid = elevation[~mask]
                if valid.size == 0:
                    raise RuntimeError("DEM contains no valid data")
                med = float(np.median(valid))
                elevation[mask] = med

            self.progress.emit(15)

            # Create grid and scale
            nrows, ncols = elevation.shape
            x, y = np.meshgrid(np.arange(ncols, dtype=np.float32), np.arange(nrows, dtype=np.float32))
            z_scaled = (elevation * self.vertical_scale).astype(np.float32)
            span_x = x.max() - x.min() if ncols > 1 else 1.0
            span_y = y.max() - y.min() if nrows > 1 else 1.0
            scale_factor = self.base_size / max(span_x if span_x != 0 else 1.0, span_y if span_y != 0 else 1.0)
            x_scaled = x * scale_factor
            y_scaled = y * scale_factor

            self.progress.emit(30)

            grid = pv.StructuredGrid(x_scaled, y_scaled, z_scaled)
            terrain = grid.extract_surface()

            self.progress.emit(45)

            # Smoothing
            if self.smooth and self.smooth_iters > 0:
                try:
                    terrain = terrain.smooth(n_iter=self.smooth_iters)
                except Exception:
                    logger.exception("Smoothing failed (continuing)")

            self.progress.emit(55)

            # Decimation: ensure triangles
            if 0.0 < self.decimate < 1.0:
                try:
                    try:
                        terrain = terrain.clean(tolerance=1e-6)
                    except Exception:
                        try:
                            terrain = terrain.clean()
                        except Exception:
                            pass
                    try:
                        terrain = terrain.triangulate()
                    except Exception:
                        logger.exception("Triangulation failed (continuing)")
                    if hasattr(terrain, 'decimate_pro'):
                        # newer pyvista/vtk expect 'reduction' or 'target_reduction' depending on version;
                        # we use 'reduction' which is supported in many versions; fall back to positional if needed.
                        try:
                            terrain = terrain.decimate_pro(reduction=float(self.decimate))
                        except TypeError:
                            terrain = terrain.decimate_pro(float(self.decimate))
                    else:
                        terrain = terrain.decimate(float(self.decimate))
                except Exception:
                    logger.exception("Decimation failed (continuing)")

            self.progress.emit(70)

            # Base and walls
            zmin = float(np.nanmin(z_scaled))
            base_z = zmin - (self.base_height + self.overlap_epsilon)
            xmin, xmax, ymin, ymax, _, _ = terrain.bounds
            base_center = (self.base_size / 2.0, self.base_size / 2.0, base_z)
            base = pv.Plane(center=base_center, i_size=self.base_size, j_size=self.base_size)

            walls = []
            try:
                edges = [
                    (np.full(nrows, xmin), np.linspace(ymin, ymax, nrows), z_scaled[:, 0]),
                    (np.full(nrows, xmax), np.linspace(ymin, ymax, nrows), z_scaled[:, -1]),
                    (np.linspace(xmin, xmax, ncols), np.full(ncols, ymin), z_scaled[0, :]),
                    (np.linspace(xmin, xmax, ncols), np.full(ncols, ymax), z_scaled[-1, :])
                ]
                for xv, yv, zv in edges:
                    x_pts = np.column_stack([xv, xv])
                    y_pts = np.column_stack([yv, yv])
                    z_pts = np.column_stack([np.full_like(zv, zmin - self.base_height), zv])
                    try:
                        walls.append(pv.StructuredGrid(x_pts, y_pts, z_pts).extract_surface())
                    except Exception:
                        logger.exception("Failed to build one wall edge (skipping)")
            except Exception:
                logger.exception("Wall construction failed")

            self.progress.emit(85)

            mesh = terrain
            for w in walls:
                mesh = mesh.merge(w)
            mesh = mesh.merge(base)

            # Cleanup
            try:
                mesh = mesh.clean(tolerance=1e-6)
            except Exception:
                try:
                    mesh = mesh.clean()
                except Exception:
                    pass
            try:
                mesh = mesh.triangulate()
            except Exception:
                pass

            self.progress.emit(95)

            # Save
            os.makedirs(self.stl_folder, exist_ok=True)
            ext = '.' + (self.export_format if self.export_format in ('stl','obj','ply') else 'stl')
            out_file = os.path.join(self.stl_folder, next_filename(self.stl_folder, "terrain", ext))
            mesh.save(out_file)

            self.progress.emit(100)
            self.finished.emit(out_file)
        except Exception as e:
            logger.exception("MeshWorker failed")
            try:
                self.error.emit(str(e))
            except Exception:
                traceback.print_exc()

# ----------------- Map Window -----------------
class MapWindow(QWidget):
    dem_downloaded = pyqtSignal(str)
    map_ready_signal = pyqtSignal()
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout(self)
        self.map_var_name=None
        self.search_input = QLineEdit(); self.search_input.setPlaceholderText("Enter location and press Enter")
        self.search_input.returnPressed.connect(self.search_location)
        self.layout.addWidget(self.search_input)
        self.webview = QWebEngineView()
        self.webview.setHtml("<div style='text-align:center;margin-top:30%'><h3>Loading map...</h3></div>")
        self.download_button = QPushButton("Download DEM for Selected Area")
        self.download_button.clicked.connect(self.download_dem)
        self.download_button.setEnabled(False)
        self.layout.addWidget(self.webview); self.layout.addWidget(self.download_button)
        self.channel = QWebChannel()
        self.bridge = JSBridge(); self.bridge.rectangleDrawn.connect(self.on_rectangle_drawn)
        self.channel.registerObject("bridge", self.bridge)
        self.map_loader = MapLoader(map_filename="map.html")
        self.map_loader.map_ready.connect(self.on_map_ready)
        self.map_loader.error.connect(self.on_map_error)
        self.map_loader.start()
        self.drawn_geojson=None

    def on_map_ready(self, html_path, map_var_name):
        self.map_var_name = map_var_name
        try:
            self.webview.page().setWebChannel(self.channel)
            self.webview.load(QUrl.fromLocalFile(html_path))
            self.map_ready_signal.emit()
        except Exception as e:
            QMessageBox.critical(self,"Map Load Error",str(e))
    def on_map_error(self,message):
        QMessageBox.critical(self,"Map Error",message)
    def search_location(self):
        loc=self.search_input.text()
        if not loc or not self.map_var_name: return
        import urllib.request, urllib.parse
        try:
            q = urllib.parse.quote(loc)
            url=f"https://nominatim.openstreetmap.org/search?format=json&q={q}"
            with urllib.request.urlopen(url, timeout=10) as response:
                data=json.loads(response.read())
                if data:
                    lat,lon=float(data[0]['lat']),float(data[0]['lon'])
                    js=f"if(typeof {self.map_var_name} !== 'undefined'){{ {self.map_var_name}.setView([{lat},{lon}],12); }}"
                    self.webview.page().runJavaScript(js)
        except Exception as e:
            QMessageBox.warning(self,"Search Failed",str(e))
    def on_rectangle_drawn(self, geojson_str):
        self.drawn_geojson=json.loads(geojson_str); self.download_button.setEnabled(True)
    def download_dem(self):
        if not self.drawn_geojson:
            QMessageBox.warning(self,"No Rectangle","Please draw a rectangle first!")
            return
        self.download_button.setEnabled(False)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        self.dem_thread = QThread()
        self.dem_worker = DEMDownloadWorker(self.drawn_geojson)
        self.dem_worker.moveToThread(self.dem_thread)
        self.dem_worker.finished.connect(self._on_dem_finished)
        self.dem_worker.error.connect(self._on_dem_error)
        self.dem_worker.progress.connect(self._on_dem_progress)
        self.dem_thread.started.connect(self.dem_worker.run)
        self.dem_worker.finished.connect(self.dem_thread.quit)
        self.dem_worker.finished.connect(self.dem_worker.deleteLater)
        self.dem_thread.finished.connect(self.dem_thread.deleteLater)
        # progress dialog
        self.progress_dialog = QProgressDialog('Downloading DEM...', 'Cancel', 0, 100, self)
        self.progress_dialog.setWindowModality(Qt.WindowModal)
        self.progress_dialog.canceled.connect(self._cancel_dem)
        self.progress_dialog.show()
        self.dem_thread.start()
    def _on_dem_progress(self,pct):
        if hasattr(self,'progress_dialog') and self.progress_dialog:
            self.progress_dialog.setValue(pct)
    def _cancel_dem(self):
        if hasattr(self,'dem_worker') and self.dem_worker:
            self.dem_worker.cancel()
    def _on_dem_finished(self, dem_file):
        QApplication.restoreOverrideCursor()
        self.download_button.setEnabled(True)
        if hasattr(self,'progress_dialog') and self.progress_dialog:
            self.progress_dialog.setValue(100); self.progress_dialog.close(); self.progress_dialog=None
        QMessageBox.information(self,"Download Complete",f"DEM saved as '{dem_file}'")
        self.dem_downloaded.emit(dem_file)
    def _on_dem_error(self,message):
        QApplication.restoreOverrideCursor()
        self.download_button.setEnabled(True)
        if hasattr(self,'progress_dialog') and self.progress_dialog:
            self.progress_dialog.close(); self.progress_dialog=None
        QMessageBox.critical(self,"DEM Error",message)

# ----------------- STL Viewer Window -----------------
class STLViewerWindow(QMainWindow):
    def __init__(self, stl_path, parent=None):
        # Ensure this is a top-level window (avoid parenting to main window)
        super().__init__(parent=None)
        self.setWindowTitle(f"3D Viewer - {os.path.basename(stl_path)}")
        self.resize(800,600)
        # Make it a proper window and ask Qt to delete it on close
        self.setWindowFlags(self.windowFlags() | Qt.Window)
        self.setAttribute(Qt.WA_DeleteOnClose, True)

        if not PV_AVAILABLE:
            lbl = QLabel("pyvista/pyvistaqt not available. Cannot show 3D preview.")
            self.setCentralWidget(lbl)
            return

        self.pv_widget = None
        try:
            # Ensure interactive viewer uses on-screen OpenGL
            try:
                import pyvista as _pv
                _pv.OFF_SCREEN = False
            except Exception:
                pass

            # Create the QtInteractor as a child of this top-level window
            self.pv_widget = QtInteractor(self)
            self.setCentralWidget(self.pv_widget)

            mesh = pv.read(stl_path)
            self.pv_widget.add_mesh(mesh, show_edges=False)
            self.pv_widget.reset_camera()
        except Exception:
            logger.exception("Failed to open STL in viewer")
            lbl = QLabel("Failed to load STL file. See logs.")
            self.setCentralWidget(lbl)

    def closeEvent(self, event):
        # Cleanly destroy the PyVista/VTK widget and its GL context to avoid flicker
        try:
            if hasattr(self, 'pv_widget') and self.pv_widget is not None:
                try:
                    self.pv_widget.close()
                except Exception:
                    pass
                # small delay to help some drivers release GL handles cleanly
                time.sleep(0.08)
                try:
                    self.pv_widget.deleteLater()
                except Exception:
                    pass
                self.pv_widget = None
            # Ensure pyvista remains in interactive mode for the rest of the app
            try:
                import pyvista as _pv
                _pv.OFF_SCREEN = False
            except Exception:
                pass
        except Exception:
            logger.exception("Error during STLViewerWindow.closeEvent")
        super().closeEvent(event)

# ----------------- DEM Viewer Window -----------------
class DEMViewerWindow(QMainWindow):
    def __init__(self, dem_path, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"DEM Viewer - {os.path.basename(dem_path)}")
        self.resize(700,600)
        label = QLabel(); label.setAlignment(Qt.AlignCenter)
        self.setCentralWidget(label)
        if not RASTER_AVAILABLE:
            label.setText("rasterio/numpy/matplotlib not available. Cannot show DEM image.")
            return
        try:
            with rasterio.open(dem_path) as src:
                data = src.read(out_shape=(1, 1024, 1024))
                if data.ndim == 3:
                    if data.shape[0] > 1:
                        arr = np.mean(data, axis=0)
                    else:
                        arr = data[0]
                else:
                    arr = np.squeeze(data)
                arr = np.array(arr, dtype=float)
                if arr.ndim != 2:
                    arr = arr.reshape((arr.shape[-2], arr.shape[-1]))
                gy, gx = np.gradient(arr)
                slope = np.pi/2.0 - np.arctan(np.hypot(gx, gy))
                mn = np.nanmin(slope); mx = np.nanmax(slope)
                shaded = (slope - mn) / (mx - mn + 1e-12)
                fig = plt.figure(frameon=False); fig.set_size_inches(6,6)
                ax = fig.add_axes([0,0,1,1]); ax.axis('off')
                ax.imshow(shaded, cmap='gray', origin='lower')
                tmp = resource_path(os.path.join('terrain_files','last_dem_view.png'))
                os.makedirs(os.path.dirname(tmp), exist_ok=True)
                fig.savefig(tmp, dpi=100); plt.close(fig)
                pix = QPixmap(tmp); label.setPixmap(pix.scaled(self.width()-40, self.height()-40, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        except Exception:
            logger.exception("Failed to render DEM image")
            label.setText("Failed to render DEM image. See logs.")

# ----------------- Main GUI -----------------
class TerrainApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D Terrain Generator v2.0")
        self.setGeometry(100,100,1400,800)
        self.splash=MapSplash("Loading map..."); self.splash.show()
        # keep strong refs to top-level viewer windows so they don't get GC'd
        self._viewer_windows = []

        splitter=QSplitter(Qt.Horizontal)

        # Left panel
        left_widget=QWidget(); left_layout=QVBoxLayout(left_widget)
        left_layout.addWidget(QLabel("DEM Files")); self.dem_list=QListWidget(); left_layout.addWidget(self.dem_list)
        left_layout.addWidget(QLabel("STL/3D Files")); self.stl_list=QListWidget(); left_layout.addWidget(self.stl_list)
        self.vertical_scale_spin=QDoubleSpinBox(); self.vertical_scale_spin.setRange(0.001,10.0); self.vertical_scale_spin.setValue(0.01)
        left_layout.addWidget(QLabel("Vertical Scale")); left_layout.addWidget(self.vertical_scale_spin)
        self.base_size_spin=QDoubleSpinBox(); self.base_size_spin.setRange(5.0,500.0); self.base_size_spin.setValue(200.0)
        left_layout.addWidget(QLabel("Base Size (mm)")); left_layout.addWidget(self.base_size_spin)
        self.base_height_spin=QDoubleSpinBox(); self.base_height_spin.setRange(0.1,50.0); self.base_height_spin.setValue(2.0)
        left_layout.addWidget(QLabel("Base Height (mm)")); left_layout.addWidget(self.base_height_spin)
        left_layout.addWidget(QLabel("Export Format"))
        self.export_combo=QComboBox(); self.export_combo.addItems(['STL','OBJ','PLY']); left_layout.addWidget(self.export_combo)
        left_layout.addWidget(QLabel("Mesh Smoothing & Decimation"))
        self.smooth_combo=QComboBox(); self.smooth_combo.addItems(['None','Light','Medium','Strong']); left_layout.addWidget(self.smooth_combo)
        left_layout.addStretch()
        self.stl_button=QPushButton("Convert selected DEM to 3D"); self.stl_button.clicked.connect(self.convert_to_3d); left_layout.addWidget(self.stl_button)
        self.dem_list.setContextMenuPolicy(Qt.CustomContextMenu); self.dem_list.customContextMenuRequested.connect(self.dem_context_menu)
        self.stl_list.setContextMenuPolicy(Qt.CustomContextMenu); self.stl_list.customContextMenuRequested.connect(self.stl_context_menu)
        splitter.addWidget(left_widget)

        # Right panel -- now ONLY map
        right_widget=QWidget(); right_layout=QVBoxLayout(right_widget)
        self.map_window=MapWindow(); self.map_window.dem_downloaded.connect(self.add_dem_file); self.map_window.map_ready_signal.connect(self.finish_splash)
        right_layout.addWidget(self.map_window)
        splitter.addWidget(right_widget)
        self.setCentralWidget(splitter); splitter.setSizes([360,1040])

        self.terrain_files = resource_path("terrain_files")
        self.dem_folder=resource_path(os.path.join("terrain_files","dem")); self.stl_folder=resource_path(os.path.join("terrain_files","stl"))
        os.makedirs(self.dem_folder,exist_ok=True); os.makedirs(self.stl_folder,exist_ok=True)
        self.update_file_lists(); self.selected_dem=None; self.current_worker=None
        self.load_settings()
        self.dem_list.itemSelectionChanged.connect(self.on_dem_selected)
        self.stl_list.itemSelectionChanged.connect(self.on_stl_selected)
        self._build_menu()

    def _build_menu(self):
        menubar=self.menuBar(); file_menu=menubar.addMenu("&File")
        open_folder_act = QAction("Open Folder", self);open_folder_act.triggered.connect(self.open_terrain_folder);file_menu.addAction(open_folder_act)
        save_act=QAction("Save Settings",self); save_act.triggered.connect(self.save_settings); file_menu.addAction(save_act)
        load_act=QAction("Load Settings",self); load_act.triggered.connect(self.load_settings); file_menu.addAction(load_act)
        exit_act=QAction("Exit",self); exit_act.triggered.connect(self.close); file_menu.addAction(exit_act)
        help_menu=menubar.addMenu("&Help"); 
        about_act=QAction("About",self); about_act.triggered.connect(self._about); help_menu.addAction(about_act)
        repo_act = QAction("Github", self);repo_act.triggered.connect(self._Github);help_menu.addAction(repo_act)


        act_3d = QAction("Open 3D Viewer", self); act_3d.triggered.connect(self.open_3d_viewer_blank)
        act_dem = QAction("Open DEM Viewer", self); act_dem.triggered.connect(self.open_dem_viewer_blank)
        menubar.addAction(act_3d); menubar.addAction(act_dem)

    def _about(self): QMessageBox.information(self,"About","3D Terrain Generator v2.0\nEnhanced version \nCreated by Rohon Alam \nMore Updates are Coming Soon! \nContact: \n     rohon.alam1555@gmail.coom")
    def _Github(self): QDesktopServices.openUrl(QUrl("https://github.com/RohonAlam/3D-Terrain-Generator"))
    def finish_splash(self): self.splash.close()
    def open_terrain_folder(self):
        folder_path = self.terrain_files
        if not os.path.exists(folder_path):
            QMessageBox.warning(self, "Folder Not Found", f"The folder {folder_path} does not exist.")
            return
        url = QUrl.fromLocalFile(folder_path)
        if not QDesktopServices.openUrl(url):
            QMessageBox.warning(self, "Open Folder Failed", "Could not open the folder in file explorer.")

    def load_settings(self):
        self.vertical_scale_spin.setValue(float(SETTINGS.value('vertical_scale',0.01)))
        self.base_size_spin.setValue(float(SETTINGS.value('base_size',200.0)))
        self.base_height_spin.setValue(float(SETTINGS.value('base_height',2.0)))
        self.export_combo.setCurrentText(SETTINGS.value('export_format','STL'))
        self.smooth_combo.setCurrentText(SETTINGS.value('smooth_level','Light'))

    def save_settings(self):
        SETTINGS.setValue('vertical_scale',self.vertical_scale_spin.value())
        SETTINGS.setValue('base_size',self.base_size_spin.value())
        SETTINGS.setValue('base_height',self.base_height_spin.value())
        SETTINGS.setValue('export_format',self.export_combo.currentText())
        SETTINGS.setValue('smooth_level',self.smooth_combo.currentText())
        QMessageBox.information(self,"Settings","Settings saved")

    def update_file_lists(self):
        self.dem_list.clear(); self.stl_list.clear()
        for f in sorted(os.listdir(self.dem_folder)):
            try:
                size=os.path.getsize(os.path.join(self.dem_folder,f))/1024
                self.dem_list.addItem(f"{f} ({size:.1f} KB)")
            except Exception:
                self.dem_list.addItem(f)
        for f in sorted(os.listdir(self.stl_folder)):
            try:
                size=os.path.getsize(os.path.join(self.stl_folder,f))/1024
                self.stl_list.addItem(f"{f} ({size:.1f} KB)")
            except Exception:
                self.stl_list.addItem(f)

    def add_dem_file(self,file_path):
        target_file=os.path.join(self.dem_folder,os.path.basename(file_path))
        try: os.replace(file_path,target_file)
        except Exception: import shutil; shutil.copy(file_path,target_file)
        self.selected_dem=target_file; self.update_file_lists();
        self.generate_thumbnail(target_file)

    def generate_thumbnail(self, dem_path: str):
        if not RASTER_AVAILABLE:
            return
        try:
            with rasterio.open(dem_path) as src:
                data = src.read(out_shape=(1, 256, 256))
                if data.ndim == 3:
                    if data.shape[0] > 1:
                        arr = np.mean(data, axis=0)
                    else:
                        arr = data[0]
                else:
                    arr = np.squeeze(data)
                arr = np.array(arr, dtype=float)
                if arr.ndim != 2:
                    arr = arr.reshape((arr.shape[-2], arr.shape[-1]))
                gy, gx = np.gradient(arr)
                slope = np.pi/2.0 - np.arctan(np.hypot(gx, gy))
                if np.all(np.isfinite(slope)):
                    mn = np.nanmin(slope); mx = np.nanmax(slope)
                    shaded = (slope - mn) / (mx - mn + 1e-12)
                else:
                    shaded = np.nan_to_num(slope)
                fig = plt.figure(frameon=False); fig.set_size_inches(2.56,2.56)
                ax = fig.add_axes([0,0,1,1]); ax.axis('off')
                ax.imshow(shaded, cmap='gray', origin='lower')
                tmp = resource_path(os.path.join('terrain_files','thumb.png')); os.makedirs(os.path.dirname(tmp), exist_ok=True)
                fig.savefig(tmp, dpi=100); plt.close(fig)
        except Exception:
            logger.exception("Thumbnail generation failed")

    def on_dem_selected(self):
        items = self.dem_list.selectedItems()
        if not items: return
        text = items[0].text().split(' ')[0]
        file = os.path.join(self.dem_folder, text)
        if os.path.exists(file):
            try:
                viewer = DEMViewerWindow(file, parent=None)
                viewer.setAttribute(Qt.WA_DeleteOnClose, True)
                # keep reference
                self._viewer_windows.append(viewer)
                # remove when closed/destroyed
                viewer.destroyed.connect(lambda _, v=viewer: self._viewer_windows.remove(v) if v in self._viewer_windows else None)
                viewer.show()
            except Exception:
                logger.exception("Failed to open DEM viewer")

    def on_stl_selected(self):
        items = self.stl_list.selectedItems()
        if not items: return
        text = items[0].text().split(' ')[0]
        file = os.path.join(self.stl_folder, text)
        if os.path.exists(file):
            try:
                viewer = STLViewerWindow(file, parent=None)
                viewer.setAttribute(Qt.WA_DeleteOnClose, True)
                self._viewer_windows.append(viewer)
                viewer.destroyed.connect(lambda _, v=viewer: self._viewer_windows.remove(v) if v in self._viewer_windows else None)
                viewer.show()
            except Exception:
                logger.exception("Failed to open STL viewer")

    def dem_context_menu(self,pos):
        idx=self.dem_list.indexAt(pos)
        if not idx.isValid(): return
        item=self.dem_list.item(idx.row())
        menu=QMenu(); open_act=QAction('Open in Folder',self); open_act.triggered.connect(lambda: os.startfile(self.dem_folder))
        reproc_act=QAction('Re-generate Thumbnail',self); reproc_act.triggered.connect(lambda: self.generate_thumbnail(os.path.join(self.dem_folder,item.text().split(' ')[0])))
        delete_act=QAction('Delete',self); delete_act.triggered.connect(lambda: self._delete_file(os.path.join(self.dem_folder,item.text().split(' ')[0])))
        menu.addAction(open_act); menu.addAction(reproc_act); menu.addAction(delete_act); menu.exec_(self.dem_list.mapToGlobal(pos))

    def stl_context_menu(self,pos):
        idx=self.stl_list.indexAt(pos)
        if not idx.isValid(): return
        item=self.stl_list.item(idx.row())
        menu=QMenu(); open_act=QAction('Open in Folder',self); open_act.triggered.connect(lambda: os.startfile(self.stl_folder))
        export_act=QAction('Export As...',self); export_act.triggered.connect(lambda: self._export_as(os.path.join(self.stl_folder,item.text().split(' ')[0])))
        delete_act=QAction('Delete',self); delete_act.triggered.connect(lambda: self._delete_file(os.path.join(self.stl_folder,item.text().split(' ')[0])))
        menu.addAction(open_act); menu.addAction(export_act); menu.addAction(delete_act); menu.exec_(self.stl_list.mapToGlobal(pos))

    def _delete_file(self,path):
        try: os.remove(path); self.update_file_lists()
        except Exception as e: QMessageBox.warning(self,'Delete Failed',str(e))

    def _export_as(self,path):
        fmt, ok = QInputDialog.getItem(self,'Export format','Choose format',['stl','obj','ply'],0,False)
        if not ok: return
        try:
            mesh = pv.read(path)
            out = QFileDialog.getSaveFileName(self,'Export As', resource_path(os.path.join('terrain_files','exports', os.path.basename(path))))[0]
            if out:
                mesh.save(out); QMessageBox.information(self,'Export','Exported to '+out)
        except Exception:
            logger.exception("Export failed"); QMessageBox.critical(self,'Export failed','See log for details')

    def convert_to_3d(self):
        if not self.selected_dem:
            QMessageBox.warning(self,'No DEM','Please select or download a DEM first!'); return
        self.stl_button.setEnabled(False)
        smooth_map = {'None': False, 'Light': True, 'Medium': True, 'Strong': True}
        dec_map = {'None': 0.0, 'Light': 0.25, 'Medium': 0.5, 'Strong': 0.75}
        smooth = smooth_map.get(self.smooth_combo.currentText(), True)
        decimate = dec_map.get(self.smooth_combo.currentText(), 0.5)
        export_format = self.export_combo.currentText().lower()

        self.mesh_thread = QThread()
        self.mesh_worker = MeshWorker(self.selected_dem, self.vertical_scale_spin.value(), self.base_size_spin.value(), self.base_height_spin.value(), self.stl_folder, export_format, smooth, 5, decimate, 1e-3)
        self.mesh_worker.moveToThread(self.mesh_thread)
        self.mesh_worker.progress.connect(self._on_mesh_progress)
        self.mesh_thread.started.connect(self.mesh_worker.run)
        self.mesh_worker.finished.connect(self._on_mesh_finished)
        self.mesh_worker.error.connect(self._on_mesh_error)
        self.mesh_worker.finished.connect(self.mesh_thread.quit)
        self.mesh_worker.finished.connect(self.mesh_worker.deleteLater)
        self.mesh_thread.finished.connect(self.mesh_thread.deleteLater)

        self.mesh_progress = QProgressDialog('Generating 3D mesh...', 'Cancel', 0, 100, self)
        self.mesh_progress.setWindowModality(Qt.WindowModal); self.mesh_progress.canceled.connect(self._cancel_mesh)
        self.mesh_progress.show()

        self.mesh_thread.start()

    def _on_mesh_progress(self,pct):
        if hasattr(self,'mesh_progress') and self.mesh_progress:
            self.mesh_progress.setValue(pct)

    def _cancel_mesh(self):
        if hasattr(self,'mesh_worker') and self.mesh_worker:
            self.mesh_worker.cancel()

    def _on_mesh_finished(self,stl_file):
        if hasattr(self,'mesh_progress') and self.mesh_progress:
            self.mesh_progress.close(); self.mesh_progress=None
        self.stl_button.setEnabled(True)
        QMessageBox.information(self,'Mesh Saved',f'3D file saved as {stl_file}')
        self.update_file_lists()

    def _on_mesh_error(self,message):
        if hasattr(self,'mesh_progress') and self.mesh_progress:
            self.mesh_progress.close(); self.mesh_progress=None
        self.stl_button.setEnabled(True)
        QMessageBox.critical(self,'Mesh Error',message)

    def open_3d_viewer_blank(self):
        if not PV_AVAILABLE:
            QMessageBox.warning(self, '3D Viewer', 'pyvista/pyvistaqt not available on this system.')
            return
        w = QMainWindow(None)
        w.setWindowTitle('3D Viewer')
        w.resize(800,600)
        try:
            try:
                import pyvista as _pv
                _pv.OFF_SCREEN = False
            except Exception:
                pass
            widget = QtInteractor(w)
            w.setCentralWidget(widget)
        except Exception:
            logger.exception('Failed to create blank 3D viewer')
            QMessageBox.warning(self, '3D Viewer', 'Failed to create 3D viewer window.')
            return
        w.show()

    def open_dem_viewer_blank(self):
        w = QMainWindow(None)
        w.setWindowTitle('DEM Viewer')
        w.resize(700,600)
        lbl = QLabel('Open a DEM from the list to view it here.'); lbl.setAlignment(Qt.AlignCenter)
        w.setCentralWidget(lbl)
        w.show()

# ----------------- Run -----------------
if __name__ == "__main__":
    from PyQt5.QtGui import QSurfaceFormat, QIcon

    # OpenGL setup (your existing code)
    fmt = QSurfaceFormat()
    fmt.setRenderableType(QSurfaceFormat.OpenGL)
    try:
        fmt.setProfile(QSurfaceFormat.CompatibilityProfile)
    except Exception:
        pass
    fmt.setVersion(2, 0)
    fmt.setDepthBufferSize(24)
    QSurfaceFormat.setDefaultFormat(fmt)

    app = QApplication(sys.argv)

    # ---- NEW: set global app icon ----
    try:
        app.setWindowIcon(QIcon(resource_path("app_icon.ico")))
    except Exception as e:
        print("Could not load app icon:", e)

    window = TerrainApp()

    # ---- NEW: set main window icon ----
    try:
        window.setWindowIcon(QIcon(resource_path("app_icon.ico")))
    except Exception:
        pass

    window.show()
    sys.exit(app.exec_())
