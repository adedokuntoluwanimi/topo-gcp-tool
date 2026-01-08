# 🗺️ Topographic Survey GCP Generation Tool

**Automated terrain analysis and ground control point generation for drone surveys.**

100% Pure Python — No GDAL, no system dependencies — Guaranteed to work on Streamlit Cloud!

---

## ✨ Features

- **Slope Analysis** — Calculate terrain steepness in degrees
- **Aspect Analysis** — Determine slope direction (0-360°)
- **Flow Direction** — D8 algorithm for water flow patterns
- **Flow Accumulation** — Identify drainage channels
- **Watershed Delineation** — Map catchment basins
- **Contour Generation** — Create elevation lines at custom intervals
- **Stream Extraction** — Automatically map drainage networks
- **GCP Generation** — Create optimally-placed ground control points

---

## 📁 Export Formats

| Format | Use Case |
|--------|----------|
| **CSV** | Spreadsheets, data analysis |
| **KML** | Google Earth visualization |
| **GeoJSON** | GIS software (QGIS, ArcGIS) |
| **DXF** | CAD software (AutoCAD, Civil 3D) |
| **GPX** | GPS devices for field work |

---

## 🚀 Deployment (Streamlit Cloud)

1. **Fork/Clone this repository**
2. **Go to [share.streamlit.io](https://share.streamlit.io)**
3. **Click "New app"**
4. **Select your repository → main branch → app.py**
5. **Click Deploy!**

Your app will be live in ~2 minutes at `https://your-app.streamlit.app`

---

## 💻 Local Installation

```bash
# Clone the repo
git clone https://github.com/yourusername/topo-gcp-tool.git
cd topo-gcp-tool

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

---

## 📊 Input Requirements

- **Format:** GeoTIFF (.tif, .tiff)
- **Type:** Single-band elevation raster
- **Source:** Drone photogrammetry software (Pix4D, DroneDeploy, OpenDroneMap, etc.)

---

## 🔧 Configuration Options

### Contour Settings
- Interval: 0.5m to 20m

### GCP Settings
- Spacing: 50m to 500m
- Strategies: Grid Pattern, Terrain-Adaptive, Edge + Interior

### Hydrology
- Stream threshold: 50 to 1000

---

## 📄 License

MIT License — Free for commercial and personal use.

---

## 🙏 Credits

Built for **Geoinfotech — Kaduna Drone Topographic Survey Project**

Made with ❤️ using Streamlit, NumPy, SciPy, and Matplotlib
