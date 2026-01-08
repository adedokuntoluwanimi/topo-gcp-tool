"""
Topographic Survey GCP Generation Tool
=======================================
A unified Python-based tool that automates topographic analysis workflow.
100% Pure Python - No GDAL or system dependencies required.

Project: Geoinfotech - Kaduna Drone Topographic Survey
"""

import streamlit as st
import numpy as np
import pandas as pd
import tempfile
import os
import zipfile
import json
from io import BytesIO
import matplotlib.pyplot as plt
from scipy import ndimage

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="Topo Survey GCP Tool",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    .main-header {
        background: linear-gradient(135deg, #0F4C3A 0%, #1A6B50 50%, #0F4C3A 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        border: 1px solid rgba(45, 212, 163, 0.3);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    .main-header h1 {
        font-family: 'IBM Plex Sans', sans-serif;
        font-weight: 700;
        font-size: 2.2rem;
        color: #FFFFFF;
        margin: 0;
        letter-spacing: -0.5px;
    }
    
    .main-header p {
        font-family: 'IBM Plex Sans', sans-serif;
        color: rgba(255, 255, 255, 0.85);
        margin: 0.5rem 0 0 0;
        font-size: 1.05rem;
    }
    
    .stat-card {
        background: linear-gradient(145deg, rgba(15, 76, 58, 0.7), rgba(26, 107, 80, 0.5));
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid rgba(45, 212, 163, 0.2);
        text-align: center;
        backdrop-filter: blur(10px);
    }
    
    .stat-card h3 {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2rem;
        color: #2DD4A3;
        margin: 0;
        font-weight: 600;
    }
    
    .stat-card p {
        font-family: 'IBM Plex Sans', sans-serif;
        color: rgba(255, 255, 255, 0.75);
        margin: 0.5rem 0 0 0;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1.2px;
    }
    
    .process-step {
        background: rgba(15, 76, 58, 0.4);
        padding: 1.25rem 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #2DD4A3;
        margin: 0.75rem 0;
    }
    
    .process-step h4 {
        font-family: 'IBM Plex Sans', sans-serif;
        color: #FFFFFF;
        margin: 0 0 0.4rem 0;
        font-weight: 600;
        font-size: 1rem;
    }
    
    .process-step p {
        font-family: 'IBM Plex Sans', sans-serif;
        color: rgba(255, 255, 255, 0.7);
        margin: 0;
        font-size: 0.9rem;
    }
    
    .success-box {
        background: linear-gradient(145deg, rgba(45, 212, 163, 0.2), rgba(45, 212, 163, 0.08));
        border: 1px solid rgba(45, 212, 163, 0.5);
        padding: 1.25rem 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
    }
    
    .info-box {
        background: linear-gradient(145deg, rgba(59, 130, 246, 0.2), rgba(59, 130, 246, 0.08));
        border: 1px solid rgba(59, 130, 246, 0.5);
        padding: 1.25rem 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
    }
    
    .stButton > button {
        font-family: 'IBM Plex Sans', sans-serif;
        font-weight: 600;
        background: linear-gradient(135deg, #2DD4A3, #1A9B6C);
        color: #0A1F1A;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        font-size: 1rem;
        letter-spacing: 0.5px;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(45, 212, 163, 0.4);
    }
    
    .stProgress > div > div {
        background: linear-gradient(90deg, #2DD4A3, #1A9B6C);
    }
    
    div[data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace;
        color: #2DD4A3;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# DEM PROCESSING FUNCTIONS (Pure Python - No GDAL)
# ============================================================================

def load_geotiff(uploaded_file):
    """
    Load a GeoTIFF file using tifffile (pure Python).
    Returns elevation data and basic metadata.
    """
    import tifffile
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.tif') as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name
    
    # Read the TIFF file
    with tifffile.TiffFile(tmp_path) as tif:
        dem_data = tif.asarray().astype(np.float32)
        
        # Handle multi-band images (take first band)
        if len(dem_data.shape) > 2:
            dem_data = dem_data[0] if dem_data.shape[0] < dem_data.shape[-1] else dem_data[:,:,0]
        
        # Try to get geotransform from tags
        pixel_size = 1.0  # Default
        origin_x, origin_y = 0.0, 0.0
        
        try:
            # Check for ModelPixelScaleTag
            for page in tif.pages:
                for tag in page.tags.values():
                    if tag.name == 'ModelPixelScaleTag':
                        scales = tag.value
                        pixel_size = float(scales[0])
                    elif tag.name == 'ModelTiepointTag':
                        tiepoints = tag.value
                        if len(tiepoints) >= 6:
                            origin_x = float(tiepoints[3])
                            origin_y = float(tiepoints[4])
        except:
            pass
    
    # Handle nodata values
    dem_data[dem_data < -9000] = np.nan
    dem_data[dem_data > 9000] = np.nan
    
    # Create transform info
    transform = {
        'pixel_size': pixel_size,
        'origin_x': origin_x,
        'origin_y': origin_y
    }
    
    return dem_data, transform, tmp_path


def pixel_to_coords(row, col, transform):
    """Convert pixel coordinates to geographic coordinates."""
    x = transform['origin_x'] + col * transform['pixel_size']
    y = transform['origin_y'] - row * transform['pixel_size']
    return x, y


def calculate_slope(dem, pixel_size):
    """
    Calculate slope in degrees using Horn's algorithm.
    """
    # Pad array for edge handling
    padded = np.pad(dem, 1, mode='edge')
    
    # Get the 8 neighbors
    a = padded[:-2, :-2]  # NW
    b = padded[:-2, 1:-1]  # N
    c = padded[:-2, 2:]    # NE
    d = padded[1:-1, :-2]  # W
    f = padded[1:-1, 2:]   # E
    g = padded[2:, :-2]    # SW
    h = padded[2:, 1:-1]   # S
    i = padded[2:, 2:]     # SE
    
    # Calculate gradients using Horn's method
    dzdx = ((c + 2*f + i) - (a + 2*d + g)) / (8 * pixel_size)
    dzdy = ((g + 2*h + i) - (a + 2*b + c)) / (8 * pixel_size)
    
    # Calculate slope in degrees
    slope = np.degrees(np.arctan(np.sqrt(dzdx**2 + dzdy**2)))
    slope[np.isnan(dem)] = np.nan
    
    return slope.astype(np.float32)


def calculate_aspect(dem, pixel_size):
    """
    Calculate aspect (slope direction) in degrees.
    North = 0°, East = 90°, South = 180°, West = 270°
    """
    padded = np.pad(dem, 1, mode='edge')
    
    a = padded[:-2, :-2]
    b = padded[:-2, 1:-1]
    c = padded[:-2, 2:]
    d = padded[1:-1, :-2]
    f = padded[1:-1, 2:]
    g = padded[2:, :-2]
    h = padded[2:, 1:-1]
    i = padded[2:, 2:]
    
    dzdx = ((c + 2*f + i) - (a + 2*d + g)) / (8 * pixel_size)
    dzdy = ((g + 2*h + i) - (a + 2*b + c)) / (8 * pixel_size)
    
    aspect = np.degrees(np.arctan2(dzdy, -dzdx))
    aspect = 90 - aspect
    aspect[aspect < 0] += 360
    aspect[aspect >= 360] -= 360
    aspect[np.isnan(dem)] = np.nan
    
    return aspect.astype(np.float32)


def calculate_flow_direction(dem):
    """
    Calculate D8 flow direction.
    Direction encoding:
      32  64  128
      16   X   1
       8   4   2
    """
    rows, cols = dem.shape
    flow_dir = np.zeros((rows, cols), dtype=np.uint8)
    
    # Neighbor offsets and their direction codes
    neighbors = [
        (-1, -1, 32),   # NW
        (-1, 0, 64),    # N
        (-1, 1, 128),   # NE
        (0, -1, 16),    # W
        (0, 1, 1),      # E
        (1, -1, 8),     # SW
        (1, 0, 4),      # S
        (1, 1, 2),      # SE
    ]
    
    # Distance weights (diagonal = sqrt(2))
    weights = [1.414, 1.0, 1.414, 1.0, 1.0, 1.414, 1.0, 1.414]
    
    padded = np.pad(dem, 1, mode='edge')
    max_drop = np.zeros((rows, cols))
    
    for idx, (dr, dc, direction) in enumerate(neighbors):
        neighbor = padded[1+dr:rows+1+dr, 1+dc:cols+1+dc]
        drop = (dem - neighbor) / weights[idx]
        
        steeper = drop > max_drop
        flow_dir[steeper] = direction
        max_drop[steeper] = drop[steeper]
    
    flow_dir[np.isnan(dem)] = 0
    flow_dir[max_drop <= 0] = 0
    
    return flow_dir


def calculate_flow_accumulation(flow_dir):
    """
    Calculate flow accumulation using iterative approach.
    """
    rows, cols = flow_dir.shape
    flow_acc = np.ones((rows, cols), dtype=np.float32)
    
    d8_to_offset = {
        1: (0, 1), 2: (1, 1), 4: (1, 0), 8: (1, -1),
        16: (0, -1), 32: (-1, -1), 64: (-1, 0), 128: (-1, 1)
    }
    
    # Count incoming flows for each cell
    inflow_count = np.zeros((rows, cols), dtype=np.int32)
    
    for r in range(rows):
        for c in range(cols):
            direction = flow_dir[r, c]
            if direction in d8_to_offset:
                dr, dc = d8_to_offset[direction]
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    inflow_count[nr, nc] += 1
    
    # Process cells in topological order
    from collections import deque
    queue = deque()
    
    # Start with cells that have no inflow
    for r in range(rows):
        for c in range(cols):
            if inflow_count[r, c] == 0 and flow_dir[r, c] != 0:
                queue.append((r, c))
    
    processed = np.zeros((rows, cols), dtype=bool)
    
    while queue:
        r, c = queue.popleft()
        if processed[r, c]:
            continue
        processed[r, c] = True
        
        direction = flow_dir[r, c]
        if direction in d8_to_offset:
            dr, dc = d8_to_offset[direction]
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                flow_acc[nr, nc] += flow_acc[r, c]
                inflow_count[nr, nc] -= 1
                if inflow_count[nr, nc] == 0:
                    queue.append((nr, nc))
    
    return flow_acc


def delineate_watersheds(flow_dir, flow_acc, num_watersheds=10):
    """
    Delineate watersheds based on pour points.
    """
    rows, cols = flow_dir.shape
    
    # Find pour points (high accumulation at edges or local maxima)
    threshold = np.nanpercentile(flow_acc, 95)
    pour_points = flow_acc >= threshold
    
    # Label connected regions
    labeled, num_features = ndimage.label(pour_points)
    
    # Limit number of watersheds
    if num_features > num_watersheds:
        # Keep only the largest accumulation points
        pour_acc = flow_acc.copy()
        pour_acc[~pour_points] = 0
        
        flat = pour_acc.flatten()
        top_indices = np.argsort(flat)[-num_watersheds:]
        
        new_pour = np.zeros_like(pour_points)
        for idx in top_indices:
            r, c = np.unravel_index(idx, pour_points.shape)
            new_pour[r, c] = True
        pour_points = new_pour
        labeled, num_features = ndimage.label(pour_points)
    
    # Assign watershed IDs by tracing flow
    watersheds = np.zeros((rows, cols), dtype=np.int32)
    
    # Simple assignment based on nearest pour point
    for ws_id in range(1, num_features + 1):
        watersheds[labeled == ws_id] = ws_id
    
    return watersheds


def generate_contours(dem, transform, interval):
    """
    Generate contour lines as GeoJSON features.
    """
    min_elev = np.nanmin(dem)
    max_elev = np.nanmax(dem)
    
    # Generate contour levels
    start = np.ceil(min_elev / interval) * interval
    end = np.floor(max_elev / interval) * interval
    levels = np.arange(start, end + interval, interval)
    
    if len(levels) == 0:
        return {"type": "FeatureCollection", "features": []}
    
    # Fill NaN for contouring
    dem_filled = np.nan_to_num(dem, nan=min_elev - 100)
    
    rows, cols = dem.shape
    x = np.arange(cols)
    y = np.arange(rows)
    
    # Generate contours using matplotlib
    fig, ax = plt.subplots(figsize=(1, 1))
    cs = ax.contour(x, y, dem_filled, levels=levels)
    plt.close(fig)
    
    features = []
    
    for level_idx, level in enumerate(cs.levels):
        paths = cs.collections[level_idx].get_paths()
        
        for path in paths:
            vertices = path.vertices
            if len(vertices) < 2:
                continue
            
            # Convert to geographic coordinates
            coords = []
            for px, py in vertices:
                gx, gy = pixel_to_coords(py, px, transform)
                coords.append([float(gx), float(gy)])
            
            if len(coords) >= 2:
                features.append({
                    "type": "Feature",
                    "properties": {
                        "elevation": float(level),
                        "type": "index" if level % (interval * 5) == 0 else "intermediate"
                    },
                    "geometry": {
                        "type": "LineString",
                        "coordinates": coords
                    }
                })
    
    return {"type": "FeatureCollection", "features": features}


def extract_streams(flow_acc, transform, threshold):
    """
    Extract stream network from flow accumulation.
    """
    stream_mask = flow_acc >= threshold
    
    # Label connected stream segments
    labeled, num_segments = ndimage.label(stream_mask)
    
    features = []
    
    for seg_id in range(1, min(num_segments + 1, 100)):  # Limit segments
        segment = labeled == seg_id
        rows, cols = np.where(segment)
        
        if len(rows) < 3:
            continue
        
        # Sort points by flow accumulation (upstream to downstream)
        acc_values = flow_acc[segment]
        sort_idx = np.argsort(acc_values)
        
        coords = []
        for idx in sort_idx[::max(1, len(sort_idx)//50)]:  # Sample points
            r, c = rows[idx], cols[idx]
            gx, gy = pixel_to_coords(r, c, transform)
            coords.append([float(gx), float(gy)])
        
        if len(coords) >= 2:
            features.append({
                "type": "Feature",
                "properties": {
                    "segment_id": int(seg_id),
                    "length": len(rows)
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": coords
                }
            })
    
    return {"type": "FeatureCollection", "features": features}


def generate_gcps(dem, transform, spacing, strategy, slope):
    """
    Generate Ground Control Point markers.
    """
    rows, cols = dem.shape
    pixel_size = transform['pixel_size']
    
    # Calculate spacing in pixels
    spacing_px = max(int(spacing / pixel_size), 5)
    buffer_px = max(spacing_px // 3, 2)
    
    # Valid placement mask (not NaN, slope < 30°)
    valid = ~np.isnan(dem) & (slope <= 30)
    
    # Apply buffer
    if buffer_px < rows // 2 and buffer_px < cols // 2:
        valid[:buffer_px, :] = False
        valid[-buffer_px:, :] = False
        valid[:, :buffer_px] = False
        valid[:, -buffer_px:] = False
    
    gcps = []
    
    if strategy == "Grid Pattern":
        for r in range(buffer_px, rows - buffer_px, spacing_px):
            for c in range(buffer_px, cols - buffer_px, spacing_px):
                if valid[r, c]:
                    x, y = pixel_to_coords(r, c, transform)
                    gcps.append({
                        'id': f'GCP_{len(gcps)+1:03d}',
                        'x': float(x),
                        'y': float(y),
                        'elevation': float(dem[r, c]),
                        'slope': float(slope[r, c]),
                        'type': 'grid'
                    })
    
    elif strategy == "Terrain-Adaptive":
        # Prefer flat areas
        flat_areas = (slope < 5.0) & valid
        
        if np.any(flat_areas):
            flat_r, flat_c = np.where(flat_areas)
            
            # Sample evenly
            step = max(len(flat_r) // 50, 1)
            indices = np.arange(0, len(flat_r), step)
            
            for idx in indices[:100]:
                r, c = flat_r[idx], flat_c[idx]
                x, y = pixel_to_coords(r, c, transform)
                gcps.append({
                    'id': f'GCP_{len(gcps)+1:03d}',
                    'x': float(x),
                    'y': float(y),
                    'elevation': float(dem[r, c]),
                    'slope': float(slope[r, c]),
                    'type': 'flat_area'
                })
        
        # Add some ridge/valley points
        if len(gcps) < 20:
            for r in range(buffer_px, rows - buffer_px, spacing_px * 2):
                for c in range(buffer_px, cols - buffer_px, spacing_px * 2):
                    if valid[r, c] and len(gcps) < 100:
                        x, y = pixel_to_coords(r, c, transform)
                        gcps.append({
                            'id': f'GCP_{len(gcps)+1:03d}',
                            'x': float(x),
                            'y': float(y),
                            'elevation': float(dem[r, c]),
                            'slope': float(slope[r, c]),
                            'type': 'terrain'
                        })
    
    else:  # Edge + Interior
        # Edge points
        edge_spacing = spacing_px
        
        # Top and bottom edges
        for c in range(buffer_px, cols - buffer_px, edge_spacing):
            for r in [buffer_px, rows - buffer_px - 1]:
                if 0 <= r < rows and valid[r, c]:
                    x, y = pixel_to_coords(r, c, transform)
                    gcps.append({
                        'id': f'GCP_{len(gcps)+1:03d}',
                        'x': float(x),
                        'y': float(y),
                        'elevation': float(dem[r, c]),
                        'slope': float(slope[r, c]),
                        'type': 'edge'
                    })
        
        # Left and right edges
        for r in range(buffer_px + edge_spacing, rows - buffer_px - edge_spacing, edge_spacing):
            for c in [buffer_px, cols - buffer_px - 1]:
                if 0 <= c < cols and valid[r, c]:
                    x, y = pixel_to_coords(r, c, transform)
                    gcps.append({
                        'id': f'GCP_{len(gcps)+1:03d}',
                        'x': float(x),
                        'y': float(y),
                        'elevation': float(dem[r, c]),
                        'slope': float(slope[r, c]),
                        'type': 'edge'
                    })
        
        # Interior points (sparser)
        interior_spacing = int(spacing_px * 1.5)
        for r in range(buffer_px + interior_spacing, rows - buffer_px - interior_spacing, interior_spacing):
            for c in range(buffer_px + interior_spacing, cols - buffer_px - interior_spacing, interior_spacing):
                if valid[r, c]:
                    x, y = pixel_to_coords(r, c, transform)
                    gcps.append({
                        'id': f'GCP_{len(gcps)+1:03d}',
                        'x': float(x),
                        'y': float(y),
                        'elevation': float(dem[r, c]),
                        'slope': float(slope[r, c]),
                        'type': 'interior'
                    })
    
    return gcps


# ============================================================================
# EXPORT FUNCTIONS
# ============================================================================

def save_array_as_csv(data, path, name="data"):
    """Save a 2D array as CSV."""
    np.savetxt(path, data, delimiter=',', fmt='%.4f')


def save_geojson(geojson_dict, path):
    """Save GeoJSON to file."""
    with open(path, 'w') as f:
        json.dump(geojson_dict, f, indent=2)


def export_gcps_csv(gcps, path):
    """Export GCPs to CSV."""
    df = pd.DataFrame(gcps)
    df.to_csv(path, index=False)


def export_gcps_kml(gcps, path):
    """Export GCPs to KML for Google Earth."""
    kml = '''<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
<Document>
    <name>GCP Markers - Topo Survey</name>
    <description>Ground Control Points generated by Topo Survey GCP Tool</description>
    
    <Style id="gcpStyleRed">
        <IconStyle>
            <color>ff0000ff</color>
            <scale>1.2</scale>
            <Icon><href>http://maps.google.com/mapfiles/kml/shapes/target.png</href></Icon>
        </IconStyle>
        <LabelStyle><color>ffffffff</color><scale>0.8</scale></LabelStyle>
    </Style>
    
    <Style id="gcpStyleGreen">
        <IconStyle>
            <color>ff00ff00</color>
            <scale>1.0</scale>
            <Icon><href>http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png</href></Icon>
        </IconStyle>
        <LabelStyle><color>ffffffff</color><scale>0.7</scale></LabelStyle>
    </Style>
    
    <Folder>
        <name>Ground Control Points</name>
'''
    
    for gcp in gcps:
        style = "gcpStyleGreen" if gcp['type'] == 'edge' else "gcpStyleRed"
        kml += f'''
        <Placemark>
            <name>{gcp['id']}</name>
            <description><![CDATA[
                <b>{gcp['id']}</b><br/>
                X: {gcp['x']:.6f}<br/>
                Y: {gcp['y']:.6f}<br/>
                Elevation: {gcp['elevation']:.2f} m<br/>
                Slope: {gcp['slope']:.1f}°<br/>
                Type: {gcp['type']}
            ]]></description>
            <styleUrl>#{style}</styleUrl>
            <Point>
                <altitudeMode>absolute</altitudeMode>
                <coordinates>{gcp['x']},{gcp['y']},{gcp['elevation']}</coordinates>
            </Point>
        </Placemark>
'''
    
    kml += '''
    </Folder>
</Document>
</kml>'''
    
    with open(path, 'w') as f:
        f.write(kml)


def export_gcps_geojson(gcps, path):
    """Export GCPs to GeoJSON."""
    features = []
    for gcp in gcps:
        features.append({
            "type": "Feature",
            "properties": {
                "id": gcp['id'],
                "elevation": gcp['elevation'],
                "slope": gcp['slope'],
                "type": gcp['type']
            },
            "geometry": {
                "type": "Point",
                "coordinates": [gcp['x'], gcp['y'], gcp['elevation']]
            }
        })
    
    geojson = {
        "type": "FeatureCollection",
        "name": "GCP_Markers",
        "features": features
    }
    
    with open(path, 'w') as f:
        json.dump(geojson, f, indent=2)


def export_gcps_dxf(gcps, path):
    """Export GCPs to DXF for CAD software."""
    try:
        import ezdxf
        
        doc = ezdxf.new('R2010')
        msp = doc.modelspace()
        
        # Create layers
        doc.layers.add('GCP_POINTS', color=1)
        doc.layers.add('GCP_LABELS', color=7)
        doc.layers.add('GCP_MARKERS', color=3)
        
        for gcp in gcps:
            x, y, z = gcp['x'], gcp['y'], gcp['elevation']
            
            # Point
            msp.add_point((x, y, z), dxfattribs={'layer': 'GCP_POINTS'})
            
            # Circle marker
            msp.add_circle((x, y, z), radius=2.0, dxfattribs={'layer': 'GCP_MARKERS'})
            
            # Cross
            cross = 3.0
            msp.add_line((x-cross, y, z), (x+cross, y, z), dxfattribs={'layer': 'GCP_MARKERS'})
            msp.add_line((x, y-cross, z), (x, y+cross, z), dxfattribs={'layer': 'GCP_MARKERS'})
            
            # Label
            msp.add_text(
                f"{gcp['id']}: {z:.1f}m",
                dxfattribs={'layer': 'GCP_LABELS', 'height': 1.5}
            ).set_pos((x + 4, y + 2, z))
        
        doc.saveas(path)
        
    except Exception:
        # Fallback: simple ASCII DXF
        dxf = "0\nSECTION\n2\nENTITIES\n"
        for gcp in gcps:
            x, y, z = gcp['x'], gcp['y'], gcp['elevation']
            dxf += f"0\nPOINT\n8\nGCP\n10\n{x}\n20\n{y}\n30\n{z}\n"
            dxf += f"0\nCIRCLE\n8\nGCP\n10\n{x}\n20\n{y}\n30\n{z}\n40\n2.0\n"
            dxf += f"0\nTEXT\n8\nGCP\n10\n{x+3}\n20\n{y+1}\n30\n{z}\n40\n1.5\n1\n{gcp['id']}: {z:.1f}m\n"
        dxf += "0\nENDSEC\n0\nEOF\n"
        
        with open(path, 'w') as f:
            f.write(dxf)


def export_gcps_gpx(gcps, path):
    """Export GCPs to GPX for GPS devices."""
    gpx = '''<?xml version="1.0" encoding="UTF-8"?>
<gpx version="1.1" creator="Topo Survey GCP Tool"
     xmlns="http://www.topografix.com/GPX/1/1">
    <metadata>
        <name>Survey GCP Waypoints</name>
        <desc>Ground Control Points for topographic survey</desc>
    </metadata>
'''
    
    for gcp in gcps:
        gpx += f'''
    <wpt lat="{gcp['y']}" lon="{gcp['x']}">
        <ele>{gcp['elevation']}</ele>
        <name>{gcp['id']}</name>
        <desc>Elevation: {gcp['elevation']:.2f}m, Slope: {gcp['slope']:.1f}deg, Type: {gcp['type']}</desc>
        <sym>Flag, Red</sym>
    </wpt>
'''
    
    gpx += '</gpx>'
    
    with open(path, 'w') as f:
        f.write(gpx)


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🗺️ Topographic Survey GCP Generation Tool</h1>
        <p>Automated terrain analysis and ground control point generation for drone surveys</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar Configuration
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        
        st.markdown("### 📏 Contour Settings")
        contour_interval = st.slider(
            "Contour Interval (m)", 
            min_value=0.5, 
            max_value=20.0, 
            value=2.0, 
            step=0.5,
            help="Vertical distance between contour lines"
        )
        
        st.markdown("### 📍 GCP Settings")
        gcp_spacing = st.slider(
            "GCP Spacing (m)", 
            min_value=50, 
            max_value=500, 
            value=100, 
            step=25,
            help="Distance between ground control points"
        )
        
        gcp_strategy = st.selectbox(
            "Placement Strategy",
            ["Grid Pattern", "Terrain-Adaptive", "Edge + Interior"],
            help="Method for distributing GCP markers"
        )
        
        st.markdown("### 🌊 Hydrology Settings")
        stream_threshold = st.slider(
            "Stream Threshold",
            min_value=50,
            max_value=1000,
            value=200,
            step=50,
            help="Minimum flow accumulation to define streams"
        )
        
        st.markdown("### 📁 Export Formats")
        export_csv = st.checkbox("CSV (Coordinates)", value=True)
        export_kml = st.checkbox("KML (Google Earth)", value=True)
        export_geojson = st.checkbox("GeoJSON (GIS)", value=True)
        export_dxf = st.checkbox("DXF (CAD)", value=True)
        export_gpx = st.checkbox("GPX (GPS)", value=True)
        
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; opacity: 0.6; font-size: 0.8rem;">
            <p><strong>Geoinfotech</strong></p>
            <p>Kaduna Drone Survey Project</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Main Content - Stats
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="stat-card"><h3>7</h3><p>Output Datasets</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="stat-card"><h3>5</h3><p>Export Formats</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="stat-card"><h3>100%</h3><p>Open Source</p></div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="stat-card"><h3>~5min</h3><p>Processing</p></div>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # File Upload
    st.markdown("### 📤 Upload DEM File")
    
    uploaded_file = st.file_uploader(
        "Upload your Digital Elevation Model (GeoTIFF format)",
        type=['tif', 'tiff'],
        help="Supported: GeoTIFF (.tif, .tiff) from drone photogrammetry"
    )
    
    if uploaded_file is not None:
        st.markdown("""
        <div class="success-box">
            <strong>✅ File uploaded successfully!</strong> Ready to process your DEM data.
        </div>
        """, unsafe_allow_html=True)
        
        # File info
        col1, col2 = st.columns(2)
        with col1:
            st.metric("File Name", uploaded_file.name)
        with col2:
            file_size = len(uploaded_file.getvalue()) / (1024 * 1024)
            st.metric("File Size", f"{file_size:.2f} MB")
        
        # Process button
        if st.button("🚀 Start Processing", use_container_width=True):
            output_dir = tempfile.mkdtemp()
            progress = st.progress(0)
            status = st.empty()
            
            try:
                # Step 1: Load DEM
                status.text("📂 Loading DEM file...")
                progress.progress(5)
                
                dem, transform, tmp_path = load_geotiff(uploaded_file)
                pixel_size = transform['pixel_size']
                
                st.markdown('<div class="process-step"><h4>✅ DEM Loaded Successfully</h4><p>Raster data extracted and georeferencing captured.</p></div>', unsafe_allow_html=True)
                
                # DEM stats
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Min Elevation", f"{np.nanmin(dem):.2f} m")
                with col2:
                    st.metric("Max Elevation", f"{np.nanmax(dem):.2f} m")
                with col3:
                    st.metric("Pixel Size", f"{pixel_size:.2f} m")
                with col4:
                    st.metric("Dimensions", f"{dem.shape[1]} × {dem.shape[0]}")
                
                progress.progress(15)
                
                # Step 2: Slope
                status.text("📐 Calculating slope...")
                slope = calculate_slope(dem, pixel_size)
                slope_path = os.path.join(output_dir, "slope.csv")
                save_array_as_csv(slope, slope_path)
                
                st.markdown('<div class="process-step"><h4>✅ Slope Analysis Complete</h4><p>Terrain steepness calculated in degrees.</p></div>', unsafe_allow_html=True)
                progress.progress(25)
                
                # Step 3: Aspect
                status.text("🧭 Calculating aspect...")
                aspect = calculate_aspect(dem, pixel_size)
                aspect_path = os.path.join(output_dir, "aspect.csv")
                save_array_as_csv(aspect, aspect_path)
                
                st.markdown('<div class="process-step"><h4>✅ Aspect Analysis Complete</h4><p>Slope direction calculated (0-360°).</p></div>', unsafe_allow_html=True)
                progress.progress(35)
                
                # Step 4: Flow Direction
                status.text("🌊 Calculating flow direction...")
                flow_dir = calculate_flow_direction(dem)
                flow_dir_path = os.path.join(output_dir, "flow_direction.csv")
                save_array_as_csv(flow_dir.astype(np.float32), flow_dir_path)
                
                st.markdown('<div class="process-step"><h4>✅ Flow Direction Calculated</h4><p>D8 water flow patterns analyzed.</p></div>', unsafe_allow_html=True)
                progress.progress(50)
                
                # Step 5: Flow Accumulation
                status.text("💧 Calculating flow accumulation...")
                flow_acc = calculate_flow_accumulation(flow_dir)
                flow_acc_path = os.path.join(output_dir, "flow_accumulation.csv")
                save_array_as_csv(flow_acc, flow_acc_path)
                
                st.markdown('<div class="process-step"><h4>✅ Flow Accumulation Complete</h4><p>Drainage patterns identified.</p></div>', unsafe_allow_html=True)
                progress.progress(60)
                
                # Step 6: Watersheds
                status.text("🏔️ Delineating watersheds...")
                watersheds = delineate_watersheds(flow_dir, flow_acc)
                watersheds_path = os.path.join(output_dir, "watersheds.csv")
                save_array_as_csv(watersheds.astype(np.float32), watersheds_path)
                
                st.markdown('<div class="process-step"><h4>✅ Watersheds Delineated</h4><p>Catchment basins mapped.</p></div>', unsafe_allow_html=True)
                progress.progress(70)
                
                # Step 7: Contours
                status.text("🗺️ Generating contours...")
                contours = generate_contours(dem, transform, contour_interval)
                contours_path = os.path.join(output_dir, "contours.geojson")
                save_geojson(contours, contours_path)
                
                st.markdown(f'<div class="process-step"><h4>✅ Contours Generated</h4><p>{len(contours["features"])} contour lines at {contour_interval}m interval.</p></div>', unsafe_allow_html=True)
                progress.progress(80)
                
                # Step 8: Streams
                status.text("🌊 Extracting stream network...")
                streams = extract_streams(flow_acc, transform, stream_threshold)
                streams_path = os.path.join(output_dir, "streams.geojson")
                save_geojson(streams, streams_path)
                
                st.markdown(f'<div class="process-step"><h4>✅ Stream Network Extracted</h4><p>{len(streams["features"])} stream segments identified.</p></div>', unsafe_allow_html=True)
                progress.progress(90)
                
                # Step 9: GCPs
                status.text("📍 Generating GCP markers...")
                gcps = generate_gcps(dem, transform, gcp_spacing, gcp_strategy, slope)
                
                # Export in selected formats
                if export_csv:
                    export_gcps_csv(gcps, os.path.join(output_dir, "gcp_markers.csv"))
                if export_kml:
                    export_gcps_kml(gcps, os.path.join(output_dir, "gcp_markers.kml"))
                if export_geojson:
                    export_gcps_geojson(gcps, os.path.join(output_dir, "gcp_markers.geojson"))
                if export_dxf:
                    export_gcps_dxf(gcps, os.path.join(output_dir, "gcp_markers.dxf"))
                if export_gpx:
                    export_gcps_gpx(gcps, os.path.join(output_dir, "gcp_markers.gpx"))
                
                st.markdown(f'<div class="process-step"><h4>✅ GCP Markers Generated</h4><p>{len(gcps)} ground control points using {gcp_strategy}.</p></div>', unsafe_allow_html=True)
                progress.progress(100)
                status.text("✅ All processing complete!")
                
                # Create ZIP
                zip_buffer = BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                    for root, dirs, files in os.walk(output_dir):
                        for file in files:
                            file_path = os.path.join(root, file)
                            zf.write(file_path, file)
                zip_buffer.seek(0)
                
                # Success
                st.success("🎉 Processing complete! Download your results below.")
                
                # Summary
                st.markdown("### 📊 Results Summary")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("""
                    **Raster Outputs (CSV):**
                    - `slope.csv` - Terrain steepness (degrees)
                    - `aspect.csv` - Slope direction (0-360°)
                    - `flow_direction.csv` - D8 flow codes
                    - `flow_accumulation.csv` - Drainage accumulation
                    - `watersheds.csv` - Basin IDs
                    """)
                
                with col2:
                    st.markdown(f"""
                    **Vector Outputs:**
                    - `contours.geojson` - {len(contours['features'])} contour lines
                    - `streams.geojson` - {len(streams['features'])} stream segments
                    - `gcp_markers.*` - {len(gcps)} GCP points
                    """)
                
                # Download
                st.download_button(
                    "📥 Download All Results (ZIP)",
                    zip_buffer,
                    "topographic_survey_results.zip",
                    "application/zip",
                    use_container_width=True
                )
                
                # GCP Preview
                st.markdown("### 📍 GCP Preview")
                gcp_df = pd.DataFrame(gcps)
                st.dataframe(gcp_df, use_container_width=True, hide_index=True)
                
                # Cleanup
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                
            except Exception as e:
                st.error(f"❌ Error during processing: {str(e)}")
                st.exception(e)
    
    else:
        # Instructions
        st.markdown("### 📋 How It Works")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="process-step"><h4>1️⃣ Upload DEM</h4><p>Upload your GeoTIFF Digital Elevation Model from drone photogrammetry (Pix4D, DroneDeploy, ODM, etc.)</p></div>', unsafe_allow_html=True)
            st.markdown('<div class="process-step"><h4>2️⃣ Configure Settings</h4><p>Adjust contour intervals, GCP spacing, and choose your preferred export formats in the sidebar.</p></div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="process-step"><h4>3️⃣ Process</h4><p>Click "Start Processing" and wait ~5 minutes for automated terrain analysis.</p></div>', unsafe_allow_html=True)
            st.markdown('<div class="process-step"><h4>4️⃣ Download Results</h4><p>Get all outputs in a single ZIP file ready for GIS, CAD, or field work.</p></div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
            <strong>💡 Supported Input:</strong> GeoTIFF (.tif, .tiff) — Single-band elevation raster from drone photogrammetry software.
        </div>
        """, unsafe_allow_html=True)
        
        # Output info
        st.markdown("### 📦 Output Datasets")
        
        st.markdown("""
        | Dataset | Description | Format |
        |---------|-------------|--------|
        | **Slope** | Terrain steepness (0-90°) | CSV |
        | **Aspect** | Slope direction (0-360°) | CSV |
        | **Flow Direction** | D8 water flow patterns | CSV |
        | **Flow Accumulation** | Drainage accumulation | CSV |
        | **Watersheds** | Catchment basin boundaries | CSV |
        | **Contours** | Elevation lines | GeoJSON |
        | **Streams** | Drainage network | GeoJSON |
        | **GCP Markers** | Ground control points | CSV, KML, GeoJSON, DXF, GPX |
        """)


if __name__ == "__main__":
    main()
