import streamlit as st
import folium
from streamlit_folium import st_folium
import osmnx as ox
ox.settings.overpass_url = "https://overpass.kumi.systems/api/interpreter"
ox.settings.overpass_endpoint = "https://overpass.kumi.systems/api"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap
import io
import math
import time
import warnings
import requests
import numpy as np
from astral import LocationInfo
from astral.sun import elevation, azimuth
import datetime
import zipfile
import pydeck as pdk
import pandas as pd
import json
import geopandas as gpd
from shapely.geometry import Point, LineString, Polygon, box
from shapely.affinity import translate
from shapely.ops import unary_union
from scipy.interpolate import griddata
from scipy.spatial import cKDTree
import pyproj
from mpl_toolkits.axes_grid1 import make_axes_locatable
try:
    import ezdxf
    _HAS_EZDXF = True
except ImportError:
    _HAS_EZDXF = False

warnings.filterwarnings("ignore")

# ==============================================================================
# PAGE CONFIG + GLOBAL CSS
# ==============================================================================

st.set_page_config(
    page_title="Site Analysis",
    page_icon="⬛",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&family=JetBrains+Mono:wght@300;400&display=swap');

:root {
  --ink:    #0f0f0f;
  --paper:  #ffffff;
  --accent: #2b2b2b;
  --muted:  #707070;
  --line:   #d1d1d1;
  --card:   #f8f9fa;
}

html, body, .stApp, [data-testid="stAppViewContainer"], [class*="css"] {
  font-family: 'JetBrains Mono', monospace !important;
  background-color: var(--paper) !important;
  color: var(--ink) !important;
}

#MainMenu, footer { visibility: hidden; }
header { background: transparent !important; }
.stDeployButton { display: none; }

.main .block-container {
  max-width: 900px;
  padding: 3rem 2.5rem 5rem;
  margin: 0 auto;
}

h1, h2, h3 {
  font-family: 'Inter', sans-serif !important;
  color: var(--ink) !important;
  font-weight: 300 !important;
  letter-spacing: -0.03em;
}

h1 {
  font-size: 3rem !important;
  border-bottom: 2px solid var(--ink);
  padding-bottom: 0.5rem;
  margin-bottom: 2rem !important;
}

h2 {
  font-size: 1.4rem !important;
  border-top: 1px solid var(--line);
  padding-top: 1.5rem;
  margin-top: 3rem !important;
  text-transform: lowercase;
}

.stCaption, [data-testid="stCaptionContainer"] p {
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 0.7rem !important;
  color: var(--muted) !important;
  letter-spacing: 0.05em;
  text-transform: uppercase;
}

section[data-testid="stSidebar"] {
  background-color: var(--card) !important;
  border-right: 1px solid var(--line);
}
section[data-testid="stSidebar"] * { color: var(--ink) !important; }
section[data-testid="stSidebar"] hr { border-color: var(--line) !important; }
section[data-testid="stSidebar"] .stMarkdown a {
  font-family: 'Inter', sans-serif !important;
  font-size: 0.8rem;
  letter-spacing: 0.02em;
  text-transform: uppercase;
  text-decoration: none;
  color: var(--muted) !important;
  transition: color 0.2s ease;
}
section[data-testid="stSidebar"] .stMarkdown a:hover { color: var(--ink) !important; }

.stButton > button, .stDownloadButton > button {
  border-radius: 0px !important;
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 0.75rem !important;
  letter-spacing: 0.05em;
  text-transform: uppercase;
  border: 1px solid var(--line) !important;
  background-color: var(--paper) !important;
  color: var(--ink) !important;
  transition: all 0.2s ease;
  width: 100%;
}
.stButton > button:hover, .stDownloadButton > button:hover {
  border-color: var(--ink) !important;
}
.stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"] {
  background-color: var(--ink) !important;
  color: var(--paper) !important;
  border: 1px solid var(--ink) !important;
}
.stButton > button[kind="primary"]:hover, .stDownloadButton > button[kind="primary"]:hover {
  background-color: var(--paper) !important;
  color: var(--ink) !important;
}

[data-testid="metric-container"] {
  background: transparent;
  border-left: 2px solid var(--ink);
  border-radius: 0;
  padding: 0.5rem 1rem !important;
}
[data-testid="stMetricLabel"] {
  font-family: 'Inter', sans-serif !important;
  font-size: 0.7rem !important;
  letter-spacing: 0.05em;
  text-transform: uppercase;
  color: var(--muted) !important;
}
[data-testid="stMetricValue"] {
  font-family: 'Inter', sans-serif !important;
  font-size: 2.5rem !important;
  font-weight: 300 !important;
  color: var(--ink) !important;
}

input[type="text"], .stTextInput input {
  background-color: var(--paper) !important;
  border: 1px solid var(--line) !important;
  border-radius: 0 !important;
  font-family: 'JetBrains Mono', monospace !important;
}
[data-testid="stSlider"] > div > div > div { background-color: var(--ink) !important; }

.stAlert {
  border-radius: 0 !important;
  border: 1px solid var(--line) !important;
  border-left: 4px solid var(--ink) !important;
  background-color: var(--paper) !important;
  font-family: 'JetBrains Mono', monospace !important;
}

[data-testid="stProgress"] > div > div { background-color: var(--ink) !important; }
[data-testid="stProgress"] { background-color: var(--line) !important; }
[data-testid="stSpinner"] { color: var(--ink) !important; }

iframe, [data-testid="stImage"] img, .stPyplot img {
  border: 1px solid var(--line) !important;
  border-radius: 0 !important;
}
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# CACHED DATA FETCHERS
# ==============================================================================

EMPTY_GDF_CRS = "EPSG:4326"


def _empty_gdf():
    return gpd.GeoDataFrame(geometry=[], crs=EMPTY_GDF_CRS)


def _fetch_elevations(pts):
    """Query OpenTopoData in batches of 100 (public API: max 1 request/second)."""
    all_elevs = []
    for start in range(0, len(pts), 100):
        batch = pts[start:start + 100]
        url = ("https://api.opentopodata.org/v1/srtm30m?locations="
               + "|".join(f"{p[0]},{p[1]}" for p in batch))
        elevs = [None] * len(batch)
        for attempt in range(2):
            try:
                if start > 0 or attempt > 0:
                    time.sleep(1.2)
                res = requests.get(url, timeout=25).json()
                if res.get("status") == "OK":
                    elevs = [r["elevation"] for r in res["results"]]
                    break
            except Exception:
                pass
        all_elevs.extend(elevs)
    return all_elevs


def _elevation_points(lat, lon, radius, n):
    dlat = radius / 111320
    dlon = radius / (111320 * math.cos(math.radians(lat)))
    return [
        (lat - dlat + i * (2 * dlat / (n - 1)),
         lon - dlon + j * (2 * dlon / (n - 1)))
        for i in range(n) for j in range(n)
    ]


@st.cache_data(show_spinner=False)
def fetch_buildings(lat, lon, radius):
    gdf = ox.features_from_point((lat, lon), {"building": True}, dist=radius)
    return gdf[gdf.geom_type.isin(["Polygon", "MultiPolygon"])]


@st.cache_data(show_spinner=False)
def fetch_graph(lat, lon, radius):
    G = ox.graph_from_point((lat, lon), dist=radius, network_type="all")
    return ox.graph_to_gdfs(G, nodes=False, edges=True)


@st.cache_data(show_spinner=False)
def fetch_wind_data(lat, lon):
    url = (
        f"https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={lat}&longitude={lon}"
        f"&start_date=2023-01-01&end_date=2023-12-31"
        f"&daily=wind_direction_10m_dominant,wind_speed_10m_max&timezone=auto"
    )
    res = requests.get(url, timeout=15).json()
    return res["daily"]["wind_direction_10m_dominant"], res["daily"]["wind_speed_10m_max"]



@st.cache_data(show_spinner=False)
def fetch_flood_layers(lat, lon, radius):
    waterway_tags = {"waterway": ["river", "stream", "canal", "drain",
                                  "ditch", "culvert", "pressurised"]}
    water_area_tags = {
        "natural": ["water", "wetland", "mud"],
        "landuse": ["reservoir", "basin", "floodplain"],
        "water":   True,
    }
    pad = 1.05
    dlat_c = (radius * pad) / 111320
    dlon_c = (radius * pad) / (111320 * math.cos(math.radians(lat)))
    clip_box = box(lon - dlon_c, lat - dlat_c, lon + dlon_c, lat + dlat_c)

    try:
        waterways = ox.features_from_point((lat, lon), waterway_tags, dist=radius)
        if waterways is not None and not waterways.empty:
            waterways = gpd.clip(waterways, clip_box)
    except Exception:
        waterways = None
    try:
        water_bodies = ox.features_from_point((lat, lon), water_area_tags, dist=radius)
        water_bodies = water_bodies[water_bodies.geom_type.isin(["Polygon", "MultiPolygon"])]
        if not water_bodies.empty:
            water_bodies = gpd.clip(water_bodies, clip_box)
    except Exception:
        water_bodies = None

    pts = _elevation_points(lat, lon, radius, 13)
    all_elevs = _fetch_elevations(pts)

    elev_grid = None
    if any(e is not None for e in all_elevs):
        elev_grid = {
            "lats":       [p[0] for p in pts],
            "lons":       [p[1] for p in pts],
            "elevations": all_elevs,
        }
    return waterways, water_bodies, elev_grid


@st.cache_data(show_spinner=False)
def fetch_access_points(lat, lon, radius):
    edges = fetch_graph(lat, lon, radius)
    tags_transit = {
        "highway":          ["bus_stop", "crossing", "traffic_signals"],
        "public_transport": ["stop_position", "platform"],
        "amenity":          ["bus_station", "ferry_terminal", "taxi"],
        "railway":          ["station", "halt", "tram_stop", "subway_entrance"],
    }
    try:
        transit = ox.features_from_point((lat, lon), tags_transit, dist=radius)
        transit = transit[transit.geom_type == "Point"]
    except Exception:
        transit = None
    return edges, transit


@st.cache_data(show_spinner=False)
def fetch_landmarks(lat, lon, radius):
    tags = {
        "amenity": [
            "school", "university", "college",
            "hospital", "clinic", "doctors", "pharmacy",
            "bus_station", "ferry_terminal",
            "restaurant", "cafe", "fast_food", "food_court",
            "convenience", "supermarket", "marketplace",
            "bank", "atm", "place_of_worship",
            "library", "community_centre",
        ],
        "railway": ["station", "halt", "tram_stop", "subway_entrance"],
        "shop":    ["convenience", "supermarket", "general", "mall"],
        "leisure": ["park"],
    }
    try:
        return ox.features_from_point((lat, lon), tags, dist=radius)
    except Exception:
        return _empty_gdf()


@st.cache_data(show_spinner=False)
def fetch_land_use(lat, lon, radius):
    tags = {
        "landuse": [
            "residential", "commercial", "industrial", "retail",
            "education", "institutional", "religious", "cemetery",
            "recreation_ground", "allotments", "farmland", "forest",
            "grass", "meadow", "orchard", "vineyard",
            "quarry", "landfill", "brownfield", "greenfield",
            "construction", "military", "village_green",
            "basin", "reservoir", "floodplain",
        ],
        "leisure": ["park", "playground", "sports_centre", "pitch",
                    "garden", "golf_course", "nature_reserve"],
        "amenity": ["school", "university", "college", "hospital",
                    "place_of_worship", "parking"],
        "natural": ["wood", "scrub", "heath", "grassland", "water",
                    "wetland", "beach", "sand", "mud", "bare_rock"],
    }
    try:
        gdf = ox.features_from_point((lat, lon), tags, dist=radius)
        return gdf[gdf.geom_type.isin(["Polygon", "MultiPolygon"])]
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def fetch_utilities(lat, lon, radius):
    power_tags = {
        "power": ["pole", "line", "minor_line", "cable", "transformer",
                  "switch", "junction", "connection"],
    }
    sewer_tags = {
        "man_made": ["manhole", "sewer", "wastewater_plant",
                     "pumping_station", "storage_tank", "pipeline"],
        "waterway": ["drain", "ditch", "culvert", "pressurised"],
    }
    water_tags = {
        "emergency": ["fire_hydrant"],
        "man_made":  ["water_meter", "valve", "water_tap",
                      "water_tower", "water_works"],
        "waterway":  ["canal", "stream"],
    }
    telecom_tags = {
        "telecom":  ["distribution_point", "connection_point",
                     "street_cabinet", "exchange", "service_device"],
        "man_made": ["street_cabinet", "antenna"],
    }
    pad = 1.05
    dlat_c = (radius * pad) / 111320
    dlon_c = (radius * pad) / (111320 * math.cos(math.radians(lat)))
    clip_box = box(lon - dlon_c, lat - dlat_c, lon + dlon_c, lat + dlat_c)

    results = {}
    for name, tags in [
        ("power",   power_tags),
        ("sewer",   sewer_tags),
        ("water",   water_tags),
        ("telecom", telecom_tags),
    ]:
        try:
            gdf = ox.features_from_point((lat, lon), tags, dist=radius)
            if gdf is not None and not gdf.empty:
                gdf = gpd.clip(gdf, clip_box)
            results[name] = gdf
        except Exception:
            results[name] = None
    return results


@st.cache_data(show_spinner=False)
def fetch_topography_detailed(lat, lon, radius):
    pts = _elevation_points(lat, lon, radius, 15)
    all_elevs = _fetch_elevations(pts)
    return {"lats": [p[0] for p in pts],
            "lons": [p[1] for p in pts],
            "elevations": all_elevs}

# ==============================================================================
# HELPERS
# ==============================================================================

def road_width(highway_val, scale=1.0):
    t = str(highway_val[0] if isinstance(highway_val, list) else highway_val).lower()
    if any(x in t for x in ["motorway", "trunk", "primary"]): return 5.5 * scale
    if "secondary"   in t: return 3.5 * scale
    if "tertiary"    in t: return 2.5 * scale
    if "residential" in t or "unclassified" in t: return 1.5 * scale
    return 0.6 * scale


def edge_widths(edges, scale=1.0, default=1.0):
    if "highway" in edges.columns:
        return edges["highway"].apply(lambda x: road_width(x, scale))
    return pd.Series(default * scale, index=edges.index)


def lonlat_to_xy(lons, lats, crs):
    """Transform lon/lat arrays into the given projected CRS (always lon, lat order)."""
    tr = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    xs, ys = tr.transform(np.asarray(lons, dtype=float), np.asarray(lats, dtype=float))
    return np.asarray(xs), np.asarray(ys)


def project_center(lat, lon, crs):
    x, y = lonlat_to_xy([lon], [lat], crs)
    return float(x[0]), float(y[0])


def load_base(lat, lon, radius):
    """Buildings + street edges in a common projected CRS, plus the site centre."""
    buildings = fetch_buildings(lat, lon, radius)
    buildings_p = ox.projection.project_gdf(buildings)
    crs = buildings_p.crs
    edges_p = fetch_graph(lat, lon, radius).to_crs(crs)
    cx, cy = project_center(lat, lon, crs)
    return buildings_p, edges_p, crs, cx, cy

def get_carto_key():
    """Returns the personal CARTO Basemaps API key."""
    return "cb1_4dpu_1_858a2eddbeb8483dfe150952"


def site_tz(lon):
    """Approximate local time zone from longitude (Manila ≈ UTC+8)."""
    return datetime.timezone(datetime.timedelta(hours=round(lon / 15)))


def sun_path_points(lat, lon, date, radius_m, cx, cy):
    loc   = LocationInfo("Site", "Region", "UTC", lat, lon)
    tz    = site_tz(lon)
    max_r = radius_m * 0.85
    pts   = []
    for h in range(24):
        for m in [0, 10, 20, 30, 40, 50]:
            t  = datetime.datetime.combine(date, datetime.time(h, m), tzinfo=tz)
            el = elevation(loc.observer, t)
            if el > 0:
                az    = azimuth(loc.observer, t)
                r     = max_r * (1.0 - el / 90.0)
                angle = math.radians(90.0 - az)
                pts.append((t, cx + r * math.cos(angle), cy + r * math.sin(angle)))
    pts.sort(key=lambda x: x[0])
    return pts


def building_heights(df, default):
    heights = []
    for _, row in df.iterrows():
        h = default
        raw = row.get("height")
        lev = row.get("building:levels")
        if raw is not None and pd.notnull(raw):
            try:
                h = float(str(raw).lower().replace("m", "").replace(",", ".").strip())
                heights.append(h); continue
            except ValueError:
                pass
        if lev is not None and pd.notnull(lev):
            try:
                h = float(str(lev).strip()) * 3.0
            except ValueError:
                pass
        heights.append(h)
    return heights


def shadow_polygon(geom, dx, dy):
    """Swept shadow: footprint + translated footprint + quads along every wall."""
    parts = [geom, translate(geom, xoff=dx, yoff=dy)]
    polys = [geom] if geom.geom_type == "Polygon" else list(getattr(geom, "geoms", []))
    for p in polys:
        c = list(p.exterior.coords)
        for a, b in zip(c[:-1], c[1:]):
            q = Polygon([(a[0], a[1]), (b[0], b[1]),
                         (b[0] + dx, b[1] + dy), (a[0] + dx, a[1] + dy)])
            if q.is_valid and q.area > 0:
                parts.append(q)
    return unary_union(parts)


def save_fig_to_svg(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="svg", transparent=True, bbox_inches="tight", pad_inches=0)
    buf.seek(0)
    return buf


def save_fig_to_dxf(fig):
    """Export plotted geometry to DXF in data coordinates (projected metres)."""
    if not _HAS_EZDXF:
        return save_fig_to_svg(fig).getvalue(), "svg"
    try:
        from matplotlib.collections import (
            PathCollection, LineCollection, PolyCollection, PatchCollection
        )

        doc = ezdxf.new("R2010")
        doc.header["$INSUNITS"] = 6   # metres
        msp = doc.modelspace()

        def _pts(arr):
            return [(float(p[0]), float(p[1])) for p in arr
                    if np.isfinite(p[0]) and np.isfinite(p[1])]

        for ax in fig.get_axes():
            for line in ax.get_lines():
                if not line.get_visible():
                    continue
                pts = _pts(zip(line.get_xdata(), line.get_ydata()))
                if len(pts) >= 2:
                    msp.add_lwpolyline(pts, dxfattribs={"layer": "LINES"})

            for artist in ax.collections:
                if not artist.get_visible():
                    continue
                if isinstance(artist, LineCollection):
                    for seg in artist.get_segments():
                        pts = _pts(seg)
                        if len(pts) >= 2:
                            msp.add_lwpolyline(pts, dxfattribs={"layer": "EDGES"})
                elif isinstance(artist, (PatchCollection, PolyCollection)):
                    for path in artist.get_paths():
                        for ring in path.to_polygons(closed_only=False):
                            pts = _pts(ring)
                            if len(pts) >= 3:
                                msp.add_lwpolyline(pts, close=True,
                                                   dxfattribs={"layer": "BUILDINGS"})
                elif isinstance(artist, PathCollection):
                    for p in _pts(artist.get_offsets()):
                        msp.add_point((p[0], p[1], 0), dxfattribs={"layer": "POINTS"})

        buf = io.StringIO()
        doc.write(buf)
        return buf.getvalue().encode(doc.encoding or "cp1252", errors="replace"), "dxf"

    except Exception as e:
        st.session_state.setdefault("_dxf_errors", []).append(str(e))
        return save_fig_to_svg(fig).getvalue(), "svg"


def classify_landmark(row):
    amenity = str(getattr(row, "amenity", "nan")).lower()
    railway = str(getattr(row, "railway", "nan")).lower()
    shop    = str(getattr(row, "shop",    "nan")).lower()
    leisure = str(getattr(row, "leisure", "nan")).lower()

    if amenity in {"school", "university", "college"}:               return "Education",         "#3498DB", "^"
    if amenity in {"hospital", "clinic", "doctors", "pharmacy"}:     return "Health",            "#E74C3C", "P"
    if railway in {"station", "halt", "tram_stop", "subway_entrance"} \
       or amenity in {"bus_station", "ferry_terminal"}:              return "Transit Hub",       "#9B59B6", "D"
    if amenity in {"restaurant", "cafe", "fast_food", "food_court"}: return "Food & Dining",    "#E67E22", "o"
    if amenity in {"convenience", "supermarket", "marketplace"} \
       or shop  in {"convenience", "supermarket", "general", "mall"}:return "Retail / Shop",    "#27AE60", "s"
    if amenity in {"bank", "atm"}:                                   return "Finance",          "#F39C12", "p"
    if amenity == "place_of_worship":                                return "Place of Worship", "#1ABC9C", "h"
    if amenity in {"library", "community_centre"}:                   return "Community",        "#95A5A6", "8"
    if leisure == "park":                                            return "Park / Green",      "#2ECC71", "*"
    return "Other", "#BDC3C7", "."


LAND_USE_PALETTE = {
    "residential":       ("#FFEAA7", "Residential"),
    "commercial":        ("#FAB1A0", "Commercial"),
    "industrial":        ("#B2BEC3", "Industrial"),
    "retail":            ("#E17055", "Retail"),
    "education":         ("#74B9FF", "Education"),
    "institutional":     ("#A29BFE", "Institutional"),
    "school":            ("#74B9FF", "Education"),
    "university":        ("#74B9FF", "Education"),
    "college":           ("#74B9FF", "Education"),
    "hospital":          ("#FD79A8", "Healthcare"),
    "place_of_worship":  ("#DCDDE1", "Religious"),
    "religious":         ("#DCDDE1", "Religious"),
    "cemetery":          ("#BDC3C7", "Cemetery"),
    "park":              ("#55EFC4", "Park / Green"),
    "garden":            ("#00B894", "Garden"),
    "grass":             ("#00CEC9", "Grassland"),
    "meadow":            ("#81ECEC", "Meadow"),
    "grassland":         ("#81ECEC", "Grassland"),
    "forest":            ("#27AE60", "Forest"),
    "wood":              ("#27AE60", "Forest"),
    "scrub":             ("#2ECC71", "Scrub"),
    "orchard":           ("#6AB04C", "Orchard"),
    "farmland":          ("#F9CA24", "Farmland"),
    "nature_reserve":    ("#00B894", "Nature Reserve"),
    "water":             ("#0984E3", "Water"),
    "basin":             ("#3498DB", "Water Basin"),
    "reservoir":         ("#3498DB", "Reservoir"),
    "wetland":           ("#6C5CE7", "Wetland"),
    "floodplain":        ("#74B9FF", "Floodplain"),
    "recreation_ground": ("#FEA47F", "Recreation"),
    "sports_centre":     ("#F8C291", "Sports"),
    "pitch":             ("#78E08F", "Pitch"),
    "playground":        ("#FFC312", "Playground"),
    "golf_course":       ("#A3CB38", "Golf"),
    "parking":           ("#636E72", "Parking"),
    "construction":      ("#D63031", "Construction"),
    "brownfield":        ("#C0392B", "Brownfield"),
    "military":          ("#6D1E07", "Military"),
    "quarry":            ("#7F8C8D", "Quarry"),
    "landfill":          ("#95A5A6", "Landfill"),
}

UTILITY_STYLES = {
    "power":   ("#F1C40F", "#FFD700", 1.4, "--",  "^", 60,  "Powerline / Electrical"),
    "sewer":   ("#8B4513", "#A0522D", 1.2, "-.",  "o", 50,  "Sewer / Drain"),
    "water":   ("#3498DB", "#5DADE2", 1.3, "-",   "s", 55,  "Water Supply / Hydrant"),
    "telecom": ("#9B59B6", "#AF7AC5", 1.0, ":",   "D", 45,  "Telecom / Cabinet"),
}

WATER_LABELS = {"Water", "Water Basin", "Reservoir", "Wetland", "Floodplain"}


def classify_land_use(row):
    for field in ["landuse", "leisure", "amenity", "natural"]:
        val = str(getattr(row, field, "nan")).lower().strip()
        if val not in ("nan", "none", "") and val in LAND_USE_PALETTE:
            color, label = LAND_USE_PALETTE[val]
            return color, label
    return "#ECF0F1", "Other"

# ==============================================================================
# SESSION STATE
# ==============================================================================

MODULES = ["base_map", "figure_ground", "porosity", "mobility",
           "landmarks", "synthesis", "sun_path", "massing", "shadow",
           "land_use", "utilities", "flood_map", "topography"]

for mod in MODULES:
    if mod not in st.session_state:
        st.session_state[mod] = False

if "svg_exports" not in st.session_state:
    st.session_state.svg_exports = {}

if "gallery_images" not in st.session_state:
    st.session_state.gallery_images = {}

# Widget-backed keys are initialised here (not via value=/index=) so they can be
# safely updated from callbacks.
if "radius_val" not in st.session_state:
    st.session_state.radius_val = 500
if "_radius_slider" not in st.session_state:
    st.session_state._radius_slider = st.session_state.radius_val
if "_radius_text" not in st.session_state:
    st.session_state._radius_text = str(st.session_state.radius_val)
if "view_mode" not in st.session_state:
    st.session_state.view_mode = "Sections"

# ==============================================================================
# CALLBACKS
# ==============================================================================

def _sync_slider():
    v = int(st.session_state._radius_slider)
    st.session_state.radius_val = v
    st.session_state._radius_text = str(v)


def _sync_text():
    try:
        v = max(100, min(2000, int(st.session_state._radius_text)))
    except ValueError:
        v = st.session_state.radius_val
    st.session_state.radius_val = v
    st.session_state._radius_slider = v
    st.session_state._radius_text = str(v)


def _load_all():
    for mod in MODULES:
        st.session_state[mod] = True


def _clear_all():
    for mod in MODULES:
        st.session_state[mod] = False
    st.session_state.svg_exports = {}
    st.session_state.gallery_images = {}


def _generate_all_and_go_to_sections():
    # Runs as a callback, i.e. before the `view_mode` radio is instantiated,
    # which is the only time its session-state value may be changed.
    _load_all()
    st.session_state.view_mode = "Sections"

# ==============================================================================
# SIDEBAR
# ==============================================================================

st.sidebar.title("Site Configurator")
st.sidebar.markdown("Paste your Google Maps coordinates below.")

coord_input = st.sidebar.text_input("Coordinates (Lat, Lon)", value="14.0700, 121.3255")

try:
    sep = "," if "," in coord_input else None
    lat_str, lon_str = coord_input.split(sep, 1)
    lat, lon = float(lat_str.strip()), float(lon_str.strip())
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise ValueError("out of range")
except Exception:
    st.sidebar.error("⚠️ Invalid format. Using default.")
    lat, lon = 14.0700, 121.3255

st.sidebar.slider(
    "Analysis Radius (m)", 100, 2000,
    key="_radius_slider",
    on_change=_sync_slider,
)
st.sidebar.text_input(
    "Radius (m) — type exact value",
    key="_radius_text",
    on_change=_sync_text,
    placeholder="100 – 2000",
)
radius = st.session_state.radius_val

col_a, col_b = st.sidebar.columns(2)
col_a.button("Load All", type="primary", on_click=_load_all)
col_b.button("Clear All", on_click=_clear_all)

st.sidebar.markdown("---")

view_mode = st.sidebar.radio(
    "View",
    ["Sections", "Gallery"],
    key="view_mode",
    help="Sections — load and inspect maps one at a time.\n"
         "Gallery — see every map generated so far together, in a grid.",
)

st.sidebar.markdown("---")

if st.session_state.svg_exports:
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname, data in st.session_state.svg_exports.items():
            zf.writestr(fname, data)
    st.sidebar.download_button(
        label="⬇ Download All Maps (.zip)",
        data=zip_buf.getvalue(),
        file_name="site_analysis_export.zip",
        mime="application/zip",
        type="primary",
        use_container_width=True,
    )
    st.sidebar.markdown("---")

st.sidebar.markdown("### Export Format")
export_fmt = st.sidebar.radio(
    "Per-map download format",
    ["SVG", "DXF (AutoCAD / DWG-compatible)"],
    index=0,
    help="SVG — scalable vector for Illustrator / Inkscape.\n"
         "DXF — opens directly in AutoCAD, BricsCAD, etc. (DWG-compatible).",
)
_use_dxf = export_fmt.startswith("DXF")
if _use_dxf and not _HAS_EZDXF:
    st.sidebar.warning("⚠ ezdxf not installed. Run `pip install ezdxf` then restart.")
    _use_dxf = False
st.session_state["use_dxf"] = _use_dxf

st.sidebar.markdown("---")
st.sidebar.markdown("""
* [Base Map](#base-map)
* [Figure-Ground](#figure-ground)
* [Porosity](#porosity)
* [Mobility Network](#mobility-network)
* [Nearby Landmarks](#nearby-landmarks)
* [Environmental Synthesis](#environmental-synthesis)
* [Sun Path Analysis](#sun-path-analysis)
* [Massing Heatmap](#massing-heatmap)
* [3D Shadow Study](#3d-shadow-study)
* [Land Use](#land-use)
* [Utility Infrastructure](#utility-lines)
* [Flood Risk](#flood-risk)
* [Land Topography](#land-topography)
""")

# ==============================================================================
# HEADER
# ==============================================================================

st.markdown(
    '<p style="font-family:\'JetBrains Mono\',monospace;font-size:0.68rem;'
    'letter-spacing:0.14em;text-transform:uppercase;color:#707070;margin-bottom:0.1rem">'
    'Architectural Site Intelligence</p>',
    unsafe_allow_html=True,
)
st.title("Site Analysis")
st.caption(f"↗ {lat}°N  ·  {lon}°E  ·  {radius} m radius  ·  OpenStreetMap + Open-Meteo")

# ==============================================================================
# REUSABLE BUTTON HELPERS
# ==============================================================================

def load_button(mod_key, label="Load Map"):
    if not st.session_state[mod_key]:
        if st.button(label, key=f"load_{mod_key}", type="primary"):
            st.session_state[mod_key] = True
            st.rerun()
        return False
    return True


def set_site_extent(ax, cx, cy, radius, pad=1.05):
    ax.set_xlim(cx - radius * pad, cx + radius * pad)
    ax.set_ylim(cy - radius * pad, cy + radius * pad)
    ax.set_aspect("equal")


DISPLAY_WIDTH = 620


def display_and_store(fig, base_name, display_width=DISPLAY_WIDTH):
    png_buf = io.BytesIO()
    fig.savefig(png_buf, format="png", dpi=150, bbox_inches="tight",
                pad_inches=0.05, facecolor=fig.get_facecolor())
    png_bytes = png_buf.getvalue()
    st.session_state.gallery_images[base_name] = png_bytes
    st.image(png_bytes, width=display_width)


def download_button(filename, data, mime="image/svg+xml"):
    ext = filename.split(".")[-1].upper()
    st.download_button(
        label=f"⬇ Download as {ext}",
        data=data,
        file_name=filename,
        mime=mime,
        key=f"dl_{filename}",
        use_container_width=True,
    )


def export_buttons(base_name, fig, svg_data):
    use_dxf = st.session_state.get("use_dxf", False)
    if use_dxf:
        col1, col2 = st.columns(2)
        with col1:
            download_button(base_name + ".svg", svg_data)
        with col2:
            dxf_data, ext = save_fig_to_dxf(fig)
            if ext == "dxf":
                download_button(base_name + ".dxf", dxf_data, "application/dxf")
            else:
                dxf_errors = st.session_state.get("_dxf_errors", [])
                reason = f" ({dxf_errors[-1]})" if dxf_errors else ""
                st.warning(f"⚠ DXF unavailable for this layer — fell back to SVG{reason}.")
    else:
        download_button(base_name + ".svg", svg_data)


def finish_figure(fig, key, base_name):
    """Shared tail: SVG export, on-screen PNG, download buttons, close figure."""
    svg_data = save_fig_to_svg(fig).getvalue()
    st.session_state.svg_exports[f"{key}.svg"] = svg_data
    display_and_store(fig, key)
    export_buttons(key, fig, svg_data)
    plt.close(fig)


if view_mode == "Sections":
    # ==========================================================================
    # 1. BASE MAP
    # ==========================================================================

    st.header("Base Map", anchor="base-map")

    if load_button("base_map"):
        with st.spinner("Loading base map…"):
            m = folium.Map(location=[lat, lon], zoom_start=15, tiles="OpenStreetMap")
            folium.Circle(radius=radius, location=[lat, lon], color="#2b2b2b",
                          fill=True, fill_opacity=0.07, weight=1.5).add_to(m)
            folium.Marker([lat, lon], tooltip="Site Centre").add_to(m)
            # returned_objects=[] stops map pan/zoom from triggering full-app reruns
            st_folium(m, width=820, height=460, returned_objects=[])
            html_data = m.get_root().render().encode("utf-8")
            st.session_state.svg_exports["01_base_map.html"] = html_data
            download_button("01_base_map.html", html_data, "text/html")

    st.markdown("---")

    # ==========================================================================
    # 2. FIGURE-GROUND
    # ==========================================================================

    st.header("Figure-Ground", anchor="figure-ground")

    if load_button("figure_ground"):
        with st.spinner("Rendering figure-ground diagram…"):
            try:
                buildings_p, edges_p, crs, cx, cy = load_base(lat, lon, radius)

                fig, ax = plt.subplots(figsize=(10, 10), facecolor="white")
                ax.set_facecolor("white")
                edges_p["w"] = edge_widths(edges_p, 0.8)
                edges_p.plot(ax=ax, linewidth=edges_p["w"], color="black", zorder=1)
                buildings_p.plot(ax=ax, facecolor="black", edgecolor="none", zorder=2)
                set_site_extent(ax, cx, cy, radius)
                ax.set_axis_off()

                finish_figure(fig, "02_figure_ground", "Figure-Ground")
            except Exception as e:
                st.error(f"Figure-Ground error: {e}")

    st.markdown("---")

    # ==========================================================================
    # 3. POROSITY
    # ==========================================================================

    st.header("Porosity", anchor="porosity")

    if load_button("porosity", "Calculate Site Porosity"):
        with st.spinner("Calculating site density…"):
            try:
                buildings = fetch_buildings(lat, lon, radius)
                buildings_p = ox.projection.project_gdf(buildings)
                cx, cy = project_center(lat, lon, buildings_p.crs)

                # Buildings are fetched in a square window: clip to the analysis circle
                # and dissolve overlaps so built area can never exceed the site area.
                circle     = Point(cx, cy).buffer(radius, resolution=64)
                built_geom = unary_union(buildings_p.geometry.buffer(0)).intersection(circle)
                total_area = circle.area
                built_area = built_geom.area
                open_area  = total_area - built_area
                built_pct  = int(min(100, max(0, round(built_area / total_area * 100))))
                open_pct   = 100 - built_pct

                c1, c2, c3 = st.columns(3)
                c1.metric("Total Site Area",  f"{int(total_area):,} m²")
                c2.metric("Total Built Area", f"{int(built_area):,} m²")
                c3.metric("Total Open Space", f"{int(open_area):,} m²")
                st.caption(f"Site Density — {built_pct}% Built  /  {open_pct}% Open")
                st.progress(built_pct)
            except Exception as e:
                st.error(f"Porosity error: {e}")

    st.markdown("---")

    # ==========================================================================
    # 4. MOBILITY NETWORK
    # ==========================================================================

    st.header("Mobility Network", anchor="mobility-network")

    if load_button("mobility"):
        with st.spinner("Mapping streets and transit…"):
            try:
                buildings_p, edges_p, crs, scx, scy = load_base(lat, lon, radius)
                _, transit = fetch_access_points(lat, lon, radius)
                transit_p = transit.to_crs(crs) if transit is not None and not transit.empty else None

                ROAD_COLORS = {
                    "motorway": "#E74C3C", "trunk": "#E74C3C", "primary":  "#E67E22",
                    "secondary": "#F1C40F", "tertiary": "#2ECC71",
                    "residential": "#AED6F1", "unclassified": "#AED6F1",
                    "footway": "#D5DBDB", "path": "#D5DBDB",
                    "cycleway": "#A9CCE3", "service": "#D5DBDB",
                }
                DEFAULT_ROAD_COLOR = "#ECF0F1"

                def edge_color(highway_val):
                    t = str(highway_val[0] if isinstance(highway_val, list) else highway_val).lower()
                    for key, col in ROAD_COLORS.items():
                        if key in t: return col
                    return DEFAULT_ROAD_COLOR

                fig, ax = plt.subplots(figsize=(10, 10), facecolor="#1C1C2E")
                ax.set_facecolor("#1C1C2E")   # was an assignment that overwrote the method

                edges_p["lw"]  = edge_widths(edges_p)
                edges_p["col"] = edges_p["highway"].apply(edge_color) \
                    if "highway" in edges_p.columns else DEFAULT_ROAD_COLOR
                for col_val in edges_p["col"].unique():
                    subset = edges_p[edges_p["col"] == col_val]
                    subset.plot(ax=ax, linewidth=subset["lw"], color=col_val, zorder=1)

                buildings_p.plot(ax=ax, facecolor="#2C2C4E", edgecolor="#3D3D6B",
                                 linewidth=0.4, zorder=2)

                TRANSIT_STYLES = {
                    "bus_stop":        ("#F1C40F", "^", 120, "Bus Stop"),
                    "platform":        ("#F1C40F", "^", 120, "Bus Stop"),
                    "stop_position":   ("#F1C40F", "^", 100, "Bus Stop"),
                    "bus_station":     ("#E67E22", "D", 160, "Bus Station"),
                    "ferry_terminal":  ("#3498DB", "D", 160, "Ferry Terminal"),
                    "taxi":            ("#F39C12", "s",  90, "Taxi Stand"),
                    "station":         ("#9B59B6", "D", 180, "Train / Transit Station"),
                    "halt":            ("#9B59B6", "D", 140, "Train Halt"),
                    "tram_stop":       ("#1ABC9C", "^", 120, "Tram Stop"),
                    "subway_entrance": ("#E74C3C", "o", 120, "Subway Entrance"),
                    "crossing":        ("#ECF0F1", "P",  80, "Pedestrian Crossing"),
                    "traffic_signals": ("#2ECC71", "o",  60, "Traffic Signal"),
                }
                plotted_labels = {}

                if transit_p is not None and not transit_p.empty:
                    by_label = {}
                    for geom, row in zip(transit_p.geometry.centroid, transit_p.itertuples()):
                        for field in ["highway", "railway", "amenity", "public_transport"]:
                            val = str(getattr(row, field, "nan")).lower()
                            if val in TRANSIT_STYLES:
                                color, marker, size, label = TRANSIT_STYLES[val]
                                by_label.setdefault(
                                    label, {"xs": [], "ys": [], "color": color,
                                            "marker": marker, "size": size})
                                by_label[label]["xs"].append(geom.x)
                                by_label[label]["ys"].append(geom.y)
                                plotted_labels[label] = (color, marker)
                                break

                    for label, d in by_label.items():
                        ax.scatter(d["xs"], d["ys"], color=d["color"], s=d["size"],
                                   marker=d["marker"], edgecolor="white", linewidth=0.6,
                                   zorder=10, alpha=0.95)

                ax.scatter([scx], [scy], color="white", s=220, marker="*",
                           edgecolor="#E74C3C", linewidth=1.5, zorder=15)

                road_legend = [Line2D([0], [0], color=c, lw=2, label=l) for c, l in [
                    ("#E74C3C", "Primary"), ("#E67E22", "Secondary"), ("#F1C40F", "Tertiary"),
                    ("#AED6F1", "Residential"), ("#A9CCE3", "Path"),
                ]]
                transit_legend = [
                    Line2D([0], [0], marker=mk, color="w", markerfacecolor=c,
                           markersize=8, label=lbl, linestyle="None")
                    for lbl, (c, mk) in plotted_labels.items()
                ] + [Line2D([0], [0], marker="*", color="w", markerfacecolor="white",
                            markersize=10, label="Site", linestyle="None")]
                ax.legend(handles=road_legend + transit_legend, loc="upper right",
                          frameon=True, facecolor="#1C1C2E", labelcolor="white", fontsize=7.5)
                set_site_extent(ax, scx, scy, radius)
                ax.set_axis_off()

                finish_figure(fig, "04_mobility", "Mobility")
            except Exception as e:
                st.error(f"Mobility error: {e}")

    st.markdown("---")

    # ==========================================================================
    # 5. NEARBY LANDMARKS
    # ==========================================================================

    st.header("Nearby Landmarks", anchor="nearby-landmarks")

    if load_button("landmarks"):
        with st.spinner("Mapping amenities…"):
            try:
                buildings_p, edges_p, crs, scx, scy = load_base(lat, lon, radius)
                landmarks   = fetch_landmarks(lat, lon, radius)
                landmarks_p = landmarks.to_crs(crs) if not landmarks.empty else landmarks

                fig, ax = plt.subplots(figsize=(11, 11), facecolor="#F8F9FA")
                ax.set_facecolor("#F8F9FA")

                edges_p["lw"] = edge_widths(edges_p, 0.7, 0.8 / 0.7)
                edges_p.plot(ax=ax, linewidth=edges_p["lw"], color="#CCD1D1", zorder=1)
                buildings_p.plot(ax=ax, facecolor="#E8EAED", edgecolor="#BFC9CA",
                                 linewidth=0.3, zorder=2)

                counts_by_cat  = {}
                legend_entries = {}

                if not landmarks_p.empty:
                    by_cat = {}
                    for geom, row in zip(landmarks_p.geometry.centroid, landmarks_p.itertuples()):
                        cat, color, marker = classify_landmark(row)
                        counts_by_cat[cat] = counts_by_cat.get(cat, 0) + 1
                        legend_entries[cat] = (color, marker)
                        by_cat.setdefault(cat, {"xs": [], "ys": [], "color": color, "marker": marker})
                        by_cat[cat]["xs"].append(geom.x)
                        by_cat[cat]["ys"].append(geom.y)

                    for cat, d in by_cat.items():
                        ax.scatter(d["xs"], d["ys"], color=d["color"], s=130, marker=d["marker"],
                                   edgecolor="white", linewidth=0.7, zorder=8, alpha=0.92)

                ax.scatter([scx], [scy], color="#0f0f0f", s=220, marker="*",
                           edgecolor="white", linewidth=1.5, zorder=15)

                handles = [
                    Line2D([0], [0], marker=mk, color="w", markerfacecolor=c,
                           markersize=10, label=lbl, linestyle="None")
                    for lbl, (c, mk) in sorted(legend_entries.items())
                ]
                if handles:
                    ax.legend(handles=handles, loc="upper right", frameon=True,
                              facecolor="white", fontsize=8)
                set_site_extent(ax, scx, scy, radius)
                ax.set_axis_off()

                finish_figure(fig, "05_landmarks", "Landmarks")

                cols = st.columns(4)
                for i, (cat, cnt) in enumerate(sorted(counts_by_cat.items())):
                    cols[i % 4].metric(cat, cnt)
            except Exception as e:
                st.error(f"Landmarks error: {e}")

    st.markdown("---")

    # ==========================================================================
    # 6. ENVIRONMENTAL SYNTHESIS
    # ==========================================================================

    st.header("Environmental Synthesis", anchor="environmental-synthesis")

    if load_button("synthesis"):
        with st.spinner("Layering environmental data…"):
            try:
                buildings_p, edges_p, crs, cx, cy = load_base(lat, lon, radius)
                wind_dirs, wind_speeds = fetch_wind_data(lat, lon)
                waterways, water_bodies, elev_grid = fetch_flood_layers(lat, lon, radius)

                fig, ax = plt.subplots(figsize=(12, 12))
                ax.set_facecolor("#F0F4F8")

                if elev_grid is not None:
                    xs_all, ys_all = lonlat_to_xy(elev_grid["lons"], elev_grid["lats"], crs)
                    zs_all = np.array(
                        [e if e is not None else float("nan")
                         for e in elev_grid["elevations"]], dtype=float)
                    valid = ~np.isnan(zs_all)
                    if valid.sum() >= 4:
                        xs, ys, zs = xs_all[valid], ys_all[valid], zs_all[valid]
                        xi, yi = np.linspace(xs.min(), xs.max(), 120), np.linspace(ys.min(), ys.max(), 120)
                        Xi, Yi = np.meshgrid(xi, yi)
                        try:
                            Zi = griddata((xs, ys), zs, (Xi, Yi), method="cubic")
                        except Exception:
                            Zi = griddata((xs, ys), zs, (Xi, Yi), method="linear")
                        nan_mask = np.isnan(Zi)
                        if nan_mask.any():
                            Zi_nn = griddata((xs, ys), zs, (Xi, Yi), method="nearest")
                            Zi[nan_mask] = Zi_nn[nan_mask]
                        flood_cmap = LinearSegmentedColormap.from_list("flood", [
                            (0.08, 0.35, 0.75, 0.65),
                            (0.35, 0.65, 0.95, 0.35),
                            (0.85, 0.93, 1.0,  0.0),
                        ])
                        if np.nanmax(Zi) - np.nanmin(Zi) > 1e-6:
                            cf = ax.contourf(Xi, Yi, Zi, levels=12, cmap=flood_cmap,
                                             vmin=np.nanmin(Zi), vmax=np.nanmax(Zi), zorder=0)
                            cbar = plt.colorbar(cf, ax=ax, fraction=0.025, pad=0.01)
                            cbar.set_label("Elevation (m)")

                if water_bodies is not None and not water_bodies.empty:
                    try:
                        water_bodies.to_crs(crs).plot(
                            ax=ax, facecolor="#4A90D9", edgecolor="#2471A3",
                            linewidth=0.8, alpha=0.55, zorder=1)
                    except Exception:
                        pass

                if waterways is not None and not waterways.empty:
                    try:
                        waterways.to_crs(crs).plot(ax=ax, linewidth=1.5, color="#1A6FA8",
                                                   alpha=0.85, zorder=2)
                    except Exception:
                        pass

                edges_p["w"] = edge_widths(edges_p, 0.8, 1.2 / 0.8)
                edges_p.plot(ax=ax, linewidth=edges_p["w"], color="#B2BEC3", alpha=0.7, zorder=3)
                buildings_p.plot(ax=ax, facecolor="#DFE6E9", edgecolor="#636E72",
                                 linewidth=0.4, alpha=0.9, zorder=4)

                valid_dirs = [d for d in wind_dirs if d is not None]
                n_sectors  = 16
                sector_deg = 360.0 / n_sectors
                counts, _  = np.histogram(valid_dirs, bins=np.linspace(0, 360, n_sectors + 1))
                max_c      = counts.max() if counts.max() > 0 else 1
                max_rose_r = radius * 0.48
                wind_cmap  = matplotlib.colormaps["cool"]

                for i in range(n_sectors):
                    petal_r = (counts[i] / max_c) * max_rose_r
                    if petal_r < max_rose_r * 0.04: continue
                    center_az    = i * sector_deg
                    half         = sector_deg / 2.0
                    wedge_angles = np.linspace(
                        math.radians(90.0 - (center_az + half)),
                        math.radians(90.0 - (center_az - half)), 20)
                    wx = [cx] + [cx + petal_r * math.cos(a) for a in wedge_angles] + [cx]
                    wy = [cy] + [cy + petal_r * math.sin(a) for a in wedge_angles] + [cy]
                    ax.fill(wx, wy, color=wind_cmap(counts[i] / max_c), alpha=0.52, zorder=9)

                set_site_extent(ax, cx, cy, radius)
                ax.set_axis_off()

                finish_figure(fig, "06_synthesis", "Synthesis")
            except Exception as e:
                st.error(f"Synthesis error: {e}")

    st.markdown("---")

    # ==========================================================================
    # 6.5 SUN PATH ANALYSIS
    # ==========================================================================

    st.header("Sun Path Analysis", anchor="sun-path-analysis")

    if load_button("sun_path"):
        with st.spinner("Mapping solar paths and directions…"):
            try:
                buildings_p, edges_p, crs, cx, cy = load_base(lat, lon, radius)

                fig, ax = plt.subplots(figsize=(12, 12))
                ax.set_facecolor("#F0F4F8")

                edges_p["w"] = edge_widths(edges_p, 0.8, 1.2 / 0.8)
                edges_p.plot(ax=ax, linewidth=edges_p["w"], color="#B2BEC3", alpha=0.7, zorder=3)
                buildings_p.plot(ax=ax, facecolor="#DFE6E9", edgecolor="#636E72",
                                 linewidth=0.4, alpha=0.9, zorder=4)

                SUN_DATES = [
                    (datetime.date(2024, 6, 21),  "#FF7675", "Summer Solstice"),
                    (datetime.date(2024, 3, 21),  "#FDCB6E", "Equinox"),
                    (datetime.date(2024, 12, 21), "#74B9FF", "Winter Solstice"),
                ]
                for date, color, label in SUN_DATES:
                    pts = sun_path_points(lat, lon, date, radius, cx, cy)
                    if len(pts) > 1:
                        ax.plot([p[1] for p in pts], [p[2] for p in pts],
                                color=color, linewidth=2.5, linestyle="--", zorder=8, label=label)

                max_rose_r = radius * 0.48
                for lbl, az_deg in [("N", 0), ("E", 90), ("S", 180), ("W", 270)]:
                    a = math.radians(90.0 - az_deg)
                    ax.text(cx + (max_rose_r * 1.12) * math.cos(a),
                            cy + (max_rose_r * 1.12) * math.sin(a),
                            lbl, ha="center", va="center", fontsize=11, fontweight="bold", zorder=12)

                ax.legend(loc="upper right", frameon=True, facecolor="white", fontsize=8)
                set_site_extent(ax, cx, cy, radius)
                ax.set_axis_off()

                finish_figure(fig, "06b_sun_path", "Sun Path")
            except Exception as e:
                st.error(f"Sun Path error: {e}")

    st.markdown("---")

    # ==========================================================================
    # 7. MASSING HEATMAP
    # ==========================================================================

    st.header("Massing Heatmap", anchor="massing-heatmap")

    if load_button("massing"):
        with st.spinner("Extracting building heights…"):
            try:
                buildings_p, edges_p, crs, cx, cy = load_base(lat, lon, radius)
                buildings_p["calc_height"] = building_heights(buildings_p, 3.0)

                fig, ax = plt.subplots(figsize=(12, 12), facecolor="#F8F9FA")
                ax.set_facecolor("#F8F9FA")

                edges_p["w"] = edge_widths(edges_p, 0.6, 0.8 / 0.6)
                edges_p.plot(ax=ax, linewidth=edges_p["w"], color="#CCD1D1", zorder=1)

                hmin = float(buildings_p["calc_height"].min())
                hmax = float(buildings_p["calc_height"].max())
                if hmax <= hmin:
                    hmax = hmin + 1.0
                norm = plt.Normalize(vmin=hmin, vmax=hmax)
                buildings_p.plot(ax=ax, column="calc_height", cmap="magma_r", norm=norm,
                                 edgecolor="#BDC3C7", linewidth=0.3, legend=False, zorder=2)

                divider = make_axes_locatable(ax)
                cax     = divider.append_axes("right", size="3%", pad=0.1)
                sm      = plt.cm.ScalarMappable(cmap="magma_r", norm=norm)
                sm.set_array([])
                cbar = fig.colorbar(sm, cax=cax)
                cbar.set_label("Building Height (m)", rotation=270, labelpad=15, fontweight="bold")
                set_site_extent(ax, cx, cy, radius)
                ax.set_axis_off()

                finish_figure(fig, "07_massing_heatmap", "Massing")
            except Exception as e:
                st.error(f"Massing heatmap error: {e}")

    st.markdown("---")

    # ==========================================================================
    # 8. 3D SHADOW STUDY
    # ==========================================================================

    st.header("3D Shadow Study", anchor="3d-shadow-study")
    st.markdown("Adjust the time of day to visualize dynamic sun shading across the site's urban massing.")

    col1, col2 = st.columns(2)
    with col1:
        shadow_date = st.date_input("Date for Sun Path", datetime.date(2024, 6, 21))
    with col2:
        shadow_time = st.slider(
            "Time of Day (local solar zone)",
            min_value=datetime.time(6, 0),
            max_value=datetime.time(18, 0),
            value=datetime.time(12, 0),
            format="HH:mm",
        )

    if load_button("shadow"):
        with st.spinner("Projecting 3D shadows…"):
            try:
                buildings = fetch_buildings(lat, lon, radius).copy()
                buildings["calc_height"] = building_heights(buildings, 15.0)

                tz  = site_tz(lon)
                loc = LocationInfo("Site", "Region", "UTC", lat, lon)
                dt_local = datetime.datetime.combine(shadow_date, shadow_time, tzinfo=tz)
                el        = elevation(loc.observer, dt_local)
                az        = azimuth(loc.observer, dt_local)
                sun_is_up = el > 2
                el_rad    = math.radians(max(el, 1))
                az_rad    = math.radians(az)

                buildings_proj = ox.projection.project_gdf(buildings)
                target_crs     = buildings_proj.crs
                shadow_geoms   = []

                if sun_is_up:
                    for _, row in buildings_proj.iterrows():
                        geom = row.geometry
                        if geom is None or geom.is_empty: continue
                        shadow_len = row["calc_height"] / math.tan(el_rad)
                        dx = -math.sin(az_rad) * shadow_len
                        dy = -math.cos(az_rad) * shadow_len
                        shadow_geoms.append(shadow_polygon(geom, dx, dy))

                if shadow_geoms:
                    shadow_gdf  = gpd.GeoDataFrame(geometry=shadow_geoms, crs=target_crs).to_crs("EPSG:4326")
                    shadow_dict = json.loads(shadow_gdf.to_json())
                else:
                    shadow_dict = {"type": "FeatureCollection", "features": []}

                m_per_deg_lon = 111320 * math.cos(math.radians(lat))
                sun_arc_radius = radius * 0.72

                def sun_pos(az_deg, el_deg, r):
                    a = math.radians(az_deg)
                    k = 1 - el_deg / 90
                    return [lon + (r / m_per_deg_lon) * math.sin(a) * k,
                            lat + (r / 111320) * math.cos(a) * k]

                sun_path_coords = []
                for h_i in range(6, 19):
                    for m_i in [0, 10, 20, 30, 40, 50]:
                        t = datetime.datetime.combine(shadow_date, datetime.time(h_i, m_i), tzinfo=tz)
                        el_i = elevation(loc.observer, t)
                        if el_i <= 0: continue
                        sun_path_coords.append(sun_pos(azimuth(loc.observer, t), el_i, sun_arc_radius))

                key_sun_points = []
                for h_i in sorted({6, shadow_time.hour, 18}):
                    t = datetime.datetime.combine(shadow_date, datetime.time(h_i, 0), tzinfo=tz)
                    el_i = elevation(loc.observer, t)
                    if el_i <= 0: continue
                    az_i   = azimuth(loc.observer, t)
                    is_cur = (h_i == shadow_time.hour)
                    ampm   = "AM" if h_i < 12 else "PM"
                    h12    = h_i if h_i <= 12 else h_i - 12
                    key_sun_points.append({
                        "position": sun_pos(az_i, el_i, sun_arc_radius),
                        "label":  f"{h12}:00 {ampm}",
                        "size":   16 if is_cur else 11,
                        "color":  [255, 225, 0, 255] if is_cur else [255, 225, 0, 160],
                        "radius": 18 if is_cur else 10,
                    })

                cardinal_r = sun_arc_radius * 1.18
                cardinals  = [
                    {"position": [
                        lon + (cardinal_r / m_per_deg_lon) * math.sin(math.radians(az_c)),
                        lat + (cardinal_r / 111320) * math.cos(math.radians(az_c)),
                     ], "label": lbl}
                    for lbl, az_c in [("N", 0), ("E", 90), ("S", 180), ("W", 270)]
                ]

                # reset_index: OSM features have a (element_type, osmid) MultiIndex,
                # which GeoDataFrame.to_json cannot serialise reliably.
                buildings_dict = json.loads(
                    buildings[["geometry", "calc_height"]].reset_index(drop=True).to_json())
                brightness = int(max(180, min(230, 180 + el * 0.8))) if sun_is_up else 160
                fill_color = [brightness, brightness - 8, brightness - 15, 255]

                shadow_layer = pdk.Layer(
                    "GeoJsonLayer", data=shadow_dict,
                    opacity=1.0, filled=True, stroked=False,
                    get_fill_color=[40, 35, 25, 170], pickable=False,
                )
                building_layer = pdk.Layer(
                    "GeoJsonLayer", data=buildings_dict,
                    opacity=1.0, stroked=True, filled=True,
                    extruded=True, wireframe=False,
                    get_elevation="properties.calc_height",
                    get_fill_color=fill_color,
                    get_line_color=[200, 190, 175, 80],
                    get_line_width=1, pickable=True,
                )
                sun_path_layer = pdk.Layer(
                    "PathLayer", data=[{"path": sun_path_coords}],
                    get_path="path", get_color=[255, 220, 50, 120],
                    get_width=4, width_min_pixels=1, pickable=False,
                )
                sun_label_layer = pdk.Layer(
                    "TextLayer", data=key_sun_points,
                    get_position="position", get_text="label",
                    get_size="size", get_color="color",
                    billboard=True, pickable=False,
                )
                sun_dot_layer = pdk.Layer(
                    "ScatterplotLayer", data=key_sun_points,
                    get_position="position", get_radius="radius",
                    radius_min_pixels=6, radius_max_pixels=20,
                    get_fill_color="color", stroked=False, pickable=False,
                )
                cardinal_layer = pdk.Layer(
                    "TextLayer", data=cardinals,
                    get_position="position", get_text="label",
                    get_size=16, get_color=[140, 130, 115, 200],
                    font_weight=700, pickable=False, billboard=False,
                )

                view_state = pdk.ViewState(
                    latitude=lat, longitude=lon,
                    zoom=15.5, pitch=55, bearing=15,
                )
                r = pdk.Deck(
                    layers=[shadow_layer, building_layer, sun_path_layer,
                            sun_label_layer, sun_dot_layer, cardinal_layer],
                    initial_view_state=view_state,
                    map_provider=None,
                )

                st.pydeck_chart(r)

                deck_html = r.to_html(as_string=True).encode("utf-8")
                st.session_state.svg_exports["08_3d_shadow.html"] = deck_html
                download_button("08_3d_shadow.html", deck_html, "text/html")

                c1, c2, c3 = st.columns(3)
                c1.metric("Sun Elevation", f"{el:.1f}°")
                c2.metric("Sun Azimuth",   f"{az:.1f}°")
                c3.metric(
                    "Shadow Length (15 m bldg)",
                    f"{(15 / math.tan(el_rad)):.1f} m" if sun_is_up else "Below Horizon",
                )
            except Exception as e:
                st.error(f"3D Shadow error: {e}")

    st.markdown("---")

    # ==========================================================================
    # 9. LAND USE MAP
    # ==========================================================================

    st.header("Land Use", anchor="land-use")
    st.caption("OSM landuse, leisure, amenity, and natural polygons classified by use type.")

    if load_button("land_use", "Map Land Use"):
        with st.spinner("Classifying land parcels…"):
            try:
                buildings_p, edges_p, crs, scx, scy = load_base(lat, lon, radius)
                land_use = fetch_land_use(lat, lon, radius)

                fig, ax = plt.subplots(figsize=(12, 12), facecolor="#F0F0EE")
                ax.set_facecolor("#F0F0EE")

                legend_entries = {}
                counts = {}

                if land_use is not None and not land_use.empty:
                    land_use_p = land_use.to_crs(crs).copy()

                    cats = [classify_land_use(row) for row in land_use_p.itertuples()]
                    land_use_p["_cat_color"] = [c[0] for c in cats]
                    land_use_p["_cat_label"] = [c[1] for c in cats]

                    # Water is drawn by the flood map; drop it *before* building the legend
                    land_use_p = land_use_p[~land_use_p["_cat_label"].isin(WATER_LABELS)]

                    for color, label in zip(land_use_p["_cat_color"], land_use_p["_cat_label"]):
                        legend_entries[label] = color
                        counts[label] = counts.get(label, 0) + 1

                    for color in land_use_p["_cat_color"].unique():
                        subset = land_use_p[land_use_p["_cat_color"] == color]
                        subset.plot(ax=ax, facecolor=color, edgecolor="white",
                                    linewidth=0.3, alpha=0.75, zorder=1)

                edges_p["w"] = edge_widths(edges_p, 0.9)
                edges_p.plot(ax=ax, linewidth=edges_p["w"], color="white", alpha=0.9, zorder=3)
                edges_p.plot(ax=ax, linewidth=edges_p["w"] * 2.2, color="#BBBBBB",
                             alpha=0.25, zorder=2)

                buildings_p.plot(ax=ax, facecolor="#3D3D3D", edgecolor="none",
                                 alpha=0.70, zorder=4)

                ax.scatter([scx], [scy], color="white", s=260, marker="*",
                           edgecolor="#E74C3C", linewidth=2.0, zorder=15)

                sorted_cats = sorted(legend_entries.items(), key=lambda kv: -counts.get(kv[0], 0))
                handles = [
                    Patch(facecolor=c, edgecolor="#AAAAAA", label=lbl, linewidth=0.4, alpha=0.85)
                    for lbl, c in sorted_cats
                ] + [Line2D([0], [0], marker="*", color="w", markerfacecolor="white",
                            markersize=10, label="Site", linestyle="None",
                            markeredgecolor="#E74C3C")]
                ax.legend(handles=handles, loc="upper right", frameon=True,
                          facecolor="white", fontsize=7, ncol=2,
                          edgecolor="#DDDDDD", title="Land Use", title_fontsize=7)
                set_site_extent(ax, scx, scy, radius)
                ax.set_axis_off()

                finish_figure(fig, "09_land_use", "Land Use")

                if counts:
                    st.caption("Parcel counts by land use category")
                    cols = st.columns(4)
                    for i, (cat, cnt) in enumerate(sorted(counts.items(), key=lambda x: -x[1])[:12]):
                        cols[i % 4].metric(cat, cnt)
            except Exception as e:
                st.error(f"Land Use error: {e}")

    st.markdown("---")

    # ==========================================================================
    # 10. UTILITY INFRASTRUCTURE
    # ==========================================================================

    st.header("Utility Infrastructure", anchor="utility-lines")
    st.caption("Street-level electrical poles/cables, sewer/drain lines, water supply, and telecom from OSM.")

    if load_button("utilities", "Map Utilities"):
        with st.spinner("Tracing utility networks…"):
            try:
                buildings_p, edges_p, crs, scx, scy = load_base(lat, lon, radius)
                util_data = fetch_utilities(lat, lon, radius)

                fig, ax = plt.subplots(figsize=(12, 12), facecolor="#1A1A2E")
                ax.set_facecolor("#1A1A2E")

                edges_p["w"] = edge_widths(edges_p, 0.55)
                edges_p.plot(ax=ax, linewidth=edges_p["w"], color="#2C2C4E", alpha=0.85, zorder=1)
                buildings_p.plot(ax=ax, facecolor="#16213E", edgecolor="#1a2a50",
                                 linewidth=0.3, alpha=0.95, zorder=2)

                legend_lines = []
                plotted_any  = False

                for util_key, (lc, pc, lw, ls, pm, ps, label) in UTILITY_STYLES.items():
                    gdf = util_data.get(util_key)
                    if gdf is None or gdf.empty:
                        continue

                    lines  = gdf[gdf.geom_type.isin(["LineString", "MultiLineString"])]
                    points = gdf[gdf.geom_type == "Point"]
                    added_to_legend = False

                    # Poles are only nodes in OSM: join nearby ones to suggest the line route
                    if util_key == "power" and len(points) > 1:
                        pts_p  = points.to_crs(crs)
                        coords = np.array([(g.x, g.y) for g in pts_p.geometry])
                        pairs  = cKDTree(coords).query_pairs(r=65.0)
                        synth_lines = [LineString([coords[i], coords[j]]) for i, j in pairs]
                        if synth_lines:
                            gpd.GeoDataFrame(geometry=synth_lines, crs=crs).plot(
                                ax=ax, linewidth=lw * 0.8, color=lc, linestyle=ls,
                                alpha=0.7, zorder=4)
                            legend_lines.append(Line2D([0], [0], color=lc, lw=lw, linestyle=ls, label=label))
                            added_to_legend = True
                            plotted_any = True

                    if not lines.empty:
                        lines.to_crs(crs).plot(ax=ax, linewidth=lw, color=lc,
                                               linestyle=ls, alpha=0.92, zorder=5)
                        if not added_to_legend:
                            legend_lines.append(Line2D([0], [0], color=lc, lw=lw, linestyle=ls, label=label))
                            added_to_legend = True
                        plotted_any = True

                    if not points.empty:
                        pts_p = points.to_crs(crs)
                        ax.scatter(pts_p.geometry.x, pts_p.geometry.y, color=pc, s=ps, marker=pm,
                                   edgecolor="white", linewidth=0.4, zorder=10, alpha=0.95)
                        if not added_to_legend:
                            legend_lines.append(
                                Line2D([0], [0], marker=pm, color="w", markerfacecolor=pc,
                                       markersize=7, label=label, linestyle="None"))
                            added_to_legend = True
                        plotted_any = True

                ax.scatter([scx], [scy], color="white", s=240, marker="*",
                           edgecolor="#E74C3C", linewidth=1.5, zorder=15)
                legend_lines.append(
                    Line2D([0], [0], marker="*", color="w", markerfacecolor="white",
                           markersize=10, label="Site", linestyle="None",
                           markeredgecolor="#E74C3C"))

                ax.legend(handles=legend_lines, loc="upper right", frameon=True,
                          facecolor="#1A1A2E", labelcolor="white", fontsize=8,
                          edgecolor="#333355")
                set_site_extent(ax, scx, scy, radius)
                ax.set_axis_off()

                if not plotted_any:
                    st.warning(
                        "No street-level utility data found in OSM for this area. "
                        "OSM utility coverage is best in dense urban cores — try increasing "
                        "the radius, or the area may not have utility tagging yet.")

                finish_figure(fig, "10_utilities", "Utilities")
            except Exception as e:
                st.error(f"Utilities error: {e}")

    st.markdown("---")

    # ==========================================================================
    # 11. FLOOD RISK MAP
    # ==========================================================================

    st.header("Flood Risk", anchor="flood-risk")
    st.caption("Elevation-derived inundation risk, OSM waterways, and flood-prone land use zones.")

    if load_button("flood_map", "Generate Flood Map"):
        with st.spinner("Analysing flood risk layers…"):
            try:
                buildings_p, edges_p, crs, cx, cy = load_base(lat, lon, radius)
                waterways, water_bodies, elev_grid = fetch_flood_layers(lat, lon, radius)
                land_use = fetch_land_use(lat, lon, radius)

                fig, ax = plt.subplots(figsize=(12, 12), facecolor="#EAF4FB")
                ax.set_facecolor("#EAF4FB")

                elev_min = elev_max = None
                xs_v = ys_v = zs_v = None

                if elev_grid is not None:
                    xs, ys = lonlat_to_xy(elev_grid["lons"], elev_grid["lats"], crs)
                    zs = np.array(
                        [e if e is not None else float("nan")
                         for e in elev_grid["elevations"]], dtype=float)
                    valid = ~np.isnan(zs)
                    if valid.sum() >= 9:
                        xs_a, ys_a, zs_a = xs[valid], ys[valid], zs[valid]

                        pad  = (xs_a.max() - xs_a.min()) * 0.02
                        xi   = np.linspace(xs_a.min() - pad, xs_a.max() + pad, 250)
                        yi   = np.linspace(ys_a.min() - pad, ys_a.max() + pad, 250)
                        Xi, Yi = np.meshgrid(xi, yi)

                        Zi = griddata((xs_a, ys_a), zs_a, (Xi, Yi), method="linear")
                        nan_mask = np.isnan(Zi)
                        if nan_mask.any():
                            Zi_nn = griddata((xs_a, ys_a), zs_a, (Xi, Yi), method="nearest")
                            Zi[nan_mask] = Zi_nn[nan_mask]

                        elev_min = float(np.nanmin(Zi))
                        elev_max = float(np.nanmax(Zi))
                        xs_v, ys_v, zs_v = xs_a, ys_a, zs_a

                        # Contours need a non-flat surface
                        if elev_max - elev_min > 1e-6:
                            flood_cmap = LinearSegmentedColormap.from_list("flood_risk", [
                                "#08306B", "#1565C0", "#1976D2",
                                "#42A5F5", "#90CAF9", "#BBDEFB", "#F5F9FF",
                            ])
                            n_levels = min(16, max(6, int((elev_max - elev_min) * 2)))
                            cf = ax.contourf(Xi, Yi, Zi, levels=n_levels,
                                             cmap=flood_cmap, zorder=0, alpha=0.85)
                            try:
                                cs = ax.contour(Xi, Yi, Zi, levels=min(8, n_levels),
                                                colors="#1A4D7A", linewidths=0.6,
                                                alpha=0.55, zorder=1)
                                ax.clabel(cs, inline=True, fontsize=6.5,
                                          fmt="%.1f m", colors="#1A4D7A")
                            except Exception:
                                pass
                            cbar = plt.colorbar(cf, ax=ax, fraction=0.025, pad=0.01)
                            cbar.set_label("Elevation (m asl)", fontsize=8)
                            cbar.ax.tick_params(labelsize=7)

                FLOOD_LANDUSE = {"floodplain", "basin", "reservoir", "water", "wetland", "mud"}
                if land_use is not None and not land_use.empty:
                    mask = []
                    for row in land_use.itertuples():
                        mask.append(any(
                            str(getattr(row, f, "nan")).lower() in FLOOD_LANDUSE
                            for f in ["landuse", "natural", "water"]))
                    fz = land_use[np.array(mask, dtype=bool)]
                    fz = fz[fz.geom_type.isin(["Polygon", "MultiPolygon"])]
                    if not fz.empty:
                        fz.to_crs(crs).plot(ax=ax, facecolor="#1565C0", edgecolor="#0D47A1",
                                            linewidth=0.7, alpha=0.45, zorder=2)

                if water_bodies is not None and not water_bodies.empty:
                    try:
                        water_bodies.to_crs(crs).plot(
                            ax=ax, facecolor="#1976D2", edgecolor="#0D47A1",
                            linewidth=0.9, alpha=0.80, zorder=3)
                    except Exception:
                        pass

                if waterways is not None and not waterways.empty:
                    try:
                        ww_p = waterways.to_crs(crs).copy()

                        def ww_lw(val):
                            v = str(val[0] if isinstance(val, list) else val).lower()
                            if "river" in v or "canal" in v: return 3.0
                            if "stream" in v: return 2.0
                            return 1.2

                        ww_p["_lw"] = ww_p["waterway"].apply(ww_lw) \
                            if "waterway" in ww_p.columns else 1.5
                        for lw_val in ww_p["_lw"].unique():
                            ww_p[ww_p["_lw"] == lw_val].plot(
                                ax=ax, linewidth=float(lw_val), color="#0D47A1",
                                alpha=0.90, zorder=4)
                    except Exception:
                        pass

                edges_p["w"] = edge_widths(edges_p, 0.85)
                edges_p.plot(ax=ax, linewidth=edges_p["w"], color="#4A6080", alpha=0.55, zorder=5)

                if xs_v is not None and elev_min is not None and elev_max > elev_min:
                    b_cx = np.array([g.centroid.x for g in buildings_p.geometry])
                    b_cy = np.array([g.centroid.y for g in buildings_p.geometry])
                    b_elev = griddata((xs_v, ys_v), zs_v, (b_cx, b_cy), method="nearest")
                    safe = np.clip((b_elev - elev_min) / (elev_max - elev_min), 0.0, 1.0)
                    r_ch = np.clip(0.78 + safe * 0.07, 0, 1)
                    g_ch = np.clip(0.45 + safe * 0.37, 0, 1)
                    b_ch = np.clip(0.50 + safe * 0.32, 0, 1)
                    colors_arr = np.column_stack([r_ch, g_ch, b_ch, np.full(len(safe), 0.88)])
                    bins = np.round(safe * 9).astype(int)
                    buildings_reset = buildings_p.reset_index(drop=True)
                    for bin_val in range(10):
                        idx = np.where(bins == bin_val)[0]
                        if len(idx) == 0:
                            continue
                        buildings_reset.iloc[idx].plot(
                            ax=ax, facecolor=tuple(colors_arr[idx[0]]),
                            edgecolor="#4A6080", linewidth=0.2, zorder=6)
                else:
                    buildings_p.plot(ax=ax, facecolor="#CFDBE8",
                                     edgecolor="#7F9AB0", linewidth=0.25, zorder=6)

                ax.scatter([cx], [cy], color="white", s=250, marker="*",
                           edgecolor="#E74C3C", linewidth=1.8, zorder=15)

                risk_legend = [
                    Patch(facecolor="#1976D2", alpha=0.80, label="Water Body"),
                    Patch(facecolor="#1565C0", alpha=0.45, label="Flood-prone Zone"),
                    Patch(facecolor="#DDA0A0", alpha=0.90, label="High Risk (low ground)"),
                    Patch(facecolor="#CFDBE8", alpha=0.90, label="Low Risk (high ground)"),
                    Line2D([0], [0], color="#0D47A1", lw=2.0, label="Waterway / Drain"),
                    Line2D([0], [0], marker="*", color="w", markerfacecolor="white",
                           markersize=10, label="Site", linestyle="None",
                           markeredgecolor="#E74C3C"),
                ]
                ax.legend(handles=risk_legend, loc="upper right", frameon=True,
                          facecolor="white", fontsize=8, edgecolor="#AAAAAA")
                set_site_extent(ax, cx, cy, radius)
                ax.set_axis_off()

                finish_figure(fig, "11_flood_risk", "Flood Risk")

                if elev_min is not None:
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Min Elevation",   f"{elev_min:.1f} m")
                    c2.metric("Max Elevation",   f"{elev_max:.1f} m")
                    c3.metric("Elevation Range", f"{elev_max - elev_min:.1f} m")
                elif elev_grid is None:
                    st.warning("Elevation service (OpenTopoData) returned no data — "
                               "elevation layers were skipped. Try again in a moment.")
            except Exception as e:
                st.error(f"Flood Risk error: {e}")

    st.markdown("---")

    # ==========================================================================
    # 12. LAND TOPOGRAPHY
    # ==========================================================================

    st.header("Land Topography", anchor="land-topography")
    st.caption("High-resolution elevation surface with hillshade and slope analysis (dual-panel).")

    if load_button("topography", "Generate Topography Map"):
        with st.spinner("Fetching elevation surface — this may take ~20 s…"):
            try:
                buildings_p, edges_p, crs, cx, cy = load_base(lat, lon, radius)
                topo = fetch_topography_detailed(lat, lon, radius)

                xs, ys = lonlat_to_xy(topo["lons"], topo["lats"], crs)
                zs = np.array(
                    [e if e is not None else float("nan")
                     for e in topo["elevations"]], dtype=float)
                valid = ~np.isnan(zs)

                surf = None
                if valid.sum() >= 4:
                    xi = np.linspace(xs[valid].min(), xs[valid].max(), 300)
                    yi = np.linspace(ys[valid].min(), ys[valid].max(), 300)
                    Xi, Yi = np.meshgrid(xi, yi)
                    try:
                        Zi = griddata((xs[valid], ys[valid]), zs[valid], (Xi, Yi), method="cubic")
                    except Exception:
                        Zi = griddata((xs[valid], ys[valid]), zs[valid], (Xi, Yi), method="linear")
                    # Fill the NaN border left by cubic interpolation
                    nan_mask = np.isnan(Zi)
                    if nan_mask.any():
                        Zi_nn = griddata((xs[valid], ys[valid]), zs[valid], (Xi, Yi), method="nearest")
                        Zi[nan_mask] = Zi_nn[nan_mask]
                    if np.nanmax(Zi) - np.nanmin(Zi) > 1e-6:
                        surf = (xi, yi, Xi, Yi, Zi)

                if surf is None:
                    st.warning("Not enough elevation data returned (or terrain is flat) — "
                               "try again in a moment or use a larger radius.")
                else:
                    xi, yi, Xi, Yi, Zi = surf
                    dZdx = np.gradient(Zi, xi, axis=1)
                    dZdy = np.gradient(Zi, yi, axis=0)
                    slope_rad = np.arctan(np.sqrt(dZdx ** 2 + dZdy ** 2))

                    fig, axes = plt.subplots(1, 2, figsize=(18, 9), facecolor="#F2F3F4")

                    for ax_i, ax in enumerate(axes):
                        ax.set_facecolor("#F2F3F4")

                        if ax_i == 0:
                            topo_cmap = LinearSegmentedColormap.from_list("terrain_c", [
                                "#1A6FA8", "#52BE80", "#F9E79F",
                                "#E59866", "#E74C3C", "#FDFEFE",
                            ])
                            cf = ax.contourf(Xi, Yi, Zi, levels=20, cmap=topo_cmap, zorder=0)
                            light_az  = math.radians(315)
                            light_alt = math.radians(45)
                            aspect    = np.arctan2(-dZdy, dZdx)
                            hillshade = np.clip(
                                np.cos(light_alt) * np.cos(slope_rad)
                                + np.sin(light_alt) * np.sin(slope_rad)
                                * np.cos(light_az - aspect), 0, 1)
                            ax.imshow(hillshade,
                                      extent=[xi.min(), xi.max(), yi.min(), yi.max()],
                                      origin="lower", cmap="gray",
                                      alpha=0.25, zorder=1, interpolation="bilinear")
                            cs = ax.contour(Xi, Yi, Zi, levels=10,
                                            colors="white", linewidths=0.5, alpha=0.5, zorder=2)
                            ax.clabel(cs, inline=True, fontsize=6, fmt="%.0f m", colors="white")
                            cbar = plt.colorbar(cf, ax=ax, fraction=0.025, pad=0.02)
                            cbar.set_label("Elevation (m asl)", fontsize=8)
                            ax.set_title("Elevation Surface + Hillshade", fontsize=10, pad=8)
                        else:
                            slope_cmap = LinearSegmentedColormap.from_list("slope", [
                                "#2ECC71", "#F1C40F", "#E67E22", "#E74C3C", "#7B241C",
                            ])
                            sf = ax.contourf(Xi, Yi, np.degrees(slope_rad), levels=np.linspace(0, 45, 16),
                                             cmap=slope_cmap, extend="max", zorder=0)
                            cbar2 = plt.colorbar(sf, ax=ax, fraction=0.025, pad=0.02)
                            cbar2.set_label("Slope (degrees)", fontsize=8)
                            ax.set_title("Slope Analysis", fontsize=10, pad=8)

                        edges_p["w"] = edge_widths(edges_p, 0.7, 0.8 / 0.7)
                        edges_p.plot(ax=ax, linewidth=edges_p["w"], color="white", alpha=0.45, zorder=5)
                        buildings_p.plot(ax=ax, facecolor="white", edgecolor="#BDC3C7",
                                         linewidth=0.2, alpha=0.55, zorder=6)
                        ax.scatter([cx], [cy], color="white", s=180, marker="*",
                                   edgecolor="#E74C3C", linewidth=1.5, zorder=15)
                        set_site_extent(ax, cx, cy, radius)
                        ax.set_axis_off()

                    plt.tight_layout(pad=1.5)
                    finish_figure(fig, "12_topography", "Topography")

                    ev = zs[valid]
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Min Elevation",  f"{ev.min():.1f} m")
                    c2.metric("Max Elevation",  f"{ev.max():.1f} m")
                    c3.metric("Mean Elevation", f"{ev.mean():.1f} m")
                    c4.metric("Relief",         f"{ev.max() - ev.min():.1f} m")
            except Exception as e:
                st.error(f"Topography error: {e}")

    st.markdown("---")

else:
    st.header("Gallery", anchor="gallery")
    st.caption("Every map you've generated so far, together in one grid. "
               "Switch to Sections to load more, or generate everything below.")

    GALLERY_LABELS = {
        "02_figure_ground":   "Figure-Ground",
        "04_mobility":        "Mobility Network",
        "05_landmarks":       "Nearby Landmarks",
        "06_synthesis":       "Environmental Synthesis",
        "06b_sun_path":       "Sun Path Analysis",
        "07_massing_heatmap": "Massing Heatmap",
        "09_land_use":        "Land Use",
        "10_utilities":       "Utility Infrastructure",
        "11_flood_risk":      "Flood Risk",
        "12_topography":      "Land Topography",
    }

    available = [k for k in GALLERY_LABELS if k in st.session_state.gallery_images]

    if not available:
        st.info(
            "No maps generated yet. Switch to **Sections** in the sidebar and load "
            "a few maps, or generate everything at once below."
        )
        st.button("Generate All Maps", type="primary",
                  on_click=_generate_all_and_go_to_sections)
    else:
        cols = st.columns(3)
        for i, key in enumerate(available):
            with cols[i % 3]:
                st.image(st.session_state.gallery_images[key], use_container_width=True)
                st.caption(GALLERY_LABELS[key])

        missing = [GALLERY_LABELS[k] for k in GALLERY_LABELS if k not in available]
        if missing:
            st.markdown("---")
            st.caption("Not generated yet: " + ", ".join(missing))
            st.button("Generate Remaining Maps", type="primary",
                      on_click=_generate_all_and_go_to_sections)
