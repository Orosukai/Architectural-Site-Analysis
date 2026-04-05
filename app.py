import streamlit as st
import folium
from streamlit_folium import st_folium
import osmnx as ox
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
import io
import math
import warnings
import requests
import numpy as np
from astral import LocationInfo
from astral.sun import elevation, azimuth
import datetime
import zipfile

warnings.filterwarnings("ignore")

# ==============================================================================
# CACHED DATA FETCHERS
# ==============================================================================

@st.cache_data(show_spinner=False)
def fetch_buildings(lat, lon, radius):
    tags = {"building": True}
    gdf = ox.features_from_point((lat, lon), tags, dist=radius)
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
    wind_dirs = res["daily"]["wind_direction_10m_dominant"]
    wind_speeds = res["daily"]["wind_speed_10m_max"]
    return wind_dirs, wind_speeds

@st.cache_data(show_spinner=False)
def fetch_flood_layers(lat, lon, radius):
    import numpy as np

    # 1. OSM water features
    waterway_tags  = {"waterway": ["river", "stream", "canal", "drain", "ditch"]}
    water_area_tags = {
        "natural":  ["water", "wetland", "mud"],
        "landuse":  ["reservoir", "basin", "floodplain"],
        "water":    True,
        "flood_prone": True,
    }
    try:
        waterways = ox.features_from_point((lat, lon), waterway_tags, dist=radius)
    except Exception:
        waterways = None
    try:
        water_bodies = ox.features_from_point((lat, lon), water_area_tags, dist=radius)
        water_bodies = water_bodies[water_bodies.geom_type.isin(["Polygon", "MultiPolygon"])]
    except Exception:
        water_bodies = None

    # 2. Elevation grid via Open-Topo Data (SRTM 30m) — 10x10 sample grid
    step   = (radius * 2) / 10
    import math as _math
    dlat = (radius / 111320)
    dlon = (radius / (111320 * _math.cos(_math.radians(lat))))
    grid_pts = []
    for i in range(11):
        for j in range(11):
            glat = lat - dlat + i * (2 * dlat / 10)
            glon = lon - dlon + j * (2 * dlon / 10)
            grid_pts.append((glat, glon))

    # Batch query Open-Topo Data
    locations_str = "|".join(f"{p[0]},{p[1]}" for p in grid_pts)
    topo_url = f"https://api.opentopodata.org/v1/srtm30m?locations={locations_str}"
    elev_grid = None
    try:
        topo_res = requests.get(topo_url, timeout=20).json()
        if topo_res.get("status") == "OK":
            results = topo_res["results"]
            elev_vals = [r["elevation"] for r in results]
            elev_grid = {
                "lats": [p[0] for p in grid_pts],
                "lons": [p[1] for p in grid_pts],
                "elevations": elev_vals,
            }
    except Exception:
        elev_grid = None

    return waterways, water_bodies, elev_grid

@st.cache_data(show_spinner=False)
def fetch_access_points(lat, lon, radius):
    edges = fetch_graph(lat, lon, radius)
    tags_transit = {
        "highway": ["bus_stop", "crossing", "traffic_signals"],
        "public_transport": ["stop_position", "platform"],
        "amenity": ["bus_station", "ferry_terminal", "taxi"],
        "railway": ["station", "halt", "tram_stop", "subway_entrance"],
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
            "bank", "atm",
            "place_of_worship",
            "library", "community_centre",
        ],
        "railway": ["station", "halt", "tram_stop", "subway_entrance"],
        "shop": ["convenience", "supermarket", "general", "mall"],
        "leisure": ["park"],
    }
    return ox.features_from_point((lat, lon), tags, dist=radius)

# ==============================================================================
# HELPERS
# ==============================================================================

def road_width(highway_val, scale=1.0):
    t = str(highway_val[0] if isinstance(highway_val, list) else highway_val).lower()
    if any(x in t for x in ["motorway", "trunk", "primary"]):
        return 5.5 * scale
    if "secondary" in t:
        return 3.5 * scale
    if "tertiary" in t:
        return 2.5 * scale
    if "residential" in t or "unclassified" in t:
        return 1.5 * scale
    return 0.6 * scale

def project_center(lat, lon):
    import pyproj
    proj = pyproj.Proj(proj="utm", zone=int((lon + 180) / 6) + 1, ellps="WGS84")
    return proj(lon, lat)

def sun_path_points(lat, lon, date, radius_m):
    loc = LocationInfo("Site", "Region", "UTC", lat, lon)
    cx, cy = project_center(lat, lon)
    max_r = radius_m * 0.85
    pts = []
    for h in range(24):
        for m in [0, 10, 20, 30, 40, 50]:
            t = datetime.datetime.combine(date, datetime.time(h, m, tzinfo=datetime.timezone.utc))
            el = elevation(loc.observer, t)
            if el > 0:
                az = azimuth(loc.observer, t)
                r = max_r * (1.0 - el / 90.0)
                angle_rad = math.radians(90.0 - az)
                pts.append((t, cx + r * math.cos(angle_rad), cy + r * math.sin(angle_rad)))
    pts.sort(key=lambda x: x[0])
    return pts, cx, cy

def save_fig_to_svg(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="svg", transparent=True, bbox_inches="tight", pad_inches=0)
    buf.seek(0)
    return buf

def classify_landmark(row):
    amenity  = str(getattr(row, "amenity",  "nan")).lower()
    railway  = str(getattr(row, "railway",  "nan")).lower()
    shop     = str(getattr(row, "shop",     "nan")).lower()
    leisure  = str(getattr(row, "leisure",  "nan")).lower()

    if amenity in {"school", "university", "college"}: return "Education",        "#3498DB", "^"
    if amenity in {"hospital", "clinic", "doctors", "pharmacy"}: return "Health",           "#E74C3C", "+"
    if railway in {"station", "halt", "tram_stop", "subway_entrance"} or amenity in {"bus_station", "ferry_terminal"}: return "Transit Hub",      "#9B59B6", "D"
    if amenity in {"restaurant", "cafe", "fast_food", "food_court"}: return "Food & Dining",    "#E67E22", "o"
    if amenity in {"convenience", "supermarket", "marketplace"} or shop in {"convenience", "supermarket", "general", "mall"}: return "Retail / Shop",    "#27AE60", "s"
    if amenity in {"bank", "atm"}: return "Finance",          "#F39C12", "p"
    if amenity == "place_of_worship": return "Place of Worship", "#1ABC9C", "h"
    if amenity in {"library", "community_centre"}: return "Community",        "#95A5A6", "8"
    if leisure == "park": return "Park / Green",     "#2ECC71", "*"
    return "Other",                "#BDC3C7", "."

# ==============================================================================
# UI - CONTINUOUS SCROLL WITH SIDEBAR NAVIGATION
# ==============================================================================

# --- INITIALIZE SESSION STATE MEMORY ---
if "report_generated" not in st.session_state:
    st.session_state.report_generated = False

st.sidebar.title("Site Configurator")
st.sidebar.markdown("Paste your Google Maps coordinates below.")

coord_input = st.sidebar.text_input("Paste Coordinates (Lat, Lon)", value="14.0700, 121.3255")

try:
    sep = "," if "," in coord_input else None
    lat_str, lon_str = coord_input.split(sep, 1)
    lat, lon = float(lat_str.strip()), float(lon_str.strip())
except Exception:
    st.sidebar.error("⚠️ Invalid format. Using default.")
    lat, lon = 14.0700, 121.3255

radius = st.sidebar.slider("Analysis Radius (meters)", 100, 2000, 500)

# Master trigger button
if st.sidebar.button("🏗️ Generate Full Site Report", type="primary"):
    st.session_state.report_generated = True

# --- DYNAMIC PROGRESS BAR CONTAINER ---
progress_container = st.sidebar.empty()

st.sidebar.markdown("---")

# Sidebar Navigation
st.sidebar.markdown("### 🗺️ Navigation")
st.sidebar.markdown("""
* [Base Map](#base-map)
* [Figure-Ground](#figure-ground)
* [Porosity](#porosity)
* [Mobility Network](#mobility-network)
* [Nearby Landmarks](#nearby-landmarks)
* [Environmental Synthesis](#environmental-synthesis)
* [Massing Heatmap](#massing-heatmap)
""")

# Container for the Master ZIP Download Button
download_container = st.sidebar.empty()

st.title("Architectural Site Analysis")
st.caption(f"Analyzing {radius}m radius around {lat}, {lon}")

# --- USE THE MEMORY INSTEAD OF THE BUTTON DIRECTLY ---
if st.session_state.report_generated:
    
    # Dictionary to hold all files for the ZIP export
    files_to_export = {}
    
    # ------------------------------------------------------------------------------
    # 1. BASE MAP
    # ------------------------------------------------------------------------------
    progress_container.progress(1/7, text="1/7: Initializing Base Map...")
    st.header("Base Map", anchor="base-map")
    m = folium.Map(location=[lat, lon], zoom_start=15, tiles="CartoDB positron")
    folium.Circle(radius=radius, location=[lat, lon], color="#3388ff", fill=True, fill_opacity=0.1).add_to(m)
    folium.Marker([lat, lon]).add_to(m)
    st_folium(m, width=700, height=450)
    
    # Save base map as HTML
    files_to_export["01_base_map.html"] = m.get_root().render().encode('utf-8')
    st.markdown("---")

    # ------------------------------------------------------------------------------
    # 2. FIGURE-GROUND (Now with Roads!)
    # ------------------------------------------------------------------------------
    progress_container.progress(2/7, text="2/7: Generating Figure-Ground with Roads...")
    st.header("Figure-Ground", anchor="figure-ground")
    with st.spinner("Generating Figure-Ground SVG..."):
        try:
            buildings = fetch_buildings(lat, lon, radius)
            edges = fetch_graph(lat, lon, radius)
            
            buildings_proj = ox.projection.project_gdf(buildings)
            target_crs = buildings_proj.crs
            edges_proj = ox.projection.project_gdf(edges).to_crs(target_crs)
            
            fig, ax = plt.subplots(figsize=(10, 10), facecolor="white")
            ax.set_facecolor("white")
            
            # Roads underneath
            edges_proj["w"] = edges_proj["highway"].apply(lambda x: road_width(x, scale=0.8)) if "highway" in edges_proj.columns else 1.0
            edges_proj.plot(ax=ax, linewidth=edges_proj["w"], color="black", zorder=1)
            
            # Buildings on top
            buildings_proj.plot(ax=ax, facecolor="black", edgecolor="none", zorder=2)
            ax.set_axis_off()
            
            svg_buf = save_fig_to_svg(fig)
            files_to_export["02_figure_ground.svg"] = svg_buf.getvalue() # Add to ZIP collection
            
            plt.close(fig)
            st.pyplot(fig)
            st.download_button("Download Figure-Ground (SVG)", svg_buf, "site_figure_ground.svg", "image/svg+xml")
        except Exception as e:
            st.error(f"Error: {e}")
    st.markdown("---")

    # ------------------------------------------------------------------------------
    # 3. POROSITY
    # ------------------------------------------------------------------------------
    progress_container.progress(3/7, text="3/7: Calculating Site Density...")
    st.header("Porosity", anchor="porosity")
    with st.spinner("Calculating site density..."):
        try:
            buildings = fetch_buildings(lat, lon, radius)
            buildings_proj = ox.projection.project_gdf(buildings)
            total_area = math.pi * radius ** 2
            built_area = buildings_proj.geometry.area.sum()
            open_area = total_area - built_area
            built_pct = int((built_area / total_area) * 100)
            open_pct = 100 - built_pct

            c1, c2, c3 = st.columns(3)
            c1.metric("Total Site Area", f"{int(total_area):,} m²")
            c2.metric("Total Built Area", f"{int(built_area):,} m²")
            c3.metric("Total Open Space", f"{int(open_area):,} m²")
            st.caption(f"**Site Density:** {built_pct}% Built / {open_pct}% Porous (Open)")
            st.progress(built_pct)
        except Exception as e:
            st.error(f"Error: {e}")
    st.markdown("---")

    # ------------------------------------------------------------------------------
    # 4. MOBILITY
    # ------------------------------------------------------------------------------
    progress_container.progress(4/7, text="4/7: Mapping Mobility Network...")
    st.header("Mobility Network", anchor="mobility-network")
    with st.spinner("Mapping streets and transit..."):
        try:
            import geopandas as gpd
            from shapely.geometry import Point

            buildings = fetch_buildings(lat, lon, radius)
            edges, transit = fetch_access_points(lat, lon, radius)

            buildings_p = ox.projection.project_gdf(buildings)
            target_crs  = buildings_p.crs
            edges_p     = ox.projection.project_gdf(edges).to_crs(target_crs)
            transit_p = transit.to_crs(target_crs) if transit is not None and not transit.empty else None

            center_pt   = gpd.GeoDataFrame(geometry=[Point(lon, lat)], crs="EPSG:4326")
            center_p    = center_pt.to_crs(target_crs)
            scx, scy    = center_p.geometry.iloc[0].x, center_p.geometry.iloc[0].y

            ROAD_COLORS = {
                "motorway": "#E74C3C", "trunk": "#E74C3C", "primary": "#E67E22",
                "secondary": "#F1C40F", "tertiary": "#2ECC71",
                "residential": "#AED6F1", "unclassified": "#AED6F1",
                "footway": "#D5DBDB", "path": "#D5DBDB", "cycleway": "#A9CCE3", "service": "#D5DBDB"
            }
            DEFAULT_ROAD_COLOR = "#ECF0F1"

            def edge_color(highway_val):
                t = str(highway_val[0] if isinstance(highway_val, list) else highway_val).lower()
                for key, col in ROAD_COLORS.items():
                    if key in t: return col
                return DEFAULT_ROAD_COLOR

            fig, ax = plt.subplots(figsize=(10, 10), facecolor="#1C1C2E")
            ax.set_facecolor("#1C1C2E")

            edges_p["lw"]  = edges_p["highway"].apply(road_width) if "highway" in edges_p.columns else 1.0
            edges_p["col"] = edges_p["highway"].apply(edge_color) if "highway" in edges_p.columns else DEFAULT_ROAD_COLOR
            for col_val in edges_p["col"].unique():
                subset = edges_p[edges_p["col"] == col_val]
                subset.plot(ax=ax, linewidth=subset["lw"].values[0], color=col_val, zorder=1)

            buildings_p.plot(ax=ax, facecolor="#2C2C4E", edgecolor="#3D3D6B", linewidth=0.4, zorder=2)

            TRANSIT_STYLES = {
                "bus_stop": ("#F1C40F", "^", 120, "Bus Stop"), "platform": ("#F1C40F", "^", 120, "Bus Stop"),
                "stop_position": ("#F1C40F", "^", 100, "Bus Stop"), "bus_station": ("#E67E22", "D", 160, "Bus Station"),
                "ferry_terminal": ("#3498DB", "D", 160, "Ferry Terminal"), "taxi": ("#F39C12", "s",  90, "Taxi Stand"),
                "station": ("#9B59B6", "D", 180, "Train / Transit Station"), "halt": ("#9B59B6", "D", 140, "Train Halt"),
                "tram_stop": ("#1ABC9C", "^", 120, "Tram Stop"), "subway_entrance": ("#E74C3C", "o", 120, "Subway Entrance"),
                "crossing": ("#ECF0F1", "+",  80, "Pedestrian Crossing"), "traffic_signals": ("#2ECC71", "o",  60, "Traffic Signal"),
            }
            plotted_labels = {}

            if transit_p is not None and not transit_p.empty:
                centroids = transit_p.geometry.centroid
                for geom, row in zip(centroids, transit_p.itertuples()):
                    for field in ["highway", "railway", "amenity", "public_transport"]:
                        val = str(getattr(row, field, "nan")).lower()
                        if val in TRANSIT_STYLES:
                            color, marker, size, label = TRANSIT_STYLES[val]
                            ax.scatter(geom.x, geom.y, color=color, s=size, marker=marker, edgecolor="white", linewidth=0.6, zorder=10, alpha=0.95)
                            plotted_labels[label] = (color, marker)
                            break

            ax.scatter([scx], [scy], color="white", s=220, marker="*", edgecolor="#E74C3C", linewidth=1.5, zorder=15)
            
            road_legend = [Line2D([0],[0], color=c, lw=2, label=l) for c, l in [("#E74C3C", "Primary"), ("#E67E22", "Secondary"), ("#F1C40F", "Tertiary"), ("#AED6F1", "Residential"), ("#A9CCE3", "Path")]]
            transit_legend = [Line2D([0],[0], marker=mk, color="w", markerfacecolor=c, markersize=8, label=lbl, linestyle="None") for lbl, (c, mk) in plotted_labels.items()] + [Line2D([0],[0], marker="*", color="w", markerfacecolor="white", markersize=10, label="Site", linestyle="None")]
            ax.legend(handles=road_legend + transit_legend, loc="upper right", frameon=True, facecolor="#1C1C2E", labelcolor="white", fontsize=7.5)
            ax.set_axis_off()
            
            svg_buf = save_fig_to_svg(fig)
            files_to_export["04_mobility.svg"] = svg_buf.getvalue() # Add to ZIP collection

            st.pyplot(fig)
            plt.close(fig)
            st.download_button("Download Mobility Map (SVG)", svg_buf, "site_mobility.svg", "image/svg+xml")
        except Exception as e:
            st.error(f"Error: {e}")
    st.markdown("---")

    # ------------------------------------------------------------------------------
    # 5. LANDMARKS
    # ------------------------------------------------------------------------------
    progress_container.progress(5/7, text="5/7: Plotting Local Amenities...")
    st.header("Nearby Landmarks", anchor="nearby-landmarks")
    with st.spinner("Mapping amenities..."):
        try:
            import geopandas as gpd
            from shapely.geometry import Point

            buildings = fetch_buildings(lat, lon, radius)
            edges     = fetch_graph(lat, lon, radius)
            landmarks = fetch_landmarks(lat, lon, radius)

            buildings_p = ox.projection.project_gdf(buildings)
            target_crs  = buildings_p.crs
            edges_p     = ox.projection.project_gdf(edges).to_crs(target_crs)
            landmarks_p = landmarks.to_crs(target_crs) if not landmarks.empty else landmarks

            center_pt = gpd.GeoDataFrame(geometry=[Point(lon, lat)], crs="EPSG:4326")
            center_p  = center_pt.to_crs(target_crs)
            scx, scy  = center_p.geometry.iloc[0].x, center_p.geometry.iloc[0].y

            fig, ax = plt.subplots(figsize=(11, 11), facecolor="#F8F9FA")
            ax.set_facecolor("#F8F9FA")

            edges_p["lw"] = edges_p["highway"].apply(lambda x: road_width(x, scale=0.7)) if "highway" in edges_p.columns else 0.8
            edges_p.plot(ax=ax, linewidth=edges_p["lw"], color="#CCD1D1", zorder=1)
            buildings_p.plot(ax=ax, facecolor="#E8EAED", edgecolor="#BFC9CA", linewidth=0.3, zorder=2)

            counts_by_cat  = {}
            legend_entries = {}

            if not landmarks_p.empty:
                centroids = landmarks_p.geometry.centroid
                for geom, row in zip(centroids, landmarks_p.itertuples()):
                    cat, color, marker = classify_landmark(row)
                    counts_by_cat[cat] = counts_by_cat.get(cat, 0) + 1
                    ax.scatter(geom.x, geom.y, color=color, s=130, marker=marker, edgecolor="white", linewidth=0.7, zorder=8, alpha=0.92)
                    legend_entries[cat] = (color, marker)

            ax.scatter([scx], [scy], color="#E74C3C", s=220, marker="*", edgecolor="white", linewidth=1.5, zorder=15)
            
            handles = [Line2D([0],[0], marker=mk, color="w", markerfacecolor=c, markersize=10, label=lbl, linestyle="None") for lbl, (c, mk) in sorted(legend_entries.items())]
            ax.legend(handles=handles, loc="upper right", frameon=True, facecolor="white", fontsize=8)
            ax.set_axis_off()

            svg_buf = save_fig_to_svg(fig)
            files_to_export["05_landmarks.svg"] = svg_buf.getvalue() # Add to ZIP collection

            st.pyplot(fig)
            plt.close(fig)
            st.download_button("Download Landmarks Map (SVG)", svg_buf, "site_landmarks.svg", "image/svg+xml")

            cols = st.columns(4)
            for i, (cat, cnt) in enumerate(sorted(counts_by_cat.items())):
                cols[i % 4].metric(cat, cnt)
        except Exception as e:
            st.error(f"Error: {e}")
    st.markdown("---")

    # ------------------------------------------------------------------------------
    # 6. SYNTHESIS OVERLAY
    # ------------------------------------------------------------------------------
    progress_container.progress(6/7, text="6/7: Layering Environmental Synthesis...")
    st.header("Environmental Synthesis", anchor="environmental-synthesis")
    with st.spinner("Calculating environmental data..."):
        try:
            import geopandas as gpd
            from shapely.geometry import Point
            from matplotlib.colors import LinearSegmentedColormap

            buildings  = fetch_buildings(lat, lon, radius)
            buildings = buildings[buildings.geom_type.isin(["Polygon", "MultiPolygon"])]
            edges      = fetch_graph(lat, lon, radius)
            wind_dirs, wind_speeds  = fetch_wind_data(lat, lon)
            waterways, water_bodies, elev_grid = fetch_flood_layers(lat, lon, radius)

            buildings_proj = ox.projection.project_gdf(buildings)
            target_crs     = buildings_proj.crs
            edges_proj     = ox.projection.project_gdf(edges).to_crs(target_crs)
            cx, cy         = project_center(lat, lon)

            fig, ax = plt.subplots(figsize=(12, 12))
            ax.set_facecolor("#F0F4F8")

            if elev_grid is not None:
                import pyproj
                proj = pyproj.Proj(proj="utm", zone=int((lon + 180) / 6) + 1, ellps="WGS84")
                xs = [proj(lo, la)[0] for la, lo in zip(elev_grid["lats"], elev_grid["lons"])]
                ys = [proj(lo, la)[1] for la, lo in zip(elev_grid["lats"], elev_grid["lons"])]
                zs = np.array(elev_grid["elevations"], dtype=float)

                from scipy.interpolate import griddata
                xi, yi = np.linspace(min(xs), max(xs), 120), np.linspace(min(ys), max(ys), 120)
                Xi, Yi = np.meshgrid(xi, yi)
                Zi = griddata((xs, ys), zs, (Xi, Yi), method="cubic")

                flood_cmap = LinearSegmentedColormap.from_list("flood", [(0.08, 0.35, 0.75, 0.65), (0.35, 0.65, 0.95, 0.35), (0.85, 0.93, 1.0, 0.0)])
                elev_min, elev_max = np.nanmin(Zi), np.nanmax(Zi)
                cf = ax.contourf(Xi, Yi, Zi, levels=12, cmap=flood_cmap, vmin=elev_min, vmax=elev_max, zorder=0)
                cbar = plt.colorbar(cf, ax=ax, fraction=0.025, pad=0.01)
                cbar.set_label("Elevation (m)")

            if water_bodies is not None and not water_bodies.empty:
                try:
                    wb_proj = water_bodies.to_crs(target_crs)
                    wb_proj.plot(ax=ax, facecolor="#4A90D9", edgecolor="#2471A3", linewidth=0.8, alpha=0.55, zorder=1)
                except Exception: pass

            if waterways is not None and not waterways.empty:
                try:
                    ww_proj = waterways.to_crs(target_crs)
                    ww_proj["wlw"] = 1.5
                    ww_proj.plot(ax=ax, linewidth=ww_proj["wlw"], color="#1A6FA8", alpha=0.85, zorder=2)
                except Exception: pass

            edges_proj["w"] = edges_proj["highway"].apply(lambda x: road_width(x, scale=0.8)) if "highway" in edges_proj.columns else 1.2
            edges_proj.plot(ax=ax, linewidth=edges_proj["w"], color="#B2BEC3", alpha=0.7, zorder=3)
            buildings_proj.plot(ax=ax, facecolor="#DFE6E9", edgecolor="#636E72", linewidth=0.4, alpha=0.9, zorder=4)

            SUN_DATES = [(datetime.date(2024, 6, 21), "#FF7675", "Summer Solstice"), (datetime.date(2024, 3, 21), "#FDCB6E", "Equinox"), (datetime.date(2024, 12, 21), "#74B9FF", "Winter Solstice")]
            for date, color, label in SUN_DATES:
                pts, _, _ = sun_path_points(lat, lon, date, radius)
                if len(pts) > 1: ax.plot([p[1] for p in pts], [p[2] for p in pts], color=color, linewidth=2.5, linestyle="--", zorder=8, label=label)

            valid_dirs = [d for d in wind_dirs if d is not None]
            n_sectors, sector_deg = 16, 360.0 / 16
            counts, _ = np.histogram(valid_dirs, bins=np.linspace(0, 360, n_sectors + 1))
            max_c = counts.max() if counts.max() > 0 else 1
            max_rose_r, wind_cmap = radius * 0.48, plt.cm.get_cmap("cool")

            for i in range(n_sectors):
                petal_r = (counts[i] / max_c) * max_rose_r
                if petal_r < max_rose_r * 0.04: continue
                center_az, half = i * sector_deg, sector_deg / 2.0
                wedge_angles = np.linspace(math.radians(90.0 - (center_az + half)), math.radians(90.0 - (center_az - half)), 20)
                wx = [cx] + [cx + petal_r * math.cos(a) for a in wedge_angles] + [cx]
                wy = [cy] + [cy + petal_r * math.sin(a) for a in wedge_angles] + [cy]
                ax.fill(wx, wy, color=wind_cmap(counts[i] / max_c), alpha=0.52, zorder=9)

            for lbl, az_deg in [("N", 0), ("E", 90), ("S", 180), ("W", 270)]:
                a = math.radians(90.0 - az_deg)
                ax.text(cx + (max_rose_r * 1.12) * math.cos(a), cy + (max_rose_r * 1.12) * math.sin(a), lbl, ha="center", va="center", fontsize=9, fontweight="bold", zorder=12)

            ax.set_axis_off()
            ax.legend(loc="upper right", frameon=True, facecolor="white", fontsize=8)
            
            svg_buf = save_fig_to_svg(fig)
            files_to_export["06_synthesis.svg"] = svg_buf.getvalue() # Add to ZIP collection

            st.pyplot(fig)
            plt.close(fig)
            st.download_button("Download Synthesis Map (SVG)", svg_buf, "site_synthesis.svg", "image/svg+xml")
        except Exception as e:
            st.error(f"Synthesis failed: {e}")
    st.markdown("---")

    # ------------------------------------------------------------------------------
    # 7. MASSING HEATMAP
    # ------------------------------------------------------------------------------
    progress_container.progress(7/7, text="7/7: Extracting Massing Heights...")
    st.header("Massing Heatmap", anchor="massing-heatmap")
    with st.spinner("Generating height heatmap..."):
        try:
            import pandas as pd
            from mpl_toolkits.axes_grid1 import make_axes_locatable

            buildings = fetch_buildings(lat, lon, radius)
            buildings_proj = ox.projection.project_gdf(buildings)
            edges = fetch_graph(lat, lon, radius)
            edges_proj = ox.projection.project_gdf(edges)

            def calculate_heights(df):
                heights = []
                for idx, row in df.iterrows():
                    h = 3.0 
                    if 'height' in df.columns and pd.notnull(row['height']):
                        try:
                            h = float(str(row['height']).replace('m', '').replace(',', '.').strip())
                            heights.append(h)
                            continue
                        except ValueError: pass
                    if 'building:levels' in df.columns and pd.notnull(row['building:levels']):
                        try:
                            h = float(str(row['building:levels']).strip()) * 3.0
                            heights.append(h)
                            continue
                        except ValueError: pass
                    heights.append(h)
                return heights

            buildings_proj['calc_height'] = calculate_heights(buildings_proj)

            fig, ax = plt.subplots(figsize=(12, 12), facecolor="#F8F9FA")
            ax.set_facecolor("#F8F9FA")

            edges_proj["w"] = edges_proj["highway"].apply(lambda x: road_width(x, scale=0.6)) if "highway" in edges_proj.columns else 0.8
            edges_proj.plot(ax=ax, linewidth=edges_proj["w"], color="#CCD1D1", zorder=1)

            buildings_proj.plot(ax=ax, column='calc_height', cmap='magma_r', edgecolor='#BDC3C7', linewidth=0.3, legend=False, zorder=2)

            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="3%", pad=0.1)
            sm = plt.cm.ScalarMappable(cmap='magma_r', norm=plt.Normalize(vmin=buildings_proj['calc_height'].min(), vmax=buildings_proj['calc_height'].max()))
            sm._A = []
            cbar = fig.colorbar(sm, cax=cax)
            cbar.set_label('Building Height (Meters)', rotation=270, labelpad=15, fontweight='bold')

            ax.set_axis_off()
            
            svg_buf = save_fig_to_svg(fig)
            files_to_export["07_massing_heatmap.svg"] = svg_buf.getvalue() # Add to ZIP collection

            st.pyplot(fig)
            plt.close(fig)
            st.download_button("Download Height Heatmap (SVG)", svg_buf, "site_height_heatmap.svg", "image/svg+xml")

        except Exception as e:
            st.error(f"Error generating heatmap: {e}")

    # --- FINISH PROGRESS BAR ---
    progress_container.progress(1.0, text="✅ Analysis Complete!")

    # --- CREATE AND DISPLAY THE MASTER ZIP BUTTON ---
    if files_to_export:
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for filename, file_bytes in files_to_export.items():
                zip_file.writestr(filename, file_bytes)
        
        # Display the button in the placeholder we created earlier in the sidebar
        download_container.markdown("---")
        download_container.download_button(
            label="📦 Download All Maps (.zip)",
            data=zip_buffer.getvalue(),
            file_name="architectural_site_analysis.zip",
            mime="application/zip",
            type="primary",
            use_container_width=True
        )

else:
    st.info("👈 Set your coordinates in the sidebar and click **Generate Full Site Report** to begin.")