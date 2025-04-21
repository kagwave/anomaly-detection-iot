import os
import zipfile
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from utils.zone_mapper import get_zone_for_station, get_station_coords

def plot_zone_anomalies_from_train():

    def ensure_natural_earth_map():
        folder = "data/maps/ne_110m_admin_0_countries"
        shp_path = os.path.join(folder, "ne_110m_admin_0_countries.shp")

        if os.path.exists(shp_path):
            return shp_path

        print("[Downloading] Natural Earth shapefile...")
        url = "https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip"
        zip_path = "data/maps/countries.zip"
        os.makedirs("data/maps", exist_ok=True)

        with open(zip_path, "wb") as f:
            f.write(requests.get(url).content)

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(folder)

        os.remove(zip_path)
        return shp_path

    # Load and filter anomalies from train.csv
    df = pd.read_csv("data/raw/train.csv")
    df['timestamp'] = pd.to_datetime(df['Ob'])
    df = df[df['target'] == 1]

    # Map station to zone and coordinates
    df['zone'] = df['Station'].map(get_zone_for_station)
    coords_dict = get_station_coords()
    df[['lat', 'lon']] = df['Station'].apply(lambda s: pd.Series(coords_dict.get(s.strip(), (np.nan, np.nan))))

    df = df.dropna(subset=['lat', 'lon'])

    # Convert to GeoDataFrame
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat), crs="EPSG:4326")

    # Load and validate world map
    shp_path = ensure_natural_earth_map()
    world = gpd.read_file(shp_path)
    world = world[world.is_valid & ~world.is_empty]

    # Filter to United States
    usa = world[world["ADMIN"].str.contains("United States", na=False, case=False)]

    if usa.empty:
        raise ValueError("United States shape not found in shapefile")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    usa.plot(ax=ax, color='lightgrey')

    gdf.plot(ax=ax, markersize=25, alpha=0.7, column='zone', legend=True,
             cmap="coolwarm", edgecolor="black")

    ax.set_xlim([-85, -75])
    ax.set_ylim([33, 37])
    plt.title("Anomaly Detections by Zone (EcoNET Stations)", fontsize=14)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.tight_layout()
    plt.savefig("outputs/zone_map.png")
    plt.show()
