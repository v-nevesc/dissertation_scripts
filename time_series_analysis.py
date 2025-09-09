import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import os
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.stats import linregress

# -------------------------------
# Define bounding boxes (regions)
# -------------------------------
# Area 1 = North, Area 2 = Ocean
min_latitudeNorth, max_latitudeNorth, min_longitudeNorth, max_longitudeNorth = (-32.3, -32.15, -52.15, -51)
min_latitudeOcean, max_latitudeOcean, min_longitudeOcean, max_longitudeOcean = (-32.59, -32.45, -51.7, -51.55)

# -------------------------------
# Load datasets (replace paths)
# -------------------------------
chl_total = xr.open_dataset('path/to/full_data/chl_RG.nc').to_array()
dom_total = xr.open_dataset('path/to/full_data/dom_RG.nc').to_array()
tsm_total = xr.open_dataset('path/to/full_data/tsm_RG.nc').to_array()

# Directories where individual daily files are stored
caminho_do_diretorio_chl = "path/to/chl"
caminho_do_diretorio_dom = "path/to/dom"
caminho_do_diretorio_tsm = "path/to/tsm"

# Extract dates from filenames (assumes date is between positions 11 and 21)
days_chl = np.array(pd.to_datetime([i[11:21] for i in os.listdir(caminho_do_diretorio_chl)]))
days_dom = np.array(pd.to_datetime([i[11:21] for i in os.listdir(caminho_do_diretorio_dom)]))
days_tsm = np.array(pd.to_datetime([i[11:21] for i in os.listdir(caminho_do_diretorio_tsm)]))

# =====================================================
# CHLOROPHYLL (CHL) - Extract median values in each area
# =====================================================
chl_north_median = chl_total.sel(
    latitude=slice(min_latitudeNorth, max_latitudeNorth),
    longitude=slice(min_longitudeNorth, max_longitudeNorth)
).median(dim=["latitude", "longitude"], skipna=True).squeeze().values

chl_ocean_median = chl_total.sel(
    latitude=slice(min_latitudeOcean, max_latitudeOcean),
    longitude=slice(min_longitudeOcean, max_longitudeOcean)
).median(dim=["latitude", "longitude"], skipna=True).squeeze().values

# Extract timestamps from dataset
time_stamps = chl_total.sel(
    latitude=slice(min_latitudeNorth, max_latitudeNorth),
    longitude=slice(min_longitudeNorth, max_longitudeNorth)
).median(dim=["latitude", "longitude"], skipna=True).squeeze()['Time'].values

# Create masks to filter out NaNs
valid_mask_north = ~np.isnan(chl_north_median)
valid_mask_ocean = ~np.isnan(chl_ocean_median)

# Keep only valid data
chl_north_median_valid = chl_north_median[valid_mask_north]
chl_time_stamps_valid_north = time_stamps[valid_mask_north]

chl_ocean_median_valid = chl_ocean_median[valid_mask_ocean]
chl_time_stamps_valid_ocean = time_stamps[valid_mask_ocean]

# =====================================================
# CDOM (DOM) - Repeat the same workflow
# =====================================================
dom_north_median = dom_total.sel(
    latitude=slice(min_latitudeNorth, max_latitudeNorth),
    longitude=slice(min_longitudeNorth, max_longitudeNorth)
).median(dim=["latitude", "longitude"], skipna=True).squeeze().values

dom_ocean_median = dom_total.sel(
    latitude=slice(min_latitudeOcean, max_latitudeOcean),
    longitude=slice(min_longitudeOcean, max_longitudeOcean)
).median(dim=["latitude", "longitude"], skipna=True).squeeze().values

time_stamps = dom_total.sel(
    latitude=slice(min_latitudeNorth, max_latitudeNorth),
    longitude=slice(min_longitudeNorth, max_longitudeNorth)
).median(dim=["latitude", "longitude"], skipna=True).squeeze()['Time'].values

valid_mask_north = ~np.isnan(dom_north_median)
valid_mask_ocean = ~np.isnan(dom_ocean_median)

dom_north_median_valid = dom_north_median[valid_mask_north]
dom_time_stamps_valid_north = time_stamps[valid_mask_north]

dom_ocean_median_valid = dom_ocean_median[valid_mask_ocean]
dom_time_stamps_valid_ocean = time_stamps[valid_mask_ocean]

# =====================================================
# TSM - Repeat the same workflow
# =====================================================
tsm_north_median = tsm_total.sel(
    latitude=slice(min_latitudeNorth, max_latitudeNorth),
    longitude=slice(min_longitudeNorth, max_longitudeNorth)
).median(dim=["latitude", "longitude"], skipna=True).squeeze().values

tsm_ocean_median = tsm_total.sel(
    latitude=slice(min_latitudeOcean, max_latitudeOcean),
    longitude=slice(min_longitudeOcean, max_longitudeOcean)
).median(dim=["latitude", "longitude"], skipna=True).squeeze().values

time_stamps = tsm_total.sel(
    latitude=slice(min_latitudeNorth, max_latitudeNorth),
    longitude=slice(min_longitudeNorth, max_longitudeNorth)
).median(dim=["latitude", "longitude"], skipna=True).squeeze()['Time'].values

valid_mask_north = ~np.isnan(tsm_north_median)
valid_mask_ocean = ~np.isnan(tsm_ocean_median)

tsm_north_median_valid = tsm_north_median[valid_mask_north]
tsm_time_stamps_valid_north = time_stamps[valid_mask_north]

tsm_ocean_median_valid = tsm_ocean_median[valid_mask_ocean]
tsm_time_stamps_valid_ocean = time_stamps[valid_mask_ocean]

# =====================================================
# Data cleaning (remove unrealistic high values in chl)
# =====================================================
chl_north_median_valid[chl_north_median_valid > 100] = np.nan
chl_ocean_median_valid[chl_ocean_median_valid > 100] = np.nan

# =====================================================
# LOOP through variables (chl, dom, tsm)
# =====================================================
for i in ['chl', 'dom', 'tsm']:
    
    # Label and y-axis limits per variable
    if i == 'chl':
        plot_ylabel = 'log-transformed chlorophyll-a concentration (mg/m³)'
        y_limits = [-1.05, 1.75]
    elif i == 'dom':
        plot_ylabel = 'log-transformed CDOM absorption coefficient (m-1)'
        y_limits = [-2.73, 1.62]
    else:
        plot_ylabel = 'log-transformed TSM concentration (mg/m³)'
        y_limits = [-2.01, 2.18]
    
    # Define reference years for vertical lines
    years = pd.date_range("2015-01-01", "2025-05-01", freq="A-DEC")
    
    # -----------------------------
    # Build timeseries for Area 1
    # -----------------------------
    var_TimeSeries_north = pd.DataFrame({ 
        'var': globals()[i + '_north_median_valid'],
        'date': globals()[i + '_time_stamps_valid_north']})
    
    var_TimeSeries_north['date'] = pd.to_datetime(var_TimeSeries_north['date'])
    var_TimeSeries_north.set_index('date', inplace=True)
    
    # Apply rolling median smoothing
    var_TimeSeries_north['var_median'] = var_TimeSeries_north['var'].rolling(window=8, center=True).median()
    
    # -----------------------------
    # Build timeseries for Area 2
    # -----------------------------
    var_TimeSeries_ocean = pd.DataFrame({ 
        'var': globals()[i + '_ocean_median_valid'],
        'date': globals()[i + '_time_stamps_valid_ocean']})
    
    var_TimeSeries_ocean['date'] = pd.to_datetime(var_TimeSeries_ocean['date'])
    var_TimeSeries_ocean.set_index('date', inplace=True)
    
    var_TimeSeries_ocean['var_median'] = var_TimeSeries_ocean['var'].rolling(window=8, center=True).median()
    
    # Group by unique dates (if duplicates exist, take mean)
    var_TimeSeries_north = var_TimeSeries_north.groupby(var_TimeSeries_north.index).mean()
    var_TimeSeries_ocean = var_TimeSeries_ocean.groupby(var_TimeSeries_ocean.index).mean()
    
    # Fill missing days with full daily range
    full_date_range = pd.date_range(start=var_TimeSeries_north.index.min(), 
                                    end=var_TimeSeries_north.index.max(), 
                                    freq='D')
    
    var_TimeSeries_north = var_TimeSeries_north.reindex(full_date_range)
    var_TimeSeries_ocean = var_TimeSeries_ocean.reindex(full_date_range)
    
    # Interpolate and backfill missing values
    var_TimeSeries_north['var'] = var_TimeSeries_north['var'].interpolate(method='linear')
    var_TimeSeries_north['var'] = var_TimeSeries_north['var'].fillna(method='bfill')
    var_TimeSeries_north['var_median'] = var_TimeSeries_north['var_median'].interpolate(method='linear')
    var_TimeSeries_north['var_median'] = var_TimeSeries_north['var_median'].fillna(method='bfill')
    
    var_TimeSeries_ocean['var'] = var_TimeSeries_ocean['var'].interpolate(method='linear')
    var_TimeSeries_ocean['var'] = var_TimeSeries_ocean['var'].fillna(method='bfill')
    var_TimeSeries_ocean['var_median'] = var_TimeSeries_ocean['var_median'].interpolate(method='linear')
    var_TimeSeries_ocean['var_median'] = var_TimeSeries_ocean['var_median'].fillna(method='bfill')
    
    # -----------------------------
    # Seasonal decomposition (extract trend component)
    # -----------------------------
    trend_north = seasonal_decompose(np.log10(var_TimeSeries_north['var_median']), 
                                     model='additive', 
                                     period=30).trend
    trend_ocean = seasonal_decompose(np.log10(var_TimeSeries_ocean['var_median']), 
                                     model='additive', 
                                     period=30).trend
    
    # -----------------------------
    # Linear regression on trend
    # -----------------------------
    slope_north, intercept_north, r_value_north, p_value_north, std_err_north = linregress(
        range(len(trend_north[15:-15])), trend_north[15:-15])
    regression_line_north = slope_north * np.arange(len(trend_north[15:-15])) + intercept_north
    
    slope_ocean, intercept_ocean, r_value_ocean, p_value_ocean, std_err_ocean = linregress(
        range(len(trend_ocean[15:-15])), trend_ocean[15:-15])
    regression_line_ocean = slope_ocean * np.arange(len(trend_ocean[15:-15])) + intercept_ocean
    
    # -----------------------------
    # PLOT time series
    # -----------------------------
    fig, ax = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    fig.subplots_adjust(hspace=0.45) 

    # --- Area 1 ---
    ax[0].plot(var_TimeSeries_north.index, np.log10(var_TimeSeries_north['var_median']), color='black', alpha=0.3)
    ax[0].set_title('Area 1', fontsize=19)
    for year in years:
        ax[0].axvline(x=year, color='gray', linestyle='--')
    ax[0].plot(var_TimeSeries_north.index, trend_north, color='black')
    ax[0].plot(var_TimeSeries_north.index[15:-15], regression_line_north, color='blue', linewidth=3, linestyle=':')
    
    # --- Area 2 ---
    ax[1].plot(var_TimeSeries_ocean.index, np.log10(var_TimeSeries_ocean['var_median']), color='black', alpha=0.3)
    ax[1].set_title('Area 2', fontsize=19)
    for year in years:
        ax[1].axvline(x=year, color='gray', linestyle='--')
    ax[1].plot(var_TimeSeries_ocean.index, trend_ocean, color='black')
    ax[1].plot(var_TimeSeries_ocean.index[15:-15], regression_line_ocean, color='blue', linewidth=3, linestyle=':')
    
    # Y-axis label
    fig.text(0.03, 0.5, plot_ylabel, va='center', rotation='vertical', fontsize=22)
    
    # Save option (currently commented)
    # plt.savefig('path/to/output/time_series_plots/' + i + '_FullTimeSeries.png')
    
    plt.show()
