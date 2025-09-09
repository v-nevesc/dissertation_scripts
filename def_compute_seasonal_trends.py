import xarray as xr
import numpy as np
from tqdm import tqdm
from scipy import stats

# --------------------------
# Function to compute seasonal linear trends per pixel
# --------------------------
def compute_seasonal_trends(ds, data_var=None, time_dim='Time', season_coord='season', min_obs=5):
    """
    Compute robust seasonal linear trends for each pixel with improved NaN handling.
    
    Parameters:
    -----------
    ds : xarray.Dataset
        Input dataset containing the data variable and coordinates
    data_var : str, optional
        Name of the data variable to analyze. If None, uses the first data variable.
    time_dim : str, default='Time'
        Name of the time dimension
    season_coord : str, default='season'
        Name of the season coordinate
    min_obs : int, default=5
        Minimum number of valid observations required to compute trend
        
    Returns:
    --------
    xarray.Dataset
        Dataset containing slope, intercept, r_value, p_value, std_err, and n_obs
    """
    
    # Determine the data variable if not specified
    if data_var is None:
        data_vars = [var for var in ds.variables if var not in ds.coords]
        if not data_vars:
            raise ValueError("No data variables found in the dataset")
        data_var = data_vars[0]
        print(f"Using data variable: {data_var}")

    # Get unique seasons in order of appearance (preserves seasonal cycle order)
    unique_seasons = np.unique(ds[season_coord].values, return_index=True)[0]
    
    # Convert time to numeric (days since first observation)
    time_numeric = (ds[time_dim] - ds[time_dim][0]).astype('timedelta64[D]').astype(float)
    
    # Prepare output arrays with reduced precision to save memory
    shape = (len(unique_seasons), ds.dims['latitude'], ds.dims['longitude'])
    results = {
        'slope': np.full(shape, np.nan, dtype=np.float32),
        'intercept': np.full(shape, np.nan, dtype=np.float32),
        'r_value': np.full(shape, np.nan, dtype=np.float32),
        'p_value': np.full(shape, np.nan, dtype=np.float32),
        'std_err': np.full(shape, np.nan, dtype=np.float32),
        'n_obs': np.zeros(shape, dtype=np.uint16)
    }
    
    # --------------------------
    # Loop through each season
    # --------------------------
    for i, season in enumerate(unique_seasons):
        print(f"\nProcessing season: {season}")
        
        # Mask data for current season
        season_mask = (ds[season_coord] == season)
        season_data = ds[data_var].where(season_mask, drop=True)
        season_time = time_numeric.where(season_mask, drop=True)
        
        # Loop through all pixels (latitude and longitude)
        for lat in tqdm(range(ds.dims['latitude']), desc='Latitude', leave=False):
            for lon in range(ds.dims['longitude']):
                y = season_data[:, lat, lon].values
                x = season_time.values
                
                # Identify valid (finite) observations
                valid_mask = np.isfinite(y) & np.isfinite(x)
                x_valid = x[valid_mask]
                y_valid = y[valid_mask]
                n_valid = valid_mask.sum()
                
                results['n_obs'][i, lat, lon] = n_valid
                
                # Only compute regression if enough valid points
                if n_valid >= min_obs:
                    try:
                        # Center x to reduce numerical errors
                        x_centered = x_valid - x_valid.mean()
                        
                        # Perform linear regression
                        slope, intercept, r_value, p_value, std_err = stats.linregress(
                            x_centered, y_valid
                        )
                        
                        # Adjust intercept back to original scale
                        intercept = intercept - slope * x_valid.mean()
                        
                        results['slope'][i, lat, lon] = slope
                        results['intercept'][i, lat, lon] = intercept
                        results['r_value'][i, lat, lon] = r_value
                        results['p_value'][i, lat, lon] = p_value
                        results['std_err'][i, lat, lon] = std_err
                        
                    except Exception:
                        # Skip any numerical issues
                        continue
    
    # --------------------------
    # Create output xarray.Dataset
    # --------------------------
    result_ds = xr.Dataset(
        {name: (('season', 'latitude', 'longitude'), data) for name, data in results.items()},
        coords={
            'season': unique_seasons,
            'latitude': ds.latitude,
            'longitude': ds.longitude
        }
    )
    
    # Add metadata
    result_ds['slope'].attrs = {
        'units': f'{ds[data_var].attrs.get("units", "value")}/day',
        'description': 'Slope of linear trend (change per day)'
    }
    result_ds['intercept'].attrs = {
        'units': ds[data_var].attrs.get('units', 'value'),
        'description': 'Intercept at time=0 (first observation)'
    }
    result_ds.attrs = {
        'source_data': data_var,
        'time_range': f"{ds[time_dim][0].item()} to {ds[time_dim][-1].item()}",
        'processing': 'Seasonal linear trends computed with centered x-values',
        'minimum_observations': str(min_obs),
        'regression_method': 'scipy.stats.linregress with centered x-values'
    }
    
    return result_ds

# --------------------------
# Define seasons by month
# --------------------------
seasons = xr.DataArray(
    ['Summer', 'Summer', 'Autumn', 'Autumn', 'Autumn', 
     'Winter', 'Winter', 'Winter', 'Spring', 'Spring', 
     'Spring', 'Summer'],
    dims='month',
    coords={'month': range(1, 13)}
)

# --------------------------
# Define dataset file paths (generic)
# --------------------------
chl_path = "PATH_TO_CHL_NETCDF_FILE/chl_RG.nc"
dom_path = "PATH_TO_DOM_NETCDF_FILE/dom_RG.nc"
tsm_path = "PATH_TO_TSM_NETCDF_FILE/tsm_RG.nc"

# --------------------------
# Open datasets using xarray
# --------------------------
chl = xr.open_dataset(chl_path)
dom = xr.open_dataset(dom_path)
tsm = xr.open_dataset(tsm_path)

# --------------------------
# Assign seasonal coordinate to each dataset
# --------------------------
chl = chl.assign_coords(season=('Time', seasons.sel(month=chl['Time.month']).data))
dom = dom.assign_coords(season=('Time', seasons.sel(month=dom['Time.month']).data))
tsm = tsm.assign_coords(season=('Time', seasons.sel(month=tsm['Time.month']).data))

# --------------------------
# Rename variables for clarity
# --------------------------
chl = chl.rename({"__xarray_dataarray_variable__": "chl"})
dom = dom.rename({"__xarray_dataarray_variable__": "CDM_absorption_coefficient"})
tsm = tsm.rename({"__xarray_dataarray_variable__": "tsm"})

# --------------------------
# Compute trends and save results
# --------------------------
print("Processing chlorophyll data...")
chl_trends = compute_seasonal_trends(chl, data_var='chl')
chl_trends.to_netcdf("PATH_TO_SAVE_RESULTS/chl_trends.nc")

print("\nProcessing DOM data...")
dom_trends = compute_seasonal_trends(dom, data_var='CDM_absorption_coefficient')
dom_trends.to_netcdf("PATH_TO_SAVE_RESULTS/dom_trends.nc")

print("\nProcessing TSM data...")
tsm_trends = compute_seasonal_trends(tsm, data_var='tsm')
tsm_trends.to_netcdf("PATH_TO_SAVE_RESULTS/tsm_trends.nc")
