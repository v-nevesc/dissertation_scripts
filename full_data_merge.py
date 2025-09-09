import xarray as xr
import numpy as np
import os
import pandas as pd
from scipy.interpolate import griddata

# ==========================================================
# Function to interpolate a dataset to a uniform resolution
# ==========================================================
def interpolate_to_resolution(dataarray, min_latitude, max_latitude, min_longitude, max_longitude, resolution=300):
    # Distance per degree of latitude (constant)
    distance_per_degree_latitude = 111320  # meters

    # Number of latitude points based on resolution
    num_lat_points = int((max_latitude - min_latitude) * distance_per_degree_latitude / resolution) + 1

    # Distance per degree of longitude depends on latitude
    average_latitude = (min_latitude + max_latitude) / 2
    distance_per_degree_longitude = 111320 * np.cos(np.radians(average_latitude))

    # Number of longitude points based on resolution
    num_lon_points = int((max_longitude - min_longitude) * distance_per_degree_longitude / resolution) + 1

    # Create new latitude/longitude grid
    new_latitudes = np.linspace(min_latitude, max_latitude, num=num_lat_points)
    new_longitudes = np.linspace(min_longitude, max_longitude, num=num_lon_points)

    # Flatten existing coordinates and values
    existing_latitudes = dataarray.latitude.values.flatten()
    existing_longitudes = dataarray.longitude.values.flatten()
    existing_values = dataarray.values.flatten()

    # Build point list for interpolation
    points = np.array(list(zip(existing_longitudes, existing_latitudes)))

    # Perform linear interpolation on new grid
    interpolated_values = griddata(points, existing_values,
                                   (new_longitudes[None, :], new_latitudes[:, None]),
                                   method='linear')

    # Check shape consistency
    if interpolated_values.shape != (num_lat_points, num_lon_points):
        raise ValueError(f"Expected shape ({num_lat_points}, {num_lon_points}), got {interpolated_values.shape}")

    # Return new interpolated DataArray
    return xr.DataArray(interpolated_values,
                        coords=[new_latitudes, new_longitudes],
                        dims=["latitude", "longitude"])


# ==========================================================
# Define input directory and region limits
# ==========================================================
caminho_do_diretorio = "path/to/NewDataVariables"
min_latitude, max_latitude, min_longitude, max_longitude = (-32.6, -31.7, -52.3, -51.3)

# Extract sensing dates and variable types from filenames
dias_img = [i[-22:-12] for i in os.listdir(caminho_do_diretorio)]
type_var = [i[0:3] for i in os.listdir(caminho_do_diretorio)]


# ==========================================================
# Interpolate all input files to the chosen resolution
# ==========================================================
for i in range(len(os.listdir(caminho_do_diretorio))):
    sensing_date = dias_img[i]          # Extract sensing date from filename
    var_type_filename = type_var[i]     # Extract variable type (chl, dom, tsm)

    # Open dataset
    var = xr.open_dataset(os.path.join(caminho_do_diretorio, os.listdir(caminho_do_diretorio)[i])).to_array()

    # Apply interpolation
    interpolated = interpolate_to_resolution(var,
                                             min_latitude, max_latitude,
                                             min_longitude, max_longitude,
                                             resolution=300)

    # Save result in interpolated data folder (organized by variable type)
    interpolated_filename = f"path/to/interpolated_data/{var_type_filename}/{var_type_filename}_interpolated_{sensing_date}.nc"
    interpolated.to_netcdf(interpolated_filename)

    print(f'File {i+1} out of {len(os.listdir(caminho_do_diretorio))} interpolated!')


# ==========================================================
# Process interpolated CHL files (concatenate daily means)
# ==========================================================
sensing_dates_chl = np.array([i[17:27] for i in os.listdir('path/to/interpolated_data/chl')])
chl_padronizados = pd.DataFrame(columns=['File', 'Sensing_date'])

for i in range(len(sensing_dates_chl)):
    print(f'CHL count {i+1} out of {len(sensing_dates_chl)}')

    # Select all interpolated files from this day
    files_day = [f for f in os.listdir('path/to/interpolated_data/chl') if sensing_dates_chl[i] in f]

    # Open all files and concatenate along new dimension
    list_var = [xr.open_dataset(f'path/to/interpolated_data/chl/{x}').to_array() for x in files_day]
    concat_imgs = xr.concat(list_var, dim='new')

    # Save daily mean as NetCDF
    concat_imgs.mean(dim='new', skipna=True).to_netcdf(f'path/to/concat_data/chl/Concat_chl_{sensing_dates_chl[i]}.nc')

    # Store reference in DataFrame
    chl_padronizados.loc[i] = [concat_imgs.mean(dim='new', skipna=True), sensing_dates_chl[i]]


# ==========================================================
# Process interpolated DOM files
# ==========================================================
sensing_dates_dom = np.array([i[17:27] for i in os.listdir('path/to/interpolated_data/dom')])
dom_padronizados = pd.DataFrame(columns=['File', 'Sensing_date'])

for i in range(len(sensing_dates_dom)):
    print(f'DOM count {i+1} out of {len(sensing_dates_dom)}')
    files_day = [f for f in os.listdir('path/to/interpolated_data/dom') if sensing_dates_dom[i] in f]
    list_var = [xr.open_dataset(f'path/to/interpolated_data/dom/{x}').to_array() for x in files_day]
    concat_imgs = xr.concat(list_var, dim='new')

    # Save daily mean as NetCDF
    concat_imgs.mean(dim='new', skipna=True).to_netcdf(f'path/to/concat_data/dom/Concat_dom_{sensing_dates_dom[i]}.nc')
    dom_padronizados.loc[i] = [concat_imgs.mean(dim='new', skipna=True), sensing_dates_dom[i]]


# ==========================================================
# Process interpolated TSM files
# ==========================================================
sensing_dates_tsm = np.array([i[17:27] for i in os.listdir('path/to/interpolated_data/tsm')])
tsm_padronizados = pd.DataFrame(columns=['File', 'Sensing_date'])

for i in range(len(sensing_dates_tsm)):
    print(f'TSM count {i+1} out of {len(sensing_dates_tsm)}')
    files_day = [f for f in os.listdir('path/to/interpolated_data/tsm') if sensing_dates_tsm[i] in f]
    list_var = [xr.open_dataset(f'path/to/interpolated_data/tsm/{x}').to_array() for x in files_day]
    concat_imgs = xr.concat(list_var, dim='new')

    # Save daily mean as NetCDF
    concat_imgs.mean(dim='new', skipna=True).to_netcdf(f'path/to/concat_data/tsm/Concat_tsm_{sensing_dates_tsm[i]}.nc')
    tsm_padronizados.loc[i] = [concat_imgs.mean(dim='new', skipna=True), sensing_dates_tsm[i]]


# ==========================================================
# Merge all daily CHL means into a single NetCDF
# ==========================================================
dias_chl = np.array(pd.to_datetime([i[11:21] for i in os.listdir('path/to/concat_data/chl')]))
list_chl = [xr.open_dataset(os.path.join('path/to/concat_data/chl', i)).to_array().squeeze(dim="variable", drop=True)
            for i in os.listdir('path/to/concat_data/chl')]
concat_chl = xr.concat(list_chl, dim='Time')
concat_chl = concat_chl.assign_coords(Time=dias_chl[:concat_chl.sizes['Time']])
concat_chl.to_netcdf('path/to/concat_data/full_data/chl_RG.nc')


# ==========================================================
# Merge all daily DOM means into a single NetCDF
# ==========================================================
dias_dom = np.array(pd.to_datetime([i[11:21] for i in os.listdir('path/to/concat_data/dom')]))
list_dom = [xr.open_dataset(os.path.join('path/to/concat_data/dom', i)).to_array().squeeze(dim="variable", drop=True)
            for i in os.listdir('path/to/concat_data/dom')]
concat_dom = xr.concat(list_dom, dim='Time')
concat_dom = concat_dom.assign_coords(Time=dias_dom[:concat_dom.sizes['Time']])
concat_dom.to_netcdf('path/to/concat_data/full_data/dom_RG.nc')


# ==========================================================
# Merge all daily TSM means into a single NetCDF
# ==========================================================
dias_tsm = np.array(pd.to_datetime([i[11:21] for i in os.listdir('path/to/concat_data/tsm')]))
list_tsm = [xr.open_dataset(os.path.join('path/to/concat_data/tsm', i)).to_array().squeeze(dim="variable", drop=True)
            for i in os.listdir('path/to/concat_data/tsm')]
concat_tsm = xr.concat(list_tsm, dim='Time')
concat_tsm = concat_tsm.assign_coords(Time=dias_tsm[:concat_tsm.sizes['Time']])
concat_tsm.to_netcdf('path/to/concat_data/full_data/tsm_RG.nc')
