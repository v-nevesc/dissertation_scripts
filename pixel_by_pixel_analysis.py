import matplotlib.pyplot as plt
import xarray as xr
from matplotlib.colors import LogNorm
import cartopy.crs as ccrs

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

# =========================
# PROCESS CHLOROPHYLL-a DATA
# =========================
chl_file = "PATH_TO_CHL_NETCDF_FILE/chl_RG.nc"
chl = xr.open_dataset(chl_file)

# Rename default variable
chl = chl.rename({"__xarray_dataarray_variable__": "Chlorophyll-a concentration (mg/m³)"})

# Assign season coordinate
chl = chl.assign_coords(season=('Time', seasons.sel(month=chl['Time.month']).data))

# Compute monthly and seasonal statistics
monthly_median = chl.groupby('Time.month').median()
monthly_mean = chl.groupby('Time.month').mean()
monthly_std = chl.groupby('Time.month').std()
monthly_cv = monthly_std / monthly_mean

seasonal_median = chl.groupby('season').median()
seasonal_std = chl.groupby('season').std()

# Reorder seasons
season_order = ['Summer', 'Autumn', 'Winter', 'Spring']
seasonal_median = seasonal_median.reindex(season=season_order)
seasonal_std = seasonal_std.reindex(season=season_order)

# --------------------------
# PLOT Chlorophyll-a Monthly Median
# --------------------------
g = monthly_median["Chlorophyll-a concentration (mg/m³)"].plot(
    robust=True,
    x="longitude",
    y="latitude",
    col="month",
    col_wrap=3,
    cmap=plt.cm.jet,
    transform=ccrs.PlateCarree(),
    norm=LogNorm(vmin=0.1, vmax=3.5),
    subplot_kws={"projection": ccrs.PlateCarree()}
)

g.cbar.ax.set_ylabel("Chlorophyll-a concentration (mg/m³)", fontsize=15)
g.cbar.ax.tick_params(labelsize=15)

for i, ax in enumerate(g.axs.flat):
    ax.set_extent([-52.3, -51.45, -32.6, -31.81], crs=ccrs.PlateCarree())
    ax.coastlines(resolution='10m', color='black', linewidth=0.8)
    gl = ax.gridlines(draw_labels=True, crs=ccrs.PlateCarree(),
                      linewidth=0.5, color='gray', alpha=0.7, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.left_labels = (i % 3 == 0)
    gl.bottom_labels = (i >= 9)
    gl.xlabel_style = {"size": 15, "color": "black"}
    gl.ylabel_style = {"size": 15, "color": "black"}

season_titles = ['January', 'February', 'March', 'April', 'May', 'June',
                 'July', 'August','September', 'October', 'November', 'December']

for ax, month_name in zip(g.axs.flat, season_titles):
    ax.set_title(f"{month_name}", fontsize=16)

plt.suptitle('Chlorophyll-a Monthly Median', y=1, x=0.46)
# plt.savefig("PATH_TO_SAVE_FIGURES/CHL_MonMedian.png")
plt.show()

# --------------------------
# PLOT Chlorophyll-a Monthly Coefficient of Variation
# --------------------------
h = monthly_cv["Chlorophyll-a concentration (mg/m³)"].plot(
    robust=True,
    x="longitude",
    y="latitude",
    col="month",
    col_wrap=3,
    cmap=plt.cm.jet,
    transform=ccrs.PlateCarree(),
    norm=LogNorm(vmin=0.1, vmax=2),
    subplot_kws={"projection": ccrs.PlateCarree()}
)

h.cbar.ax.set_ylabel("Coefficient of variation", fontsize=15)
h.cbar.ax.tick_params(labelsize=15)

for i, ax in enumerate(h.axs.flat):
    ax.set_extent([-52.3, -51.45, -32.6, -31.81], crs=ccrs.PlateCarree())
    ax.coastlines(resolution='10m', color='black', linewidth=0.8)
    gl = ax.gridlines(draw_labels=True, crs=ccrs.PlateCarree(),
                      linewidth=0.5, color='gray', alpha=0.7, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.left_labels = (i % 3 == 0)
    gl.bottom_labels = (i >= 9)
    gl.xlabel_style = {"size": 15, "color": "black"}
    gl.ylabel_style = {"size": 15, "color": "black"}

for ax, month_name in zip(h.axs.flat, season_titles):
    ax.set_title(f"{month_name}", fontsize=16)

plt.suptitle('Chlorophyll-a Monthly Coefficient of Variation', y=1, x=0.46)
# plt.savefig("PATH_TO_SAVE_FIGURES/CHL_MonCV.png")
plt.show()

# =========================
# PROCESS DOM DATA (CDOM)
# =========================
dom_file = "PATH_TO_DOM_NETCDF_FILE/dom_RG.nc"
dom = xr.open_dataset(dom_file)
dom = dom.rename({"__xarray_dataarray_variable__": "CDM absorption coefficient (m-1)"})
dom = dom.assign_coords(season=('Time', seasons.sel(month=dom['Time.month']).data))

monthly_median = dom.groupby('Time.month').median()
monthly_std = dom.groupby('Time.month').std()
monthly_mean = dom.groupby('Time.month').mean()
monthly_cv = monthly_std / monthly_mean

seasonal_median = dom.groupby('season').median()
seasonal_std = dom.groupby('season').std()
seasonal_median = seasonal_median.reindex(season=season_order)
seasonal_std = seasonal_std.reindex(season=season_order)

# --------------------------
# PLOT DOM Monthly Median
# --------------------------
g = monthly_median["CDM absorption coefficient (m-1)"].plot(
    robust=True,
    x="longitude",
    y="latitude",
    col="month",
    col_wrap=3,
    cmap=plt.cm.PiYG,
    transform=ccrs.PlateCarree(),
    norm=LogNorm(vmin=0.1, vmax=2),
    subplot_kws={"projection": ccrs.PlateCarree()}
)

g.cbar.ax.set_ylabel("CDOM absorption coefficient (m-1)", fontsize=15)
g.cbar.ax.tick_params(labelsize=15)

for i, ax in enumerate(g.axs.flat):
    ax.set_extent([-52.3, -51.45, -32.6, -31.81], crs=ccrs.PlateCarree())
    ax.coastlines(resolution='10m', color='black', linewidth=0.8)
    gl = ax.gridlines(draw_labels=True, crs=ccrs.PlateCarree(),
                      linewidth=0.5, color='gray', alpha=0.7, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.left_labels = (i % 3 == 0)
    gl.bottom_labels = (i >= 9)
    gl.xlabel_style = {"size": 15, "color": "black"}
    gl.ylabel_style = {"size": 15, "color": "black"}

for ax, month_name in zip(g.axs.flat, season_titles):
    ax.set_title(f"{month_name}", fontsize=16)

plt.suptitle('CDOM Monthly Median', y=1, x=0.46)
# plt.savefig("PATH_TO_SAVE_FIGURES/CDOM_MonMedian.png")
plt.show()

# --------------------------
# PLOT DOM Monthly Coefficient of Variation
# --------------------------
h = monthly_cv["CDM absorption coefficient (m-1)"].plot(
    robust=True,
    x="longitude",
    y="latitude",
    col="month",
    col_wrap=3,
    cmap=plt.cm.PiYG,
    transform=ccrs.PlateCarree(),
    norm=LogNorm(vmin=0.1, vmax=6),
    subplot_kws={"projection": ccrs.PlateCarree()},
    cbar_kwargs={"label": "Coefficient of variation"}
)

h.cbar.ax.set_ylabel("Coefficient of variation", fontsize=15)
h.cbar.ax.tick_params(labelsize=15)

for i, ax in enumerate(h.axs.flat):
    ax.set_extent([-52.3, -51.45, -32.6, -31.81], crs=ccrs.PlateCarree())
    ax.coastlines(resolution='10m', color='black', linewidth=0.8)
    gl = ax.gridlines(draw_labels=True, crs=ccrs.PlateCarree(),
                      linewidth=0.5, color='gray', alpha=0.7, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.left_labels = (i % 3 == 0)
    gl.bottom_labels = (i >= 9)
    gl.xlabel_style = {"size": 15, "color": "black"}
    gl.ylabel_style = {"size": 15, "color": "black"}

for ax, month_name in zip(h.axs.flat, season_titles):
    ax.set_title(f"{month_name}", fontsize=16)

plt.suptitle('CDOM Monthly Coefficient of Variation', y=1, x=0.46)
# plt.savefig("PATH_TO_SAVE_FIGURES/CDOM_MonCV.png")
plt.show()

# =========================
# PROCESS TSM DATA
# =========================
tsm_file = "PATH_TO_TSM_NETCDF_FILE/tsm_RG.nc"
tsm = xr.open_dataset(tsm_file)
tsm = tsm.rename({"__xarray_dataarray_variable__": "Total suspended matter concentration (mg/m³)"})
tsm = tsm.assign_coords(season=('Time', seasons.sel(month=tsm['Time.month']).data))

monthly_median = tsm.groupby('Time.month').median()
monthly_mean = tsm.groupby('Time.month').mean()
monthly_std = tsm.groupby('Time.month').std()
monthly_cv = monthly_std / monthly_mean

seasonal_median = tsm.groupby('season').median()
seasonal_std = tsm.groupby('season').std()
seasonal_median = seasonal_median.reindex(season=season_order)
seasonal_std = seasonal_std.reindex(season=season_order)

# --------------------------
# PLOT TSM Monthly Median
# --------------------------
g = monthly_median["Total suspended matter concentration (mg/m³)"].plot(
    robust=True,
    x="longitude",
    y="latitude",
    col="month",
    col_wrap=3,
    cmap=plt.cm.PuOr,
    transform=ccrs.PlateCarree(),
    norm=LogNorm(vmin=0.1, vmax=40),
    subplot_kws={"projection": ccrs.PlateCarree()}
)

g.cbar.ax.set_ylabel("TSM concentration (mg/m³)", fontsize=15)
g.cbar.ax.tick_params(labelsize=15)

for i, ax in enumerate(g.axs.flat):
    ax.set_extent([-52.3, -51.45, -32.6, -31.81], crs=ccrs.PlateCarree())
    ax.coastlines(resolution='10m', color='black', linewidth=0.8)
    gl = ax.gridlines(draw_labels=True, crs=ccrs.PlateCarree(),
                      linewidth=0.5, color='gray', alpha=0.7, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.left_labels = (i % 3 == 0)
    gl.bottom_labels = (i >= 9)
    gl.xlabel_style = {"size": 15, "color": "black"}
    gl.ylabel_style = {"size": 15, "color": "black"}

for ax, month_name in zip(g.axs.flat, season_titles):
    ax.set_title(f"{month_name}", fontsize=16)

plt.suptitle('TSM Monthly Median', y=1, x=0.46)
# plt.savefig("PATH_TO_SAVE_FIGURES/TSM_MonMedian.png")
plt.show()

# --------------------------
# PLOT TSM Monthly Coefficient of Variation
# --------------------------
h = monthly_cv["Total suspended matter concentration (mg/m³)"].plot(
    robust=True,
    x="longitude",
    y="latitude",
    col="month",
    col_wrap=3,
    cmap=plt.cm.PuOr,
    transform=ccrs.PlateCarree(),
    norm=LogNorm(vmin=0.1, vmax=6),
    subplot_kws={"projection": ccrs.PlateCarree()}
)

h.cbar.ax.set_ylabel("Coefficient of variation", fontsize=15)
h.cbar.ax.tick_params(labelsize=15)

for i, ax in enumerate(h.axs.flat):
    ax.set_extent([-52.3, -51.45, -32.6, -31.81], crs=ccrs.PlateCarree())
    ax.coastlines(resolution='10m', color='black', linewidth=0.8)
    gl = ax.gridlines(draw_labels=True, crs=ccrs.PlateCarree(),
                      linewidth=0.5, color='gray', alpha=0.7, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.left_labels = (i % 3 == 0)
    gl.bottom_labels = (i >= 9)
    gl.xlabel_style = {"size": 15, "color": "black"}
    gl.ylabel_style = {"size": 15, "color": "black"}

for ax, month_name in zip(h.axs.flat, season_titles):
    ax.set_title(f"{month_name}", fontsize=16)

plt.suptitle('TSM Monthly Coefficient of Variation', y=1, x=0.46)
# plt.savefig("PATH_TO_SAVE_FIGURES/TSM_MonCV.png")
plt.show()


