import matplotlib.pyplot as plt
import xarray as xr
from matplotlib import colors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.ticker import MultipleLocator

# --------------------------
# Load seasonal trend datasets (generic paths)
# --------------------------
chl_trends = xr.open_dataset("PATH_TO_RESULTS/chl_trends.nc")
dom_trends = xr.open_dataset("PATH_TO_RESULTS/dom_trends.nc")
tsm_trends = xr.open_dataset("PATH_TO_RESULTS/tsm_trends.nc")

# --------------------------
# Define normalization and ordered seasons
# --------------------------
ordered_seasons = ['Summer', 'Autumn', 'Winter', 'Spring']

# Select slopes in the proper seasonal order
slope_ordered_chl = chl_trends.slope.sel(season=ordered_seasons)
slope_ordered_dom = dom_trends.slope.sel(season=ordered_seasons)
slope_ordered_tsm = tsm_trends.slope.sel(season=ordered_seasons)

##########--CHL--##########
g = slope_ordered_chl.plot(
    x='longitude',
    y='latitude',
    col='season',
    col_wrap=2,
    cmap='jet',
    norm=colors.SymLogNorm(linthresh=1e-10, vmin=-1e-5, vmax=1e-5, base=10, linscale=2),
    cbar_kwargs={'label': 'Trend'},
    subplot_kws={'projection': ccrs.PlateCarree()}
)

# Map season names to axes manually
season_to_ax = dict(zip(ordered_seasons, g.axes.flat))

# Customize each subplot
for season, ax in season_to_ax.items():
    ax.coastlines(resolution='10m')
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.set_extent([-52.27, -51.40, -32.6, -31.83], crs=ccrs.PlateCarree())

    # Add gridlines with 0.2° intervals
    gl = ax.gridlines(draw_labels=True, crs=ccrs.PlateCarree(),
                      linestyle='--', linewidth=0.5, color='gray')
    gl.xlocator = MultipleLocator(0.2)
    gl.ylocator = MultipleLocator(0.2)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 10}
    gl.ylabel_style = {'size': 10}

    # Remove left (latitude) labels for Autumn and Spring
    if season in ['Autumn', 'Spring']:
        gl.left_labels = False

    # Set clean season-only titles
    ax.set_title(season, fontsize=12)

plt.suptitle("CHL", fontsize=16, y=1.05)
plt.show()


##########--DOM--##########
g = slope_ordered_dom.plot(
    x='longitude',
    y='latitude',
    col='season',
    col_wrap=2,
    cmap='PiYG',
    norm=colors.SymLogNorm(linthresh=1e-10, vmin=-1e-8, vmax=1e-8, base=10, linscale=2),
    cbar_kwargs={'label': 'Trend'},
    subplot_kws={'projection': ccrs.PlateCarree()}
)

# Map season names to axes manually
season_to_ax = dict(zip(ordered_seasons, g.axes.flat))

# Customize each subplot
for season, ax in season_to_ax.items():
    ax.coastlines(resolution='10m')
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.set_extent([-52.27, -51.40, -32.6, -31.83], crs=ccrs.PlateCarree())

    # Add gridlines with 0.2° intervals
    gl = ax.gridlines(draw_labels=True, crs=ccrs.PlateCarree(),
                      linestyle='--', linewidth=0.5, color='gray')
    gl.xlocator = MultipleLocator(0.2)
    gl.ylocator = MultipleLocator(0.2)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 10}
    gl.ylabel_style = {'size': 10}

    # Remove left (latitude) labels for Autumn and Spring
    if season in ['Autumn', 'Spring']:
        gl.left_labels = False

    # Set clean season-only titles
    ax.set_title(season, fontsize=12)

plt.suptitle("DOM", fontsize=16, y=1.05)
plt.show()


##########--TSM--##########
g = slope_ordered_tsm.plot(
    x='longitude',
    y='latitude',
    col='season',
    col_wrap=2,
    cmap='PuOr',
    norm=colors.SymLogNorm(linthresh=1e-8, vmin=-1e-7, vmax=1e-7, base=10, linscale=2),
    cbar_kwargs={'label': 'Trend'},
    subplot_kws={'projection': ccrs.PlateCarree()}
)

# Map season names to axes manually
season_to_ax = dict(zip(ordered_seasons, g.axes.flat))

# Customize each subplot
for season, ax in season_to_ax.items():
    ax.coastlines(resolution='10m')
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.set_extent([-52.27, -51.40, -32.6, -31.83], crs=ccrs.PlateCarree())

    # Add gridlines with 0.2° intervals
    gl = ax.gridlines(draw_labels=True, crs=ccrs.PlateCarree(),
                      linestyle='--', linewidth=0.5, color='gray')
    gl.xlocator = MultipleLocator(0.2)
    gl.ylocator = MultipleLocator(0.2)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 10}
    gl.ylabel_style = {'size': 10}

    # Remove left (latitude) labels for Autumn and Spring
    if season in ['Autumn', 'Spring']:
        gl.left_labels = False

    # Set clean season-only titles
    ax.set_title(season, fontsize=12)

plt.suptitle("TSM", fontsize=16, y=1.05)
plt.show()
