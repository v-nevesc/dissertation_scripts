import os
import hda
import xarray as xr
import zipfile
import eumartools
import numpy as np
import shutil


# ==============================
# USER SETTINGS
# ==============================

# WEkEO credentials
USERNAME = "YOUR_USERNAME"
PASSWORD = "YOUR_PASSWORD"

# Local directories
download_dir = "path/to/zips"          # where .zip files will be downloaded
unzip_dir = "path/to/unzipped"         # temporary folder for extraction
output_dir = "path/to/output"          # where final NetCDF files will be saved

# Bounding box (lat_min, lat_max, lon_min, lon_max)
lat_min, lat_max, lon_min, lon_max = (
    -32.627592826722825,
    -31.81826053944898,
    -52.33189055409943,
    -51.4102852508839,
)


# ==============================
# HDA Client
# ==============================

c = hda.Client(hda.Configuration(user=USERNAME, password=PASSWORD))

query = {
    "datasetId": "EO:EUM:DAT:SENTINEL-3:OL_2_WFR___",
    "boundingBoxValues": [
        {
            "name": "bbox",
            "bbox": [
                lon_min,
                lat_min,
                lon_max,
                lat_max,
            ],
        }
    ],
    "dateRangeSelectValues": [
        {
            "name": "position",
            "start": "2023-12-29T00:00:00.000Z",
            "end": "2024-12-31T00:00:00.000Z",
        }
    ],
    "stringChoiceValues": [
        {"name": "type", "value": "OL_2_WFR___"},
        {"name": "timeliness", "value": "NT"},
    ],
}


matches = c.search(query)
print(matches)


# ==============================
# MAIN LOOP
# ==============================

for i in range(len(matches)):
    print(f"loop: {i} out of {len(matches)}")
    aaaa = 1

    # download
    matches[i].download(download_dir)

    # unzip all files
    for z in os.listdir(download_dir):
        zip_path = os.path.join(download_dir, z)
        try:
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(unzip_dir)
            print(f"Successfully extracted {z}")

            # remove zip
            os.remove(zip_path)

            # remove extra files
            for file_to_remove in ["browse.jpg", "EOPMetadata.xml", "manifest.xml"]:
                try:
                    os.remove(os.path.join(unzip_dir, file_to_remove))
                except FileNotFoundError:
                    print(f"Warning: {file_to_remove} not found and could not be deleted.")

        except zipfile.BadZipFile:
            print(f"Error: The file {z} is not a valid zip file.")
            os.remove(zip_path)
            aaaa = "skip"
        except Exception as e:
            print(f"An unexpected error occurred with {z}: {e}")
            os.remove(zip_path)
            aaaa = "skip"

    if aaaa == "skip":
        print("Skipping this iteration")
        continue

    # open required files
    base_dir = os.path.join(unzip_dir, os.listdir(unzip_dir)[0])

    nome_arquivo_tsm = os.path.join(base_dir, "tsm_nn.nc")
    nome_arquivo_dom = os.path.join(base_dir, "iop_nn.nc")
    nome_arquivo_coo = os.path.join(base_dir, "geo_coordinates.nc")
    nome_arquivo_oa6 = os.path.join(base_dir, "Oa06_reflectance.nc")
    nome_arquivo_oa4 = os.path.join(base_dir, "Oa04_reflectance.nc")
    nome_arquivo_flag = os.path.join(base_dir, "wqsf.nc")

    arquivo_tsm = xr.open_dataset(nome_arquivo_tsm)
    arquivo_dom = xr.open_dataset(nome_arquivo_dom)
    coordenadas = xr.open_dataset(nome_arquivo_coo)
    arquivo_oa6 = xr.open_dataset(nome_arquivo_oa6)
    arquivo_oa4 = xr.open_dataset(nome_arquivo_oa4)
    arquivo_flags = xr.open_dataset(nome_arquivo_flag)

    binary_mask = ~eumartools.flag_mask(
        nome_arquivo_flag,
        "WQSF",
        [
            "LAND",
            "CLOUD",
            "CLOUD_AMBIGUOUS",
            "INVALID",
            "COSMETIC",
            "SATURATED",
            "SUSPECT",
            "HISOLZEN",
            "AC_FAIL",
            "HIGHGLINT",
            "WHITECAPS",
            "RWNEG_O2",
            "RWNEG_O3",
            "RWNEG_O4",
            "RWNEG_O5",
            "RWNEG_O6",
            "RWNEG_O7",
            "RWNEG_O8",
            "OCNN_FAIL",
        ],
    )

    sensing_time = arquivo_tsm.attrs["start_time"][0:19]

    tsm = arquivo_tsm["TSM_NN"]
    dom = arquivo_dom["ADG443_NN"]
    latitude = coordenadas["latitude"]
    longitude = coordenadas["longitude"]
    oa4 = arquivo_oa4["Oa04_reflectance"]
    oa6 = arquivo_oa6["Oa06_reflectance"]

    # assign coords
    tsm = tsm.assign_coords(longitude=longitude, latitude=latitude)
    dom = dom.assign_coords(longitude=longitude, latitude=latitude)
    oa4 = oa4.assign_coords(longitude=longitude, latitude=latitude)
    oa6 = oa6.assign_coords(longitude=longitude, latitude=latitude)

    dummy_netcdf_mask = tsm.copy()
    binary_mask = binary_mask[: dummy_netcdf_mask.shape[0], :]
    dummy_netcdf_mask.values = binary_mask
    binary_mask = dummy_netcdf_mask

    # subset
    lat_cond = (tsm["latitude"] > lat_min) & (tsm["latitude"] < lat_max)
    lon_cond = (tsm["longitude"] > lon_min) & (tsm["longitude"] < lon_max)
    coord_lim_cond = lat_cond & lon_cond

    tsm_recortada = tsm.where(coord_lim_cond, drop=True)
    dom_recortada = dom.where(coord_lim_cond, drop=True)
    oa4_recortada = oa4.where(coord_lim_cond, drop=True)
    oa6_recortada = oa6.where(coord_lim_cond, drop=True)
    binary_mask_recortada = binary_mask.where(coord_lim_cond, drop=True)

    # transform
    tsm_recortada = 10**tsm_recortada
    dom_recortada = 10**dom_recortada

    r = oa4_recortada / oa6_recortada
    r.values = np.where(r <= 0, np.nan, r)

    oc2lp = 10 ** (
        0.0949
        - 1.6329 * np.log10(r)
        + 0.1551 * pow(np.log10(r), 2)
        - 0.9824 * pow(np.log10(r), 3)
    ) - 0.071

    # apply mask
    tsm_recortada.values = np.where(binary_mask_recortada, tsm_recortada.values, np.nan)
    dom_recortada.values = np.where(binary_mask_recortada, dom_recortada.values, np.nan)
    oc2lp.values = np.where(binary_mask_recortada, oc2lp.values, np.nan)

    # metadata
    tsm_recortada.attrs["units"] = "(Neural Net) Total suspended matter concentration (g.m-3)"
    tsm_recortada.attrs["sensing date"] = sensing_time
    dom_recortada.attrs["units"] = "(Neural Net) CDM absorption coefficient (m-1)"
    dom_recortada.attrs["sensing date"] = sensing_time
    oc2lp.attrs["units"] = "(OC2-LP) Algal pigment concentration (mg.m-3)"
    oc2lp.attrs["sensing date"] = sensing_time

    # save outputs
    tsm_recortada.to_netcdf(os.path.join(output_dir, f"TSM_{sensing_time.replace(':', '_')}.nc"))
    dom_recortada.to_netcdf(os.path.join(output_dir, f"DOM_{sensing_time.replace(':', '_')}.nc"))
    oc2lp.to_netcdf(os.path.join(output_dir, f"chl_OC2LP_{sensing_time.replace(':', '_')}.nc"))

    # close
    arquivo_tsm.close()
    arquivo_dom.close()
    arquivo_oa6.close()
    arquivo_oa4.close()
    arquivo_flags.close()
    coordenadas.close()

    # cleanup
    for zz in os.listdir(unzip_dir):
        shutil.rmtree(os.path.join(unzip_dir, zz))

    print("LOOP COMPLETED")



