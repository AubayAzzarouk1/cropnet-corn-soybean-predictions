#When you want to work on your project, just:
#Open Terminal
#Activate your environment:
#conda activate cropnet_api
#Then run your scripts in that environment (e.g., python cropNet.py)

#Run this PATH: python /Users/aubayazzarouk/Downloads/DS340/CropNet_Test.py
#activate crop net with: conda activate cropnet_api

from cropnet.data_downloader import DataDownloader
from cropnet.data_retriever import DataRetriever
from cropnet.dataset.hrrr_computed_dataset import HRRRComputedDataset
from cropnet.dataset.sentinel2_imagery import Sentinel2Imagery
from cropnet.dataset.usda_crop_dataset import USDACropDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


downloader = DataDownloader(target_dir="./data")

#Example USDA soybean data-> Passing lists of county code (LA,DE) on specific year (2022)
downloader.download_USDA("Soybean", fips_codes=["10003", "22007"], years=["2022"])
#downloader.download_USDA("Corn", fips_codes=["10003", "22007", "17031"], years=["2019", "2020", "2021"])
#downloading two other modalities (WRF/ Sentinel 2img):
#AG and NDVI
downloader.download_Sentinel2(fips_codes = ["10003","22007"], years = ["2022"], image_type = "AG")
downloader.download_Sentinel2(fips_codes=["10003", "22007"], years=["2022"], image_type="NDVI")
#Weather Data (WRF-HRRR data)
#HIgh resolution rapid refresh dataset provides hourly weather forecast
downloader.download_HRRR(
    fips_codes=["10003", "22007"],
    years=["2022"],
)

#LATER! CropNet to download only the weather data for May–July 2022 for run time/space.
#rm -r data/HRRR/hrrr/202201*

#load data with dataloader 
config_file = "corn_train_soybean.json"
base_dir = "./data"

sentinel2_loader = DataLoader(Sentinel2Imagery(base_dir, config_file), batch_size=1)
hrrr_loader = DataLoader(HRRRComputedDataset(base_dir, config_file), batch_size=1)
usda_loader = DataLoader(USDACropDataset(base_dir, config_file), batch_size=1)

# === Step 3: Preview One Batch ===
for s, h, u in zip(sentinel2_loader, hrrr_loader, usda_loader):
    x = s[0]          # satellite image
    ys, yl = h[0], h[1]  # short-term & long-term weather
    z = u[0]          # crop yield

    print("\n📦 Satellite Image Shape:", x.shape)
    print("🌦️ Short-term Weather Shape:", ys.shape)
    print("📊 Long-term Weather Shape:", yl.shape)
    print("🌽 Yield Value:", z.item())
    break  # Preview only one batch











#retreive all 3 modality data from local data strg.
'''
retrieve = DataRetriever(base_dir="./data")

try:
    ag_images = retriever.retrieve_Sentinel2(
        fips_codes=["10003", "22007"],
        years=["2022"],
        image_type="AG"
    )
    print("\n✅ AG Image Loaded. Shape of first image:", ag_images[0].shape)

    # Show a sample AG image
    print("\n🖼️ Displaying sample AG image:")
    plt.imshow(ag_images[0])
    plt.title("Sample Sentinel-2 AG Image")
    plt.axis("off")
    plt.show()

except Exception as e:
    print("\n Could not retrieve AG images:", e)

# Sentinel-2 NDVI Imagery data 
try:
    ndvi_images = retriever.retrieve_Sentinel2(
        fips_codes=["10003", "22007"],
        years=["2022"],
        image_type="NDVI"
    )
    print("\n✅ NDVI Image Loaded. Shape of first image:", ndvi_images[0].shape)

except Exception as e:
    print("\n Could not retrieve NDVI images:", e)

#Retrieve WRF-HRRR Weather Data
try:
    hrrr_data = retriever.retrieve_HRRR(
        fips_codes=["10003", "22007"],
        years=["2022"]
    )
    print("\n✅ HRRR Weather Data Loaded. Shape of first entry:", hrrr_data[0].shape)

except Exception as e:
    print("\n Could not retrieve WRF-HRRR weather data:", e)
'''