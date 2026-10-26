import ee
import os
from dotenv import load_dotenv

load_dotenv()

service_account = os.getenv("GEE_SERVICE_ACCOUNT")
project_id = os.getenv("GEE_PROJECT_ID")
key_path = os.getenv("GEE_KEY_PATH")

credentials = ee.ServiceAccountCredentials(
    service_account,
    key_path
)

ee.Initialize(credentials, project=project_id)

print("✅ Earth Engine Connected Successfully!")

collection = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
print("Total Images Available:", collection.size().getInfo())
