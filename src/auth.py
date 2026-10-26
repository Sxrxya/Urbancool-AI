"""
UrbanCool-AI — Auth Module
Handles Google Earth Engine authentication using service account credentials.
"""

import ee
import os
from dotenv import load_dotenv


def initialize_gee():
    """
    Initialize Google Earth Engine with service account credentials.
    Loads credentials from .env file.
    Returns the authenticated ee module.
    """
    load_dotenv()

    service_account = os.getenv("GEE_SERVICE_ACCOUNT")
    project_id = os.getenv("GEE_PROJECT_ID")
    key_path = os.getenv("GEE_KEY_PATH") or os.getenv("GEE_PRIVATE_KEY")

    if not service_account:
        raise ValueError("GEE_SERVICE_ACCOUNT not set in .env")
    if not project_id:
        raise ValueError("GEE_PROJECT_ID not set in .env")
    if not key_path:
        raise ValueError("GEE_KEY_PATH / GEE_PRIVATE_KEY not set in .env")

    credentials = ee.ServiceAccountCredentials(service_account, key_path)
    ee.Initialize(credentials, project=project_id)

    print("✅ Earth Engine initialized successfully.")
    return ee


if __name__ == "__main__":
    initialize_gee()
    print("Auth module test passed.")
