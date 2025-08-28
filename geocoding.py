# geocode_addresses.py
import requests # Note: How you request something from a website Live
from pathlib import Path
from nearest_library import find_closest_library
from nearest_farmers_market import find_closest_market
from nearest_art import find_closest_art
from nearest_trail_poi import find_closest_trail
from nearest_hospital import find_closest_hospital
from nearest_county_office import find_closest_county_office
from nearest_school import find_closest_school
from nearest_public_safety import find_closest_fire_station
from nearest_recreation_center import find_closest_recreation_center
from nearest_senior_housing import find_closest_senior_housing
from nearest_transportation import find_closest_transport
from nearest_business import find_closest_business


HERE = Path(__file__).resolve().parent.parent
ART_CSV = HERE / "datasets/Arts and Recreation.csv"
SCHOOL_CSV = HERE / "datasets/Education.csv"
FARM_CSV = HERE / "datasets/farmers_markets.csv"
HOSP_CSV = HERE / "datasets/hospitals.csv"
LIB_CSV = HERE / "datasets/libraries.csv"
COUNTY_CSV = HERE / "datasets/Municipal Services.csv"
SAFETY_CSV = HERE / "datasets/Public Safety.csv"
TRAIL_CSV = HERE / "datasets/Trails_POI.csv"
REC_CSV = HERE / "datasets/Recreation_Centers.csv"
SENIOR_CSV = HERE / "datasets/Senior_Housing.csv"
TRANSPORT_CSV = HERE / "datasets/Transportation.csv"
BUSINESS_CSV = HERE / "datasets/TTC_Business_License.csv"

# Example: Downtown LA

# Base URL for the OpenStreetMap Nominatim API
BASE_URL = "https://nominatim.openstreetmap.org/search" # Note: URL we are searching from

def geocode_address(address: str):
    """
    Takes a string address and sends it to the OpenStreetMap Nominatim API.
    Returns a dictionary with the full formatted address, latitude, and longitude.
    """

    # Parameters we send to the API
    params = {
        "q": address,           # the address string to search
        "format": "json",       # we want results in JSON format
        "limit": 1,             # only return the top result
        "addressdetails": 1,    # include detailed breakdown of address
    }

    # OSM policy requires a User-Agent with a valid contact email
    headers = {
        "User-Agent": "GeocodingScript/1.0 (your_email@example.com)"  
    }

    # Send the request to the API
    response = requests.get(BASE_URL, params=params, headers=headers)

    # Raise an error if the request failed (e.g., bad connection, 500 error)
    response.raise_for_status()

    # Parse the JSON response Note: reads through the response given by the Website
    data = response.json()

    # If nothing is found, return None
    if not data:
        return None

    # We only asked for one result, so grab the first item
    result = data[0]

    # Return a cleaned-up dictionary with key details
    return {
        "display_name": result.get("display_name"),  # full standardized address
        "lat": result.get("lat"),                    # latitude
        "lon": result.get("lon"),                    # longitude
    }

def main():
    """
    Main function to ask the user for number of addresses,
    prompt for each one, and print the geocoding results.
    """
    # Helpful sanity check
    if not LIB_CSV.is_file():
        print(f"❌ CSV not found at: {LIB_CSV}")
        print(f"🧭 Current working directory is: {Path.cwd()}")
        print("Tip: Make sure the file is named exactly 'libraries.csv' and is in the same folder as this script.")
        return

    try:
        count = int(input("How many addresses do you want to enter? "))
    except ValueError:
        print("Invalid number. Please enter an integer.")
        return

    for i in range(count):
        address = input(f"Enter address {i+1}: ")
        result = geocode_address(address)

        if result:
            print("\n✅ Found Address:")
            print("Full Address:", result["display_name"])
            print("Latitude:", result["lat"])
            print("Longitude:", result["lon"])

            # 👉 convert to float and use an absolute path to the CSV
            nearest = find_closest_market(
                float(result["lat"]),
                float(result["lon"]),
                data_path=str(FARM_CSV)
            )
            print("Nearest Farmers Market:", nearest)
            nearest = find_closest_school(
                float(result["lat"]),
                float(result["lon"]),
                data_path=str(SCHOOL_CSV)
            )
            print("Nearest School:", nearest)
            nearest = find_closest_business(
                float(result["lat"]),
                float(result["lon"]),
                data_path=str(BUSINESS_CSV)
            )
            print("Nearest School:", nearest)
            nearest = find_closest_recreation_center(
                float(result["lat"]),
                float(result["lon"]),
                data_path=str(REC_CSV)
            )
            print("Nearest Recreation Center:", nearest)
            nearest = find_closest_hospital(
                float(result["lat"]),
                float(result["lon"]),
                data_path=str(HOSP_CSV)
            )
            print("Nearest Hospital:", nearest)
            nearest = find_closest_senior_housing(
                float(result["lat"]),
                float(result["lon"]),
                data_path=str(SENIOR_CSV)
            )
            print("Nearest Hospital:", nearest)
            
            nearest = find_closest_transport(
                float(result["lat"]),
                float(result["lon"]),
                data_path=str(TRANSPORT_CSV)
            )
            print("Nearest Transport Station:", nearest)
            
            nearest = find_closest_county_office(
                float(result["lat"]),
                float(result["lon"]),
                data_path=str(COUNTY_CSV)
            )
            print("Nearest County Office:", nearest)
            nearest = find_closest_library(
                float(result["lat"]),
                float(result["lon"]),
                data_path=str(LIB_CSV)
            )
            print("Nearest library:", nearest)
            nearest = find_closest_trail(
                float(result["lat"]),
                float(result["lon"]),
                data_path=str(TRAIL_CSV)
            )
            print("Nearest Trail:", nearest)

            nearest = find_closest_fire_station(
                float(result["lat"]),
                float(result["lon"]),
                data_path=str(SAFETY_CSV)
            )
            print("Nearest Fire Station:", nearest)

            nearest = find_closest_art(
                float(result["lat"]),
                float(result["lon"]),
                data_path=str(ART_CSV)
            )
            print("Nearest Art Center:", nearest)

            
        else:
            print("❌ Address not found.")

            

# Run the program if executed directly
if __name__ == "__main__":
    main()
