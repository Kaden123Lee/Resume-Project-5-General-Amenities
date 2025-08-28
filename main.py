from senior_housing_data import find_closest_senior_housing, get_all_senior_housing
from map_visualization import create_senior_housing_map, save_and_open_map


def main():
    print("Senior Housing Locator")
    print("======================")
    
    # Example: Find nearest to LA City Hall
    my_lat, my_lon = 34.05, -118.25
    
    try:
        nearest = find_closest_senior_housing(my_lat, my_lon)
        
        print(f"\nNearest senior housing to LA City Hall ({my_lat}, {my_lon}):")
        print(f"Name: {nearest['name']}")
        print(f"Address: {nearest['address']}, {nearest['city']}, {nearest['state']} {nearest['zip']}")
        print(f"Distance: {nearest['distance_km']:.2f} km")
        
        # Create and display the map
        print("\nCreating map with all senior housing locations...")
        housing_map = create_senior_housing_map()
        save_and_open_map(housing_map)
        
        # Show some statistics
        all_housing = get_all_senior_housing()
        print(f"\nTotal senior housing facilities in database: {len(all_housing)}")
    
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please check that the CSV file exists at the specified path.")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()