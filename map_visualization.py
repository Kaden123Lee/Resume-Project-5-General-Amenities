import folium
import webbrowser
import os
from senior_housing_data import get_all_senior_housing


def create_senior_housing_map(center_lat=34.0522, center_lon=-118.2437, zoom_start=11):
    """
    Create a Folium map with all senior housing locations marked.
    """
    # Create the map
    m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_start, tiles="OpenStreetMap")
    
    # Add all senior housing locations
    housing_data = get_all_senior_housing()
    
    for housing in housing_data:
        # Create popup content
        popup_content = f"""
        <div style="width: 250px;">
            <h4>{housing.get('Name', 'Unknown')}</h4>
            <p><b>Address:</b> {housing.get('Address', 'N/A')}<br>
            {housing.get('City', 'N/A')}, {housing.get('State', 'N/A')} {housing.get('Zipcode', 'N/A')}</p>
            <p><b>Coordinates:</b> {housing.get('Lat', 'N/A')}, {housing.get('Lon', 'N/A')}</p>
        </div>
        """
        
        # Add marker to map
        folium.Marker(
            [housing.get('Lat'), housing.get('Lon')],
            popup=folium.Popup(popup_content, max_width=300),
            tooltip=housing.get('Name', 'Senior Housing'),
            icon=folium.Icon(color='blue', icon='home', prefix='fa')
        ).add_to(m)
    
    return m


def save_and_open_map(map_obj, filename="senior_housing_map.html"):
    """
    Save the map to an HTML file and open it in the default browser.
    """
    map_obj.save(filename)
    print(f"Map saved as {filename} — opening in your browser now...")
    webbrowser.open(f"file://{os.path.realpath(filename)}")


if __name__ == "__main__":
    # Create and display the map
    housing_map = create_senior_housing_map()
    save_and_open_map(housing_map)