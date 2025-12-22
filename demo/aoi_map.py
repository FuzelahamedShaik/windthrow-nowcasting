import folium
from folium.plugins import Draw

def aoi_draw_map(center=(64.5, 26.0), zoom=5):
    m = folium.Map(location=center, zoom_start=zoom, tiles="CartoDB positron")
    Draw(
        export=False,
        draw_options={
            "polyline": False,
            "circle": False,
            "circlemarker": False,
            "marker": False,
            "rectangle": True,
            "polygon": True,
        },
        edit_options={"edit": True, "remove": True},
    ).add_to(m)
    return m