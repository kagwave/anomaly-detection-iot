ZONE_MAP = {
    "AURO": "Urban", "BAHA": "Unknown", "BALD": "Unknown", "BEAR": "Rural",
    "BUCK": "Rural", "BURN": "Unknown", "CAST": "Unknown", "CHAP": "Unknown",
    "CLA2": "Rural", "CLAY": "Rural", "CLIN": "Unknown", "DURH": "Urban",
    "FLET": "Unknown", "FRYI": "Unknown", "GOLD": "Urban", "HAML": "Unknown",
    "JACK": "Rural", "JEFF": "Unknown", "KINS": "Unknown", "LAKE": "Unknown",
    "LAUR": "Unknown", "LEWS": "Unknown", "LILE": "Unknown", "MITC": "Unknown",
    "NCAT": "Campus", "NEWL": "Unknown", "OXFO": "Unknown", "PLYM": "Coastal",
    "RALE": "Urban", "REED": "Unknown", "REID": "Unknown", "ROCK": "Unknown",
    "SALI": "Urban", "SASS": "Unknown", "SILR": "Unknown", "SPIN": "Unknown",
    "SPRU": "Unknown", "TAYL": "Unknown", "UNCA": "Campus", "WAKE": "Urban",
    "WAYN": "Unknown", "WHIT": "Unknown", "WILD": "Unknown", "WILL": "Unknown",
    "WINE": "Unknown"
}

def get_zone_for_station(station_id):
    return ZONE_MAP.get(station_id, "Unknown")

def get_station_coords():
    return {
        "AURO": (35.706, -78.651), "BAHA": (35.390, -77.963), "BALD": (35.243, -83.419),
        "BEAR": (36.414, -77.712), "BUCK": (35.949, -81.670), "BURN": (35.912, -82.307),
        "CAST": (36.472, -79.151), "CHAP": (35.928, -79.030), "CLA2": (34.773, -78.778),
        "CLAY": (35.042, -83.822), "CLIN": (36.307, -82.349), "DURH": (36.015, -78.923),
        "FLET": (35.430, -82.500), "FRYI": (35.594, -82.629), "GOLD": (35.387, -77.978),
        "HAML": (35.348, -81.122), "JACK": (35.519, -77.421), "JEFF": (36.303, -81.690),
        "KINS": (35.812, -77.584), "LAKE": (36.455, -78.207), "LAUR": (34.787, -79.465),
        "LEWS": (36.111, -77.774), "LILE": (36.121, -80.064), "MITC": (35.587, -82.312),
        "NCAT": (36.076, -79.773), "NEWL": (36.484, -77.099), "OXFO": (36.307, -78.590),
        "PLYM": (35.865, -76.755), "REED": (35.971, -78.468), "REID": (35.928, -80.404),
        "ROCK": (36.383, -79.750), "SALI": (36.096, -80.247), "SASS": (36.078, -79.294),
        "SILR": (35.373, -82.768), "SPIN": (35.943, -77.790), "SPRU": (35.679, -81.978),
        "TAYL": (36.270, -81.447), "UNCA": (35.617, -82.566), "WAYN": (35.478, -77.999),
        "WHIT": (35.933, -78.169), "WILD": (36.364, -81.096), "WILL": (35.064, -77.055),
        "WINE": (36.117, -80.266)
    }