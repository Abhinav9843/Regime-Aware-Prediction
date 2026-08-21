import pvlib
import pandas as pd
import numpy as np

# -------------------------
# Location and period
# -------------------------

country = input("Enter Country:\t")

if country == 'germany':
    latitude, longitude = 48.40, 9.99  # Ulm
    start, end = "2021-01-01", "2023-12-31"
elif country == 'norway':
    latitude, longitude = 59.91333, 10.73889 # Oslo
    start, end = "2023-01-01", "2023-12-31"
elif country == 'france':
    latitude, longitude = 48.85667, 2.35222 # Paris
    start, end = "2023-01-01", "2023-12-31"



# -------------------------
# PV system config
# -------------------------
peak_power_kW = 5.0
surface_tilt = 30
surface_azimuth = 180

# -------------------------
# Fetch PVGIS hourly data
# -------------------------
data, meta = pvlib.iotools.get_pvgis_hourly(
    latitude=latitude,
    longitude=longitude,
    start=start,
    end=end,
    raddatabase="PVGIS-ERA5",
    components=True,
    surface_tilt=surface_tilt,
    surface_azimuth=surface_azimuth,
    pvcalculation=True,
    peakpower=peak_power_kW,
    map_variables=True,
    # strongly recommended for realism:
    # loss=14, mountingplace="building"
)

data = data.copy()

# -------------------------
# Time alignment
# PVGIS ERA5 often returns hourly-average values timestamped at HH:30 (UTC).
# For typical electricity price series indexed at start-of-hour local time (HH:00),
# convert to Europe/Berlin and shift by -30 min to represent start-of-hour.
# -------------------------
data.index = data.index.tz_convert("Europe/Berlin") - pd.Timedelta(minutes=30)

# -------------------------
# PV energy per hour in MWh
# PVGIS column P is power (typically W). Hourly energy (MWh) = P(W) * 1h / 1e6.
# For 1-hour timestep, that's simply P / 1e6.
# -------------------------
data["pv_MWh"] = data["P"] / 1e6

# -------------------------
# Simulate hourly load directly in MWh with TRUE daily total
# daily_kWh_total means TOTAL daily household energy (incl. base load).
# -------------------------
def simulate_load_MWh(index, daily_kWh_total=10.0, base_load_kW=0.3, seed=42):
    rng = np.random.default_rng(seed)

    # shape over 24 hours (unitless weights)
    profile = np.array(
        [
            0.4, 0.3, 0.3, 0.3, 0.4, 0.6,  # 0–5
            0.9, 1.1, 1.2, 1.0, 0.9, 0.8,  # 6–11
            0.7, 0.8, 1.0, 1.3, 1.5, 1.8,  # 12–17
            2.0, 2.3, 2.0, 1.5, 1.0, 0.6   # 18–23
        ],
        dtype=float,
    )
    profile /= profile.sum()

    # Convert daily target to MWh
    daily_MWh_total = daily_kWh_total / 1000.0

    # Base load per hour in MWh (kW * 1h -> kWh; /1000 -> MWh)
    base_MWh = (base_load_kW * 1.0) / 1000.0

    # Variable daily energy so that TOTAL daily energy = daily_MWh_total
    variable_daily_MWh = daily_MWh_total - 24.0 * base_MWh
    if variable_daily_MWh < 0:
        variable_daily_MWh = 0.0

    load_MWh = np.empty(len(index), dtype=float)
    for i, ts in enumerate(index):
        h = ts.hour
        var = variable_daily_MWh * profile[h]

        # Noise in MWh (0.05 kWh = 0.00005 MWh)
        noise_MWh = rng.normal(0.0, 0.00005)

        load = base_MWh + var + noise_MWh
        load_MWh[i] = max(load, 0.0)

    return pd.Series(load_MWh, index=index, name="load_MWh")


data["load_MWh"] = simulate_load_MWh(
    data.index, daily_kWh_total=10.0, base_load_kW=0.3, seed=42
)

# -------------------------
# Net grid exchange BEFORE battery (MWh)
# Positive means export, negative means import.
# -------------------------
data["net_grid_MWh_no_batt"] = data["pv_MWh"] - data["load_MWh"]

# -------------------------
# Export revenue example (price already in €/MWh, energy in MWh -> clean)
# -------------------------
price_eur_per_MWh = 150.0
data["export_MWh_no_batt"] = data["net_grid_MWh_no_batt"].clip(lower=0.0)
data["revenue_eur_no_batt"] = data["export_MWh_no_batt"] * price_eur_per_MWh

print("PV total (MWh):", data["pv_MWh"].sum())
print("Load total (MWh):", data["load_MWh"].sum())
print("Export total (MWh):", data["export_MWh_no_batt"].sum())
print("Revenue (€):", data["revenue_eur_no_batt"].sum())

if country == 'norway' or country == 'france':
    np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Strategies/sim_load_data_'+ country + '.txt', data["load_MWh"].iloc[:].values.reshape(-1,24))
    np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Strategies/sim_pv_data_'+ country + '.txt', data["pv_MWh"].iloc[:].values.reshape(-1,24))
if country == 'germany':
    np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Strategies/sim_load_data_'+ country + '.txt', data["load_MWh"].iloc[:].values.reshape(-1,24)[730:])
    np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Strategies/sim_load_data_'+ country + '_2021.txt', data["load_MWh"].iloc[:].values.reshape(-1,24)[:365])
    np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Strategies/sim_load_data_'+ country + '_2022.txt', data["load_MWh"].iloc[:].values.reshape(-1,24)[365:730])

    np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Strategies/sim_pv_data_'+ country + '.txt', data["pv_MWh"].iloc[:].values.reshape(-1,24)[730:])
    np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Strategies/sim_pv_data_'+ country + '_2021.txt', data["pv_MWh"].iloc[:].values.reshape(-1,24)[:365])
    np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Strategies/sim_pv_data_'+ country + '_2022.txt', data["pv_MWh"].iloc[:].values.reshape(-1,24)[365:730])

