# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.dates as mdates
import matplotlib_inline.backend_inline

matplotlib_inline.backend_inline.set_matplotlib_formats("retina")

# %% [markdown]
# ### Read CMT and GNSS data from NumPy archive files

# %%
# Read CMT earthquake data and GPS/GNSS data from npz files

# Load CMT earthquake data from npz file
from types import SimpleNamespace

with np.load("cmt.npz", allow_pickle=True) as data:
    cmt = SimpleNamespace(
        dates=data["dates"],
        lons=data["lons"],
        lats=data["lats"],
        deps=data["deps"],
        mags=data["mags"],
    )

# Load GPS data from npz file
with np.load("gps_arrays_contiguous_N400_K5.npz", allow_pickle=True) as data:
    gnss = SimpleNamespace(
        names=data["gps_names"],
        dates=data["gps_dates"],
        lons=data["gps_lons"],
        lats=data["gps_lats"],
        e=data["gps_e"],
        n=data["gps_n"],
        h=data["gps_h"],
    )

neq = len(cmt.dates)
nsta = np.shape(gnss.e)[0]
ndays = np.shape(gnss.e)[1]

# %% [markdown]
# ## Make offset matrices
# - `nSta`-by-`nDays`
# - Includes equipment maintenance
# - Includes earthquakes from CMT search
# - Earthquakes are deemed to cause an offset if a station lies within $10^{0.36M_W - 0.15}$ km of the epicenter
#     - This yields distance thresholds of:
#         - 102 km for $M_W=6.0$
#         - 234 km for $M_W=7.0$
#         - 537 km for $M_W=8.0$
#         - 1230 km for $M_W=9.0$

# %% [markdown]
# ### Helper function for spherical distance calculation, used by `scipy.spatial.distance.cdist`


# %%
def sph_distance(point1, point2):
    R = 6370
    lat1 = np.radians(point1[0])  # insert value
    lon1 = np.radians(point1[1])
    lat2 = np.radians(point2[0])
    lon2 = np.radians(point2[1])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c
    return distance


# %% [markdown]
# ### Find dates on which equipment maintenance occurred for each station

# %%
# Load equipment maintenance offsets
offsets = pd.read_csv("F3_offset_var221231.csv")
# Get rid of unparseable characters
offsets = offsets[offsets.month != "?"]
# Assemble date columns into datetime
offsets["date"] = pd.to_datetime(offsets[["year", "month", "day"]])
# Reset index
offsets.reset_index(inplace=True, drop=True)

# Allocate space
maint_offs = np.zeros_like(gnss.e, dtype=bool)
# For each offset,

for i in range(len(offsets)):
    siteidx = np.where(gnss.names == str(offsets.loc[i, "site"]))[0]  # Station index
    dateidx = np.where(gnss.dates == offsets.loc[i, "date"])[0]  # Date index
    if len(siteidx) > 0 and len(dateidx) > 0:
        maint_offs[siteidx[0], dateidx[0]] = True  # Set to True on day of offset

# %% [markdown]
# ### Set offset dates for stations within a threshold distance of an earthquake

# %%
from scipy.spatial.distance import cdist

# Distance threshold exponent
threshexp = np.array(0.36 * cmt.mags - 0.15)
threshdist = np.power(10, threshexp)
threshdist = np.repeat(threshdist, nsta).reshape(neq, nsta).T

# Distance between stations and earthquakes
eq_points = np.array([cmt.lats, cmt.lons])
sta_points = np.array([gnss.lats, gnss.lons])
sta_eq_dist = cdist(sta_points.T, eq_points.T, sph_distance)
# Logical array indicating if there should be an earthquake jump
eqjump = np.less(sta_eq_dist, threshdist)
# Insert at correct dates
eqjumpdates = np.zeros((nsta, ndays), dtype=bool)

for i in range(neq):
    dateidx = np.where(gnss.dates == np.array(cmt.dates[i], dtype="datetime64[D]"))[
        0
    ]  # Date index
    if len(dateidx) != 0:
        eqjumpdates[:, dateidx[0]] = eqjumpdates[:, dateidx[0]] + eqjump[:, i]

# Combine earthquakes with maintenance
offsets = maint_offs + eqjumpdates
# Cumulative sum
offsets_cumulative = np.cumsum(offsets, axis=1)

# %% [markdown]
# ### Visualize offsets

# %%
# Image plots showing offsets as step functions
# Original arrays are True for (station, date) pairs where there's maintenance/threshold earthquake
# Step functions are cumulative sums across the dates
fig, ax = plt.subplots(1, 3, figsize=(12, 2))
max_val = np.max(offsets_cumulative)
ax[0].imshow(np.cumsum(maint_offs, axis=1), vmax=max_val, aspect="equal")
ax[0].set_title("Maintenance")
ax[1].imshow(np.cumsum(eqjumpdates, axis=1), vmax=max_val, aspect="equal")
ax[1].set_title("Earthquakes")
ax[2].imshow(offsets_cumulative, aspect="equal")
ax[2].set_title("Combined")
plt.show()

# %% [markdown]
# ### Visualize earthquakes based on distance

# %%
fig, ax = plt.subplots()
dist_cutoff = 150
# Find distance between each earthquake and closest station
min_eq_sta_dist = np.min(sta_eq_dist, axis=0)
close_eq_idx = min_eq_sta_dist <= dist_cutoff

# Plot: small points for all stations and all earthquakes; overlay colors showing min. distance
ax.plot(gnss.lons, gnss.lats, ".r", markersize=2, label="Station")
ax.plot(cmt.lons, cmt.lats, ".k", markersize=2, label="Earthquake")
do = ax.scatter(
    cmt.lons[close_eq_idx], cmt.lats[close_eq_idx], c=min_eq_sta_dist[close_eq_idx]
)
plt.colorbar(do, label="Minimum distance to station (km)")
plt.legend()
ax.set_aspect("equal")
plt.show()

# %% [markdown]
# ## Example time series with offset dates marked


# %%
def get_nice_bounds(data, margin=0.03):
    data_min, data_max = np.min(data), np.max(data)
    data_range = data_max - data_min

    # Add margin
    extended_min = data_min - margin * data_range
    extended_max = data_max + margin * data_range

    # Determine the magnitude/scale
    magnitude = 10 ** np.floor(np.log10(data_range))

    # Nice step sizes (multiples of the magnitude)
    nice_steps = np.array([0.1, 1, 2, 5, 10, 100]) * magnitude

    # Choose the step that gives a reasonable number of intervals (3-10)
    target_intervals = 100
    step_errors = np.abs(data_range / nice_steps - target_intervals)
    best_step = nice_steps[np.argmin(step_errors)]

    # Round bounds to nearest nice step
    lower_bound = np.floor(extended_min / best_step) * best_step
    upper_bound = np.ceil(extended_max / best_step) * best_step

    return int(lower_bound), int(upper_bound)


site_idx = 100

plt.clf()  # Clear current figure
plt.close()  # Close it fully
plt.figure(figsize=(12, 3))
plt.plot(
    gnss.dates,
    gnss.e[site_idx, :],
    ".",
    markersize=3,
    label=f"east ({gnss.names[site_idx]})",
    linewidth=1.0,
    color="tab:blue",
)
plt.plot(
    gnss.dates,
    gnss.n[site_idx, :],
    ".",
    markersize=3,
    label=f"north ({gnss.names[site_idx]})",
    linewidth=1.0,
    color="tab:orange",
)
ymin, ymax = plt.ylim()
# plt.vlines(gnss.dates[offsets[site_idx, :]], ymin, ymax, linewidth=2, color="lightgray", label="Offset", zorder=1)
# plt.vlines(gnss.dates[maint_offs[site_idx, :]], ymin, ymax, linewidth=0.25, color="m", label="Maintenance", zorder=1)
# plt.vlines(gnss.dates[eqjumpdates[site_idx, :]], ymin, ymax, linewidth=0.25, color="gray", label="coseismic", zorder=1)

# Find first and last non-Nan indices
non_nan_mask = ~np.isnan(gnss.e[site_idx, :])
non_nan_indices = np.where(non_nan_mask)[0]
first_idx = non_nan_indices[0]
last_idx = non_nan_indices[-1]
non_nan_day_count = np.sum(~np.isnan(gnss.e[site_idx, first_idx : last_idx + 1]))

# Get approximate min and max limits for nice plotting.  I don't love this.
y_min, y_max = get_nice_bounds(
    np.concatenate(
        (gnss.e[site_idx, :][non_nan_indices], gnss.n[site_idx, :][non_nan_indices])
    )
)

# Plot maintanance offsets
idx = np.where(maint_offs[site_idx, :] == True)[0]
for i in range(len(idx)):
    max_y = np.max([gnss.e[site_idx, idx[i]], gnss.n[site_idx, idx[i]]])
    plt.plot(
        [gnss.dates[idx[i]], gnss.dates[idx[i]]],
        [y_min, max_y],
        "-",
        linewidth=0.25,
        color="black",
    )

# Plot event magnitudes and number of events
idx = np.where(eqjumpdates[site_idx, :] == True)[0]
n_eq_days = len(idx)
eq_dates = gnss.dates[eqjumpdates[site_idx, :]]
eq_dates_orig = np.copy(eq_dates)
eq_mags = np.zeros(len(idx))
eq_count = np.zeros(len(idx))
for i in range(len(idx)):
    these_eq_idx = np.where(eqjump[site_idx, :])[0]
    match_idx = np.where(
        gnss.dates[idx[i]] == np.array(cmt.dates[these_eq_idx], dtype="datetime64[D]")
    )[0]
    eq_mags[i] = np.max(cmt.mags[these_eq_idx[match_idx]])
    eq_count[i] = len(match_idx)
idx_sorted = np.argsort(eq_mags)
eq_dates = eq_dates[idx_sorted]
eq_mags = eq_mags[idx_sorted]
eq_count = eq_count[idx_sorted]
for i in range(len(idx)):
    min_y = np.min([gnss.e[site_idx, idx[i]], gnss.n[site_idx, idx[i]]])
    plt.plot(
        [eq_dates_orig[i], eq_dates_orig[i]],
        [min_y, y_max],
        "-",
        linewidth=0.25,
        color="black",
    )
for i in range(len(idx)):
    plt.text(
        eq_dates[i],
        y_max,
        f"M{eq_mags[i]:0.1f} (n={eq_count[i]:0.0f})",
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="white",
            linewidth=0.5,
            edgecolor="black",
        ),
        rotation=45,
        fontsize=8,
        color="black",
        ha="left",
    )

# Add east and north text at far RHS
plt.text(
    gnss.dates[last_idx],
    gnss.e[site_idx, last_idx],
    "east",
    bbox=dict(
        boxstyle="round,pad=0.3", facecolor="white", linewidth=0.5, edgecolor="tab:blue"
    ),
    rotation=0,
    fontsize=8,
    color="tab:blue",
    va="center",
    ha="left",
)
plt.text(
    gnss.dates[last_idx],
    gnss.n[site_idx, last_idx],
    "north",
    bbox=dict(
        boxstyle="round,pad=0.3",
        facecolor="white",
        linewidth=0.5,
        edgecolor="tab:orange",
    ),
    rotation=0,
    fontsize=8,
    color="tab:orange",
    va="center",
    ha="left",
)


# x-axis date formatting
year_min = gnss.dates[first_idx].astype("datetime64[Y]").astype("datetime64[D]")
year_max = (
    gnss.dates[last_idx].astype("datetime64[Y]") + np.timedelta64(1, "Y")
).astype("datetime64[D]")
start_year = np.datetime64(year_min, "Y")
end_year = np.datetime64(year_max, "Y")
years = np.arange(start_year, end_year + np.timedelta64(1, "Y"), dtype="datetime64[Y]")
dates = years.astype("datetime64[D]")
plt.xticks(dates)
plt.xticks(rotation=45)
plt.gca().xaxis.set_major_locator(mdates.YearLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
plt.xlabel(
    rf"$t$ ({non_nan_day_count:,} GNSS data days, {np.sum(eq_count):0.0f} earthquakes on {n_eq_days} days)"
)
x_start = np.datetime64(year_min, "Y").astype("datetime64[D]")
x_end = np.datetime64(year_max, "Y").astype("datetime64[D]")
plt.xlim([x_start, x_end])

# y-axis date formatting
# y_min, y_max = get_nice_bounds(np.concatenate((gnss.e[site_idx, :][non_nan_indices], gnss.n[site_idx, :][non_nan_indices])))
y_ticks = np.array([y_min, y_max])
# If there's a zero crossing insert a tick at zero
if (y_ticks[0] > 0 and y_ticks[1] < 0) or (y_ticks[0] < 0 and y_ticks[1] > 0):
    # Insert zero in the middle
    y_ticks = np.array([y_ticks[0], 0, y_ticks[1]])

plt.ylim(y_min, y_max)
plt.yticks(y_ticks)
plt.ylabel(rf"$\Delta p$ (mm)")

# Remove top and right hand side axes
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
plt.show()
