# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib_inline.backend_inline

matplotlib_inline.backend_inline.set_matplotlib_formats("retina")

# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib_inline.backend_inline

matplotlib_inline.backend_inline.set_matplotlib_formats("retina")


def load_gps_data(file_path):
    with np.load(file_path, allow_pickle=True) as data:
        gps_names = data["gps_names"]
        gps_dates = data["gps_dates"]
        gps_lons = data["gps_lons"]
        gps_lats = data["gps_lats"]
        gps_e = data["gps_e"]
        gps_n = data["gps_n"]
        gps_h = data["gps_h"]

    return gps_names, gps_dates, gps_lons, gps_lats, gps_e, gps_n, gps_h


def plot_station_time_series(
    gps_site_idx, gps_names, gps_dates, gps_lons, gps_lats, gps_e, gps_n, gps_h
):
    plt.figure(figsize=(10, 6))
    plt.subplot(3, 1, 1)
    plt.plot(gps_dates, gps_e[gps_site_idx, :], ".", color="tab:orange", markersize=1)
    plt.ylabel(r"$\Delta p_\mathrm{e}$ (mm)")

    plt.subplot(3, 1, 2)
    plt.plot(gps_dates, gps_n[gps_site_idx, :], ".", color="tab:blue", markersize=1)
    plt.ylabel(r"$\Delta p_\mathrm{n}$ (mm)")

    plt.subplot(3, 1, 3)
    plt.plot(gps_dates, gps_h[gps_site_idx, :], ".", color="tab:green", markersize=1)
    plt.xlabel("date (yr)")
    plt.ylabel(r"$\Delta p_\mathrm{h}$ (mm)")

    plt.suptitle(
        f"station {gps_names[gps_site_idx]} (lon={gps_lons[gps_site_idx]:0.3f}, lat={gps_lats[gps_site_idx]:0.3f})",
        fontsize=10,
    )
    plt.show()


# %% [markdown]
# # Load contiguous GPS data from Japan

# %%
file_path = "gps_arrays_contiguous_N400_K5.npz"
gps_names, gps_dates, gps_lons, gps_lats, gps_e, gps_n, gps_h = load_gps_data(file_path)

# %% [markdown]
# # Plot a few GPS site time series

# %%
# Nothing special about these indices
gps_site_idx_list = [13, 100, 200, 350]

for gps_site_idx in gps_site_idx_list:
    plot_station_time_series(
        gps_site_idx, gps_names, gps_dates, gps_lons, gps_lats, gps_e, gps_n, gps_h
    )
