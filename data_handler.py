from dataclasses import dataclass
import numpy as np
import pickle


@dataclass
class ROI:
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    sog_min: float
    sog_max: float
    cog_min: float
    cog_max: float


def load_data(data_path, interval_seconds, min_seqlen):
    print(f"Loading {data_path}...")

    with np.load(data_path, allow_pickle=True) as data:
        # Shape: [Total_Points, 6] -> [time, lat, lon, cog, sog, type]
        all_points = data['trajectories']
        traj_indices = pickle.loads(data['trajectory_idxes'].item())

    formatted_data = []
    num_trajectories = len(traj_indices)

    for i in range(num_trajectories):
        start_idx = traj_indices[i]
        end_idx = traj_indices[i+1] if i + \
            1 < num_trajectories else len(all_points)

        full_res_segment = all_points[start_idx:end_idx]
        segment = _interpolate_trajectory(
            full_res_segment, interval_seconds=interval_seconds)

        if len(segment) < min_seqlen:
            continue

        standardized_traj = np.stack([
            segment[:, 1],  # LAT
            segment[:, 2],  # LON
            segment[:, 4],  # SOG
            segment[:, 3],  # COG
            segment[:, 0]  # TIMESTAMP
        ], axis=1)

        formatted_data.append(standardized_traj)

    normalized_data = []
    roi = _get_metadata(formatted_data)

    for traj in formatted_data:
        lat_norm = (traj[:, 0] - roi.lat_min) / (roi.lat_max - roi.lat_min)
        lon_norm = (traj[:, 1] - roi.lon_min) / (roi.lon_max - roi.lon_min)
        sog_norm = (traj[:, 2] - roi.sog_min) / (roi.sog_max - roi.sog_min)
        cog_norm = (traj[:, 3] - roi.cog_min) / (roi.cog_max - roi.cog_min)

        normalized_traj = np.stack([
            lat_norm,
            lon_norm,
            sog_norm,
            cog_norm,
            traj[:, 4]  # TIMESTAMP remains unchanged
        ], axis=1)

        normalized_data.append(normalized_traj)

    final_data = []
    for i in range(len(normalized_data)):
        final_data.append({
            "mmsi": i,
            "traj": normalized_data[i]
        })

    print(f"Successfully loaded {len(formatted_data)} trajectories.")
    return final_data, roi


def _interpolate_trajectory(segment, interval_seconds=600):
    """
    segment: numpy array [N, 6] -> [time, lat, lon, cog, sog, type]
    returns: interpolated segment with fixed time intervals
    """

    times = segment[:, 0]
    lat = segment[:, 1]
    lon = segment[:, 2]
    cog = segment[:, 3]
    sog = segment[:, 4]

    sort_idx = np.argsort(times)
    times = times[sort_idx]
    lat = lat[sort_idx]
    lon = lon[sort_idx]
    cog = cog[sort_idx]
    sog = sog[sort_idx]

    start_time = times[0]
    end_time = times[-1]

    new_times = np.arange(start_time, end_time, interval_seconds)

    lat_i = np.interp(new_times, times, lat)
    lon_i = np.interp(new_times, times, lon)
    sog_i = np.interp(new_times, times, sog)

    cog_rad = np.deg2rad(cog)
    sin_cog = np.sin(cog_rad)
    cos_cog = np.cos(cog_rad)

    sin_i = np.interp(new_times, times, sin_cog)
    cos_i = np.interp(new_times, times, cos_cog)

    cog_i = np.rad2deg(np.arctan2(sin_i, cos_i)) % 360

    interpolated = np.stack([
        new_times,
        lat_i,
        lon_i,
        cog_i,
        sog_i,
    ], axis=1)

    return interpolated


def _get_metadata(formatted_data):
    # Concatenate all trajectories into one giant array [Total_Points, 5]
    # Indices: 0:LAT, 1:LON, 2:SOG, 3:COG, 4:TIMESTAMP
    all_points = np.concatenate(formatted_data, axis=0)

    lat_min, lon_min = all_points[:, 0].min(), all_points[:, 1].min()
    sog_min, cog_min = all_points[:, 2].min(), all_points[:, 3].min()

    lat_max, lon_max = all_points[:, 0].max(), all_points[:, 1].max()
    sog_max, cog_max = all_points[:, 2].max(), all_points[:, 3].max()

    return ROI(
        lat_min=lat_min,
        lat_max=lat_max,
        lon_min=lon_min,
        lon_max=lon_max,
        sog_min=sog_min,
        sog_max=sog_max,
        cog_min=cog_min,
        cog_max=cog_max
    )
