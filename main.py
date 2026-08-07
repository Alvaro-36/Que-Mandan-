import json
import re
import datetime
import statistics
import numpy as np
from scipy.stats import gaussian_kde
from typing import Union

# Imported for debugging / visualization purposes
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt


def parse_to_iso_datetime(date_str: str, time_str: str) -> str:
    """
    Combines date and time strings into a single standard Python ISO 8601 dateTime string ('YYYY-MM-DDTHH:MM:SS').
    """
    clean_time = time_str.replace('\u202f', ' ').replace('\xa0', ' ').strip()
    dt_str = f"{date_str.strip()} {clean_time}"
    for fmt in ("%d/%m/%Y %I:%M %p", "%d/%m/%Y %I:%M:%S %p", "%d/%m/%Y %H:%M", "%d/%m/%Y %H:%M:%S"):
        try:
            dt = datetime.datetime.strptime(dt_str, fmt)
            return dt.isoformat()
        except ValueError:
            pass
    return dt_str


def convert_txt_to_json(
    txt_filename: str,
    json_filename: str,
    burst_threshold_seconds: float = 30.0
) -> str:
    """
    Reads a .txt file where each line is a text message with date, time, and author/message content,
    applies Burst Merging (Debouncing) to combine consecutive messages from the same author sent
    within sub-minute intervals (less than burst_threshold_seconds), and saves the resulting JSON.

    Parameters:
        txt_filename (str): Input .txt file path or name.
        json_filename (str): Output .JSON file path or name.
        burst_threshold_seconds (float): Max time difference in seconds between messages from the same author to be merged. Defaults to 30.0.

    Returns:
        str: The path or name of the created JSON file.
    """
    pattern = re.compile(r'^([^,]+),\s*([^-]+)\s*-\s*(.*)$')
    parsed_messages = []

    with open(txt_filename, 'r', encoding='utf-8') as file:
        for line in file:
            clean_line = line.rstrip('\r\n')
            if not clean_line:
                continue

            match = pattern.match(clean_line)
            if match:
                date_str = match.group(1).strip()
                time_str = match.group(2).strip()
                iso_dt = parse_to_iso_datetime(date_str, time_str)
                content = match.group(3).strip()

                if ": " in content:
                    author, message_text = content.split(": ", 1)
                else:
                    author, message_text = "", content

                parsed_messages.append({
                    "dateTime": iso_dt,
                    "author": author,
                    "message": message_text
                })
            else:
                parsed_messages.append({
                    "dateTime": "",
                    "author": "",
                    "message": clean_line
                })

    # Apply Burst Merging / Debouncing
    merged_messages = []
    for item in parsed_messages:
        if not merged_messages:
            merged_messages.append(item.copy())
            continue

        prev = merged_messages[-1]
        same_author = (item["author"] != "" and item["author"] == prev["author"])

        time_diff = float('inf')
        if prev["dateTime"] and item["dateTime"]:
            try:
                dt_prev = datetime.datetime.fromisoformat(prev["dateTime"])
                dt_curr = datetime.datetime.fromisoformat(item["dateTime"])
                time_diff = abs((dt_curr - dt_prev).total_seconds())
            except ValueError:
                pass

        if same_author and (time_diff < burst_threshold_seconds):
            # Merge current message into the existing burst
            prev["message"] += "\n" + item["message"]
        else:
            merged_messages.append(item.copy())

    with open(json_filename, 'w', encoding='utf-8') as file:
        json.dump(merged_messages, file, ensure_ascii=False, indent=4)

    return json_filename


def _get_message_distances(json_filename: str, ignore_zeros: bool = False) -> list[float]:
    """
    Helper function to parse message timestamps from a JSON file and compute consecutive distances in seconds.
    """
    with open(json_filename, 'r', encoding='utf-8') as file:
        messages = json.load(file)

    timestamps = []
    for item in messages:
        dt_str = item.get("dateTime", "")
        if not dt_str:
            continue

        dt = None
        try:
            dt = datetime.datetime.fromisoformat(dt_str)
        except ValueError:
            for fmt in ("%d/%m/%Y %H:%M:%S", "%d/%m/%Y %I:%M %p", "%d/%m/%Y %H:%M"):
                try:
                    dt = datetime.datetime.strptime(dt_str, fmt)
                    break
                except ValueError:
                    pass

        if dt is not None:
            timestamps.append(dt)

    if len(timestamps) < 2:
        return []

    distances = [
        abs((timestamps[i + 1] - timestamps[i]).total_seconds())
        for i in range(len(timestamps) - 1)
    ]

    if ignore_zeros:
        distances = [d for d in distances if d > 0]

    return distances


def calculate_median_message_distance(json_filename: str, ignore_zeros: bool = False) -> float:
    """
    Calculates the median time distance between consecutive message bursts in seconds.
    """
    distances = _get_message_distances(json_filename, ignore_zeros)
    return statistics.median(distances) if distances else 0.0


def calculate_max_message_distance(json_filename: str, ignore_zeros: bool = False) -> float:
    """
    Calculates the maximum time distance between consecutive message bursts in seconds.
    """
    distances = _get_message_distances(json_filename, ignore_zeros)
    return max(distances) if distances else 0.0


def calculate_min_message_distance(json_filename: str, ignore_zeros: bool = False) -> float:
    """
    Calculates the minimum time distance between consecutive message bursts in seconds.
    """
    distances = _get_message_distances(json_filename, ignore_zeros)
    return min(distances) if distances else 0.0


def calculate_mean_message_distance(json_filename: str, ignore_zeros: bool = False) -> float:
    """
    Calculates the average (mean) time distance between consecutive message bursts in seconds.
    """
    distances = _get_message_distances(json_filename, ignore_zeros)
    return statistics.mean(distances) if distances else 0.0


def calculate_activity_kde(
    json_filename: str,
    num_points: int = 200,
    bw_method: str = None,
    show_plot: bool = True
) -> np.ndarray:
    """
    Calculates a continuous activity density vector using Kernel Density Estimation (KDE)
    over a 24-hour cycle from message timestamps in the JSON file.

    Parameters:
        json_filename (str): Input JSON file path or name.
        num_points (int): Number of discrete points in the output vector (0 to 24 hours). Defaults to 200.
        bw_method (str): Bandwidth method for gaussian_kde. Defaults to None.
        show_plot (bool): If True, opens a GUI window displaying the plot for debugging. Defaults to True.

    Returns:
        np.ndarray: A 1D numpy array (vector) representing the discrete activity density function values.
    """
    with open(json_filename, 'r', encoding='utf-8') as file:
        messages = json.load(file)

    hours = []
    for item in messages:
        dt_str = item.get("dateTime", "")
        if not dt_str:
            continue

        try:
            dt = datetime.datetime.fromisoformat(dt_str)
            hours.append(dt.hour + dt.minute / 60.0 + dt.second / 3600.0)
        except ValueError:
            pass

    if not hours:
        return np.zeros(num_points)

    hours_arr = np.array(hours)
    # Periodic extension to handle wrap-around smooth density across midnight
    hours_ext = np.concatenate([hours_arr - 24, hours_arr, hours_arr + 24])

    kde = gaussian_kde(hours_ext, bw_method=bw_method)
    grid = np.linspace(0, 24, num_points)
    density = kde(grid) * 3.0  # Scale density for 3x extended domain

    if show_plot:
        plt.figure(figsize=(10, 5))
        plt.plot(grid, density, color='#1f77b4', linewidth=2.5, label='KDE Activity Density')
        plt.fill_between(grid, density, color='#1f77b4', alpha=0.3)
        plt.title('24-Hour Discrete Chat Activity Density Vector (KDE)')
        plt.xlabel('Hour of Day (0 - 24)')
        plt.ylabel('Density')
        plt.xticks(range(0, 25, 2))
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.show()

    return density


def calculate_peak_hour_coefficient(
    hour: float,
    density_vector_or_json: Union[np.ndarray, str],
    num_points: int = 200
) -> float:
    """
    Calculates the 'Peak Hour Coefficient' c(h) = 2 - f(h) for a given hour h (0.0 to 24.0),
    where f(h) is the image of hour h in the KDE probability density function mapped to [-0.5, 1.5].
    The resulting coefficient c(h) ranges from 0.5 to 2.5.

    Parameters:
        hour (float): Target hour of the day (e.g., 14.5 for 14:30).
        density_vector_or_json (np.ndarray | str): Either the pre-calculated KDE density numpy vector or the JSON file path.
        num_points (int): Number of points in the KDE grid. Defaults to 200.

    Returns:
        float: The peak hour coefficient c(h) in [0.5, 2.5].
    """
    if isinstance(density_vector_or_json, str):
        density_vector = calculate_activity_kde(density_vector_or_json, num_points=num_points, show_plot=False)
    else:
        density_vector = density_vector_or_json

    if len(density_vector) == 0:
        return 2.0

    hour_mod = hour % 24.0
    grid = np.linspace(0, 24, len(density_vector))
    d = float(np.interp(hour_mod, grid, density_vector))

    v_min = float(density_vector.min())
    v_max = float(density_vector.max())

    if v_max > v_min:
        f_h = -0.5 + 2.0 * (d - v_min) / (v_max - v_min)
    else:
        f_h = 0.5

    c_h = 2.0 - f_h
    return float(np.clip(c_h, 0.5, 2.5))


if __name__ == '__main__':
    json_filename = convert_txt_to_json('chatWPP.txt', 'chatWPP.json', burst_threshold_seconds=30.0)
    print(f"JSON file generated with Burst Merging: {json_filename}")

    max_sec = calculate_max_message_distance(json_filename)
    min_sec = calculate_min_message_distance(json_filename)
    mean_sec = calculate_mean_message_distance(json_filename)
    median_sec = calculate_median_message_distance(json_filename)

    print(f"Max distance: {max_sec} seconds ({max_sec / 3600:.2f} hours)")
    print(f"Min distance: {min_sec} seconds")
    print(f"Mean distance: {mean_sec:.2f} seconds ({mean_sec / 60:.2f} minutes)")
    print(f"Median distance: {median_sec} seconds ({median_sec / 60:.2f} minutes)")

    density_vector = calculate_activity_kde(json_filename, show_plot=False)
    print(f"Density vector generated with shape: {density_vector.shape}")

    # Test peak hour coefficient c(h) = 2 - f(h) for different hours
    c_3am = calculate_peak_hour_coefficient(3.0, density_vector)
    c_230pm = calculate_peak_hour_coefficient(14.5, density_vector)
    c_9pm = calculate_peak_hour_coefficient(21.0, density_vector)

    print(f"Peak Hour Coeff at 03:00 (3 AM): c(h) = {c_3am:.4f}")
    print(f"Peak Hour Coeff at 14:30 (2:30 PM): c(h) = {c_230pm:.4f}")
    print(f"Peak Hour Coeff at 21:00 (9 PM - peak activity): c(h) = {c_9pm:.4f}")
