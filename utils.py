import irisreader as ir
from sunpy.map import Map
from reproject import reproject_interp
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import rotate
from skimage.registration import phase_cross_correlation
from scipy.ndimage import shift
from pathlib import Path
import matplotlib.animation as animation
from joblib import Parallel, delayed
import pandas as pd
from astropy.time import Time
from sunpy.net import Fido, attrs as a
from aiapy.calibrate import register, update_pointing
from aiapy.calibrate.util import get_pointing_table
from sunpy.map import MapSequence
from tqdm import tqdm
import csv
import drms
import gzip, shutil
from pathlib import Path
import smtplib, ssl
from email.message import EmailMessage
from image_registration import chi2_shift
import gc
import os

# mutes some output
import logging
log = logging.getLogger('sunpy')
log.setLevel('WARNING')
log_reprj = logging.getLogger('reproject')
log_reprj.setLevel('WARNING')






def iris_to_sunpy_map(iris_sji, frame):
    """
    Converts sji data from irisreader to a SunPy Map. 
    It extracts a single frame from an IRIS SJI data cube and creates a SunPy Map.
    There are some ASCII characters in the header that are not supported by SunPy's Map function.
    These are removed to create the Map.
    
    Parameters
    ----------
    iris_sji: SJI data from an IRIS observation obtained using irisreader (https://i4ds.github.io/IRISreader/html/index.html)
    frame: int
        To select the frame in the sji data cube to convert to a SunPy Map.

    Returns
    ----------
    sunpy.map.Map 
        SunPy Map created with the sji image data and the header data from the selected frame.
    """
    iris_data = iris_sji.get_image_step(frame)
    iris_header = iris_sji.headers[frame]
    map_header = fits.Header()
    for key, value in iris_header.items():
        try:
            map_header[key] = value
        except Exception:
            pass
    return Map(iris_data, map_header)






def crop_frame(aia_map, iris_map):
    """
    Crops AIA image to IRIS field of view.

    Parameters
    ----------
    aia_map: sunpy.map.Map
        SunPy Map containing AIA data to crop.
    iris_map: sunpy.map.Map
        SunPy Map containing IRIS data.
    
    Returns
    ----------
    sunpy.map.Map
        SunPy Map containing the cropped AIA data.
    """
    ny, nx = iris_map.data.shape
    bl_coord = iris_map.pixel_to_world(0 * u.pix, 0 * u.pix)
    tr_coord = iris_map.pixel_to_world((nx - 1) * u.pix, (ny - 1) * u.pix)
    return aia_map.submap(bottom_left=bl_coord, top_right=tr_coord)






def normalize_data(data, low=1, high=99):
    """
    Normalizes image data using logarithmic scaling and percentile clipping.

    Parameters
    ----------
    data: numpy.ndarray
        2D array containing the data that is normalized.
    low: float, optional
        Lower percentile used for normalization. Default is 1.
    high: float, optional
        Upper perecentile used for normalization. Default is 99.

    Returns
    ----------
    normalized_data: numpy.ndarray
        2D array containing the normalized data. Values have been scaled to range [0,1].

    """
    if data.min() < 0:
        data[data < 0] = 0 # If data contains negative values they are set to 0
    data = np.log10(data + 0.01) # Adding 0.01 to avoid log(0)
    vmin, vmax = np.percentile(data[np.isfinite(data)], (low, high))
    normalized_data = np.clip((data - vmin) / (vmax - vmin), 0, 1)
    normalized_data[np.isnan(normalized_data)] = 0
    return normalized_data

# def normalize_data(data):
#     norm = data / np.max(data)
#     return norm



def find_rotation(aia_data, iris_data, method='chi2', min_angle=-2.0, max_angle=2.001, step=0.05):
    """
    Goes through given angle range and returns the angle with the best cross-correlation
    and best shifts based on skimage.registration.phase_cross_correlation.
    Included image_registration.chi2_shift as a second method to obtain the shifts.

    Parameters
    ----------
    aia_data: numpy.ndarray
        2D array containing the image data from the AIA SunPy Map.
    iris_data: numpy.ndarray
        2D array containing the image data from the IRIS SunPy Map.
    method: string, optional
        Uses chi2_shift for 'chi2' or phase_cross_correlation for 'phase' as alignment method. Default is 'chi2'.
    min_angle: float, optional
        Minimum rotation angle in degrees. Default is -2.
    max_angle: float, optional
        Maximum rotation angle in degrees. Default is 2.001.
    step: float, optional
        Step size in degrees used to loop through the angle range. Default is 0.05.

    Returns
    ----------
    best_angle: float
        The angle the IRIS data was rotated with that gave the best result.
    best_shift: numpy.ndarray or list
        Array with the best shifts in y and x obtained by phase_cross_correlation or chi2_shift.
        Note: It returns the shift in y in the first position and the shift in x in the second
        -> [y,x].
    best_err: float
        Normalized RMS returned from phase_cross_correlation or chi2 value from chi2_shift.
    errx: float or None
        Uncertainty from the x-shift from chi2_shift, None if phase_cross_correlation was called.
    erry: float or None
        Uncertainty from the y-shift from chi2_shift, None if phase_cross_correlation was called.
    """
    best_err = np.inf
    best_angle = None
    best_shift = None
    best_errx = None
    best_erry = None
    aia_normalized = normalize_data(aia_data)
    aia_crop = aia_normalized[30:-30,30:-30]

    for angle in np.arange(min_angle, max_angle, step):
        iris_rotated = rotate(iris_data, angle=angle, reshape=False, order=1)
        iris_normalized = normalize_data(iris_rotated)
        iris_crop = iris_normalized[30:-30,30:-30]

        # uses chi2_shift to find best rotation angle and best shift
        if method == 'chi2':
            dx, dy, errx, erry, chi2_output = chi2_shift(
                aia_crop, 
                iris_crop,
                upsample_factor='auto', 
                return_error=True, 
                return_chi2array=True, 
                zeromean=True
                )
            
            _, _, chi2 = chi2_output

            if chi2.min() < best_err:
                best_err = chi2.min()
                best_angle = angle
                best_shift = [dy, dx] # matches convention of phase_cross_correlation
                best_errx = errx
                best_erry = erry

        # usese phase_cross_correlation to find best rotation angle and best shift
        elif method == 'phase':
            shift_phase, error, _ = phase_cross_correlation(
                aia_crop, 
                iris_crop, 
                upsample_factor=100
                )

            if error < best_err:
                best_err = error
                best_angle = angle
                best_shift = shift_phase

        else:
            raise ValueError("method must be 'chi2' or 'phase'")
    
    return best_angle, best_shift, best_err, best_errx, best_erry


def align_aia_iris(aia_data, aia_header, iris_data, iris_header, method):
    """
    Method that calls the whole aligment process. Needs an AIA and an IRIS frame
    as input and gives out the results obtained by find_rotation.

    Parameters
    ----------
    aia_data: numpy.ndarray
        2D array containing the image data from the AIA SunPy Map.
    aia_header: astropy.io.fits.Header or dict
        Header information to the AIA image data.
    iris_data: numpy.ndarray
        2D array containing the image data from the IRIS SunPy Map.
    iris_header: astropy.io.fits.Header or dict
        Header information to the IRIS image data. 
    method: string
        Uses chi2_shift when 'chi2' and phase_cross_correlation when 'phase'.

    Returns
    ----------
    best_shift: numpy.ndarray
        2D array with the best shifts in y and x obtained by phase_cross_correlation or chi2_shift.
        Note: It returns the shift in y in the first position and the shift in x in the second
        -> [y,x].
    best_angle: float
        The angle the IRIS data was rotated with that gave the best result.
    best_errx: float
        Uncertainty from the x-shift from chi2_shift, None if phase_cross_correlation was called.
    best_erry: float
        Uncertainty from the y-shift from chi2_shift, None if phase_cross_correlation was called.
    best_err: float
        Normalized RMS returned from phase_cross_correlation or chi2 value from chi2_shift.
    """
    aia_map = Map(aia_data, aia_header)
    iris_map = Map(iris_data, iris_header)
    iris_map = iris_map.rotate(order=3, missing=-200) 
    cropped_aia_map = crop_frame(aia_map, iris_map)

    # Reproject IRIS onto AIA WCS
    iris_reprojected = iris_map.reproject_to(cropped_aia_map.wcs)


    best_angle, best_shift, best_err, best_errx, best_erry = find_rotation(cropped_aia_map.data, iris_reprojected.data, method=method)
    return best_shift, best_angle, best_errx, best_erry, best_err




def find_matching_frames(iris_times_s, aia_times_s, start_s, end_s, delta_t = 24):
    """
    Matches IRIS frames to the closest AIA frames in an given time window. 

    Parameters
    ----------
    iris_times_s: numpy.ndarray
        Array of IRIS timestamps in seconds.
    aia_times_s: numpy.ndarray
        Array of AIA timestamps in seconds.
    start_s: float
        Start time of the target window in seconds.
    end_s: float
        End time of the target window in seconds.
    delta_t: float, optional
        Time to add before and after the window for both IRIS and AIA. Default is 24 seconds.

    Returns
    -----------
    matching_frames: list of tuple
        List of matched IRIS and AIA frame indices (aia_idx, iris_idx).
    """
    matching_frames = []

    iris_in_window = np.where((iris_times_s >= start_s - delta_t) & (iris_times_s <= end_s + delta_t))[0]
    aia_in_window = np.where((aia_times_s >= start_s - delta_t) & (aia_times_s <= end_s + delta_t))[0]

    for iris_idx in iris_in_window:
        iris_t = iris_times_s[iris_idx]
        matched_aia_frame = np.argmin(np.abs(aia_times_s[aia_in_window] - iris_t))
        aia_idx = aia_in_window[matched_aia_frame]
        matching_frames.append((aia_idx, iris_idx))
    return matching_frames






def write_to_file(out_file, observation_id, matches, results, iris_times_s, aia_times_s):
    """
    Writes the results of the alignment into a .tsv file. 

    Parameters
    ----------
    out_file: string
        Path to the output file.
    observation_id: string
        String containing the current IRIS observation ID.
    matches: list of tuple
        A list containing tuples that store the frames of the AIA and IRIS observation
        that are aligned (aia_idx, iris_idx).
    results: list of tuple
        A list containing tuples that store the obtained shifts and angle for the alignment
        (shifts, angle, errx_pix, erry_pix, error).
    iris_times_s: numpy.array
        Contains the timestamps for every aligned frame in seconds.
    aia_times_s: numpy.array
        Contains the timestamps for every aligned frame in seconds.
    """
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    header = [
        "Observation ID", "AIA frame", "IRIS frame",
        "AIA time", "IRIS time", "Delta t_sec",
        "Shift pixel x", "Error pixel x",
        "Shift pixel y", "Error pixel y",
        "Shift arcsec x", "Error arcsec x", 
        "Shift arcsec y", "Error arcsec y",
        "Angle", "RMSE or Chi2", "Status"   
    ]

    file_empty = (not out_file.exists()) or (out_file.stat().st_size == 0)

    with out_file.open("a", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        if file_empty:
            w.writerow(header)

        iris_iso = Time(iris_times_s, format="unix").isot
        aia_iso = Time(aia_times_s, format="unix").isot

        wrote_any = False
        for (aia_idx, iris_idx), (shifts, angle, errx_pix, erry_pix, error) in zip(matches, results):

            dy_pix, dx_pix = shifts[:2]
            dy_arc = dy_pix * 0.6 # factor 0.6 comes from AIA plate scale (0.6 arcsec/pixel)
            dx_arc = dx_pix * 0.6
            erry_arc = erry_pix * 0.6 if erry_pix is not None else None
            errx_arc = errx_pix * 0.6 if errx_pix is not None else None
            dt_s = float(aia_times_s[aia_idx] - iris_times_s[iris_idx])

            w.writerow([
                observation_id, 
                aia_idx, 
                iris_idx,
                aia_iso[aia_idx], 
                iris_iso[iris_idx], 
                f"{dt_s:.3f}",
                f"{dx_pix:.6f}", 
                f"{errx_pix:.6f}" if errx_pix is not None else "",
                f"{dy_pix:.6f}", 
                f"{erry_pix:.6f}" if erry_pix is not None else "",
                f"{dx_arc:.6f}", 
                f"{errx_arc:.6f}" if errx_arc is not None else "",
                f"{dy_arc:.6f}", 
                f"{erry_arc:.6f}" if erry_arc is not None else "",
                f"{angle:.3f}", 
                f"{error}",
                "OK"
            ])
            wrote_any = True

        if wrote_any:
            w.writerow([])

            

def write_error_row(out_file, observation_id, message):
    """
    If there was an error while aligning an observation, this method writes the error message to the .tsv file.

    Parameters
    ----------
    out_file: string
        Path to the output file.
    observation_id: string
        String containing the current IRIS observation ID.
    message: string
        String containing a message in case of an error.
    """
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    header = [
        "Observation ID", "AIA frame", "IRIS frame",
        "AIA time", "IRIS time", "Delta t_sec",
        "Shift pixel x", "Error pixel x",
        "Shift pixel y", "Error pixel y",
        "Shift arcsec x", "Error arcsec x", 
        "Shift arcsec y", "Error arcsec y",
        "Angle", "RMSE or Chi2", "Status"  
    ]
    file_empty = (not out_file.exists()) or (out_file.stat().st_size == 0)

    with out_file.open("a", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        if file_empty:
            w.writerow(header)
        w.writerow([observation_id,"NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA",f"ERROR: {message}"])
        w.writerow([])



def unpack_gz_files(paths):
    """
    Unpacks files with .gz compression. Deletes the .gz afterwards, leaving only the data.

    Parameters
    ----------
    paths: list of strings
        Contains the paths to observations that have to be unpack before being usable.

    Returns
    ----------
    out: list of strings
        Contains the paths to the unpacked data.
    """
    out = []
    for path in map(Path, paths):
        if path.suffix.lower() == ".gz":
            new_path = path.with_suffix("")
            if not new_path.exists():
                with gzip.open(path, "rb") as fin, open(new_path, "wb") as fout:
                    shutil.copyfileobj(fin, fout)
            path.unlink()
            out.append(str(new_path))
        else:
            out.append(str(path))
    return out



def fetch_iris_sji(start_time, end_time, outdir):
    """
    Downloads 1400 Å or 1330 Å IRIS SJI data based on the given start and end time.

    Parameters
    ----------
    start_time:
        Start time of the observation in the format of 'YYYY-MM-DDThh:mm:ss'
    end_time:
        End time of the observation in the format of 'YYYY-MM-DDThh:mm:ss'
    outdir: string
        Path to the folder to save the IRIS SJI data.

    Returns
    ----------
    Path to the downloaded data or empty list if no data was found.
    """
    for wavelength in (1400, 1330):
        results = Fido.search(
            a.Time(start_time, end_time),
            a.Instrument.iris,
            a.Wavelength(wavelength*u.Angstrom),
            a.Physobs.intensity,
            a.Level(2)
        )

        if len(results) and results.file_num > 0:
            files = Fido.fetch(results, path=str(outdir/"{file}"))

            iris_obs = [str(f) for f in files if "_SJI_" in str(f)]
            if not iris_obs:
                print(f"No {wavelength} Å was found...")
                continue

            return iris_obs

    print("No observation found...")
    return []



def fetch_cropped_aia(bottom_left, top_right, start_time, end_time, outdir, email):
    """
    Downloads AIA data of the given time window that is already cropped to the IRIS field of view.
    IMPORTANT: To use this you have to register a email to the JSOC data export system (http://jsoc.stanford.edu/ajax/register_email.html)

    Parameters
    ----------
    bottom_left: astropy.coordinates.SkyCoord
        Coordinates of the bottom left corner of the IRIS field of view.
    top_right: astropy.coordinates.SkyCoord
        Coordinates of the top right corner of the IRIS field of view.
    start_time: string
        Start time of the observation in the format of 'YYYY-MM-DDThh:mm:ss'.
    end_time: string
        End time of the observation in the format of 'YYYY-MM-DDThh:mm:ss'.
    outdir:
        Path to the folder to save the AIA data.
    email: string
        Registered email for the download.
    Returns
    ----------
    List of the paths to the downloaded AIA data or empty list if no data was found.
    """
    results = Fido.search(
        a.Time(start_time, end_time),
        a.Wavelength(1600*u.Angstrom),
        a.jsoc.Series.aia_lev1_uv_24s,
        a.jsoc.Notify(email), # use registered email
        a.jsoc.Segment.image,
        a.jsoc.Cutout(bottom_left, top_right, tracking=True)
    )
    if len(results) and results.file_num > 0:
        files = Fido.fetch(results, path=str(outdir/"{file}"))
        return [str(f) for f in files]
    print("No 1600 Å was found.")
    return []





def fetch_cropped_l15_aia(start_time, end_time, bottom_left, top_right, outdir, email):
    """
    Downloads AIA data of the given time window that is already cropped to the IRIS field of view.
    The exported data are processed to AIA level 1.5 during the JSOC export.
    IMPORTANT: To use this you have to register a email to the JSOC data export system (http://jsoc.stanford.edu/ajax/register_email.html)

    Parameters
    ----------
    bottom_left: astropy.coordinates.SkyCoord
        Coordinates of the bottom left corner of the IRIS field of view.
    top_right: astropy.coordinates.SkyCoord
        Coordinates of the top right corner of the IRIS field of view.
    start_time: string
        Start time of the observation in the format of 'YYYY-MM-DDThh:mm:ss'.
    end_time: string
        End time of the observation in the format of 'YYYY-MM-DDThh:mm:ss'.
    outdir:
        Path to the folder to save the AIA data.
    email: string
        Registered email for the download.

    Returns
    ----------
    List of the paths to the downloaded AIA data or empty list if no data was found.
    """
    duration = (Time(end_time) - Time(start_time)).to_value("sec")
    duration = int(round(duration))
    duration = f"{duration}s"
    q = f"aia.lev1_uv_24s[{start_time}/{duration}][1600]{{image}}"

    width  = (top_right.Tx - bottom_left.Tx + 120*u.arcsec).to(u.arcsec).value
    height = (top_right.Ty - bottom_left.Ty + 120*u.arcsec).to(u.arcsec).value
    cx = ((bottom_left.Tx + top_right.Tx) / 2).to(u.arcsec).value
    cy = ((bottom_left.Ty + top_right.Ty) / 2).to(u.arcsec).value

    im_patch = {
        "t_ref": start_time,
        "locunits": "arcsec",
        "boxunits": "arcsec",
        "x": cx,
        "y": cy,
        "width": width,
        "height": height,
        "t": 0,
        "r": 1,
        "c": 1
    }

    process = {
        "im_patch": im_patch,
        "aia_scale_aialev1": {}
    }

    client = drms.Client()

    request = client.export(q, method="url", protocol="fits", email=email, process=process)
    request.wait(timeout=3600, sleep=5)

    if not request.has_finished():
        raise drms.DrmsExportError(
            "Timed out after 3600s"
        )
    
    if not request.has_succeeded():
        raise drms.DrmsExportError(
            "Export failed or incomplete"
        )

    files = request.download(outdir)
    return [str(f) for f in files["download"]]

    




def get_level1_5_maps(aia_files):
    """
    Takes a list of AIA files and applies pointing correction and registration.
    This converts the level 1 data to level 1.5.

    Parameters
    ----------
    aia_files: list of strings
        List of paths to the AIA files to be processed.

    Returns
    ----------
    maps: list of sunpy.map.Map
        List of processed AIA maps after pointing correction and registration.
    """
    maps = []
    for file in aia_files:
        aia_map = Map(file)
        pointing_table = get_pointing_table("JSOC", time_range=(aia_map.date - 12 * u.h, aia_map.date + 12 * u.h))
        aia_map_updated_pointing = update_pointing(aia_map, pointing_table=pointing_table)
        aia_map_registered = register(aia_map_updated_pointing)
        maps.append(aia_map_registered)
    return maps