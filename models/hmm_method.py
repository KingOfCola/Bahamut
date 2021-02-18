import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import time
import numpy as np
from hmmlearn.hmm import GaussianHMM
import pandas as pd

NAMESPACES = {
    "xsi": "http://www.topografix.com/GPX/1/1",
    "gpxtpx": "http://www.garmin.com/xmlschemas/TrackPointExtension/v1",
}


def split(file, **kwargs):
    ### PARAMETERS
    ## gpx file location

    ## Filtering parameters
    # Sliding window size
    window_size = kwargs.get("window_size", 1)

    # Gaussian filtering size (for 4 sigma)
    fsize = kwargs.get("fsize", 21)

    ## Model parameters
    # Number of hidden states
    n = kwargs.get("n", 120)
    # Maximal mean speed
    max_speed = kwargs.get("max_speed", 80)
    # STD of speed in each state
    sigma = kwargs.get("sigma", 8)
    # Transition probability
    p = kwargs.get("p", 1e-8)

    ## Plot
    # activate plots
    intermediate_plot = kwargs.get("intermediate_plot", False)
    plot = kwargs.get("plot", True)

    # Bar plot color
    color = kwargs.get("color", np.array([[1.0, 0.0, 0.0]]))

    ### GPX PARSING
    ## construct xml treee
    gpx_data = parse_gpx(file)

    elevation = gpx_data["elevation"]
    timestamp = gpx_data["timestamp"]
    latitude = gpx_data["latitude"]
    longitude = gpx_data["longitude"]

    ## plots
    if intermediate_plot:
        # Trace plot
        plt.plot(longitude, latitude)
        plt.show()

        # elevation plot
        plt.plot(timestamp, elevation)
        plt.show()

    ### GPX HANDLING (coputation of distance and speed)
    ## Computation of distances from start
    distance = latlon_to_distance(latitude, longitude)

    if intermediate_plot:
        plt.plot(timestamp[:-1], d * 3.6 / np.float_(timestamp[1:] - timestamp[:-1]))
        plt.show()

        plt.plot(timestamp, distance)
        plt.show()

    ## Filtering
    # Sliding window mean
    speed = derivate(distance, timestamp, window_size)
    speed = filter_gaussian(speed, fsize)

    if intermediate_plot:
        plt.plot(timestamp[:-window_size], speed * 3.6)
        plt.show()

    ### GAUSSIAN HIDDEN MARKOV MODEL
    ## Model initialization
    # Gaussian HMM creation
    ghmm = create_hmm_model(n, p, sigma, max_speed)

    ## Gathering of training paces
    # Prediction of hidden states
    paces = ghmm.predict((speed * 3.6).reshape(-1, 1))
    paces = extend(paces, len(timestamp))

    bounds = get_bounds(paces)

    widths = bounds[1:] - bounds[:-1]
    centers = (bounds[1:] + bounds[:-1]) * 0.5

    interval_means = get_interval_means(distance, bounds, timestamp=timestamp) * 3.6

    if plot:
        plt.plot(speed * 3.6, alpha=0.7, c="C1")
        plt.bar(
            centers,
            interval_means,
            width=widths,
            color=interval_means.reshape(-1, 1)
            * color.reshape(1, -1)
            / np.max(interval_means),
        )
        plt.show()

    df = pd.DataFrame(
        {
            "latitude": latitude,
            "longitude": longitude,
            "timestamp": timestamp,
            "elevation": elevation,
            "distance": distance,
            "paces": paces,
            "interval": get_index(paces),
        }
    )

    df = df.astype({"paces": "int16", "interval": "int16"})
    return df


def latlon_to_distance(latitude, longitude):
    ### GPX HANDLING (coputation of distance and speed)
    ## Computation of distances
    # Angles as radians
    latitude_rad = latitude * np.pi / 180
    longitude_rad = longitude * np.pi / 180

    # Earth ray in meters
    R = 6371000

    # Computation of geodesic angles
    C = np.cos(latitude_rad[1:]) * np.cos(latitude_rad[:-1]) * (
        np.cos(longitude_rad[1:] - longitude_rad[:-1])
    ) + np.sin(latitude_rad[:-1]) * np.sin(latitude_rad[1:])

    # Computation of geodesic distances in meters (distance between two successive points)
    d = R * np.arccos(C * ((C > -1) & (C < 1)) - 1 * (C <= -1) + 1 * (C >= 1))

    # Distance form start
    return np.concatenate([[0], np.cumsum(d)])


def parse_gpx(file):
    ### GPX PARSING
    ## construct xml treee
    gpx = ET.parse(file)
    gpx_root = gpx.getroot()

    ## Gathering
    # Initiate interesting data
    points = []

    # Crawl through tree and gather interesting data
    for trkseg in gpx_root.findall("xsi:trk/xsi:trkseg", NAMESPACES):
        for trkpt in trkseg:
            data = {}
            data["latitude"] = float(trkpt.get("lat"))
            data["longitude"] = float(trkpt.get("lon"))
            data["elevation"] = float(trkpt.find("xsi:ele", NAMESPACES).text)
            data["timestamp"] = np.datetime64(trkpt.find("xsi:time", NAMESPACES).text)

            extensions = trkpt.find("xsi:extensions", NAMESPACES)

            if extensions is not None:
                power = extensions.find("xsi:power", NAMESPACES)
                if power is not None:
                    data["power"] = float(power.text)

                tkptExt = extensions.find("gpxtpx:TrackPointExtension", NAMESPACES)
                if tkptExt is not None:
                    hr = tkptExt.find("gpxtpx:hr", NAMESPACES)
                    if hr is not None:
                        data["hr"] = float(hr.text)

                    cadence = tkptExt.find("gpxtpx:cad", NAMESPACES)
                    if cadence is not None:
                        data["cad"] = float(hr.text)

            points.append(data)

    # numpy-zation
    return pd.DataFrame(points)


def derivate(sig, timestamp, step):
    # Sliding window mean
    return (sig[step:] - sig[:-step]) / (
        (timestamp[step:] - timestamp[:-step]) / np.timedelta64(1, "s")
    )


def filter_gaussian(sig, fsize):
    # Gaussian filter
    X = np.linspace(-4, 4, fsize)
    filt = np.exp(-(X ** 2) / 2)
    filt /= filt.sum()

    return np.convolve(sig, filt, mode="same")


def get_bounds(states):
    """
    Gets the bounds of each constant state sequence in the input sequence. 
    Returns the indices of changes in the states sequence: 
    {0 <= i <= len(states) s.t. states[i-1] != states[i]}
    with 0 and len(states) by default.
    """
    # Gathering of mean pace and location of sgements
    bounds = [0]
    left = 0
    x_left = states[0]

    for i, x in enumerate(states):
        if x != x_left:
            bounds.append(i)

            left = i
            x_left = x

    bounds.append(len(states))

    # numpy-zation
    return np.array(bounds)


def get_index(states):
    """
    Gets the bounds of each constant state sequence in the input sequence. 
    Returns the indices of changes in the states sequence: 
    {0 <= i <= len(states) s.t. states[i-1] != states[i]}
    with 0 and len(states) by default.
    """
    # Gathering of mean pace and location of sgements
    index = np.zeros(len(states))
    n = 0
    x_left = states[0]

    for i, x in enumerate(states[1:], 1):
        if x != x_left:
            n += 1
            x_left = x
        index[i] = n

    # numpy-zation
    return index


def create_hmm_model(n, p, sigma, max_speed):
    """
    Crates a new hidden markov model with given parameters:
    - n: number of hidden states.
    - p: probability of exiting a state.
    - sigma: std of speed emissions in current state.
    - max_speed: maximal speed encoded in hidden states.
    """
    ghmm = GaussianHMM(n_components=n, covariance_type="diag")

    # Model parameters initiatlization
    MEANS = np.linspace(0, max_speed, n).reshape(-1, 1)
    COVARS = np.arange(n, n * 2).reshape(-1, 1) * sigma ** 2 / n / 2
    PRIORS = np.ones(n) / n
    TRANSMAT = np.ones((n, n)) * p / (n - 1)
    TRANSMAT[np.arange(n), np.arange(n)] = 1 - p

    ghmm.means_ = MEANS
    ghmm.covars_ = COVARS
    ghmm.startprob_ = PRIORS
    ghmm.transmat_ = TRANSMAT

    return ghmm


def extend(signal, n, fill=None):
    """
    Fills the signal with a value (default its last value) so that it contins n values
    """
    if fill == None:
        fill = signal[-1]

    return np.concatenate([np.array(signal), [fill for _ in range(n - len(signal))]])


def get_default_timestamp(timestamp, sig):
    if timestamp is None:
        timestamp = np.arange(len(sig))

    timestamp = extend(timestamp, len(sig))
    return timestamp


def get_interval_means(integral, bounds, timestamp=None):
    """
    Computes the means of the signal on the intervals defined by the bounds.
    - integral: integral of the signal to average
    - bounds: boundaries of the intervals
    - timestamp: timestamp (for unequal sampling frequency)
    """
    integral = extend(integral, len(integral) + 1)
    timestamp = get_default_timestamp(timestamp, integral)
    return get_partial_integrals(bounds, integral) / (
        get_partial_integrals(bounds, timestamp) / np.timedelta64(1, "s")
    )


def get_partial_integrals(bounds, integral):
    """
    Computes the partial integrals of a signal on bounded intervals.
    - bounds: boundaries of the intervals
    - integral: integral of the signal to partially integrate
    """
    return integral[bounds[1:]] - integral[bounds[:-1]]


def get_interval_characteristics(distance, bounds, timestamp=None):
    """
    Gathers the characteristics of the segments of the track.
    """
    timestamp = get_default_timestamp(timestamp, integral)
    durations = get_partial_integrals(bounds, timestamp)
    distances = get_partial_integrals(bounds, distance)

    return pd.DataFrame(
        {"distance": distances, "duration": durations, "start": timestamp[:-1]}
    )

