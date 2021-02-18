from models.hmm_method import (
    split,
    create_hmm_model,
    extend,
    get_bounds,
    get_interval_means,
    parse_gpx,
    latlon_to_distance,
    derivate,
    filter_gaussian,
)
from matplotlib import pyplot as plt
import numpy as np


filename = "data/Power.gpx"


window_size = 1

# Gaussian filtering size (for 4 sigma)
fsize = 21
n = 120
# Maximal mean speed
max_speed = 80
# STD of speed in each state
sigma = 8
# Transition probability
p = 1e-8

color = np.array([[1.0, 0.0, 0.0]])

### GPX PARSING
## construct xml treee
gpx_data = parse_gpx(filename)

elevation = gpx_data["elevation"].values
timestamp = gpx_data["timestamp"].values
latitude = gpx_data["latitude"].values
longitude = gpx_data["longitude"].values
power = gpx_data["power"].values
hr = gpx_data["hr"].values


### GPX HANDLING (coputation of distance and speed)
## Computation of distances from start
distance = latlon_to_distance(latitude, longitude)

## Filtering
# Sliding window mean
speed = derivate(distance, timestamp, window_size)
speed = filter_gaussian(speed, fsize)

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

plt.plot(speed * 3.6, alpha=0.7, c="C1")
plt.bar(
    centers,
    interval_means,
    width=widths,
    color=interval_means.reshape(-1, 1) * color.reshape(1, -1) / np.max(interval_means),
)
plt.show()


energy = np.cumsum(power)
max_power = 600
sigma_power = 25
ghmm = create_hmm_model(n, p, sigma_power, max_power)

## Gathering of training paces
# Prediction of hidden states
paces = ghmm.predict(power.reshape(-1, 1))
paces = extend(paces, len(timestamp))

bounds = get_bounds(paces)

widths = bounds[1:] - bounds[:-1]
centers = (bounds[1:] + bounds[:-1]) * 0.5

interval_means = get_interval_means(energy, bounds, timestamp=timestamp)

fig, ax = plt.subplots(figsize=(12, 10))
ax.plot(power, alpha=0.7, c="C1")
ax.bar(
    centers,
    interval_means,
    width=widths,
    color=interval_means.reshape(-1, 1) * color.reshape(1, -1) / np.max(interval_means),
)
plt.show()


pulses = np.cumsum(hr)
max_hr = 210
sigma_hr = 10
ghmm = create_hmm_model(n, p, sigma_hr, max_hr)

## Gathering of training paces
# Prediction of hidden states
paces = ghmm.predict(hr.reshape(-1, 1))
paces = extend(paces, len(timestamp))

bounds = get_bounds(paces)

widths = bounds[1:] - bounds[:-1]
centers = (bounds[1:] + bounds[:-1]) * 0.5

interval_means = get_interval_means(pulses, bounds, timestamp=timestamp)

fig, ax = plt.subplots(figsize=(12, 10))
ax.plot(hr, alpha=0.7, c="C1")
ax.bar(
    centers,
    interval_means,
    width=widths,
    color=interval_means.reshape(-1, 1) * color.reshape(1, -1) / np.max(interval_means),
)
plt.show()
