import numpy as np
import math
import csv
import matplotlib.pyplot as plt
import pandas as pd
import mplcursors
from scipy.stats import chi2

# Define lists to store results
r = []
el = []
az = []

class CVFilter:
    def __init__(self):
        self.Sf = np.zeros((6, 1))  # Filter state vector
        self.Pf = np.eye(6)  # Filter state covariance matrix
        self.Sp = np.zeros((6, 1))
        self.plant_noise = 20  # Plant noise covariance
        self.H = np.eye(3, 6)  # Measurement matrix
        self.R = np.eye(3)  # Measurement noise covariance
        self.Meas_Time = 0  # Measured time
        self.Z = np.zeros((3, 1))
        self.gate_threshold = 9.21  # 95% confidence interval for Chi-square distribution with 3 degrees of freedom

    def initialize_filter_state(self, x, y, z, vx, vy, vz, time):
        self.Sf = np.array([[x], [y], [z], [vx], [vy], [vz]])
        self.Meas_Time = time
        print("Initialized filter state:")
        print("Sf:", self.Sf)
        print("Pf:", self.Pf)
        
    def InitializeMeasurementForFiltering(self, x, y, z, vx, vy, vz, mt):
        self.Z = np.array([[x], [y], [z], [vx], [vy], [vz]])
        self.Meas_Time = mt

    def predict_step(self, current_time):
        dt = current_time - self.Meas_Time
        Phi = np.eye(6)
        Phi[0, 3] = dt
        Phi[1, 4] = dt
        Phi[2, 5] = dt
        Q = np.eye(6) * self.plant_noise
        self.Sp = np.dot(Phi, self.Sf)
        self.Pp = np.dot(np.dot(Phi, self.Pf), Phi.T) + Q
        print("Predicted filter state:")
        print("Sp:", self.Sp)
        print("Pp:", self.Pp)

    def update_step(self, Z):
        Inn = Z - np.dot(self.H, self.Sf)
        S = np.dot(self.H, np.dot(self.Pf, self.H.T)) + self.R
        K = np.dot(np.dot(self.Pf, self.H.T), np.linalg.inv(S))
        self.Sf = self.Sf + np.dot(K, Inn)
        self.Pf = np.dot(np.eye(6) - np.dot(K, self.H), self.Pf)
        print("Updated filter state:")
        print("Sf:", self.Sf)
        print("Pf:", self.Pf)

    def gating(self, Z):
        Inn = Z - np.dot(self.H, self.Sf)
        S = np.dot(self.H, np.dot(self.Pf, self.H.T)) + self.R
        d2 = np.dot(np.dot(Inn.T, np.linalg.inv(S)), Inn)
        return d2 < self.gate_threshold

def generate_hypotheses(clusters):
    hypotheses = []
    for cluster in clusters:
        cluster_id, measurements = cluster
        for measurement in measurements:
            hypotheses.append((cluster_id, measurement))
    return hypotheses

def compute_hypothesis_likelihood(hypothesis, filter_instance):
    cluster_id, measurement = hypothesis
    Z = np.array([[measurement[0]], [measurement[1]], [measurement[2]]])
    Inn = Z - np.dot(filter_instance.H, filter_instance.Sf[:3])
    S = np.dot(filter_instance.H, np.dot(filter_instance.Pf, filter_instance.H.T)) + filter_instance.R
    likelihood = np.exp(-0.5 * np.dot(np.dot(Inn.T, np.linalg.inv(S)), Inn))
    return likelihood

def jpda(clusters, filter_instance):
    hypotheses = generate_hypotheses(clusters)
    hypothesis_likelihoods = [compute_hypothesis_likelihood(h, filter_instance) for h in hypotheses]
    total_likelihood = sum(hypothesis_likelihoods)
    
    # Handle division by zero
    if total_likelihood == 0:
        marginal_probabilities = [1.0 / len(hypotheses)] * len(hypotheses)
    else:
        marginal_probabilities = [likelihood / total_likelihood for likelihood in hypothesis_likelihoods]

    best_hypothesis = hypotheses[np.argmax(marginal_probabilities)]
    return best_hypothesis

def sph2cart(az, el, r):
    x = r * np.cos(el * np.pi / 180) * np.sin(az * np.pi / 180)
    y = r * np.cos(el * np.pi / 180) * np.cos(az * np.pi / 180)
    z = r * np.sin(el * np.pi / 180)
    return x, y, z

def cart2sph(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    el = math.atan(z / np.sqrt(x**2 + y**2)) * 180 / 3.14
    az = math.atan(y / x)    

    if x > 0.0:                
        az = 3.14 / 2 - az
    else:
        az = 3 * 3.14 / 2 - az       
        
    az = az * 180 / 3.14 

    if az < 0.0:
        az = 360 + az
    
    if az > 360:
        az = az - 360   
      
    return r, az, el

def cart2sph2(x, y, z, filtered_values_csv):
    for i in range(len(filtered_values_csv)):
        r.append(np.sqrt(x[i]**2 + y[i]**2 + z[i]**2))
        el.append(math.atan(z[i] / np.sqrt(x[i]**2 + y[i]**2)) * 180 / 3.14)
        az.append(math.atan(y[i] / x[i]))
         
        if x[i] > 0.0:                
            az[i] = 3.14 / 2 - az[i]
        else:
            az[i] = 3 * 3.14 / 2 - az[i]       
        
        az[i] = az[i] * 180 / 3.14 

        if az[i] < 0.0:
            az[i] = 360 + az[i]
    
        if az[i] > 360:
            az[i] = az[i] - 360

    return r, az, el

def read_measurements_from_csv(file_path):
    measurements = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header if exists
        for row in reader:
            mr = float(row[10])  # MR column
            ma = float(row[11])  # MA column
            me = float(row[12])  # ME column
            mt = float(row[13])  # MT column
            x, y, z = sph2cart(ma, me, mr)
            measurements.append((x, y, z, mt))
    return measurements

def form_measurement_groups(measurements, max_time_diff=50):
    measurement_groups = []
    current_group = []
    base_time = measurements[0][3]
    
    for measurement in measurements:
        if measurement[3] - base_time <= max_time_diff:
            current_group.append(measurement)
        else:
            measurement_groups.append(current_group)
            current_group = [measurement]
            base_time = measurement[3]
    
    if current_group:
        measurement_groups.append(current_group)
        
    return measurement_groups

def chi_square_clustering(group, filter_instance):
    clusters = {}
    for i, measurement in enumerate(group):
        Z = np.array([[measurement[0]], [measurement[1]], [measurement[2]]])
        if filter_instance.gating(Z).item():
            cluster_id = int(i // 3)  # Example clustering logic
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(measurement)
    return [(cid, measurements) for cid, measurements in clusters.items()]

# Create an instance of the CVFilter class
kalman_filter = CVFilter()

# Define the path to your CSV file containing measurements
csv_file_path = 'ttk_52_test.csv'  # Provide the path to your CSV file

# Read measurements from CSV file
measurements = read_measurements_from_csv(csv_file_path)

# Form measurement groups based on time
measurement_groups = form_measurement_groups(measurements)

csv_file_predicted = "ttk_52_test.csv"
df_predicted = pd.read_csv(csv_file_predicted)
filtered_values_csv = df_predicted[['FT', 'FX', 'FY', 'FZ']].values
measured_values_csv = df_predicted[['MT', 'MR', 'MA', 'ME']].values

A = cart2sph2(filtered_values_csv[:, 1], filtered_values_csv[:, 2], filtered_values_csv[:, 3], filtered_values_csv)

# Initialize lists to store plot data
plot_times = []
plot_ranges = []
plot_azimuths = []
plot_elevations = []

# Iterate through each measurement group
for group in measurement_groups:
    if not group:
        continue
    
    # Initialize filter state with the first measurement
    if len(plot_times) == 0:
        first_measurement = group[0]
        kalman_filter.initialize_filter_state(first_measurement[0], first_measurement[1], first_measurement[2], 0, 0, 0, first_measurement[3])
        plot_times.append(first_measurement[3])
        plot_ranges.append(first_measurement[0])
        plot_azimuths.append(first_measurement[1])
        plot_elevations.append(first_measurement[2])
        continue

    clusters = chi_square_clustering(group, kalman_filter)
    
    if clusters:
        best_hypothesis = jpda(clusters, kalman_filter)
        cluster_id, best_measurement = best_hypothesis

        vx = (best_measurement[0] - plot_ranges[-1]) / (best_measurement[3] - plot_times[-1])
        vy = (best_measurement[1] - plot_azimuths[-1]) / (best_measurement[3] - plot_times[-1])
        vz = (best_measurement[2] - plot_elevations[-1]) / (best_measurement[3] - plot_times[-1])

        kalman_filter.predict_step(best_measurement[3])
        kalman_filter.update_step(np.array([[best_measurement[0]], [best_measurement[1]], [best_measurement[2]], [vx], [vy], [vz]]))

        plot_times.append(best_measurement[3])
        plot_ranges.append(best_measurement[0])
        plot_azimuths.append(best_measurement[1])
        plot_elevations.append(best_measurement[2])

# Plotting
fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

axs[0].plot(A[0], A[1], label='Predicted')
axs[0].scatter(measured_values_csv[:, 0], measured_values_csv[:, 2], label='Measured', s=10)
axs[0].set_ylabel('Azimuth (degrees)')
axs[0].legend()

axs[1].plot(A[0], A[2], label='Predicted')
axs[1].scatter(measured_values_csv[:, 0], measured_values_csv[:, 3], label='Measured', s=10)
axs[1].set_ylabel('Elevation (degrees)')
axs[1].legend()

axs[2].plot(A[0], A[1], label='Predicted')
axs[2].scatter(measured_values_csv[:, 0], measured_values_csv[:, 1], label='Measured', s=10)
axs[2].set_ylabel('Range (meters)')
axs[2].set_xlabel('Time')
axs[2].legend()

plt.show()
