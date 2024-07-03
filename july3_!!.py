import numpy as np
import math
import csv
import matplotlib.pyplot as plt
import pandas as pd
import mplcursors
from scipy.spatial.distance import mahalanobis
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

def generate_hypotheses(clusters, targets):
    hypotheses = []
    for cluster in clusters:
        for target in targets:
            hypotheses.append((cluster, target))
    return hypotheses

def compute_hypothesis_likelihood(hypothesis, filter_instance):
    cluster, target = hypothesis
    Z = np.array([[cluster[0]], [cluster[1]], [cluster[2]]])
    Inn = Z - np.dot(filter_instance.H, target)
    S = np.dot(filter_instance.H, np.dot(filter_instance.Pf, filter_instance.H.T)) + filter_instance.R
    likelihood = np.exp(-0.5 * np.dot(np.dot(Inn.T, np.linalg.inv(S)), Inn))
    return likelihood

def jpda(clusters, targets, filter_instance):
    hypotheses = generate_hypotheses(clusters, targets)
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
    cluster_id = 0
    for i, measurement in enumerate(group):
        Z = np.array([[measurement[0]], [measurement[1]], [measurement[2]]])
        if filter_instance.gating(Z).item():
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(measurement)
        else:
            cluster_id += 1
            clusters[cluster_id] = [measurement]
    return clusters

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

# Lists to store the data for plotting
times = []
ranges = []
azimuths = []
elevations = []

# Initialize the filter state with the first measurement
x, y, z, time = measurement_groups[0][0]
vx, vy, vz = 0, 0, 0  # Initial velocity is assumed to be zero
kalman_filter.initialize_filter_state(x, y, z, vx, vy, vz, time)

for group in measurement_groups:
    clusters = chi_square_clustering(group, kalman_filter)
    
    # Assuming the first measurement in the group is used to initialize
    Z = np.array([[group[0][0]], [group[0][1]], [group[0][2]]])
    kalman_filter.update_step(Z)
    
    for cluster_id, cluster_measurements in clusters.items():
        if cluster_measurements:
            # Use JPDA to find the most associated measurement for each cluster
            best_hypothesis = jpda(cluster_measurements, [kalman_filter.Sf], kalman_filter)
            best_cluster, best_target = best_hypothesis
            
            # Calculate the velocity for the cluster's most associated measurement
            if len(cluster_measurements) > 1:
                time_diff = cluster_measurements[-1][3] - cluster_measurements[0][3]
                vx = (cluster_measurements[-1][0] - cluster_measurements[0][0]) / time_diff
                vy = (cluster_measurements[-1][1] - cluster_measurements[0][1]) / time_diff
                vz = (cluster_measurements[-1][2] - cluster_measurements[0][2]) / time_diff
            else:
                vx, vy, vz = 0, 0, 0

            kalman_filter.InitializeMeasurementForFiltering(
                best_cluster[0], best_cluster[1], best_cluster[2], vx, vy, vz, cluster_measurements[0][3])
            kalman_filter.update_step(np.array([[best_cluster[0]], [best_cluster[1]], [best_cluster[2]]]))

    times.append(time)
    ranges.append(r)
    azimuths.append(az)
    elevations.append(el)

# Plotting results
plt.figure(figsize=(10, 6))

plt.subplot(3, 1, 1)
plt.plot(times, ranges, label='Range')
plt.xlabel('Time')
plt.ylabel('Range')
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(times, azimuths, label='Azimuth')
plt.xlabel('Time')
plt.ylabel('Azimuth')
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(times, elevations, label='Elevation')
plt.xlabel('Time')
plt.ylabel('Elevation')
plt.grid(True)

mplcursors.cursor(hover=True)
plt.tight_layout()
plt.show()
