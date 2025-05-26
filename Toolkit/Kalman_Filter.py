import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pykalman import KalmanFilter

# Function to load data from Excel
def load_data(file_path):
    df = pd.read_excel(file_path, parse_dates=['Date'])
    df.set_index('Date', inplace=True)
    return df

# Function to resample data to weekly or monthly returns
def resample_returns(data, frequency):
    return data['Returns'].resample(frequency).sum()

# Function to initialize and run the Kalman filter
def run_kalman_filter(returns, n_states=2):
    n = len(returns)

    # Initialize transition and observation matrices for a Gaussian state-space model
    transition_matrix = np.eye(n_states)  # Identity for simplicity
    observation_matrix = np.random.rand(1, n_states)  # Random initial guess

    # Initialize the Kalman Filter
    kf = KalmanFilter(
        transition_matrices=transition_matrix,
        observation_matrices=observation_matrix,
        initial_state_mean=np.zeros(n_states),
        initial_state_covariance=np.eye(n_states),
        observation_covariance=np.eye(1),
        transition_covariance=np.eye(n_states) * 0.01  # Small process noise
    )

    # Fit the filter and smooth the state probabilities
    states_mean, states_covariance = kf.filter(returns)
    smoothed_state_means, smoothed_state_covariances = kf.smooth(returns)

    return smoothed_state_means, states_mean, transition_matrix, observation_matrix

# Function to calculate mean returns for each state
def calculate_state_means(returns, smoothed_state_means):
    state_means = {}
    most_likely_states = np.argmax(smoothed_state_means, axis=1)
    for state in range(smoothed_state_means.shape[1]):
        state_returns = returns[most_likely_states == state]
        state_means[f"State {state+1}"] = state_returns.mean() if len(state_returns) > 0 else np.nan
    return state_means

# Function to infer transition probabilities from data
def infer_transition_matrix(most_likely_states, n_states):
    transition_counts = np.zeros((n_states, n_states))
    for i in range(len(most_likely_states) - 1):
        current_state = most_likely_states[i]
        next_state = most_likely_states[i + 1]
        transition_counts[current_state, next_state] += 1
    
    # Normalize rows to convert counts to probabilities
    transition_matrix = transition_counts / transition_counts.sum(axis=1, keepdims=True)
    return transition_matrix

# Function to plot the state probabilities
def plot_states(dates, smoothed_state_means):
    plt.figure(figsize=(12, 6))
    for state_idx in range(smoothed_state_means.shape[1]):
        plt.plot(dates, smoothed_state_means[:, state_idx], label=f"State {state_idx+1}")

    plt.title("Smoothed State Probabilities")
    plt.xlabel("Date")
    plt.ylabel("Probability")
    plt.legend()
    plt.grid()
    plt.show()

# Function to plot the transition matrix
def plot_transition_matrix(transition_matrix):
    plt.figure(figsize=(8, 6))
    plt.imshow(transition_matrix, cmap="viridis", aspect="auto")
    plt.colorbar(label="Transition Probability")
    plt.title("Inferred Transition Matrix")
    plt.xlabel("To State")
    plt.ylabel("From State")
    plt.xticks(range(transition_matrix.shape[0]))
    plt.yticks(range(transition_matrix.shape[1]))
    plt.grid()
    plt.show()

# Function to print the transition matrix
def print_transition_matrix(transition_matrix):
    print("Transition Matrix:")
    print(pd.DataFrame(transition_matrix, columns=[f"State {i+1}" for i in range(transition_matrix.shape[0])],
                      index=[f"State {i+1}" for i in range(transition_matrix.shape[1])]).round(6))

# Function to plot the observation matrix
def plot_observation_matrix(observation_matrix):
    plt.figure(figsize=(8, 6))
    plt.imshow(observation_matrix, cmap="plasma", aspect="auto")
    plt.colorbar(label="Observation Coefficient")
    plt.title("Observation Matrix")
    plt.xlabel("State")
    plt.ylabel("Observation Dimension")
    plt.grid()
    plt.show()

# Function to plot the most likely state at each time step
def plot_most_likely_states(dates, smoothed_state_means):
    most_likely_states = np.argmax(smoothed_state_means, axis=1)
    plt.figure(figsize=(12, 6))
    plt.plot(dates, most_likely_states, marker="o", linestyle="-", label="Estado Más Probable")
    plt.title("Estado más probable en cada momento del Tiempo")
    plt.xlabel("Date")
    plt.ylabel("Estado")
    plt.yticks(range(smoothed_state_means.shape[1]), [f"Estado {i+1}" for i in range(smoothed_state_means.shape[1])])
    plt.grid()
    plt.legend()
    plt.show()

# Function to plot the filtered state means (raw estimates)
def plot_filtered_states(dates, states_mean):
    plt.figure(figsize=(12, 6))
    for state_idx in range(states_mean.shape[1]):
        plt.plot(dates, states_mean[:, state_idx], label=f"State {state_idx+1}")

    plt.title("Filtered State Means")
    plt.xlabel("Date")
    plt.ylabel("State Value")
    plt.legend()
    plt.grid()
    plt.show()

# Function to plot monthly returns
def plot_monthly_returns(monthly_returns):
    plt.figure(figsize=(12, 6))
    plt.plot(monthly_returns.index, monthly_returns.values, marker="o", linestyle="-", label="Retornos Mensuales En USD(Corregido)")
    plt.title("Retornos Mensuales Merval")
    plt.xlabel("Fecha")
    plt.ylabel("Retornos")
    plt.grid()
    plt.legend()
    plt.show()


def calculate_state_std(returns, smoothed_state_means):
    state_stds = {}
    most_likely_states = np.argmax(smoothed_state_means, axis=1)
    for state in range(smoothed_state_means.shape[1]):
        state_returns = returns[most_likely_states == state]
        state_stds[f"State {state+1}"] = state_returns.std() if len(state_returns) > 0 else np.nan
    return state_stds
