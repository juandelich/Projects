import numpy as np
import matplotlib.pyplot as plt

# Definimos parametros
P = np.array([[0.99, 0.01,0,0],  # Transition Matrix
              [0,0,1,0],
              [0, 0,0.99,0.01],
              [1,0,0,0]])
n_steps = 1000  # Numero de pasos

# Define return distributions
state_means = [0.05, -0.75,0.0,0.75]  # Retornos de cada estado
state_stds = [0.02, 0.0,0.0,0.0]    # STD de cada estado


current_state = 0  # Comienza en el estado 1 
states = [current_state]
returns = []

# Simulate Markov chain and bond returns
for _ in range(n_steps):
    next_state = np.random.choice([0, 1,2,3], p=P[current_state])
    states.append(next_state)
    current_state = next_state
    
    # Generate return based on current state
    state_return = np.random.normal(state_means[current_state], state_stds[current_state])
    returns.append(state_return)


plt.figure(figsize=(12, 6))
plt.plot(returns, label="Bond Returns", alpha=0.7)
plt.axhline(y=state_means[0], color="green", linestyle="--", label="Mean Return (State 1)")
plt.axhline(y=state_means[1], color="red", linestyle="--", label="Default Action Return (State 2)")
plt.axhline(y=state_means[2], color="red", linestyle="--", label="Default Return (State 3)")
plt.axhline(y=state_means[3], color="green", linestyle="--", label="Default Exit Return (State 4)")
plt.title("Simulated Bond Returns Based on Markov States")
plt.xlabel("Time Step")
plt.ylabel("Return")
plt.legend()
plt.grid()
plt.show()

print(f"Average return: {np.mean(returns):.2%}")
print(f"Volatility: {np.std(returns):.2%}")
