import numpy as np
from scipy.optimize import minimize

class NelsonSiegelSvensson:
    """
    This class implements the Nelson-Siegel-Svensson (NSS) model for yield curve estimation.
    The NSS model extends the Nelson-Siegel model by adding a fourth term (beta3 and tau2) to
    capture additional curvature in the yield curve.
    """

    def __init__(self, beta0=0, beta1=0, beta2=0, beta3=0, tau1=1, tau2=1):
        """
        Initialize the model parameters.
        
        Parameters:
        - beta0: Long-term level of the yield curve.
        - beta1: Short-term component (slope).
        - beta2: Medium-term curvature.
        - beta3: Additional curvature for more flexibility.
        - tau1: Decay factor for beta1 and beta2 terms.
        - tau2: Decay factor for beta3 term.
        """
        self.beta0 = beta0
        self.beta1 = beta1
        self.beta2 = beta2
        self.beta3 = beta3
        self.tau1 = tau1
        self.tau2 = tau2

    def nss_curve(self, durations):
        """
        Calculate the yield curve using the NSS model.
        
        Parameters:
        - durations: Array of maturities (in years).
        
        Returns:
        - Estimated yields corresponding to the input durations.
        
        The NSS formula is:
        yield = beta0 
              + beta1 * ((1 - exp(-t/tau1)) / (t/tau1)) 
              + beta2 * (((1 - exp(-t/tau1)) / (t/tau1)) - exp(-t/tau1))
              + beta3 * (((1 - exp(-t/tau2)) / (t/tau2)) - exp(-t/tau2))
        """
        term1 = self.beta0
        term2 = self.beta1 * (1 - np.exp(-durations / self.tau1)) / (durations / self.tau1)
        term3 = self.beta2 * ((1 - np.exp(-durations / self.tau1)) / (durations / self.tau1) - np.exp(-durations / self.tau1))
        term4 = self.beta3 * ((1 - np.exp(-durations / self.tau2)) / (durations / self.tau2) - np.exp(-durations / self.tau2))
        return term1 + term2 + term3 + term4

    def fit(self, durations, yields):
        """
        Fit the NSS model to observed market yield data.
        
        Parameters:
        - durations: Array of observed maturities (in years).
        - yields: Observed yields corresponding to the maturities.
        
        Returns:
        - Optimization result with estimated parameters (beta0, beta1, beta2, beta3, tau1, tau2).
        
        The fit minimizes the squared differences between observed and model-estimated yields.
        """
        def loss_function(params):
            # Unpack parameters to compute NSS yield curve
            self.beta0, self.beta1, self.beta2, self.beta3, self.tau1, self.tau2 = params
            estimated_yields = self.nss_curve(durations)
            return np.sum((yields - estimated_yields) ** 2)  # Sum of squared errors

        # Initial guess for parameters (can be improved with prior knowledge)
        initial_guess = [0, 0, 0, 0, 1, 1]

        # Use numerical optimization (L-BFGS-B) to minimize the loss function
        result = minimize(loss_function, initial_guess, method='L-BFGS-B')

        # Store optimized parameters
        self.beta0, self.beta1, self.beta2, self.beta3, self.tau1, self.tau2 = result.x
        return result

    def estimate_yield(self, new_durations):
        """
        Estimate yields for new maturities using the fitted NSS model.
        
        Parameters:
        - new_durations: Single value or array of new maturities (in years).
        
        Returns:
        - Estimated yields for the input maturities.
        """
        return self.nss_curve(new_durations)

# Example usage: fitting the NSS model and estimating yields
if __name__ == "__main__":
    # Example data: Replace these with actual market data
    durations = np.array([
        0.14, 0.18, 0.21, 0.25, 0.30, 0.34, 0.38, 0.42, 0.46,
        0.52, 0.55, 0.63, 0.67, 0.71, 0.75, 0.80, 0.85, 1.01, 1.17
    ])
    yields = np.array([
        0.027538036, 0.02801406, 0.028209196, 0.028676369, 0.029172274,
        0.029361039, 0.029195596, 0.028792161, 0.027984544, 0.028495121,
        0.028824012, 0.028666049, 0.028555778, 0.028224236, 0.027810778,
        0.027161245, 0.02725115, 0.0270447, 0.026014551
    ])

    # Initialize the NSS model with default parameters
    nss_model = NelsonSiegelSvensson()

    # Fit the model to the observed data
    nss_model.fit(durations, yields)

    # Create an array of new durations to estimate yields for (from 0 to 5.5 years)
    fechas = np.arange(0.0, 5.5, 0.1)
    estimated_yields = []

    # Estimate yields for each duration in 'fechas'
    for duration in fechas:
        estimated_yield = nss_model.estimate_yield(duration)
        estimated_yields.append(estimated_yield)

    # Now, 'estimated_yields' contains the modeled yields for all desired durations
