"""
Adaptive Proximal Conjugate Gradient Method (Prox-CGM)
Implementation matching the paper exactly:
"Adaptive Proximal Conjugate Gradient Methods for Sparse and Nonsmooth High-Dimensional Optimization Problems"

Author: mikkymaths
Date: 2025
"""

import numpy as np
from typing import Callable, Tuple, List
import warnings

class ProxCGM:
    """
    Adaptive Proximal Conjugate Gradient Method for composite optimization:
    min_x F(x) = f(x) + h(x)
    
    Parameters match Algorithm 1 in the paper:
    -----------
    max_iter : int
        Maximum number of iterations (K in paper)
    tol : float
        Stopping tolerance for relative change
    eta : float
        Restart threshold η ∈ (0,1)
    gamma : float
        Sufficient decrease constant in line search (γ in Eq. 8)
    rho : float
        Backtracking reduction factor (typically 0.5)
    verbose : bool
        Print progress information
    """
    
    def __init__(self, max_iter: int = 1000, tol: float = 1e-6, 
                 eta: float = 0.1, gamma: float = 1e-4, 
                 rho: float = 0.5, verbose: bool = True):
        
        # Parameters from paper
        self.max_iter = max_iter
        self.tol = tol
        self.eta = eta  # Restart threshold
        self.gamma = gamma  # Sufficient decrease constant (Eq. 8)
        self.rho = rho  # Backtracking reduction
        self.verbose = verbose
        
        # History tracking
        self.f_history: List[float] = []
        self.x_history: List[np.ndarray] = []
        self.alpha_history: List[float] = []
        self.restart_flags: List[int] = []
        self.theta_history: List[float] = []  # Blending parameter history
        self.g_norm_history: List[float] = []  # Gradient norm history
        
    def proximal_gradient_mapping(self, x: np.ndarray, alpha: float, 
                                  f_grad: Callable, h_prox: Callable) -> np.ndarray:
        """
        Compute proximal gradient mapping G_α(x)
        Eq. in paper: G_α(x) = 1/α [x - prox_{αh}(x - α∇f(x))]
        
        Parameters:
        -----------
        x : np.ndarray
            Current point
        alpha : float
            Step size
        f_grad : Callable
            Returns ∇f(x)
        h_prox : Callable
            Returns prox_{αh}(v)
            
        Returns:
        --------
        G : np.ndarray
            Proximal gradient mapping
        """
        gradient = f_grad(x)
        v = x - alpha * gradient
        prox_point = h_prox(v, alpha)
        G = (x - prox_point) / alpha
        return G
    
    def compute_beta_fr(self, g_k: np.ndarray, g_km1: np.ndarray) -> float:
        """
        Compute Fletcher-Reeves coefficient
        β_k^{FR} = ||g_k||^2 / ||g_{k-1}||^2
        """
        norm_gk_sq = np.dot(g_k, g_k)
        norm_gkm1_sq = np.dot(g_km1, g_km1)
        
        if norm_gkm1_sq < 1e-12:
            return 0.0
        
        return norm_gk_sq / norm_gkm1_sq
    
    def compute_beta_pr(self, g_k: np.ndarray, g_km1: np.ndarray) -> float:
        """
        Compute Polak-Ribière coefficient
        β_k^{PR} = g_k^T (g_k - g_{k-1}) / ||g_{k-1}||^2
        """
        numerator = np.dot(g_k, g_k - g_km1)
        denominator = np.dot(g_km1, g_km1)
        
        if denominator < 1e-12:
            return 0.0
        
        return numerator / denominator
    
    def compute_blending_parameter(self, g_k: np.ndarray, g_km1: np.ndarray) -> float:
        """
        Compute blending parameter θ_k based on gradient alignment
        θ_k = min{1, max{0, (g_k^T g_{k-1})/(||g_k|| ||g_{k-1}||)}}
        
        Returns:
        --------
        theta : float
            Blending parameter in [0, 1]
        """
        norm_gk = np.linalg.norm(g_k)
        norm_gkm1 = np.linalg.norm(g_km1)
        
        if norm_gk < 1e-12 or norm_gkm1 < 1e-12:
            return 0.5  # Default middle value
        
        cos_angle = np.dot(g_k, g_km1) / (norm_gk * norm_gkm1)
        
        # Clip to [0, 1] as in paper
        theta = np.clip(cos_angle, 0.0, 1.0)
        
        return theta
    
    def backtracking_line_search(self, x: np.ndarray, d: np.ndarray, alpha_init: float,
                                 f_func: Callable, f_grad: Callable, 
                                 h_prox: Callable, h_func: Callable) -> Tuple[float, np.ndarray]:
        """
        Proximal backtracking line search satisfying Eq. (8):
        F(x_{k+1}) ≤ F(x_k) + γ α_k g_k^T d_k
        
        Parameters:
        -----------
        x : np.ndarray
            Current point
        d : np.ndarray
            Search direction
        alpha_init : float
            Initial step size
        f_func, f_grad, h_prox, h_func : Callable
            Problem functions
            
        Returns:
        --------
        alpha : float
            Accepted step size
        x_new : np.ndarray
            New point after proximal update
        """
        alpha = alpha_init
        g = self.proximal_gradient_mapping(x, alpha, f_grad, h_prox)
        
        # Current function value
        F_current = f_func(x) + h_func(x)
        
        # Maximum backtracking iterations
        max_backtrack = 20
        
        for _ in range(max_backtrack):
            # Candidate update: Eq. (5) in paper
            x_candidate = h_prox(x + alpha * d, alpha)
            F_candidate = f_func(x_candidate) + h_func(x_candidate)
            
            # Check sufficient decrease condition: Eq. (8)
            descent = np.dot(g, d)
            
            if F_candidate <= F_current + self.gamma * alpha * descent:
                return alpha, x_candidate
            
            # Reduce step size
            alpha *= self.rho
        
        # If backtracking fails, return smallest step size
        if self.verbose:
            warnings.warn(f"Line search failed to satisfy descent condition. Using alpha={alpha}")
        
        return alpha, h_prox(x + alpha * d, alpha)
    
    def fit(self, f_func: Callable, f_grad: Callable, 
            h_func: Callable, h_prox: Callable, 
            x0: np.ndarray, alpha0: float = 1.0) -> np.ndarray:
        """
        Main Prox-CGM algorithm matching Algorithm 1 in paper
        
        Parameters:
        -----------
        f_func : Callable
            Returns f(x) - smooth convex function
        f_grad : Callable
            Returns ∇f(x) - gradient of smooth part
        h_func : Callable
            Returns h(x) - nonsmooth convex function
        h_prox : Callable
            Returns prox_{αh}(v) - proximal operator
        x0 : np.ndarray
            Initial point
        alpha0 : float
            Initial step size
            
        Returns:
        --------
        x : np.ndarray
            Final solution
        """
        # Initialize
        x = x0.copy()
        n = len(x0)
        
        # Initial proximal gradient and direction
        g = self.proximal_gradient_mapping(x, alpha0, f_grad, h_prox)
        d = -g  # Initial direction: steepest descent
        
        # Store initial state
        self.x_history = [x.copy()]
        self.f_history = [f_func(x) + h_func(x)]
        self.alpha_history = [alpha0]
        self.restart_flags = [0]
        self.theta_history = [0.5]  # Initial blending parameter
        self.g_norm_history = [np.linalg.norm(g)]
        
        g_prev = g.copy()
        
        if self.verbose:
            print("=" * 60)
            print("Prox-CGM: Adaptive Proximal Conjugate Gradient Method")
            print("=" * 60)
            print(f"Problem dimension: {n}")
            print(f"Parameters: max_iter={self.max_iter}, tol={self.tol}")
            print(f"            η={self.eta}, γ={self.gamma}, ρ={self.rho}")
            print("-" * 60)
            print(f"Iter 0: F(x) = {self.f_history[-1]:.6e}, ||g|| = {self.g_norm_history[-1]:.3e}")
        
        # Main optimization loop
        for k in range(1, self.max_iter + 1):
            # Step 1: Backtracking line search
            alpha, x_new = self.backtracking_line_search(
                x, d, 
                alpha0 if len(self.alpha_history) == 1 else self.alpha_history[-1],
                f_func, f_grad, h_prox, h_func
            )
            
            # Step 2: Compute proximal gradient mapping at new point
            g_new = self.proximal_gradient_mapping(x_new, alpha, f_grad, h_prox)
            
            # Step 3: Compute FR and PR coefficients
            beta_fr = self.compute_beta_fr(g_new, g_prev)
            beta_pr = self.compute_beta_pr(g_new, g_prev)
            
            # Step 4: Compute blended coefficient β_k (Eq. 7)
            theta = self.compute_blending_parameter(g_new, g_prev)
            beta = theta * beta_pr + (1.0 - theta) * beta_fr
            
            # Step 5: Restart check (gradient-direction alignment)
            norm_g_new = np.linalg.norm(g_new)
            norm_d = np.linalg.norm(d)
            
            if norm_g_new < 1e-12 or norm_d < 1e-12:
                cos_angle = 0.0
            else:
                cos_angle = np.dot(g_new, d) / (norm_g_new * norm_d)
            
            # Apply restart if alignment is poor
            if cos_angle < self.eta:
                # Restart: reset to negative gradient direction
                d_new = -g_new
                restart_flag = 1
            else:
                # Standard conjugate gradient update: Eq. (6)
                d_new = -g_new + beta * d
                restart_flag = 0
            
            # Update variables for next iteration
            x = x_new
            g_prev = g_new.copy()
            d = d_new
            
            # Store history
            current_f = f_func(x) + h_func(x)
            self.x_history.append(x.copy())
            self.f_history.append(current_f)
            self.alpha_history.append(alpha)
            self.restart_flags.append(restart_flag)
            self.theta_history.append(theta)
            self.g_norm_history.append(norm_g_new)
            
            # Print progress
            if self.verbose and (k % 100 == 0 or k == 1):
                print(f"Iter {k:4d}: F(x) = {current_f:.6e}, "
                      f"α = {alpha:.2e}, θ = {theta:.3f}, "
                      f"||g|| = {norm_g_new:.2e}, "
                      f"Restart = {restart_flag}")
            
            # Check convergence criteria
            if k > 1:
                # Relative function value change
                f_change = abs(self.f_history[-2] - self.f_history[-1])
                f_rel_change = f_change / max(1.0, abs(self.f_history[-1]))
                
                # Gradient norm
                grad_norm = norm_g_new
                
                if f_rel_change < self.tol or grad_norm < self.tol:
                    if self.verbose:
                        print("-" * 60)
                        print(f"✓ Convergence achieved at iteration {k}")
                        print(f"  Final F(x) = {current_f:.6e}")
                        print(f"  Relative change = {f_rel_change:.2e}")
                        print(f"  Gradient norm = {grad_norm:.2e}")
                    break
        
        # Final summary
        if self.verbose:
            print("=" * 60)
            print("Optimization Complete")
            print("=" * 60)
            print(f"Total iterations: {len(self.f_history)-1}")
            print(f"Final objective: {self.f_history[-1]:.6e}")
            print(f"Total restarts: {sum(self.restart_flags)}")
            print(f"Average θ (blending): {np.mean(self.theta_history):.3f}")
            print("=" * 60)
        
        return x
    
    def get_convergence_report(self) -> dict:
        """
        Generate a convergence report
        
        Returns:
        --------
        report : dict
            Dictionary with convergence metrics
        """
        if len(self.f_history) < 2:
            return {"error": "No history available"}
        
        report = {
            "n_iterations": len(self.f_history) - 1,
            "final_objective": self.f_history[-1],
            "initial_objective": self.f_history[0],
            "total_decrease": self.f_history[0] - self.f_history[-1],
            "final_gradient_norm": self.g_norm_history[-1] if self.g_norm_history else None,
            "n_restarts": sum(self.restart_flags),
            "avg_step_size": np.mean(self.alpha_history) if self.alpha_history else None,
            "avg_theta": np.mean(self.theta_history) if self.theta_history else None,
            "converged": len(self.f_history) - 1 < self.max_iter
        }
        
        return report


# ============================================================================
# Helper functions for common problems (matching paper examples)
# ============================================================================

def create_lasso_problem(A: np.ndarray, b: np.ndarray, lam: float = 0.1):
    """
    Create LASSO problem functions: min_x 0.5||Ax - b||^2 + λ||x||_1
    
    Returns:
    --------
    f_func, f_grad, h_func, h_prox : Callable
        Problem functions for ProxCGM.fit()
    """
    m, n = A.shape
    
    def f_func(x):
        residual = A @ x - b
        return 0.5 * np.dot(residual, residual)
    
    def f_grad(x):
        return A.T @ (A @ x - b)
    
    def h_func(x):
        return lam * np.sum(np.abs(x))
    
    def h_prox(v, alpha):
        # Soft thresholding for L1 norm
        threshold = alpha * lam
        return np.sign(v) * np.maximum(np.abs(v) - threshold, 0.0)
    
    return f_func, f_grad, h_func, h_prox


def create_logistic_regression_problem(X: np.ndarray, y: np.ndarray, lam: float = 0.1):
    """
    Create sparse logistic regression problem functions
    min_x ∑ log(1 + exp(-y_i x_i^T x)) + λ||x||_1
    """
    m, n = X.shape
    
    # Ensure y is ±1
    y = np.where(y > 0, 1.0, -1.0)
    
    def sigmoid(z):
        # Stable sigmoid
        z = np.clip(z, -50, 50)
        return 1.0 / (1.0 + np.exp(-z))
    
    def f_func(x):
        z = y * (X @ x)
        # Log(1 + exp(-z)) = log(1 + exp(-z)) for numerical stability
        # Use log1p(exp(-z)) for z >= 0, and z + log1p(exp(-z)) for z < 0
        return np.sum(np.log1p(np.exp(-np.abs(z))) + np.maximum(-z, 0))
    
    def f_grad(x):
        z = y * (X @ x)
        p = sigmoid(-z)  # P(y=1|x)
        return -X.T @ (y * p)
    
    def h_func(x):
        return lam * np.sum(np.abs(x))
    
    def h_prox(v, alpha):
        # Same soft thresholding as LASSO
        threshold = alpha * lam
        return np.sign(v) * np.maximum(np.abs(v) - threshold, 0.0)
    
    return f_func, f_grad, h_func, h_prox


# ============================================================================
# Example usage
# ============================================================================

if __name__ == "__main__":
    print("Testing Prox-CGM implementation...")
    
    # Generate simple test problem
    np.random.seed(42)
    n, m = 100, 50
    A = np.random.randn(m, n) / np.sqrt(m)
    b = np.random.randn(m)
    
    # Create LASSO problem
    f_func, f_grad, h_func, h_prox = create_lasso_problem(A, b, lam=0.1)
    
    # Initialize solver with paper parameters
    solver = ProxCGM(
        max_iter=500,
        tol=1e-6,
        eta=0.1,      # Restart threshold from paper
        gamma=1e-4,   # Sufficient decrease constant (Eq. 8)
        rho=0.5,      # Backtracking reduction
        verbose=True
    )
    
    # Initial point
    x0 = np.zeros(n)
    
    # Run optimization
    x_opt = solver.fit(f_func, f_grad, h_func, h_prox, x0, alpha0=1.0)
    
    # Print results
    print(f"\nSolution sparsity: {100 * np.sum(np.abs(x_opt) < 1e-4) / len(x_opt):.1f}% zeros")
    print(f"Final ||x||_1: {np.sum(np.abs(x_opt)):.4f}")
    
    # Generate convergence report
    report = solver.get_convergence_report()
    print("\nConvergence Report:")
    for key, value in report.items():
        print(f"  {key}: {value}")
