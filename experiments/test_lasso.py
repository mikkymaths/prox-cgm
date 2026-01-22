"""
LASSO Regression Experiment
Matching Section 4 of the paper with UCI Housing dataset simulation
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from prox_cgm import ProxCGM, create_lasso_problem

def generate_housing_dataset(n_samples=506, n_features=13, noise_std=0.1, seed=42):
    """
    Generate synthetic data matching UCI Housing dataset dimensions
    (506 samples, 13 features as in paper)
    """
    np.random.seed(seed)
    
    # True sparse solution (only 5 features matter)
    x_true = np.zeros(n_features)
    important_features = np.random.choice(n_features, 5, replace=False)
    x_true[important_features] = np.random.randn(5) * 2.0
    
    # Design matrix with correlation structure (realistic for housing data)
    A = np.random.randn(n_samples, n_features)
    
    # Add some correlation between features
    for i in range(1, n_features):
        A[:, i] = 0.7 * A[:, i-1] + 0.3 * A[:, i]
    
    # Scale features differently (as in real datasets)
    scales = np.linspace(0.5, 2.0, n_features)
    A = A * scales
    
    # Normalize
    A = (A - A.mean(axis=0)) / (A.std(axis=0) + 1e-8)
    
    # Generate response with noise
    b = A @ x_true + noise_std * np.random.randn(n_samples)
    
    return A, b, x_true

def run_lasso_experiment(lam=0.1, max_iter=1000, save_plots=True):
    """
    Run LASSO experiment matching paper parameters
    """
    print("=" * 70)
    print("LASSO REGRESSION EXPERIMENT")
    print("Matching paper: Adaptive Proximal Conjugate Gradient Methods")
    print("=" * 70)
    
    # Generate data (UCI Housing dimensions: 506 × 13)
    A, b, x_true = generate_housing_dataset(n_samples=506, n_features=13)
    
    print(f"Dataset: {A.shape[0]} samples × {A.shape[1]} features")
    print(f"True sparsity: {np.sum(np.abs(x_true) > 1e-4)} non-zero coefficients")
    print(f"Regularization λ: {lam}")
    print("-" * 70)
    
    # Create problem functions
    f_func, f_grad, h_func, h_prox = create_lasso_problem(A, b, lam)
    
    # Initialize Prox-CGM with paper parameters
    solver = ProxCGM(
        max_iter=max_iter,
        tol=1e-6,
        eta=0.1,      # Restart threshold η
        gamma=1e-4,   # Sufficient decrease γ
        rho=0.5,      # Backtracking reduction
        verbose=True
    )
    
    # Initial point
    x0 = np.zeros(A.shape[1])
    
    # Run optimization
    print("\nRunning Prox-CGM...")
    x_opt = solver.fit(f_func, f_grad, h_func, h_prox, x0, alpha0=1.0)
    
    # Results
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    final_f = f_func(x_opt) + h_func(x_opt)
    sparsity = 100 * np.sum(np.abs(x_opt) < 1e-4) / len(x_opt)
    true_sparsity = 100 * np.sum(np.abs(x_true) < 1e-4) / len(x_true)
    
    print(f"Final objective F(x): {final_f:.6f}")
    print(f"Solution sparsity: {sparsity:.1f}% zeros")
    print(f"True sparsity: {true_sparsity:.1f}% zeros")
    print(f"Number of iterations: {len(solver.f_history)-1}")
    print(f"Number of restarts: {sum(solver.restart_flags)}")
    print(f"Average blending parameter θ: {np.mean(solver.theta_history):.3f}")
    
    # Calculate prediction metrics
    y_pred = A @ x_opt
    mse = np.mean((y_pred - b) ** 2)
    print(f"Mean squared error: {mse:.6f}")
    
    # Create results directory
    if save_plots:
        results_dir = Path(__file__).parent.parent / 'results'
        results_dir.mkdir(exist_ok=True)
        
        # Plot 1: Convergence
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        plt.plot(solver.f_history)
        plt.xlabel('Iteration')
        plt.ylabel('Objective F(x)')
        plt.title('Prox-CGM Convergence')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        # Plot 2: Solution comparison
        plt.subplot(1, 3, 2)
        plt.stem(range(len(x_true)), x_true, 'r-', markerfmt='ro', 
                basefmt=" ", label='True', linefmt='r-')
        plt.stem(range(len(x_opt)), x_opt, 'b-', markerfmt='bx', 
                basefmt=" ", label='Estimated', linefmt='b-')
        plt.xlabel('Feature index')
        plt.ylabel('Coefficient value')
        plt.title('Solution Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Algorithm parameters
        plt.subplot(1, 3, 3)
        iterations = range(len(solver.theta_history))
        plt.plot(iterations, solver.theta_history, 'g-', label='θ (blending)')
        plt.plot(iterations, solver.alpha_history, 'b-', label='Step size α')
        restart_indices = [i for i, r in enumerate(solver.restart_flags) if r == 1]
        if restart_indices:
            plt.scatter(restart_indices, [solver.theta_history[i] for i in restart_indices], 
                       color='red', s=50, zorder=5, label='Restarts')
        plt.xlabel('Iteration')
        plt.ylabel('Parameter value')
        plt.title('Algorithm Parameters')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(results_dir / 'lasso_results.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # Save data
        np.save(results_dir / 'x_opt.npy', x_opt)
        np.save(results_dir / 'x_true.npy', x_true)
        
        print(f"\nResults saved to: {results_dir}/")
    
    return solver, x_opt, x_true

def compare_methods():
    """
    Compare Prox-CGM with gradient descent (for demonstration)
    """
    print("\n" + "=" * 70)
    print("METHOD COMPARISON: Prox-CGM vs Proximal Gradient Descent")
    print("=" * 70)
    
    # Generate smaller dataset for quick comparison
    A, b, x_true = generate_housing_dataset(n_samples=100, n_features=20)
    lam = 0.1
    
    f_func, f_grad, h_func, h_prox = create_lasso_problem(A, b, lam)
    x0 = np.zeros(A.shape[1])
    
    # Prox-CGM
    solver_pcgm = ProxCGM(max_iter=200, tol=1e-6, verbose=False)
    x_pcgm = solver_pcgm.fit(f_func, f_grad, h_func, h_prox, x0)
    
    # Simple proximal gradient descent for comparison
    class ProximalGD:
        def __init__(self, max_iter=200, tol=1e-6):
            self.max_iter = max_iter
            self.tol = tol
            self.f_history = []
            
        def fit(self, f_func, f_grad, h_func, h_prox, x0, alpha=0.01):
            x = x0.copy()
            self.f_history = [f_func(x) + h_func(x)]
            
            for k in range(self.max_iter):
                gradient = f_grad(x)
                x_new = h_prox(x - alpha * gradient, alpha)
                
                self.f_history.append(f_func(x_new) + h_func(x_new))
                
                if np.linalg.norm(x_new - x) < self.tol:
                    break
                    
                x = x_new
                
            return x
    
    # Run both methods
    solver_gd = ProximalGD(max_iter=200, tol=1e-6)
    x_gd = solver_gd.fit(f_func, f_grad, h_func, h_prox, x0, alpha=0.01)
    
    # Plot comparison
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(solver_pcgm.f_history, 'b-', label='Prox-CGM', linewidth=2)
    plt.plot(solver_gd.f_history, 'r--', label='Proximal GD', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Objective F(x)')
    plt.title('Convergence Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    plt.subplot(1, 2, 2)
    bar_width = 0.35
    indices = np.arange(3)
    
    pcgm_vals = [
        solver_pcgm.f_history[-1],
        len(solver_pcgm.f_history)-1,
        np.sum(np.abs(x_pcgm) > 1e-4)
    ]
    
    gd_vals = [
        solver_gd.f_history[-1],
        len(solver_gd.f_history)-1,
        np.sum(np.abs(x_gd) > 1e-4)
    ]
    
    plt.bar(indices - bar_width/2, pcgm_vals, bar_width, label='Prox-CGM', alpha=0.8)
    plt.bar(indices + bar_width/2, gd_vals, bar_width, label='Proximal GD', alpha=0.8)
    
    plt.xticks(indices, ['Final F(x)', 'Iterations', 'Non-zeros'])
    plt.ylabel('Value')
    plt.title('Performance Metrics')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('results/comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\nComparison Complete!")
    print(f"Prox-CGM final F(x): {solver_pcgm.f_history[-1]:.6f}")
    print(f"Prox-GD final F(x): {solver_gd.f_history[-1]:.6f}")
    print(f"Prox-CGM iterations: {len(solver_pcgm.f_history)-1}")
    print(f"Prox-GD iterations: {len(solver_gd.f_history)-1}")

if __name__ == "__main__":
    # Run main experiment
    solver, x_opt, x_true = run_lasso_experiment(lam=0.1, max_iter=500)
    
    # Optional: Run comparison
    compare_response = input("\nRun method comparison? (y/n): ")
    if compare_response.lower() == 'y':
        compare_methods()
    
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE!")
    print("=" * 70)
