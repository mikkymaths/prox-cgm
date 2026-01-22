"""
Unit tests for Prox-CGM
"""

import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from prox_cgm import ProxCGM, create_lasso_problem

def test_prox_cgm_initialization():
    """Test that ProxCGM initializes correctly"""
    solver = ProxCGM(max_iter=100, tol=1e-6, eta=0.1)
    
    assert solver.max_iter == 100
    assert solver.tol == 1e-6
    assert solver.eta == 0.1
    assert solver.gamma == 1e-4
    assert solver.rho == 0.5
    print("✓ Initialization test passed")

def test_lasso_problem_creation():
    """Test LASSO problem function creation"""
    A = np.random.randn(10, 5)
    b = np.random.randn(10)
    lam = 0.1
    
    f_func, f_grad, h_func, h_prox = create_lasso_problem(A, b, lam)
    
    # Test that functions are callable
    x_test = np.random.randn(5)
    f_val = f_func(x_test)
    grad = f_grad(x_test)
    h_val = h_func(x_test)
    prox = h_prox(x_test, 0.01)
    
    assert isinstance(f_val, float)
    assert grad.shape == (5,)
    assert isinstance(h_val, float)
    assert prox.shape == (5,)
    print("✓ LASSO problem creation test passed")

def test_small_optimization():
    """Test Prox-CGM on a tiny problem"""
    np.random.seed(42)
    
    # Very small problem for quick test
    A = np.array([[1.0, 0.5], [0.5, 1.0]])
    b = np.array([1.0, 2.0])
    lam = 0.01
    
    f_func, f_grad, h_func, h_prox = create_lasso_problem(A, b, lam)
    
    solver = ProxCGM(max_iter=10, tol=1e-6, eta=0.1, verbose=False)
    x0 = np.zeros(2)
    
    x_opt = solver.fit(f_func, f_grad, h_func, h_prox, x0)
    
    # Just check it runs without error
    assert x_opt.shape == (2,)
    assert len(solver.f_history) > 1
    print("✓ Small optimization test passed")

if __name__ == "__main__":
    print("Running Prox-CGM tests...")
    print("-" * 40)
    
    test_prox_cgm_initialization()
    test_lasso_problem_creation()
    test_small_optimization()
    
    print("-" * 40)
    print("All tests passed! ✅")
