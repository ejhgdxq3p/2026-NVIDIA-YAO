"""
Physics-Informed Test Suite for LABS Solver
============================================
Team: Solo_Yao

This test suite implements rigorous verification of AI-generated code,
specifically designed to catch common hallucinations in physics simulations.

Tests:
1. Ground Truth Verification (N=3) - Known analytical result
2. Symmetry Invariance - Physical law: E(S) == E(-S) and E(S) == E(Reverse(S))
3. Non-Negativity - Energy must be >= 0
4. CPU/GPU Consistency - Both backends must agree
"""

import pytest
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from my_labs_library import calculate_energy_cpu, calculate_energy_gpu, HAS_GPU
except ImportError:
    # Fallback: try importing from current directory
    from my_labs_library import calculate_energy_cpu, calculate_energy_gpu, HAS_GPU


# =============================================================================
# TEST 1: Ground Truth Verification (N=3)
# =============================================================================
# The Risk: AI often messes up array indexing or sign conventions.
# The Fix: Assert against analytically computed ground truth.
#
# For sequence S = [1, 1, -1] with N=3:
#   C_1 = S[0]*S[1] + S[1]*S[2] = 1*1 + 1*(-1) = 0
#   C_2 = S[0]*S[2] = 1*(-1) = -1
#   E = C_1^2 + C_2^2 = 0 + 1 = 1
# =============================================================================

class TestGroundTruth:
    """Unit Test 1: Ground Truth Verification (N=3)"""
    
    def test_known_sequence_n3(self):
        """Test that calculate_energy([1, 1, -1]) equals exactly 1.0"""
        sequence = np.array([1, 1, -1], dtype=np.float32)
        energy = calculate_energy_cpu(sequence)
        assert energy == 1.0, f"Expected E=1.0, got E={energy}"
    
    def test_all_ones_n4(self):
        """Test all-ones sequence: maximum autocorrelation"""
        # S = [1, 1, 1, 1]
        # C_1 = 3, C_2 = 2, C_3 = 1
        # E = 9 + 4 + 1 = 14
        sequence = np.array([1, 1, 1, 1], dtype=np.float32)
        energy = calculate_energy_cpu(sequence)
        assert energy == 14.0, f"Expected E=14.0, got E={energy}"
    
    def test_alternating_n4(self):
        """Test alternating sequence: low autocorrelation"""
        # S = [1, -1, 1, -1]
        # C_1 = -1 + (-1) + (-1) = -3
        # C_2 = 1 + 1 = 2
        # C_3 = -1
        # E = 9 + 4 + 1 = 14
        sequence = np.array([1, -1, 1, -1], dtype=np.float32)
        energy = calculate_energy_cpu(sequence)
        assert energy == 14.0, f"Expected E=14.0, got E={energy}"


# =============================================================================
# TEST 2: Symmetry Invariance (Physical Law)
# =============================================================================
# The Risk: GPU parallelization often introduces subtle bugs in edge cases.
# The Fix: Property test asserting physical symmetries must hold.
#
# Symmetry 1: E(S) == E(-S)  (Negation Symmetry)
# Symmetry 2: E(S) == E(Reverse(S))  (Time Reversal Symmetry)
# =============================================================================

class TestSymmetryInvariance:
    """Unit Test 2: Symmetry Invariance (Physical Law)"""
    
    @pytest.mark.parametrize("seed", [42, 123, 456, 789, 1024])
    def test_negation_symmetry(self, seed):
        """E(S) must equal E(-S) for any sequence S"""
        np.random.seed(seed)
        N = 20
        S = np.random.choice([1, -1], size=N).astype(np.float32)
        
        E_original = calculate_energy_cpu(S)
        E_negated = calculate_energy_cpu(-S)
        
        assert E_original == E_negated, \
            f"Negation symmetry violated: E(S)={E_original} != E(-S)={E_negated}"
    
    @pytest.mark.parametrize("seed", [42, 123, 456, 789, 1024])
    def test_reversal_symmetry(self, seed):
        """E(S) must equal E(Reverse(S)) for any sequence S"""
        np.random.seed(seed)
        N = 20
        S = np.random.choice([1, -1], size=N).astype(np.float32)
        
        E_original = calculate_energy_cpu(S)
        E_reversed = calculate_energy_cpu(S[::-1].copy())
        
        assert E_original == E_reversed, \
            f"Reversal symmetry violated: E(S)={E_original} != E(Rev(S))={E_reversed}"


# =============================================================================
# TEST 3: Non-Negativity Bound
# =============================================================================
# Energy E = sum(C_k^2) is a sum of squares, must be >= 0
# =============================================================================

class TestPhysicalBounds:
    """Energy must satisfy physical constraints"""
    
    @pytest.mark.parametrize("N", [5, 10, 20, 50])
    def test_energy_non_negative(self, N):
        """Energy must be non-negative (sum of squares)"""
        np.random.seed(42)
        for _ in range(10):
            S = np.random.choice([1, -1], size=N).astype(np.float32)
            E = calculate_energy_cpu(S)
            assert E >= 0, f"Energy cannot be negative, got E={E}"


# =============================================================================
# TEST 4: CPU/GPU Consistency
# =============================================================================
# Both backends must produce identical results (within floating point tolerance)
# =============================================================================

class TestCPUGPUConsistency:
    """GPU kernel must match CPU reference implementation"""
    
    @pytest.mark.skipif(not HAS_GPU, reason="GPU not available")
    @pytest.mark.parametrize("N", [10, 20, 50])
    def test_cpu_gpu_match(self, N):
        """CPU and GPU must compute the same energy"""
        np.random.seed(42)
        S = np.random.choice([1, -1], size=N).astype(np.float32)
        
        E_cpu = calculate_energy_cpu(S)
        E_gpu = calculate_energy_gpu(S)
        
        # Allow small floating point tolerance
        assert abs(E_cpu - E_gpu) < 1e-3, \
            f"CPU/GPU mismatch: CPU={E_cpu}, GPU={E_gpu}"


# =============================================================================
# MAIN: Run all tests
# =============================================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
