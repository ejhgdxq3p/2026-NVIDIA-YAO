# Product Requirements Document (PRD)

**Project Name:** QAOA-Accelerated LABS Solver
**Team Name:** Solo_Yao
**GitHub Repository:** https://github.com/ejhgdxq3p/2026-NVIDIA-YAO

---

> **Note to Students:** > The questions and examples provided in the specific sections below are **prompts to guide your thinking**, not a rigid checklist. 
> * **Adaptability:** If a specific question doesn't fit your strategy, you may skip or adapt it.
> * **Depth:** You are encouraged to go beyond these examples. If there are other critical technical details relevant to your specific approach, please include them.
> * **Goal:** The objective is to convince the reader that you have a solid plan, not just to fill in boxes.

---

## 1. Team Roles & Responsibilities

| Role | Name | GitHub Handle | Discord Handle
| :--- | :--- | :--- | :--- |
| **Project Lead** (Architect) | YaoYu | ejhgdxq3p | YaoYu |
| **GPU Acceleration PIC** (Builder) | YaoYu | ejhgdxq3p | YaoYu |
| **Quality Assurance PIC** (Verifier) | YaoYu | ejhgdxq3p | YaoYu |
| **Technical Marketing PIC** (Storyteller) | YaoYu | ejhgdxq3p | YaoYu |

**Note:** As a solo agentic hacker, I am assuming all roles, utilizing AI agents to augment coding and verification tasks.

| Role | Focus Area |
| :--- | :--- |
| **Project Lead** | Architecture design & Vibe Coding orchestration. |
| **GPU Acceleration PIC** | Migrating workflow to Brev.dev & managing CUDA-Q backends. |
| **Quality Assurance PIC** | Unit testing physics constraints & verifying QAOA energy outputs. |
| **Technical Marketing PIC** | Benchmarking CPU vs. GPU performance & visualization. |

---

## 2. The Architecture
**Owner:** Project Lead

### Choice of Quantum Algorithm
* **Algorithm:** Quantum Approximate Optimization Algorithm (QAOA) with a standard p=1 depth to start, exploring p=2 if resources allow.

* **Motivation:**
    * **Physics-Inspired:** Unlike the tutorial's Counterdiabatic approach (which requires complex pulse engineering), QAOA uses an alternating operator ansatz ($U(H_C)U(H_M)$) that intuitively maps to the LABS problem structure.
    * **Optimization Control:** QAOA allows us to treat the quantum circuit as a variational ansatz, optimizing parameters ($\beta, \gamma$) classically. This hybrid loop is ideal for demonstrating GPU acceleration on both the quantum simulation (state vector evolution) and the classical parameter update.

### Literature Review
* **Reference:** Farhi, E., Goldstone, J., & Gutmann, S. (2014). A Quantum Approximate Optimization Algorithm. arXiv:1411.4028
* **Relevance:** This is the foundational paper for QAOA. It establishes the convergence guarantees as depth $p \to \infty$. For LABS, we aim to demonstrate that even low-depth QAOA ($p=1$) provides a better seed distribution than random guessing (classical blind search).

---

## 3. The Acceleration Strategy
**Owner:** GPU Acceleration PIC

### Quantum Acceleration (CUDA-Q)
* **Strategy:**
    * We will switch the CUDA-Q backend from the CPU-based `qpp` to the NVIDIA GPU-accelerated `nvidia` backend (`cuQuantum`).
    * This allows for massive parallelization of the state vector simulation, which is the bottleneck when $N$ increases (exponential memory growth).

### Classical Acceleration (MTS)
* **Strategy:**
    * **Batch Evaluation:** The current CPU implementation evaluates neighbor energies sequentially. We will use `CuPy` (a drop-in replacement for NumPy on GPUs) to vectorize the energy calculation.
    * We will compute the autocorrelation energy of thousands of candidate sequences in parallel on the GPU tensor cores, drastically reducing the time-per-generation in the Tabu Search.

### Hardware Targets
* **Dev Environment:** Google Colab / qBraid (CPU) for initial logic verification.
* **Production Environment:** Brev.dev instance with NVIDIA L4 or A10G for performance benchmarking (Cost-efficient for Hackathon budget).

---

## 4. The Verification Plan
**Owner:** Quality Assurance PIC

### Unit Testing Strategy
* **Framework:** `pytest`
* **AI Hallucination Guardrails:**
    * We will strictly enforce that any AI-generated QAOA kernel must pass a "Sanity Check" on $N=3$ before integration.
    * We will implement a `check_energy_bounds` function to ensure no returned energy value violates mathematical limits (LABS energy cannot be negative).

### Core Correctness Checks
* **Check 1 (Ground Truth N=3):**
    * For $N=3$, the known optimal sequence is `[1, 1, -1]` (Energy=1.0). We will assert `calculate_energy([1, 1, -1]) == 1.0` in our test suite.
* **Check 2 (Symmetry Invariance):**
    * We will verify that the GPU-accelerated energy function returns the same value for a sequence $S$ and its reversed counterpart $S_{rev}$. Assert `energy_gpu(S) == energy_gpu(S.reversed())`.

---

## 5. Execution Strategy & Success Metrics
**Owner:** Technical Marketing PIC

### Agentic Workflow
* **Plan:**
    * **Agent 1 (Coder):** Uses the `cudaq` documentation to generate the QAOA ansatz kernel.
    * **Agent 2 (Reviewer):** Reviews the generated kernel against the LABS Hamiltonian definition to ensure the Cost Hamiltonian ($H_C$) matches the problem definition (autocorrelation).
    * **Human:** Orchestrates the Brev environment and runs the final benchmark.

### Success Metrics
* **Metric 1 (Correctness):** The Hybrid QAOA+MTS solver must find the known optimal solution for $N=20$ (Energy=34) consistently.
* **Metric 2 (Quantum Advantage):** The QAOA-seeded initial population should have a lower mean energy than a purely random population.
* **Metric 3 (Speedup):** The GPU-accelerated energy calculation (`CuPy`) should be at least 5x faster than the CPU NumPy implementation for large batch sizes ($10^4$ sequences).

### Visualization Plan
* **Plot 1:** Histogram of Initial Energies: Comparing "Random Seed" vs. "QAOA Seed" distributions. (Expecting QAOA to shift left/lower energy).
* **Plot 2:** Runtime Scaling: Time per iteration vs. Sequence Length ($N$) for CPU vs. GPU backends.

---

## 6. Resource Management Plan
**Owner:** GPU Acceleration PIC 

* **Plan:**
    * **Development:** All functional coding and unit testing will be done on CPU (Colab/Local) to save credits.
    * **Brev Usage:** We will only launch the Brev instance for the final "Performance Run."
    * **Shutdown Protocol:** I will set a phone alarm for 2 hours after launching the instance to ensure I manually shut it down, preventing "Zombie Instance" credit burn.
    * **Budget Allocation:** Estimated 3 hours on L4 GPU (~$3.00) leaving ample buffer.
