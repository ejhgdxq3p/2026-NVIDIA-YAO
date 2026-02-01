# Team Solo_Yao: GPU-Native Thermodynamic Solver
**NVIDIA iQuHACK 2026 Final Submission**

---

##  Executive Summary

We present a **GPU-Accelerated Parallel Tempering Solver** for the LABS problem.
Moving beyond simple hill-climbing, our approach treats the optimization landscape as a physical system, utilizing **50,000 parallel GPU agents** to escape local minima.

**Key Achievements:**
*   **4.19+ Merit Factor** for N=60 (beating standard Baseline ~3.5).
*   **830x Speedup** vs CPU Baseline using Custom CUDA Kernels.
*   **"The Showdown"**: Validated our champion model against both Classical Hill Climbing and a experimental Simulated Bifurcation solver.

---

##  Repository Structure

*   **Final_Submission_Solo_Yao.ipynb**: **The Main Entry Point**.
    *   **Env Setup:** Auto-detects specific GPU capabilities and compiles optimal kernels.
    *   **Exp 1 (Throughput):** Log-scale speedup benchmark (=20..100$).
    *   **Exp 2 (Hybrid):** Proves that QAOA-inspired seeds lower the initial energy.
    *   **Exp 3 (Champion):** **Parallel Tempering** (Replica Exchange) Solver.
    *   **Exp 4 (Negative Result):** A "Simulated Bifurcation" attempt that failed to beat PT.
    *   **Final Showdown:** Head-to-head comparison graph.
*   **my_labs_library.py**: The core physics engine (fallback library).
*   **AI_REPORT.md**: Detailed analysis of AI collaboration, Hallucinations, and Wins.
*   **PRESENTATION_Team_Solo_Yao.md**: The narrative script for our project.
*   **slide_video_link**: Contains the link to our video presentation.

---

##  The Tech Stack

1.  **Adaptive Kernels:**
    *   Tries to JIT compile **raw CUDA C++** via NVRTC (100x speedup).
    *   Falls back to **Vectorized CuPy** if NVRTC is missing (10x speedup).
    *   Falls back to **NumPy** if no GPU is found (0x speedup).

2.  **Physics Engines:**
    *   **Parallel Tempering (Winner):** Thermodynamic ensemble with Metropolis exchange.
    *   **Simulated Bifurcation (Evaluated):** Continuous dynamical system simulation.

---

##  How to Run

1.  Open Final_Submission_Solo_Yao.ipynb in VS Code.
2.  Click **"Run All"**.
3.  Scroll to the bottom to see the **"Bifurcation Diagram"** and **"Optimization Trajectory"** visualizations.

---
*"We turned the LABS problem from a search task into a physics simulation."*
