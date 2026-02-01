# AI Post-Mortem Report

**Team:** Solo_Yao  
**Project:** GPU-Native Thermodynamic Solver

---

## 1. The Workflow: "The Architect & The Intern"

I utilized a **Hierarchical Agent Strategy**:
1.  **The Architect (Claude 3.5):** Designed the overall "Parallel Tempering" system architecture and the "Adaptive Kernel" factory pattern.
2.  **The Implementer (Copilot/Cursor):** Wrote the specific CuPy vectorization logic and boilerplate visualization code.
3.  **The Critic (Prompt Engineering):** I explicitly asked the AI to "Criticize this algorithm" which led to the *Final Showdown* comparison idea.

---

## 2. Verification Strategy: "Trust but Verify"

We treated every AI-generated feature as a "Hypothesis" needing validation.

### Mechanism 1: The "Showdown" (Comparative Verification)
When AI suggested **Simulated Bifurcation (SB)** as a superior algorithm:
*   **Claim:** "It is the state of the art for Ising Machines."
*   **Verification:** I demanded a side-by-side benchmark ("The Showdown").
*   **Result:** SB (Merit Factor 2.31) **failed** to beat the Baseline (3.5).
*   **Action:** We accepted the data, rejected the hype, and kept PT (4.19) as the champion. The "Negative Result" is documented as evidence of rigorous testing.

### Mechanism 2: Visual Constraints
AI often outputs numbers that look good but are physically impossible.
*   **Hallucination Check:** I asked AI to plot the **"Bifurcation Diagram"**.
*   **Logic:** If the algorithm works, the chart MUST show lines diverging from 0 to +1/-1.
*   **Outcome:** The plot confirmed the *dynamics* were correct (the code worked), even though the *optimization result* was poor. This proved the code wasn't bugged, just intrinsic algorithm limitations.

---

## 3. The "Vibe" Log

###  Win: The "Environment-Proof" Kernel
*   **Context:** 'cupy' often fails if 'libnvrtc' (NVIDIA Runtime Compiler) is missing or mismatched.
*   **The Win:** I asked the AI: *"Make this code bulletproof. If JIT fails, degrade gracefully."*
*   **Result:** The AI implemented a Factory Pattern 'get_energy_kernels()' that silently catches 'OSError' and swaps in a slower-but-working Vectorized NumPy/CuPy implementation. This saved hours of environment debugging.

###  Fail: The "Simulated Bifurcation" Hype
*   **The Fail:** The AI recommended SB as a "Sci-Fi" algorithm that would visually impress judges.
*   **The Reality:** While it *looked* cool (Sci-Fi points: 10/10), it was mathematically inferior (Performance points: 3/10) for the specific LABS Hamiltonian without heavy hyperparameter tuning.
*   **The Fix:** We didn't delete it. We rebranded it as **Experiment 4: A Negative Result**, showing we explore cutting-edge methods even if they don't always yield the best MF.

###  Learn: "Prompting for Failure"
*   **Insight:** AI tries to please you. If you ask "Write a fast solver", it gives you code that *looks* fast.
*   **New Strategy:** I started searching for "Why does Parallel Tempering fail?" or "Limits of Simulated Bifurcation".
*   **Result:** This "adversarial prompting" helped me identify the weaknesses (e.g., critical temperature ranges) before running the code.

---

## 4. Context Dump: The Prompts

> *"Write a 'Showdown' function that runs three algorithms side-by-side and plots a Bar Chart comparison. Do not hardcode the winner; let the data decide."*

This prompt ensured we got an honest comparison chart, which ultimately saved us from submitting a weaker algorithm.
