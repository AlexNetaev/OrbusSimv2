# Master Architecture Blueprint: Autonomous AI Lab (Self-Driving Lab)

**Concept:** File-system-based multi-agent system with strictly decoupled hardware execution.
**Optimization:** Stateless agents for local LLMs with a strict context window (max `num_ctx = 8192`) designed for stable, multi-day autonomous operation.

---

## 1. Core Architecture & Philosophy

The system is designed for maximum stability and minimal VRAM consumption. To prevent exceeding the 8k context limit and avoid memory leaks, the following core rules apply:

*   **Sequential, Stateless Execution:** Only **one** agent is active in the RAM at any given time. There are no race conditions, eliminating the need for complex file-locks. A central synchronous Python loop wakes an agent, the agent reads its specific files, generates its output, saves it to the disk, and is immediately purged from memory.
*   **Closed-Loop Cycles:** Research happens in iterative cycles (Simulation -> Hardware -> Synthesis). Agents primarily read data relevant only to the current cycle.
*   **Hardware Decoupling:** The AI never controls hardware pins directly. It generates job queues (`experiment.json`). If the AI crashes, the hardware finishes the current experiment, reaches a safe state, and waits.
*   **Zero Data Loss (Shadow Memory):** Every LLM prompt and raw output is preserved within the specific cycle's folder. Nothing is ever permanently deleted; outdated knowledge is archived.

---

## 2. File System (Memory & Communication)

Since agents do not chat directly, the workspace serves as the central "mailbox" and long-term memory.

*   📂 `/workspace`
    *   📂 `/00_System/` *(The Global Brain)*
        *   `directive.md` (The main goal, e.g., "Minimize photobleaching of Fluorophore A").
        *   `hardware_limits.yaml` (Min/Max limits, toxic combinations for safety checks).
        *   `summary.md` (Global protocol by the Chef. Max 3 lines per cycle).
    *   📂 `/01_Knowledge_Base/` *(Long-Term Memory)*
        *   `theory_baseline.md` (Physical formulas and constants).
        *   📂 `/Archive/` (Outdated theories moved here by the Curator Agent).
    *   📂 `/02_Research_Cycles/` *(The active loop)*
        *   📂 `/Cycle_001/` *(Generated fresh for each loop)*
            *   📂 `A_Simulation/` (`sim_script.py`, `sim_data.csv`)
            *   📂 `B_Hardware/` (`experiment.json`, `measurement.csv`, `hardware_protocol.json`)
            *   📂 `C_Analysis/` (`plot.png`, `discrepancy.md`, `hypothesis.md`, `cycle_protocol.md`)
            *   📂 `D_Shadow_Memory/` *(Raw LLM prompts/responses of this specific cycle for debugging)*
    *   📂 `/03_Hardware_Queue/` *(The mailbox for the machine containing final JSONs)*

---

## 3. Hardware & Security Architecture

The hardware timeline is strictly separated from the AI timeline. The carousel discretely processes a `.json` from the `/03_Hardware_Queue/`.

### A. The Station Carousel (Hardware Loop)
1.  **Station 1 (Reagents):** Dosing and preparation.
2.  **Station 2 (Process):** Mixing, heating (the actual experiment execution).
3.  **Station 3 (Analytics):** Takes measurements. Generates `measurement.csv` (raw sensor data) AND `hardware_protocol.json` (Target vs. Actual states).
4.  **Station 4 (Cleanup/Switch):** Executes the mandatory `cleanup_routine` from the JSON -> Next cycle starts.

### B. Two-Stage Safety & Reality Check
*   **Stage 1 (Software Linter):** A hardcoded Python script checks every JSON in the queue against `hardware_limits.yaml` (Min/Max values).
*   **Stage 2 (Hardware SBC):** A single-board computer (e.g., Raspberry Pi) monitors temps/pressure completely independently of the AI. If danger is detected, a relay cuts the power (Hardware Emergency Stop).
*   **The Reality Check:** The `hardware_protocol.json` logs deviations (e.g., `T_target: 20.0, T_actual: 22.5`). The AI checks this *before* analyzing the science, preventing it from inventing new physics to explain a broken heater.

---

## 4. LLM Resilience & Validation Wrapper

To prevent local LLMs from returning empty strings or malformed syntax, all LLM calls pass through a central Python wrapper function (`ask_llm_with_validation()`):
1.  **Empty Response Check:** If the LLM returns an empty string or immediate EOS token, it retries silently (up to 3 times).
2.  **Pydantic Enforcement:** Output meant for hardware or voting is validated against a strict Pydantic JSON schema.
3.  **Auto-Correction:** If JSON is invalid, the traceback error is sent back to the LLM as a system prompt ("Invalid JSON: [Error]. Fix it.").

---

## 5. The AI Agents (Roles & Workflows)

### 👑 Chef Agent (The Coordinator)
*   **Task:** Maintains global oversight. 
*   **Action:** Reads `hypothesis.md` at the end of a loop. Writes exactly 3 lines into the global `summary.md`. Decides if the `directive.md` goal is met or initiates `Cycle_002`.

### 📚 Team 2: Theory & Curation
*   **2.1 The Theoretician:** Distills physical laws from local RAG literature into `theory_baseline.md`.
*   **2.2 The Fact-Checker:** Vetoes hallucinated constants against real data.
*   **2.3 The Knowledge Curator:** Token hygiene manager. Prevents `theory_baseline.md` from exceeding context limits by archiving debunked hypotheses into `/Archive/` and leaving only short reference pointers in the main file.

### 💻 Team 3: Simulation (Digital Twin)
*   **3.1 Python Architect:** Writes `sim_script.py` based on the theory baseline and next parameters.
*   **3.2 Sandbox Debugger:** Executes the code in isolation. If it crashes, it reads the traceback and patches the code (max 3 retries). Generates `sim_data.csv`.

### 🔬 Team 4: Synthesis & Hypothesis (The Scientific Core)
*   **4.1 Data Analyst:** Reads both `measurement.csv` AND `hardware_protocol.json`. Plots Reality vs. Simulation. If hardware failed (e.g., wrong temp achieved), it aborts analysis. Otherwise, it logs structural differences in `discrepancy.md`.
*   **4.2 Hypothesis Architect:** Formulates a physical explanation for the discrepancy and dictates the goal for the next experiment.
*   **4.3 The Gatekeeper (Red Teamer):** Applies Occam's razor. Is a real experiment strictly necessary, or does the simulation just need tweaking? Issues a "GO" to Team 5 or "VETO" to Team 3. Incorporates User Co-Pilot feedback.

### ⚙️ Team 5: Experiment Execution (Compiler)
*   **5.1 Machine Planner:** Translates scientific parameters into raw machine arrays/JSON for the 4-station carousel.
*   **5.2 Semantic Safety Agent:** Evaluates the JSON against chemical conflict documents. Ensures Station 4 `cleanup_routine` is correctly scheduled. Pushes the final JSON to the queue.

---

## 6. Fully Autonomous Deadlock Management

If the system gets stuck (e.g., Team 3 crashes 3 times, or Team 4 gives 3 Vetos), it heals itself without GUI popups:
1.  **Hardware continues:** The current hardware experiment finishes safely.
2.  **Crisis Intervention:** The Chef generates 3 possible workarounds (e.g., Option A: Simplify Sim, Option B: Ignore Temperature).
3.  **Chain of Thought Voting:** Lead Agents (3.1, 4.2, 5.1) vote on the options using strict JSON. They must provide reasoning first: `{"analysis": "Option B saves materials.", "vote": "Option_B"}`.
4.  **Resolution:** The Chef tallies the votes, logs the decision in `cycle_protocol.md`, and forces the next cycle with the winning parameters.

---

## 7. NiceGUI User Interface (The 4 Tabs)

The UI runs asynchronously in the background, polling the file system for live updates.

*   **Tab 1: Mission Control (Overview):** Kanban board showing the active agent. Scrolling Chef-Feed (`summary.md`). Global Software E-Stop button.
*   **Tab 2: Hardware Live-Monitor:** Visual 4-station carousel (active station pulses). Live SBC sensor streams. List of pending JSONs in the hardware queue.
*   **Tab 3: Lab Journal (Scientific Hub):** Interactive plots overlaying simulated vs. real data. Text boxes for current discrepancies/hypotheses. **Co-Pilot Chat:** User injection point to guide Team 4 ("Next time, check the pH value").
*   **Tab 4: System & LLM Settings:** Model assignment dropdowns (assigning specific Ollama models to specific teams). Hardware load indicators (RAM/CPU). Temperature and Context Size sliders.