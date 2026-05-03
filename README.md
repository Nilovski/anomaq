# AnomaQ — Web UI Setup

## Quick Start (3 steps)

### 1. Install dependencies
```bash
pip install fastapi uvicorn python-multipart qiskit qiskit-machine-learning scikit-learn pandas numpy
```

### 2. Start the server
```bash
cd anomaq_app
uvicorn main:app --reload
```

### 3. Open your browser
Go to: **http://localhost:8000**

---

## Usage
- Drag and drop any CSV file onto the upload zone
- Toggle "Include Quantum Kernel" on/off (quantum adds ~5–30s)
- Click **Run Analysis**
- Results appear below with:
  - Summary stats (classical vs quantum flagged)
  - Per-method outlier breakdown
  - Quantum similarity ranking table
  - Quantum-only detections with feature z-scores

## Requirements
- Python 3.8+
- All columns with text/IDs are auto-skipped
- Rows with NaN values are dropped automatically
- Works best with 10–500 rows, up to ~20 numeric features
  (Quantum step caps at 5 qubits via PCA for speed)
