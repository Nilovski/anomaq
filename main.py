"""
Catalyst — Domain-Aware Quantum + Classical Outlier Detection
"""
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
import io, time, warnings, json, asyncio, traceback, os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

warnings.filterwarnings('ignore')

app = FastAPI(title="Catalyst")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
executor = ThreadPoolExecutor(max_workers=2)

# Static files (dataset preview images, etc.)
_static_dir = Path(__file__).resolve().parent / "static"
if _static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")


# ══════════════════════════════════════════════════════════════════════════
#  QUANTUM METHODS — QuantumPCAResidual & QuantumAutoencoder
#  Inlined so no extra package install is needed beyond existing deps.
# ══════════════════════════════════════════════════════════════════════════

class _MagnitudePreservingEncoder:
    """Standardize -> (optional PCA) -> unit-norm amplitude encoding."""
    def __init__(self, n_qubits):
        self.n_qubits = n_qubits
        self.dim = 2 ** n_qubits
        self.usable = self.dim - 1

    def fit(self, X):
        from sklearn.preprocessing import StandardScaler as SS
        self.sc_ = SS().fit(X)
        Xs = self.sc_.transform(X)
        if X.shape[1] > self.usable:
            self.pca_ = PCA(n_components=self.usable).fit(Xs)
            Xr = self.pca_.transform(Xs)
        else:
            self.pca_ = None
            Xr = Xs
        self.r_ = float(np.linalg.norm(Xr, axis=1).max()) + 1e-9
        return self

    def transform(self, X):
        Xs = self.sc_.transform(X)
        Xr = self.pca_.transform(Xs) if self.pca_ is not None else Xs
        out = np.zeros((Xr.shape[0], self.dim))
        nf = Xr.shape[1]
        out[:, :nf] = Xr / self.r_
        norms = np.linalg.norm(out[:, :nf], axis=1, keepdims=True)
        out[:, -1] = np.sqrt(np.clip(1.0 - (norms**2).ravel(), 0, 1))
        return out


class _QuantumPCAResidual:
    """Variational qPCA residual anomaly scorer (Lloyd et al. 2014)."""
    name = "QuantumPCA"

    def __init__(self, n_qubits=3, n_components=2, reps=2, max_iter=60, n_restarts=2, seed=42):
        self.n_qubits = n_qubits; self.n_components = n_components
        self.reps = reps; self.max_iter = max_iter
        self.n_restarts = n_restarts; self.seed = seed

    def fit_score(self, X):
        try:
            from qiskit.circuit.library import efficient_su2
            from qiskit.quantum_info import Statevector
            from scipy.optimize import minimize as sp_min
            enc = _MagnitudePreservingEncoder(self.n_qubits).fit(X)
            Xa = enc.transform(X)
            rho = (Xa.T @ Xa) / Xa.shape[0]
            ansatz = efficient_su2(num_qubits=self.n_qubits, reps=self.reps, entanglement="linear")
            n_p = ansatz.num_parameters
            rng = np.random.default_rng(self.seed)
            eigenvecs, rho_def = [], rho.copy()
            nc = min(self.n_components, 2 ** self.n_qubits)
            for _ in range(nc):
                best_val, best_v = np.inf, None
                for _ in range(self.n_restarts):
                    x0 = rng.uniform(0, 2 * np.pi, n_p)
                    def loss(th, rd=rho_def, ans=ansatz):
                        sv = Statevector(ans.assign_parameters(th)).data
                        return -float(np.real(sv.conj() @ rd @ sv))
                    res = sp_min(loss, x0, method="COBYLA", options={"maxiter": self.max_iter, "rhobeg": 0.5})
                    if res.fun < best_val:
                        best_val = res.fun
                        best_v = Statevector(ansatz.assign_parameters(res.x)).data
                eigenvecs.append(best_v)
                # rho is real-symmetric by construction; take real part of the
                # outer product to keep rho_def in float64 (otherwise numpy
                # raises UFuncOutputCastingError on the in-place subtraction).
                rho_def -= best_val * np.outer(best_v, best_v.conj()).real
            V = np.column_stack([v.real for v in eigenvecs])
            residuals = np.array([float(np.linalg.norm(x - V @ (V.T @ x))**2) for x in Xa])
            mn, mx = residuals.min(), residuals.max()
            return (residuals - mn) / (mx - mn + 1e-9)
        except Exception:
            return np.zeros(len(X))


class _QuantumAutoencoder:
    """Romero-style QAE trash-fidelity anomaly scorer."""
    name = "QuantumAutoencoder"

    def __init__(self, n_qubits=3, n_trash=1, reps=2, max_iter=60, n_restarts=2, max_train=60, seed=42):
        self.n_qubits = n_qubits; self.n_trash = n_trash; self.reps = reps
        self.max_iter = max_iter; self.n_restarts = n_restarts
        self.max_train = max_train; self.seed = seed

    def fit_score(self, X):
        try:
            from qiskit.circuit.library import real_amplitudes
            from qiskit.quantum_info import Statevector, partial_trace
            from scipy.optimize import minimize as sp_min
            enc = _MagnitudePreservingEncoder(self.n_qubits).fit(X)
            Xa = enc.transform(X)
            rng = np.random.default_rng(self.seed)
            n_tr = min(self.max_train, len(Xa))
            Xa_tr = Xa[rng.choice(len(Xa), n_tr, replace=False)]
            ansatz = real_amplitudes(self.n_qubits, reps=self.reps, entanglement="linear")
            n_p = ansatz.num_parameters
            keep = list(range(self.n_qubits - self.n_trash))
            def trash_fids(theta, Xb):
                bound = ansatz.assign_parameters(theta)
                fids = np.empty(len(Xb))
                for i, x in enumerate(Xb):
                    rho_t = partial_trace(Statevector(x).evolve(bound), keep)
                    fids[i] = float(np.real(rho_t.data[0, 0]))
                return fids
            best_theta, best_val = None, np.inf
            for _ in range(self.n_restarts):
                x0 = rng.uniform(0, 2 * np.pi, n_p)
                res = sp_min(lambda th: 1.0 - float(trash_fids(th, Xa_tr).mean()),
                             x0, method="COBYLA", options={"maxiter": self.max_iter, "rhobeg": 0.5})
                if res.fun < best_val:
                    best_val, best_theta = res.fun, res.x
            scores = 1.0 - trash_fids(best_theta, Xa)
            mn, mx = scores.min(), scores.max()
            return (scores - mn) / (mx - mn + 1e-9)
        except Exception:
            return np.zeros(len(X))

# ── Domain Detection ─────────────────────────────────────────────
DOMAINS = {
    "traffic": {
        "keywords": ["speed","weight","radar","lane","reaction","accel","vehicle","brake","material_density","thermal"],
        "label": "Physical Outlier",
        "contamination": 0.04,
        "icon": "🚗",
        "description": "Detecting physically anomalous vehicles — self-driving cars, different material compositions, or unusual internal configurations",
        "outlier_names": {
            "quantum_only": "Subtle Physical Outlier (quantum-only)",
            "classical": "Flagged Vehicle",
        },
        "insight": lambda feats: _traffic_insight(feats),
    },
    "healthcare": {
        "keywords": ["bp","pressure","heart","glucose","bmi","oxygen","respiratory","temp","saturation","patient"],
        "label": "Clinical Anomaly",
        "contamination": 0.05,
        "icon": "🏥",
        "description": "Detecting patients with unusual vital sign combinations that may indicate clinical risk",
        "outlier_names": {"quantum_only": "Subtle Clinical Pattern", "classical": "Flagged Patient"},
        "insight": lambda feats: _health_insight(feats),
    },
    "finance": {
        "keywords": ["amount","transaction","balance","credit","debit","payment","v1","v2","v3","fraud"],
        "label": "Suspicious Transaction",
        "contamination": 0.03,
        "icon": "💳",
        "description": "Detecting fraudulent or anomalous financial transactions",
        "outlier_names": {"quantum_only": "Subtle Fraud Pattern", "classical": "Flagged Transaction"},
        "insight": lambda feats: _finance_insight(feats),
    },
    "industrial": {
        "keywords": ["temperature","vibration","rpm","current","voltage","pressure","torque","flow","machine"],
        "label": "Equipment Fault",
        "contamination": 0.05,
        "icon": "🏭",
        "description": "Detecting faulty industrial equipment based on sensor reading combinations",
        "outlier_names": {"quantum_only": "Subtle Fault Pattern", "classical": "Flagged Machine"},
        "insight": lambda feats: _industrial_insight(feats),
    },
    "network": {
        "keywords": ["bytes","packets","duration","port","protocol","src","dst","connection","flag"],
        "label": "Network Intrusion",
        "contamination": 0.04,
        "icon": "🔒",
        "description": "Detecting anomalous network connections that may indicate intrusion attempts",
        "outlier_names": {"quantum_only": "Subtle Intrusion Pattern", "classical": "Flagged Connection"},
        "insight": lambda feats: _network_insight(feats),
    },
    "generic": {
        "keywords": [],
        "label": "Outlier",
        "contamination": 0.05,
        "icon": "⬡",
        "description": "General-purpose outlier detection across all features",
        "outlier_names": {"quantum_only": "Quantum-Only Detection", "classical": "Flagged Entity"},
        "insight": lambda feats: "Each feature appears individually normal, but the quantum kernel detected an unusual combination of physical properties.",
    },
}

def _traffic_insight(feats):
    fd = {f["name"]: f for f in feats}
    hints = []
    if "accel_variance" in fd and fd["accel_variance"]["z_score"] > 1.5:
        hints.append("near-zero acceleration variance (superhuman consistency → possible autonomous vehicle)")
    if "lane_deviation_m" in fd and fd["lane_deviation_m"]["z_score"] > 1.5:
        hints.append("minimal lane deviation (machine-precision steering)")
    if "reaction_time_s" in fd and fd["reaction_time_s"]["z_score"] > 1.5:
        hints.append("unusually fast reaction time (sub-human latency)")
    if "material_density" in fd and fd["material_density"]["z_score"] > 1.5:
        hints.append("anomalous material density (possible legacy steel construction in modern fleet)")
    if "radar_cross_section_m2" in fd and fd["radar_cross_section_m2"]["z_score"] > 1.5:
        hints.append("elevated radar cross-section (sensor array signature)")
    if hints:
        return "Quantum kernel flagged this vehicle due to: " + "; ".join(hints) + "."
    return "The quantum kernel detected an unusual combination of physical signatures not visible in individual features."

def _health_insight(feats):
    fd = {f["name"]: f for f in feats}
    hints = []
    for col, desc in [("systolic_bp","elevated systolic pressure"),("glucose_mmol","high glucose"),
                      ("heart_rate_bpm","irregular heart rate"),("oxygen_saturation","low O₂ saturation")]:
        if col in fd and fd[col]["z_score"] > 1.2:
            hints.append(desc)
    if hints:
        return "Unusual combination of: " + ", ".join(hints) + ". Individual values may seem borderline, but together they form a high-risk pattern."
    return "The quantum kernel detected a clinically unusual combination of vitals not apparent from individual readings."

def _finance_insight(feats):
    fd = {f["name"]: f for f in feats}
    if "amount" in fd and fd["amount"]["z_score"] > 1.5:
        return f"Transaction amount is {fd['amount']['z_score']:.1f}σ above normal, combined with unusual feature correlations — classic fraud signature."
    return "The quantum kernel detected an unusual correlation pattern across transaction features, consistent with synthetic or fraudulent activity."

def _industrial_insight(feats):
    fd = {f["name"]: f for f in feats}
    hints = []
    for col, desc in [("temperature_c","overheating"),("vibration_hz","excess vibration"),
                      ("pressure_bar","overpressure"),("current_a","overcurrent")]:
        if col in fd and fd[col]["z_score"] > 1.2:
            hints.append(desc)
    if hints:
        return "Sensor anomaly: " + ", ".join(hints) + ". The combination indicates likely equipment fault."
    return "Quantum kernel detected subtle multi-sensor correlation breakdown, suggesting imminent equipment fault."

def _network_insight(feats):
    return "The quantum kernel detected an unusual combination of connection parameters consistent with network intrusion or anomalous traffic."

def detect_domain(columns):
    cols_lower = " ".join(c.lower() for c in columns)
    best, best_score = "generic", 0
    for domain, cfg in DOMAINS.items():
        if domain == "generic": continue
        score = sum(1 for kw in cfg["keywords"] if kw in cols_lower)
        if score > best_score:
            best, best_score = domain, score
    return best

def sse(event, data):
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"

def run_detection(df, use_quantum, emit, feature_map_type="ZZ", entanglement="linear", reps=2):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    non_numeric = [c for c in df.columns if c not in numeric_cols]
    label_col = non_numeric[0] if non_numeric else None

    X_raw = df[numeric_cols].values
    nan_mask = ~np.isnan(X_raw).any(axis=1)
    X_raw = X_raw[nan_mask]
    df = df[nan_mask].reset_index(drop=True)
    n_samples = len(X_raw)

    if n_samples < 5:
        emit("error", {"message": "Need at least 5 rows of data"})
        return

    # Domain detection
    domain = detect_domain(numeric_cols)
    domain_cfg = DOMAINS[domain]
    contamination = min(0.15, max(1/n_samples, domain_cfg["contamination"]))

    emit("progress", {"pct": 5, "label": f"Detected domain: {domain_cfg['icon']} {domain.upper()} — scaling features..."})

    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    def get_label(idx):
        return str(df.iloc[idx][label_col]) if label_col else f"Row {idx}"

    # ── Classical ──────────────────────────────────────────────
    emit("progress", {"pct": 12, "label": "Running Isolation Forest..."})
    iso = IsolationForest(contamination=contamination, random_state=42)
    iso_scores = iso.fit_predict(X)
    iso_s = iso.score_samples(X)
    iso_outliers = np.where(iso_scores == -1)[0].tolist()

    emit("progress", {"pct": 24, "label": "Running Local Outlier Factor..."})
    lof = LocalOutlierFactor(n_neighbors=min(5, n_samples-1), contamination=contamination)
    lof_pred = lof.fit_predict(X)
    lof_s = lof.negative_outlier_factor_
    lof_outliers = np.where(lof_pred == -1)[0].tolist()

    emit("progress", {"pct": 36, "label": "Running One-Class SVM (RBF)..."})
    svm = OneClassSVM(kernel='rbf', nu=contamination, gamma='scale')
    svm_pred = svm.fit_predict(X)
    svm_s = svm.score_samples(X)
    svm_outliers = np.where(svm_pred == -1)[0].tolist()

    emit("progress", {"pct": 46, "label": "Computing pairwise interaction features..."})
    if len(numeric_cols) <= 15:
        poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        X_poly = StandardScaler().fit_transform(poly.fit_transform(X_raw))
        iso2_outliers = np.where(IsolationForest(contamination=contamination, random_state=42).fit_predict(X_poly)==-1)[0].tolist()
        lof2_outliers = np.where(LocalOutlierFactor(n_neighbors=min(5,n_samples-1), contamination=contamination).fit_predict(X_poly)==-1)[0].tolist()
        svm2_outliers = np.where(OneClassSVM(kernel='rbf', nu=contamination, gamma='scale').fit_predict(X_poly)==-1)[0].tolist()
    else:
        iso2_outliers = lof2_outliers = svm2_outliers = []

    emit("progress", {"pct": 55, "label": "Merging classical results..."})
    all_classical = set(iso_outliers)|set(lof_outliers)|set(svm_outliers)|set(iso2_outliers)|set(lof2_outliers)|set(svm2_outliers)

    classical = {
        "isolation_forest": [{"idx":i,"label":get_label(i),"score":float(iso_s[i])} for i in iso_outliers],
        "lof": [{"idx":i,"label":get_label(i),"score":float(lof_s[i])} for i in lof_outliers],
        "svm_rbf": [{"idx":i,"label":get_label(i),"score":float(svm_s[i])} for i in svm_outliers],
        "isolation_forest_interactions": [{"idx":i,"label":get_label(i)} for i in iso2_outliers],
        "lof_interactions": [{"idx":i,"label":get_label(i)} for i in lof2_outliers],
        "svm_interactions": [{"idx":i,"label":get_label(i)} for i in svm2_outliers],
        "all_unique": [{"idx":i,"label":get_label(i)} for i in sorted(all_classical)],
    }

    # ── Quantum ────────────────────────────────────────────────
    quantum = None
    # Safety caps: keep statevector simulation under ~20s on Render free tier
    QUANTUM_MAX = 50        # max rows (50×50 = 2500 kernel evals ≈ 15s)
    MAX_QUBITS  = 4         # 2^4 = 16-dim statevector, fast
    MAX_REPS    = 2         # reps≥3 doubles gate depth, kills perf
    safe_reps   = min(reps, MAX_REPS)
    safe_ent    = "linear"  # full entanglement grows as O(n²) gates

    if use_quantum:
        try:
            from qiskit.circuit.library import ZZFeatureMap, PauliFeatureMap
            from qiskit_machine_learning.kernels import FidelityQuantumKernel

            emit("progress", {"pct": 62, "label": "Reducing dimensions with PCA..."})

            # Subsample large datasets
            if n_samples > QUANTUM_MAX:
                sample_idx = np.random.choice(n_samples, QUANTUM_MAX, replace=False)
                sample_idx = np.sort(sample_idx)
                X_q_input = X[sample_idx]
                emit("progress", {"pct": 63, "label": f"Large dataset ({n_samples} rows) — quantum kernel on {QUANTUM_MAX}-row stratified sample..."})
            else:
                sample_idx = np.arange(n_samples)
                X_q_input = X

            n_q_samples = len(sample_idx)
            n_components = min(MAX_QUBITS, len(numeric_cols), n_q_samples - 1)
            pca = PCA(n_components=n_components)
            X_pca = pca.fit_transform(X_q_input)
            variance_retained = float(pca.explained_variance_ratio_.sum())

            q_scaler = MinMaxScaler(feature_range=(0, np.pi))
            X_q = q_scaler.fit_transform(X_pca)

            emit("progress", {"pct": 70, "label": f"Building {n_components}-qubit {feature_map_type}FeatureMap (reps={safe_reps}, {safe_ent})..."})
            if feature_map_type == "Pauli":
                feature_map = PauliFeatureMap(feature_dimension=n_components, reps=safe_reps, entanglement=safe_ent)
            else:
                feature_map = ZZFeatureMap(feature_dimension=n_components, reps=safe_reps, entanglement=safe_ent)
            qk = FidelityQuantumKernel(feature_map=feature_map)

            emit("progress", {"pct": 76, "label": f"Computing {n_q_samples}×{n_q_samples} quantum kernel matrix..."})
            t0 = time.time()
            K = qk.evaluate(X_q)
            q_time = round(time.time() - t0, 2)

            emit("progress", {"pct": 91, "label": f"Kernel done in {q_time}s — fitting quantum SVM..."})
            qsvm = OneClassSVM(kernel='precomputed', nu=contamination)
            qsvm_pred = qsvm.fit_predict(K)
            qsvm_outliers_local = np.where(qsvm_pred == -1)[0].tolist()
            qsvm_outliers = [int(sample_idx[i]) for i in qsvm_outliers_local]

            emit("progress", {"pct": 95, "label": "Ranking by quantum similarity..."})
            avg_sim = np.mean(K, axis=1)
            ranked_local = np.argsort(avg_sim).tolist()
            ranked = [int(sample_idx[i]) for i in ranked_local]
            quantum_only_kernel = set(qsvm_outliers) - all_classical

            # ── QuantumPCA ──────────────────────────────────────────────
            emit("progress", {"pct": 96, "label": "Running Quantum PCA residual..."})
            n_q = min(3, len(numeric_cols), n_q_samples - 1)
            qpca = _QuantumPCAResidual(n_qubits=n_q, n_components=2, reps=2,
                                        max_iter=40, n_restarts=1)
            qpca_scores = qpca.fit_score(X_q_input)
            qpca_thresh = float(np.percentile(qpca_scores, 100 * (1 - contamination)))
            qpca_outliers_local = [int(i) for i in range(len(qpca_scores)) if qpca_scores[i] >= qpca_thresh]
            qpca_outliers = [int(sample_idx[i]) for i in qpca_outliers_local]

            # ── QuantumAutoencoder ──────────────────────────────────────
            emit("progress", {"pct": 97, "label": "Running Quantum Autoencoder..."})
            qae = _QuantumAutoencoder(n_qubits=n_q, n_trash=1, reps=2,
                                       max_iter=40, n_restarts=1,
                                       max_train=min(40, n_q_samples))
            qae_scores = qae.fit_score(X_q_input)
            qae_thresh = float(np.percentile(qae_scores, 100 * (1 - contamination)))
            qae_outliers_local = [int(i) for i in range(len(qae_scores)) if qae_scores[i] >= qae_thresh]
            qae_outliers = [int(sample_idx[i]) for i in qae_outliers_local]

            # ── Unified quantum outlier set ─────────────────────────────
            all_quantum = set(qsvm_outliers) | set(qpca_outliers) | set(qae_outliers)
            quantum_only = all_quantum - all_classical

            def get_feats(idx):
                feats = []
                for fname in numeric_cols:
                    val = float(df.iloc[idx][fname])
                    col_vals = df[fname].values
                    z = float(abs(val - col_vals.mean()) / col_vals.std()) if col_vals.std() > 0 else 0.0
                    feats.append({"name": fname, "value": round(val, 4), "z_score": round(z, 2)})
                return feats

            qonly_details = []
            for idx in quantum_only:
                feats = get_feats(idx)
                qonly_details.append({"idx": idx, "label": get_label(idx), "features": feats,
                                       "insight": domain_cfg["insight"](feats)})

            q_outlier_details = []
            for i in qsvm_outliers:
                feats = get_feats(i)
                q_outlier_details.append({"idx": i, "label": get_label(i),
                                           "also_classical": i in all_classical,
                                           "features": feats, "insight": domain_cfg["insight"](feats)})

            quantum = {
                "qubits": n_components,
                "variance_retained": round(variance_retained * 100, 1),
                "kernel_time_s": q_time,
                "feature_map": feature_map_type,
                "reps_used": safe_reps,
                "sample_size": n_q_samples,
                "outliers": q_outlier_details,
                "ranking": [{"rank": r + 1, "idx": ranked[r], "label": get_label(ranked[r]),
                              "avg_similarity": round(float(avg_sim[ranked_local[r]]), 6),
                              "also_classical": ranked[r] in all_classical}
                             for r in range(min(10, len(ranked)))],
                "quantum_only": qonly_details,
                # ── New: per-method quantum results ──────────────────────
                "qpca_outliers": [{"idx": i, "label": get_label(i), "score": round(float(qpca_scores[qpca_outliers_local[j]]), 4)}
                                   for j, i in enumerate(qpca_outliers)],
                "qae_outliers":  [{"idx": i, "label": get_label(i), "score": round(float(qae_scores[qae_outliers_local[j]]), 4)}
                                   for j, i in enumerate(qae_outliers)],
                "all_quantum_unique": [{"idx": i, "label": get_label(i)} for i in sorted(all_quantum)],
            }
        except Exception as e:
            traceback.print_exc()
            quantum = {"error": str(e)}
            emit("progress", {"pct": 96, "label": f"Quantum error: {str(e)[:80]}"})

    emit("progress", {"pct": 99, "label": "Finalizing..."})
    emit("result", {
        "meta": {
            "rows": n_samples, "features": numeric_cols,
            "label_col": label_col, "contamination": round(contamination,3),
            "domain": domain,
            "domain_icon": domain_cfg["icon"],
            "domain_label": domain_cfg["label"],
            "domain_description": domain_cfg["description"],
        },
        "classical": classical,
        "quantum": quantum,
        "summary": {
            "classical_unique": len(all_classical),
            "quantum_flagged": len(all_quantum) if quantum and "all_quantum_unique" in quantum else (len(quantum["outliers"]) if quantum and "outliers" in quantum else None),
            "quantum_only": len(quantum["quantum_only"]) if quantum and "quantum_only" in quantum else None,
        }
    })
    emit("progress", {"pct":100,"label":"Done!"})


@app.post("/analyze-stream")
async def analyze_stream(file: UploadFile = File(...), quantum: bool = True,
                         feature_map: str = "ZZ", entanglement: str = "linear", reps: int = 2):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "Please upload a .csv file.")
    contents = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(400, f"Could not parse CSV: {e}")

    queue = asyncio.Queue()
    def emit(event, data): queue.put_nowait(sse(event, data))

    async def run_bg():
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(executor, run_detection, df, quantum, emit, feature_map, entanglement, reps)
        except Exception as e:
            queue.put_nowait(sse("error", {"message": str(e)}))
        finally:
            queue.put_nowait(None)

    asyncio.create_task(run_bg())

    async def stream():
        yield sse("progress", {"pct":2,"label":"Parsing CSV..."})
        while True:
            msg = await queue.get()
            if msg is None: break
            yield msg

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})

@app.get("/", response_class=HTMLResponse)
async def root():
    with open("index.html") as f: return f.read()


@app.post("/analyze")
async def analyze(file: UploadFile = File(...), quantum: bool = True,
                  feature_map: str = "ZZ", entanglement: str = "linear", reps: int = 2):
    """Synchronous endpoint for frontends that don't use SSE streaming."""
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "Please upload a .csv file.")
    contents = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(400, f"Could not parse CSV: {e}")

    collected = {}
    errors = []

    def emit(event, data):
        if event == "result":
            collected.update(data)
        elif event == "error":
            errors.append(data.get("message", "Unknown error"))

    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(executor, run_detection, df, quantum, emit, feature_map, entanglement, reps)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, str(e))

    if errors:
        raise HTTPException(500, errors[0])
    if not collected:
        raise HTTPException(500, "Analysis produced no results")

    return JSONResponse(collected)


# ══════════════════════════════════════════════════════════════════════════════
#  IMAGE ANALYSIS ENDPOINT
#  Extracts 32 numeric features per image (color histograms + texture stats)
#  then runs the same detection pipeline as CSV mode.
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/analyze-images")
async def analyze_images(
    files: list[UploadFile] = File(...),
    quantum: bool = True,
    feature_map: str = "ZZ",
    entanglement: str = "linear",
    reps: int = 2,
):
    if len(files) < 2:
        raise HTTPException(400, "Upload at least 2 images to compare.")
    if len(files) > 20:
        raise HTTPException(400, "Maximum 20 images per request.")

    try:
        from PIL import Image as PILImage
    except ImportError:
        raise HTTPException(500, "Pillow not installed — add 'Pillow' to requirements.txt")

    rows = []
    for f in files:
        data = await f.read()
        try:
            img = PILImage.open(io.BytesIO(data)).convert("RGB").resize((64, 64))
        except Exception:
            raise HTTPException(400, f"Could not read image: {f.filename}")

        arr = np.array(img, dtype=np.float32) / 255.0  # (64,64,3)

        feats = {}
        # Colour histograms (8 bins per channel = 24 features)
        for ci, ch in enumerate(["r", "g", "b"]):
            hist, _ = np.histogram(arr[:, :, ci], bins=8, range=(0, 1))
            hist = hist.astype(float) / hist.sum()
            for bi, v in enumerate(hist):
                feats[f"{ch}_h{bi}"] = round(float(v), 5)

        # Texture: mean, std, contrast per channel (9 features)
        for ci, ch in enumerate(["r", "g", "b"]):
            ch_arr = arr[:, :, ci]
            feats[f"{ch}_mean"]     = round(float(ch_arr.mean()), 5)
            feats[f"{ch}_std"]      = round(float(ch_arr.std()),  5)
            feats[f"{ch}_contrast"] = round(float(ch_arr.max() - ch_arr.min()), 5)

        # Brightness & saturation (2 features)
        gray = 0.299 * arr[:,:,0] + 0.587 * arr[:,:,1] + 0.114 * arr[:,:,2]
        feats["brightness"] = round(float(gray.mean()), 5)
        feats["saturation"] = round(float(
            (arr.max(axis=2) - arr.min(axis=2)).mean()
        ), 5)

        feats["_label"] = f.filename
        rows.append(feats)

    df = pd.DataFrame(rows)
    label_col = "_label"
    # rename for run_detection compatibility
    df.rename(columns={"_label": label_col}, inplace=True)

    collected = {}
    errors = []

    def emit(event, d):
        if event == "result":
            collected.update(d)
        elif event == "error":
            errors.append(d.get("message", ""))

    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            executor, run_detection, df, quantum, emit, feature_map, entanglement, reps
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, str(e))

    if errors:
        raise HTTPException(500, errors[0])
    if not collected:
        raise HTTPException(500, "Analysis produced no results")

    return JSONResponse(collected)


# ══════════════════════════════════════════════════════════════════════════════
#  SYNTHETIC DEMO DATASET ENDPOINT
#  Generates realistic datasets with a single (or few) planted anomalies.
#  Used by the Demo tab on the frontend.
# ══════════════════════════════════════════════════════════════════════════════

SCENARIOS = {
    "vehicle": {
        "n_normal": 49,
        "features": {
            "speed_kmh":             (85,  8),
            "weight_kg":             (3200, 180),
            "radar_cross_section_m2":(2.1, 0.25),
            "accel_variance":        (0.42, 0.06),
            "lane_deviation_m":      (0.38, 0.05),
            "reaction_time_s":       (0.72, 0.09),
            "material_density":      (7.85, 0.30),
        },
        "outliers": [{
            "speed_kmh":             85.1,   # normal speed
            "weight_kg":             2950,   # slightly lighter (sensor array)
            "radar_cross_section_m2":3.8,    # elevated RCS
            "accel_variance":        0.004,  # near-zero (autonomous precision)
            "lane_deviation_m":      0.02,   # machine precision steering
            "reaction_time_s":       0.08,   # sub-human latency
            "material_density":      6.1,    # carbon-fibre composite
        }],
        "id_col": "vehicle_id",
        "id_prefix": "VH",
    },
    "soldier": {
        "n_normal": 79,
        "features": {
            "heart_rate_bpm":   (78,  9),
            "skin_temp_c":      (36.6, 0.4),
            "movement_speed":   (1.4,  0.3),
            "signal_strength":  (0.82, 0.07),
            "battery_pct":      (74,   12),
            "comms_latency_ms": (48,   8),
        },
        "outliers": [{
            "heart_rate_bpm":   31,    # abnormally low — device spoofing
            "skin_temp_c":      36.7,
            "movement_speed":   1.3,
            "signal_strength":  0.81,
            "battery_pct":      73,
            "comms_latency_ms": 312,   # extreme latency spike
        }],
        "id_col": "asset_id",
        "id_prefix": "ASSET",
    },
    "network": {
        "n_normal": 98,
        "features": {
            "bytes_sent":     (4200,  800),
            "bytes_recv":     (8100,  1200),
            "duration_s":     (2.1,   0.6),
            "packets":        (38,    7),
            "unique_ports":   (3,     1),
            "retransmits":    (0.8,   0.4),
            "ttl_variance":   (2.1,   0.5),
            "payload_entropy":(4.8,   0.3),
        },
        "outliers": [
            {"bytes_sent":48200,"bytes_recv":320,"duration_s":0.12,"packets":840,
             "unique_ports":94,"retransmits":0.1,"ttl_variance":18.4,"payload_entropy":7.9},
            {"bytes_sent":290,"bytes_recv":112000,"duration_s":41.2,"packets":12,
             "unique_ports":1,"retransmits":6.2,"ttl_variance":0.1,"payload_entropy":1.1},
        ],
        "id_col": "conn_id",
        "id_prefix": "CONN",
    },
    "video": {
        "n_normal": 59,
        "features": {
            "temporal_coherence":  (0.88, 0.04),
            "face_blink_rate":     (17,   3),
            "micro_expression":    (0.62, 0.08),
            "compression_artifact":(0.14, 0.03),
            "optical_flow_var":    (0.31, 0.05),
            "lip_sync_delta_ms":   (18,   5),
            "texture_freq_high":   (0.44, 0.06),
            "background_noise_db": (32,   4),
            "frame_diff_mean":     (0.08, 0.01),
            "gaze_natural_score":  (0.74, 0.07),
        },
        "outliers": [{
            "temporal_coherence":  0.997,  # too perfect
            "face_blink_rate":     4,      # unnaturally low
            "micro_expression":    0.09,   # flat affect
            "compression_artifact":0.52,   # GAN upsampling artifacts
            "optical_flow_var":    0.71,   # unnatural motion
            "lip_sync_delta_ms":   112,    # desync
            "texture_freq_high":   0.88,   # AI sharpening
            "background_noise_db": 8,      # near-silent background
            "frame_diff_mean":     0.003,  # too stable
            "gaze_natural_score":  0.12,   # unnatural gaze
        }],
        "id_col": "clip_id",
        "id_prefix": "CLIP",
    },
}


# ══════════════════════════════════════════════════════════════════════════════
#  REAL-WORLD INDUSTRY DATASETS — backend-hosted, quantum-only analysis with
#  ground-truth comparison. Each entry pre-baked into demo_data/<name>.csv with
#  an `is_anomaly` truth column.
# ══════════════════════════════════════════════════════════════════════════════

DEMO_DATA_DIR = Path(__file__).resolve().parent  # CSVs live in repo root
TRUTH_COL_NAMES = {"is_anomaly", "label", "true_label", "y", "ground_truth"}

REAL_DATASETS = {
    "vehicle": {
        "icon": "🚗",
        "industry": "Automotive",
        "gradient": "linear-gradient(135deg,#1a3050 0%,#2c4f7c 55%,#5a83b4 100%)",
        "photo_url": "https://images.unsplash.com/photo-1492144534655-ae79c964c9d7?w=1600&q=80&auto=format&fit=crop",
        "viz_image": "/static/dataset_imgs/vehicle.png",
        "title": "Vehicle Ensemble — Rogue-Vehicle Detection",
        "tagline": "Spotting the rogue vehicle in a fleet of 26.",
        "description": "26 vehicles described by 7 physical attributes — weight, "
                       "material density, average speed, acceleration variance, "
                       "engine vibration, radar cross-section, and thermal "
                       "conductivity. One vehicle in the ensemble is the physical "
                       "outlier: its internal composition doesn't match its "
                       "external design.",
        "source_url": "",
        "source_label": "",
        "anomaly_means": "VH-007 — the rogue vehicle in the fleet",
    },
    "kepler": {
        "icon": "🛰️",
        "industry": "Space exploration",
        "gradient": "linear-gradient(135deg,#08102e 0%,#241453 55%,#552378 100%)",
        "photo_url": "https://images.unsplash.com/photo-1419242902214-272b3f66ee7a?w=1600&q=80&auto=format&fit=crop",
        "viz_image": "/static/dataset_imgs/kepler.png",
        "title": "NASA Kepler — Exoplanet Filtering",
        "tagline": "Filter false-positive transit signals out of NASA's catalogue.",
        "description": "Real signals from NASA's Kepler space telescope. "
                       "Each row is a 'Kepler Object of Interest' — a brightness dip "
                       "in a star's light curve that might be a planet. Astronomers have "
                       "labeled each as either CONFIRMED exoplanet or FALSE POSITIVE "
                       "(eclipsing binary, stellar variability, instrument artifact). "
                       "The quantum pipeline picks the false alarms out of the catalogue "
                       "so astronomers don't waste follow-up telescope time on them.",
        "source_url": "https://exoplanetarchive.ipac.caltech.edu/",
        "source_label": "NASA Exoplanet Archive — KOI cumulative table",
        "anomaly_means": "FALSE POSITIVE signal that mimics a planet transit",
    },
    "bitcoin_otc": {
        "icon": "💰",
        "industry": "Cryptocurrency fraud",
        "gradient": "linear-gradient(135deg,#1a0f00 0%,#7a4a00 55%,#f0a020 100%)",
        "photo_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/1/11/Bitcoin_ATM_in_South_Africa.jpg/1280px-Bitcoin_ATM_in_South_Africa.jpg",
        "viz_image": "/static/dataset_imgs/bitcoin_otc.png",
        "title": "Bitcoin OTC — Fraud Detection",
        "tagline": "Flag fraudsters in a real peer-to-peer trading network.",
        "description": "Real trust ratings from bitcoin-otc.com, a peer-to-peer "
                       "cryptocurrency marketplace where users rate each other "
                       "from –10 (total distrust) to +10 (total trust). Each row "
                       "is one user's profile (incoming / outgoing rating stats plus "
                       "graph-spectral position in the trust network). The quantum "
                       "pipeline picks out users who accumulated many negative ratings — "
                       "the platform's actual fraudsters.",
        "source_url": "https://snap.stanford.edu/data/soc-sign-bitcoinotc.html",
        "source_label": "Stanford SNAP — soc-sign-bitcoinotc",
        "anomaly_means": "fraudster (≥30% incoming negative ratings)",
    },
    "turbofan": {
        "icon": "✈️",
        "industry": "Aerospace & Defense",
        "gradient": "linear-gradient(135deg,#0e1418 0%,#2a3744 55%,#5b7188 100%)",
        "photo_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/6/69/F100_F-15_engine.JPG/1280px-F100_F-15_engine.JPG",
        "viz_image": "/static/dataset_imgs/turbofan.png",
        "title": "NASA Turbofan — Predictive Maintenance",
        "tagline": "Catch a jet engine on its way to failure.",
        "description": "Telemetry from 100 simulated turbofan jet engines run from "
                       "healthy to mechanical failure (NASA C-MAPSS dataset, the standard "
                       "aerospace prognostics benchmark). Each row is a single-cycle "
                       "snapshot of 14 sensor channels (temperatures, pressures, "
                       "rotational speeds, fuel-flow ratios). The quantum pipeline picks "
                       "out late-life cycles where engines are degrading toward failure.",
        "source_url": "https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/",
        "source_label": "NASA Prognostics Center of Excellence — C-MAPSS FD001",
        "anomaly_means": "engine in late-life degradation",
    },
    "ecg5000": {
        "icon": "🏥",
        "industry": "Cardiology",
        "gradient": "linear-gradient(135deg,#3a0814 0%,#88172e 55%,#cf3357 100%)",
        "photo_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9e/SinusRhythmLabels.svg/1280px-SinusRhythmLabels.svg.png",
        "viz_image": "/static/dataset_imgs/ecg5000.png",
        "title": "ECG5000 — Heart-Beat Anomaly Detection",
        "tagline": "Pick rare arrhythmias out of healthy heartbeats.",
        "description": "Single-lead ECG heartbeats from a 20-hour recording of one "
                       "patient with severe congestive heart failure (BIDMC database, "
                       "PhysioNet). Each row is a heartbeat encoded as 15 frequency-band "
                       "magnitudes (FFT of the waveform). The quantum pipeline picks "
                       "out the rare-arrhythmia heartbeats — premature ventricular "
                       "contractions and unclassified rhythms — among healthy beats.",
        "source_url": "https://www.timeseriesclassification.com/description.php?Dataset=ECG5000",
        "source_label": "UCR Time Series Classification Archive — ECG5000",
        "anomaly_means": "rare arrhythmia (classes 3, 4, 5)",
    },
    "magtile": {
        "icon": "🏭",
        "industry": "Industrial QC",
        "gradient": "linear-gradient(135deg,#14161c 0%,#2c303c 55%,#525a6e 100%)",
        "photo_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8a/Automation_of_foundry_with_robot.jpg/1280px-Automation_of_foundry_with_robot.jpg",
        "viz_image": "/static/dataset_imgs/magtile.jpg",
        "title": "Magnetic Tile — Surface Defect Inspection",
        "tagline": "Find defective tiles on a manufacturing line.",
        "description": "Photographs of magnetic tiles — precision-ground ferrite "
                       "components used inside electric motors, generators, and "
                       "aerospace actuators. Each row is the HOG (Histogram-of-Oriented-"
                       "Gradients) feature vector of one tile, PCA-compressed to 15 "
                       "components. The quantum pipeline picks tiles with surface defects "
                       "(cracks, blowholes, frays, breaks, uneven surfaces) out of "
                       "defect-free production samples.",
        "source_url": "https://github.com/abin24/Magnetic-tile-defect-datasets.",
        "source_label": "Magnetic Tile Defect Dataset — Huang et al. 2020",
        "anomaly_means": "tile with one of 5 surface-defect classes",
    },
}


def _csv_path(name: str) -> Path:
    return DEMO_DATA_DIR / f"{name}.csv"


@app.get("/datasets-real")
async def list_real_datasets():
    """Metadata for the 6 backend-hosted real-world demos."""
    out = []
    for key, meta in REAL_DATASETS.items():
        path = _csv_path(key)
        if not path.exists():
            continue
        df = pd.read_csv(path)
        truth_col = next((c for c in df.columns if c.lower() in TRUTH_COL_NAMES), None)
        n_anom = int(df[truth_col].sum()) if truth_col else 0
        feat_cols = [c for c in df.columns
                     if c != truth_col and c.lower() != "id"
                     and pd.api.types.is_numeric_dtype(df[c])]
        out.append({
            "name": key,
            **meta,
            "n_samples": len(df),
            "n_features": len(feat_cols),
            "n_anomaly": n_anom,
            "feature_names": feat_cols,
            "contamination_pct": round(100 * n_anom / len(df), 1) if len(df) else 0,
        })
    return JSONResponse({"datasets": out})


def _confusion(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision": round(precision, 4), "recall": round(recall, 4),
            "f1": round(f1, 4)}


def _auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return float("nan")
    from sklearn.metrics import roc_auc_score
    try:
        return float(roc_auc_score(y_true, scores))
    except Exception:
        return float("nan")


@app.get("/run-quantum-real")
async def run_quantum_real(name: str):
    """Load demo CSV → run QPCA + QAE → compare against truth column."""
    if name not in REAL_DATASETS:
        raise HTTPException(404, f"Unknown dataset: {name}")
    path = _csv_path(name)
    if not path.exists():
        raise HTTPException(404, f"Demo CSV missing: {path.name}")

    df = pd.read_csv(path)
    truth_col = next((c for c in df.columns if c.lower() in TRUTH_COL_NAMES), None)
    if truth_col is None:
        raise HTTPException(400, "CSV has no truth column")
    y_true = df[truth_col].astype(int).values

    id_col = next((c for c in df.columns if c.lower() == "id"), None)
    ids = df[id_col].astype(str).values if id_col else np.array([f"row_{i}" for i in range(len(df))])

    feat_cols = [c for c in df.columns
                 if c not in {truth_col, id_col}
                 and pd.api.types.is_numeric_dtype(df[c])]
    X = df[feat_cols].values.astype(float)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    n_qubits = min(4, max(2, X.shape[1]))
    n_anom_truth = int(y_true.sum())

    t0 = time.time()
    qpca_scores = _QuantumPCAResidual(n_qubits=n_qubits, n_components=2,
                                       reps=2, max_iter=60, n_restarts=2).fit_score(Xs)
    qpca_secs = round(time.time() - t0, 2)

    t0 = time.time()
    qae_scores = _QuantumAutoencoder(n_qubits=n_qubits, n_trash=1, reps=2,
                                       max_iter=60, n_restarts=2,
                                       max_train=min(120, len(Xs))).fit_score(Xs)
    qae_secs = round(time.time() - t0, 2)

    # threshold = take top n_anom_truth scores → predicted-positive
    def topk_predict(scores, k):
        thr = np.sort(scores)[-k] if 0 < k <= len(scores) else np.inf
        return (scores >= thr).astype(int)

    qpca_pred = topk_predict(qpca_scores, n_anom_truth)
    qae_pred  = topk_predict(qae_scores,  n_anom_truth)
    combined_scores = (qpca_scores + qae_scores) / 2
    combined_pred = topk_predict(combined_scores, n_anom_truth)

    metrics = {
        "qpca":     {**_confusion(y_true, qpca_pred), "auc": round(_auc(y_true, qpca_scores), 4)},
        "qae":      {**_confusion(y_true, qae_pred),  "auc": round(_auc(y_true, qae_scores),  4)},
        "combined": {**_confusion(y_true, combined_pred), "auc": round(_auc(y_true, combined_scores), 4)},
    }

    samples = []
    for i in range(len(df)):
        samples.append({
            "id": ids[i],
            "true": int(y_true[i]),
            "qpca_score": round(float(qpca_scores[i]), 4),
            "qae_score":  round(float(qae_scores[i]),  4),
            "combined_score": round(float(combined_scores[i]), 4),
            "qpca_pred": int(qpca_pred[i]),
            "qae_pred":  int(qae_pred[i]),
            "combined_pred": int(combined_pred[i]),
        })

    return JSONResponse({
        "dataset": {"name": name, **REAL_DATASETS[name]},
        "stats": {
            "n_samples": int(len(df)),
            "n_features": int(len(feat_cols)),
            "n_anomaly_truth": n_anom_truth,
            "feature_names": feat_cols,
            "n_qubits": n_qubits,
            "qpca_seconds": qpca_secs,
            "qae_seconds": qae_secs,
        },
        "metrics": metrics,
        "samples": samples,
    })


@app.get("/demo-dataset")
async def demo_dataset(scenario: str = "vehicle"):
    if scenario not in SCENARIOS:
        raise HTTPException(400, f"Unknown scenario. Choose from: {list(SCENARIOS.keys())}")

    cfg = SCENARIOS[scenario]
    rng = np.random.default_rng(42)
    rows = []

    # Normal entities
    for i in range(cfg["n_normal"]):
        row = {cfg["id_col"]: f"{cfg['id_prefix']}-{i+1:03d}"}
        for feat, (mu, sigma) in cfg["features"].items():
            row[feat] = round(float(rng.normal(mu, sigma)), 4)
        rows.append(row)

    # Planted outliers
    total = cfg["n_normal"]
    for j, out in enumerate(cfg["outliers"]):
        row = {cfg["id_col"]: f"{cfg['id_prefix']}-OUT-{j+1:02d}"}
        row.update(out)
        # Insert at a random position (not always last)
        pos = rng.integers(0, total + 1)
        rows.insert(int(pos), row)
        total += 1

    df = pd.DataFrame(rows)
    csv_bytes = df.to_csv(index=False).encode("utf-8")

    return StreamingResponse(
        io.BytesIO(csv_bytes),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=demo_{scenario}.csv"},
    )
