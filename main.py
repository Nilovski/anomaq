"""
Catalyst — Domain-Aware Quantum + Classical Outlier Detection
"""
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
import io, time, warnings, json, asyncio, traceback
from concurrent.futures import ThreadPoolExecutor

warnings.filterwarnings('ignore')

app = FastAPI(title="Catalyst")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
executor = ThreadPoolExecutor(max_workers=2)

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
            quantum_only = set(qsvm_outliers) - all_classical

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
            "quantum_flagged": len(quantum["outliers"]) if quantum and "outliers" in quantum else None,
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
