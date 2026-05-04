"""
AnomaQ Web Server — FastAPI backend with SSE progress streaming
Run with: python -m uvicorn main:app --reload
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
import io, time, warnings, traceback, json, asyncio
from concurrent.futures import ThreadPoolExecutor

warnings.filterwarnings('ignore')

app = FastAPI(title="AnomaQ")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

executor = ThreadPoolExecutor(max_workers=2)

def sse(event, data):
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"

def run_detection_streaming(df, use_quantum, emit):
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

    emit("progress", {"pct": 5, "label": "Scaling features..."})
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)
    contamination = min(0.1, max(1 / n_samples, 0.02))

    def get_label(idx):
        return str(df.iloc[idx][label_col]) if label_col else f"Row {idx}"

    emit("progress", {"pct": 12, "label": "Running Isolation Forest..."})
    iso = IsolationForest(contamination=contamination, random_state=42)
    iso_pred = iso.fit_predict(X)
    iso_scores = iso.score_samples(X)
    iso_outliers = np.where(iso_pred == -1)[0].tolist()

    emit("progress", {"pct": 25, "label": "Running Local Outlier Factor..."})
    lof = LocalOutlierFactor(n_neighbors=min(5, n_samples - 1), contamination=contamination)
    lof_pred = lof.fit_predict(X)
    lof_scores = lof.negative_outlier_factor_
    lof_outliers = np.where(lof_pred == -1)[0].tolist()

    emit("progress", {"pct": 37, "label": "Running One-Class SVM (RBF)..."})
    svm = OneClassSVM(kernel='rbf', nu=contamination, gamma='scale')
    svm_pred = svm.fit_predict(X)
    svm_scores = svm.score_samples(X)
    svm_outliers = np.where(svm_pred == -1)[0].tolist()

    emit("progress", {"pct": 46, "label": "Computing pairwise interaction features..."})
    if len(numeric_cols) <= 15:
        poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        X_poly = StandardScaler().fit_transform(poly.fit_transform(X_raw))
        iso2_outliers = np.where(IsolationForest(contamination=contamination, random_state=42).fit_predict(X_poly) == -1)[0].tolist()
        lof2_outliers = np.where(LocalOutlierFactor(n_neighbors=min(5, n_samples-1), contamination=contamination).fit_predict(X_poly) == -1)[0].tolist()
        svm2_outliers = np.where(OneClassSVM(kernel='rbf', nu=contamination, gamma='scale').fit_predict(X_poly) == -1)[0].tolist()
    else:
        iso2_outliers = lof2_outliers = svm2_outliers = []

    emit("progress", {"pct": 55, "label": "Merging classical results..."})
    all_classical = set(iso_outliers)|set(lof_outliers)|set(svm_outliers)|set(iso2_outliers)|set(lof2_outliers)|set(svm2_outliers)

    classical = {
        "isolation_forest": [{"idx": i, "label": get_label(i), "score": float(iso_scores[i])} for i in iso_outliers],
        "lof": [{"idx": i, "label": get_label(i), "score": float(lof_scores[i])} for i in lof_outliers],
        "svm_rbf": [{"idx": i, "label": get_label(i), "score": float(svm_scores[i])} for i in svm_outliers],
        "isolation_forest_interactions": [{"idx": i, "label": get_label(i)} for i in iso2_outliers],
        "lof_interactions": [{"idx": i, "label": get_label(i)} for i in lof2_outliers],
        "svm_interactions": [{"idx": i, "label": get_label(i)} for i in svm2_outliers],
        "all_unique": [{"idx": i, "label": get_label(i)} for i in sorted(all_classical)],
    }

    quantum = None
    if use_quantum and n_samples <= 300:
        try:
            emit("progress", {"pct": 62, "label": "Reducing dimensions with PCA..."})
            from qiskit.circuit.library import ZZFeatureMap
            from qiskit_machine_learning.kernels import FidelityQuantumKernel

            n_components = min(5, len(numeric_cols), n_samples - 1)
            pca = PCA(n_components=n_components)
            X_pca = pca.fit_transform(X)
            variance_retained = float(pca.explained_variance_ratio_.sum())

            q_scaler = MinMaxScaler(feature_range=(0, np.pi))
            X_q = q_scaler.fit_transform(X_pca)

            emit("progress", {"pct": 70, "label": f"Building {n_components}-qubit ZZFeatureMap circuit..."})
            feature_map = ZZFeatureMap(feature_dimension=n_components, reps=2, entanglement='linear')
            qk = FidelityQuantumKernel(feature_map=feature_map)

            emit("progress", {"pct": 76, "label": f"Computing quantum kernel matrix ({n_samples}x{n_samples} pairs)..."})
            t0 = time.time()
            K = qk.evaluate(X_q)
            q_time = round(time.time() - t0, 2)

            emit("progress", {"pct": 91, "label": f"Kernel done in {q_time}s — fitting quantum SVM..."})
            qsvm = OneClassSVM(kernel='precomputed', nu=contamination)
            qsvm_pred = qsvm.fit_predict(K)
            qsvm_outliers = np.where(qsvm_pred == -1)[0].tolist()

            emit("progress", {"pct": 95, "label": "Ranking by quantum similarity score..."})
            avg_sim = np.mean(K, axis=1)
            ranked = np.argsort(avg_sim).tolist()
            quantum_only = set(qsvm_outliers) - all_classical

            qonly_details = []
            for idx in quantum_only:
                feats = [{"name": f, "value": round(float(df.iloc[idx][f]), 4),
                          "z_score": round(float(abs(df.iloc[idx][f] - df[f].mean()) / df[f].std()) if df[f].std() > 0 else 0, 2)}
                         for f in numeric_cols]
                qonly_details.append({"idx": idx, "label": get_label(idx), "features": feats})

            quantum = {
                "qubits": n_components,
                "variance_retained": round(variance_retained * 100, 1),
                "kernel_time_s": q_time,
                "outliers": [{"idx": i, "label": get_label(i), "also_classical": i in all_classical} for i in qsvm_outliers],
                "ranking": [{"rank": r+1, "idx": ranked[r], "label": get_label(ranked[r]),
                              "avg_similarity": round(float(avg_sim[ranked[r]]), 6),
                              "also_classical": ranked[r] in all_classical} for r in range(min(10, len(ranked)))],
                "quantum_only": qonly_details,
            }
        except Exception as e:
            quantum = {"error": str(e)}

    emit("progress", {"pct": 99, "label": "Finalizing..."})
    emit("result", {
        "meta": {"rows": n_samples, "features": numeric_cols, "label_col": label_col, "contamination": round(contamination, 3)},
        "classical": classical,
        "quantum": quantum,
        "summary": {
            "classical_unique": len(all_classical),
            "quantum_flagged": len(quantum["outliers"]) if quantum and "outliers" in quantum else None,
            "quantum_only": len(quantum["quantum_only"]) if quantum and "quantum_only" in quantum else None,
        }
    })
    emit("progress", {"pct": 100, "label": "Done!"})


@app.post("/analyze-stream")
async def analyze_stream(file: UploadFile = File(...), quantum: bool = True):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "Please upload a .csv file.")
    contents = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(400, f"Could not parse CSV: {e}")

    queue = asyncio.Queue()
    def emit(event, data):
        queue.put_nowait(sse(event, data))

    async def run_bg():
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(executor, run_detection_streaming, df, quantum, emit)
        except Exception as e:
            queue.put_nowait(sse("error", {"message": str(e)}))
        finally:
            queue.put_nowait(None)

    asyncio.create_task(run_bg())

    async def stream():
        yield sse("progress", {"pct": 2, "label": "Parsing CSV..."})
        while True:
            msg = await queue.get()
            if msg is None:
                break
            yield msg

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/", response_class=HTMLResponse)
async def root():
    with open("index.html") as f:
        return f.read()
