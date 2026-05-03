"""
AnomaQ Web Server — FastAPI backend
Run with: uvicorn main:app --reload
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
import io, time, warnings, traceback

warnings.filterwarnings('ignore')

app = FastAPI(title="AnomaQ")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def run_detection(df: pd.DataFrame, use_quantum: bool = True):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    non_numeric = [c for c in df.columns if c not in numeric_cols]
    label_col = non_numeric[0] if non_numeric else None

    X_raw = df[numeric_cols].values
    nan_mask = ~np.isnan(X_raw).any(axis=1)
    X_raw = X_raw[nan_mask]
    df = df[nan_mask].reset_index(drop=True)

    n_samples = len(X_raw)
    if n_samples < 5:
        raise ValueError("Need at least 5 rows of data")

    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)
    contamination = min(0.1, max(1/n_samples, 0.02))

    def get_label(idx):
        if label_col:
            return str(df.iloc[idx][label_col])
        return f"Row {idx}"

    results = {}

    # --- Classical ---
    iso = IsolationForest(contamination=contamination, random_state=42)
    iso_pred = iso.fit_predict(X)
    iso_scores = iso.score_samples(X)
    iso_outliers = np.where(iso_pred == -1)[0].tolist()

    lof = LocalOutlierFactor(n_neighbors=min(5, n_samples-1), contamination=contamination)
    lof_pred = lof.fit_predict(X)
    lof_scores = lof.negative_outlier_factor_
    lof_outliers = np.where(lof_pred == -1)[0].tolist()

    svm = OneClassSVM(kernel='rbf', nu=contamination, gamma='scale')
    svm_pred = svm.fit_predict(X)
    svm_scores = svm.score_samples(X)
    svm_outliers = np.where(svm_pred == -1)[0].tolist()

    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X_poly = StandardScaler().fit_transform(poly.fit_transform(X_raw))

    iso2_pred = IsolationForest(contamination=contamination, random_state=42).fit_predict(X_poly)
    iso2_outliers = np.where(iso2_pred == -1)[0].tolist()

    lof2_pred = LocalOutlierFactor(n_neighbors=min(5, n_samples-1), contamination=contamination).fit_predict(X_poly)
    lof2_outliers = np.where(lof2_pred == -1)[0].tolist()

    svm2_pred = OneClassSVM(kernel='rbf', nu=contamination, gamma='scale').fit_predict(X_poly)
    svm2_outliers = np.where(svm2_pred == -1)[0].tolist()

    all_classical = set(iso_outliers) | set(lof_outliers) | set(svm_outliers) | set(iso2_outliers) | set(lof2_outliers) | set(svm2_outliers)

    results["classical"] = {
        "isolation_forest": [{"idx": i, "label": get_label(i), "score": float(iso_scores[i])} for i in iso_outliers],
        "lof": [{"idx": i, "label": get_label(i), "score": float(lof_scores[i])} for i in lof_outliers],
        "svm_rbf": [{"idx": i, "label": get_label(i), "score": float(svm_scores[i])} for i in svm_outliers],
        "isolation_forest_interactions": [{"idx": i, "label": get_label(i)} for i in iso2_outliers],
        "lof_interactions": [{"idx": i, "label": get_label(i)} for i in lof2_outliers],
        "svm_interactions": [{"idx": i, "label": get_label(i)} for i in svm2_outliers],
        "all_unique": [{"idx": i, "label": get_label(i)} for i in sorted(all_classical)],
    }

    # --- Quantum ---
    quantum_result = None
    q_time = None
    if use_quantum:
        try:
            from qiskit.circuit.library import ZZFeatureMap
            from qiskit_machine_learning.kernels import FidelityQuantumKernel

            n_components = min(5, len(numeric_cols), n_samples - 1)
            pca = PCA(n_components=n_components)
            X_pca = pca.fit_transform(X)
            variance_retained = float(pca.explained_variance_ratio_.sum())

            q_scaler = MinMaxScaler(feature_range=(0, np.pi))
            X_q = q_scaler.fit_transform(X_pca)

            feature_map = ZZFeatureMap(feature_dimension=n_components, reps=2, entanglement='linear')
            quantum_kernel = FidelityQuantumKernel(feature_map=feature_map)

            t0 = time.time()
            K = quantum_kernel.evaluate(X_q)
            q_time = round(time.time() - t0, 2)

            qsvm = OneClassSVM(kernel='precomputed', nu=contamination)
            qsvm_pred = qsvm.fit_predict(K)
            qsvm_outliers = np.where(qsvm_pred == -1)[0].tolist()

            avg_sim = np.mean(K, axis=1)
            ranked = np.argsort(avg_sim).tolist()

            quantum_only = set(qsvm_outliers) - all_classical

            quantum_only_details = []
            for idx in quantum_only:
                feats = []
                for fname in numeric_cols:
                    val = float(df.iloc[idx][fname])
                    col_vals = df[fname].values
                    z = float(abs(val - col_vals.mean()) / col_vals.std()) if col_vals.std() > 0 else 0.0
                    feats.append({"name": fname, "value": round(val, 4), "z_score": round(z, 2)})
                quantum_only_details.append({"idx": idx, "label": get_label(idx), "features": feats})

            quantum_result = {
                "qubits": n_components,
                "variance_retained": round(variance_retained * 100, 1),
                "kernel_time_s": q_time,
                "outliers": [{"idx": i, "label": get_label(i), "also_classical": i in all_classical} for i in qsvm_outliers],
                "ranking": [{"rank": r+1, "idx": ranked[r], "label": get_label(ranked[r]), "avg_similarity": round(float(avg_sim[ranked[r]]), 6), "also_classical": ranked[r] in all_classical} for r in range(min(10, len(ranked)))],
                "quantum_only": quantum_only_details,
            }
        except Exception as e:
            quantum_result = {"error": str(e)}

    return {
        "meta": {
            "rows": n_samples,
            "features": numeric_cols,
            "label_col": label_col,
            "contamination": round(contamination, 3),
        },
        "classical": results["classical"],
        "quantum": quantum_result,
        "summary": {
            "classical_unique": len(all_classical),
            "quantum_flagged": len(quantum_result["outliers"]) if quantum_result and "outliers" in quantum_result else None,
            "quantum_only": len(quantum_result["quantum_only"]) if quantum_result and "quantum_only" in quantum_result else None,
        }
    }


@app.post("/analyze")
async def analyze(file: UploadFile = File(...), quantum: bool = True):
    if not file.filename.endswith(".csv"):
        raise HTTPException(400, "Please upload a CSV file")
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        result = run_detection(df, use_quantum=quantum)
        return JSONResponse(result)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, str(e))


@app.get("/", response_class=HTMLResponse)
async def root():
    with open("index.html") as f:
        return f.read()
