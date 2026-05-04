"""
ml/run.py
---------
Entry point. Orchestrates the full pipeline end-to-end.

Run with:
    cd ml
    pip install -r requirements.txt
    python run.py

This file contains NO business logic — it only wires together the modules.
"""

from data_loader import load_tox21
from preprocessing import QuantumPreprocessor, one_class_split
from quantum_kernel import QuantumKernelBuilder
from model import QuantumOCSVM, ClassicalOCSVM
from evaluation import compute_metrics, plot_results

N_COMPONENTS = 6   # qubits


def main() -> None:
    # 1. Load data
    print("Loading Tox21 SR-MMP...")
    X_train_raw, y_train_raw, X_test_raw, y_test_raw = load_tox21()

    # 2. PCA + normalize
    prep = QuantumPreprocessor(n_components=N_COMPONENTS)
    X_train_all = prep.fit_transform(X_train_raw)
    X_test_all  = prep.transform(X_test_raw)

    # 3. One-class split (train on non-toxic only)
    X_train, X_test, y_test = one_class_split(
        X_train_all, y_train_raw, n_train=200, n_test=100
    )
    # X_test_all used for 2D scatter (first 2 PCA dims)
    _, X_test_eval, _ = one_class_split(
        X_test_all, y_test_raw, n_train=200, n_test=100
    )

    # 4. Quantum kernel matrices
    print("Building quantum kernel (may take several minutes)...")
    qk = QuantumKernelBuilder(n_features=N_COMPONENTS, reps=2)
    K_train = qk.evaluate(X_train)
    K_test  = qk.evaluate(X_test, X_train)
    print("Kernel computation complete.")

    # 5. Quantum OCSVM
    q_model = QuantumOCSVM(nu=0.1).fit(K_train)
    q_pred   = q_model.predict(K_test)
    q_scores = q_model.decision_scores(K_test)

    # 6. Classical baseline
    c_model  = ClassicalOCSVM(nu=0.1).fit(X_train)
    c_pred   = c_model.predict(X_test)
    c_scores = c_model.decision_scores(X_test)

    # 7. Metrics
    q_metrics = compute_metrics("Quantum OCSVM",       y_test, q_pred, q_scores)
    c_metrics = compute_metrics("Classical RBF OCSVM", y_test, c_pred, c_scores)

    # 8. Plots
    plot_results(
        y_true=y_test,
        X_test_2d=X_test_eval,
        q_pred=q_pred,
        q_scores=q_scores,
        c_pred=c_pred,
        c_scores=c_scores,
        K_train=K_train,
        output_path="nautilus_results.png",
    )


if __name__ == "__main__":
    main()
