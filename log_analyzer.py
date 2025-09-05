import numpy as np
import pandas as pd
import tkinter as tk
from datetime import datetime
from tkinter import filedialog
from collections import defaultdict
from db_utils import get_db_path, init_db, save_to_db, check_db_dates
from log_utils import analyze_log, save_report, create_activity_chart
from ml_prepare import prepare_ml_data_from_db
from visualization_utils import generate_attack_distribution_chart
from db_utils import clear_database
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tkinter import messagebox
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

MAX_DAYS_FOR_VISUALIZATION = 124

def run_analysis():
    file_paths = filedialog.askopenfilenames(filetypes=[("Log files", "*.log")])
    if not file_paths:
        output_text.config(state='normal')
        output_text.insert(tk.END, "[DB] File selection canceled. No logs loaded.\n")
        output_text.config(state='disabled')
        return

    total_combined_results = defaultdict(int)
    output_text.config(state='normal')
    output_text.delete(1.0, tk.END)

    for file_path in file_paths:
        results, total, invalid, invalid_list = analyze_log(file_path)

        if invalid > 0:
            output_text.insert(tk.END, f"[WARN] {invalid} invalid line(s) skipped in {file_path}\n")

            if invalid > 10:
                dump_path = file_path.replace(".log", "_invalid_lines.txt")
                try:
                    with open(dump_path, "w", encoding="utf-8") as dump:
                        dump.write(f"Invalid lines in: {file_path}\n")
                        dump.write(f"Total invalid: {invalid}\n\n")
                        for ln, txt in invalid_list:
                            dump.write(f"{ln}: {txt}\n")
                    output_text.insert(
                        tk.END,
                        f"       Full list saved to: {dump_path}\n"
                    )
                except Exception as e:
                    output_text.insert(
                        tk.END,
                        f"[ERROR] Failed to write invalid lines file: {e}\n"
                    )
            else:
                for ln, txt in invalid_list:
                    snippet = (txt[:120] + "…") if len(txt) > 120 else txt
                    output_text.insert(tk.END, f"       line {ln}: {snippet}\n")

        report = save_report(results, total, file_path)
        save_to_db(file_path, results)

        output_text.insert(tk.END, f"File: {file_path}\n")
        output_text.insert(tk.END, f"Total rows: {total}\n")
        if results:
            output_text.insert(tk.END, "Suspicious activities detected:\n")
            for k, v in results.items():
                output_text.insert(tk.END, f"- {k}: {v} times\n")
                total_combined_results[k] += v
        else:
            output_text.insert(tk.END, "No suspicious activity.\n")

        chart_path = file_path.replace(".log", "_chart.png")
        output_text.insert(tk.END, f"\nRecorded report: {report}")
        output_text.insert(tk.END, f"\nSaved graphics: {chart_path}\n\n")

        create_activity_chart(results, file_path)

    output_text.insert(tk.END, "\n=== Summary of all files ===\n")
    for k, v in total_combined_results.items():
        output_text.insert(tk.END, f"- {k}: {v} times total\n")

    output_text.config(state='disabled')

def check_db():
    output_text.config(state='normal')
    output_text.insert(tk.END, "\n[DB] Checking available dates in the database:\n")

    dates = check_db_dates(get_db_path())
    if not dates:
        output_text.insert(tk.END, "No data found in the database.\n")
    else:
        for d in dates:
            output_text.insert(tk.END, f"- {d}\n")

    output_text.config(state='disabled')

def handle_clear_database():
    if not messagebox.askokcancel(
        "Confirm",
        "Delete ALL data in the database?\n\nThis action cannot be undone."
    ):
        return

    soft = messagebox.askyesno(
        "Clear mode",
        "Soft clear (keep schema)?\n\nYes = Soft (DELETE rows)\nNo = Hard (delete DB file)"
    )

    ok, msg = clear_database(soft=soft)

    output_text.config(state='normal')
    if ok:
        output_text.insert(
            tk.END,
            f"\n[DB] Database cleared ({'soft' if soft else 'hard'} mode). {msg}\n"
        )
    else:
        output_text.insert(tk.END, f"\n[DB] Clear failed: {msg}\n")
    output_text.config(state='disabled')

def load_and_validate_data():
    init_db()

    df_all = prepare_ml_data_from_db(db_path=get_db_path(), limit_days=None)
    if df_all is None or len(df_all) < 2:
        raise ValueError("At least 2 days of data are required for training.")

    attack_cols = [
        'brute_force', 'port_scan', 'sql_injection', 'ddos',
        'malware', 'ip_blocked', 'unauthorized_access', 'intrusion'
    ]
    attack_cols = [c for c in attack_cols if c in df_all.columns]

    n_days_total = len(df_all)

    if n_days_total < 30:
        output_text.config(state='normal')
        output_text.insert(
            tk.END,
            f"\n[ML] Note: only {n_days_total} day(s) loaded. "
            "For reliable ML results, ≥30 days are recommended.\n"
        )
        output_text.config(state='disabled')

    day_winners_all = df_all[attack_cols].idxmax(axis=1).astype(str)
    present_types = set(day_winners_all.unique())
    required_types = set(attack_cols)
    missing_types = sorted(required_types - present_types)

    if missing_types:
        counts = day_winners_all.value_counts().reindex(attack_cols, fill_value=0)
        fallback_attack = counts.idxmax() if counts.sum() > 0 else (attack_cols[0] if attack_cols else "unknown")

        return df_all, attack_cols, {
            "reason": "Not all attack types appear as most common on at least one day.",
            "counts": counts,
            "fallback_attack": fallback_attack
        }

    return df_all, attack_cols, None

def prepare_and_split(df_all, attack_cols):
    X_full = df_all.drop(columns=['date'])
    shifted = df_all.shift(-1)
    valid_idx = shifted.drop(columns=['date']).dropna(how='all').index
    if len(valid_idx) == 0:
        raise ValueError("Not enough consecutive days to make a prediction.")

    y = df_all.loc[valid_idx, attack_cols].idxmax(axis=1).astype(str)
    X = X_full.loc[valid_idx]

    class_counts = y.value_counts()
    rare_classes = class_counts[class_counts < 2].index
    mask_eval = ~y.isin(rare_classes)
    if mask_eval.sum() == 0:
        mask_eval = np.ones(len(y), dtype=bool)

    X_eval = X[mask_eval]
    y_eval = y[mask_eval]

    n_samples = len(y_eval)
    n_classes = y_eval.nunique()

    if n_classes < 2 or n_samples < 2 * n_classes:
        X_train, X_test, y_train, y_test = X_eval, X_eval, y_eval, y_eval
    else:
        test_size_abs = max(n_classes, int(np.ceil(0.2 * n_samples)))
        test_size = min(0.5, test_size_abs / n_samples)
        X_train, X_test, y_train, y_test = train_test_split(
            X_eval, y_eval,
            test_size=test_size,
            random_state=42,
            stratify=y_eval
        )

    return X, y, X_train, X_test, y_train, y_test

def train_and_score_models(X_train, X_test, y_train, y_test):
    accuracies = {}

    models = {
        "DecisionTree": DecisionTreeClassifier(random_state=42),
        "RandomForest": RandomForestClassifier(random_state=42),
        "LogisticRegression": LogisticRegression(max_iter=5000, random_state=42),
        "KNN": KNeighborsClassifier(),
        "SVC": SVC(probability=True),
        "XGBoost": XGBClassifier(random_state=42, tree_method="hist"),
    }

    for name, model in models.items():
        try:
            if name == "XGBoost":
                le_tr = LabelEncoder().fit(y_train)
                n_cls = len(le_tr.classes_)

                Xtr = X_train.astype("float32").values
                ytr = le_tr.transform(y_train)

                mask_seen = y_test.isin(le_tr.classes_)
                if mask_seen.sum() == 0:
                    accuracies[name] = 0.0
                    output_text.config(state='normal')
                    output_text.insert(
                        tk.END, "[ML] XGBoost note: no test samples from seen classes; skipping scoring.\n"
                    )
                    output_text.config(state='disabled')
                    continue

                Xte = X_test[mask_seen].astype("float32").values
                yte = le_tr.transform(y_test[mask_seen])

                if n_cls <= 2:
                    model.set_params(objective="binary:logistic", eval_metric="logloss")
                    model.fit(Xtr, ytr)
                    y_pred_enc = (model.predict_proba(Xte)[:, 1] >= 0.5).astype(int)
                else:
                    model.set_params(objective="multi:softprob",
                                     num_class=n_cls,
                                     eval_metric="mlogloss")
                    model.fit(Xtr, ytr)
                    y_pred_enc = model.predict_proba(Xte).argmax(axis=1)

                accuracies[name] = accuracy_score(yte, y_pred_enc)

            else:
                if name == "KNN":
                    k = min(5, len(X_train))
                    model.set_params(n_neighbors=max(1, k))

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                accuracies[name] = accuracy_score(y_test, y_pred)

        except Exception as e:
            accuracies[name] = 0.0
            output_text.config(state='normal')
            output_text.insert(
                tk.END, f"[ML] {name} error: {type(e).__name__}: {e}\n"
            )
            output_text.config(state='disabled')

    return accuracies

def fit_best_and_predict_next(X, y, accuracies):
    if not accuracies:
        raise ValueError("No accuracies computed; cannot select best model.")

    best_model_name = max(accuracies, key=accuracies.get)

    models = {
        "DecisionTree": DecisionTreeClassifier(random_state=42),
        "RandomForest": RandomForestClassifier(random_state=42),
        "LogisticRegression": LogisticRegression(max_iter=5000, random_state=42),
        "KNN": KNeighborsClassifier(),
        "SVC": SVC(probability=True),
        "XGBoost": XGBClassifier(random_state=42, tree_method="hist"),
    }
    best_model = models[best_model_name]

    if best_model_name == "XGBoost":
        le_full = LabelEncoder().fit(y)
        X_all = X.astype("float32").values
        y_all = le_full.transform(y)

        n_cls_full = len(le_full.classes_)
        if n_cls_full <= 2:
            best_model.set_params(objective="binary:logistic", eval_metric="logloss")
        else:
            best_model.set_params(objective="multi:softprob",
                                  num_class=n_cls_full,
                                  eval_metric="mlogloss")

        best_model.fit(X_all, y_all)

        avg_day = X.mean().astype("float32").to_frame().T.values
        proba = best_model.predict_proba(avg_day)
        next_day_pred = le_full.inverse_transform([proba.argmax(axis=1)[0]])[0]
    else:
        best_model.fit(X, y)
        avg_day = X.mean().to_frame().T
        next_day_pred = best_model.predict(avg_day)[0]

    return best_model_name, next_day_pred

def save_attack_distribution_chart(df_all, attack_cols, next_day_attack, label):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    days = len(df_all)
    suffix = "_NoML" if "No ML" in label or "NoML" in label else ""
    save_path = f"attack_distribution_{ts}_days{days}{suffix}.png"

    df_vis = df_all[['date'] + attack_cols].copy()
    df_vis['date'] = pd.to_datetime(df_vis['date'], errors='coerce')

    show_now = days <= MAX_DAYS_FOR_VISUALIZATION
    generate_attack_distribution_chart(
        df_vis,
        next_day_attack=next_day_attack,
        next_day_label=label,
        save_path=save_path,
        show=show_now
    )

    output_text.config(state='normal')
    if show_now:
        output_text.insert(tk.END, f"\n[ML] Saved figure: {save_path}\n")
    else:
        output_text.insert(tk.END, f"\n[ML] Visualization saved as file (too many days for direct display): {save_path}\n")
    output_text.config(state='disabled')

    return save_path

def handle_blocked_prediction(df_all, attack_cols, counts, fallback_attack):
    missing = [c for c in attack_cols if counts.get(c, 0) == 0]

    output_text.config(state='normal')
    output_text.insert(tk.END, "\n[ML] Prediction blocked: dataset is not representative.\n")
    output_text.insert(tk.END, "[ML] Not all attack types appear as most common on at least one day.\n")
    if missing:
        output_text.insert(tk.END, f"[ML] Missing types: {', '.join(missing)}\n")
    output_text.insert(tk.END, "[ML] Days per attack type in the loaded period:\n")
    for t in attack_cols:
        output_text.insert(tk.END, f"   - {t}: {int(counts.get(t, 0))} day(s)\n")
    output_text.insert(tk.END, "[ML] Please load more logs (≥ 8 representative days; ≥ 30 days recommended).\n")
    output_text.config(state='disabled')

    save_attack_distribution_chart(df_all, attack_cols, fallback_attack, "Next day (No ML)")

def run_ml_prediction():
    try:
        df_all, attack_cols, blocked = load_and_validate_data()

        if blocked is not None:
            handle_blocked_prediction(
                df_all=df_all,
                attack_cols=attack_cols,
                counts=blocked["counts"],
                fallback_attack=blocked["fallback_attack"]
            )
            return

        X, y, X_train, X_test, y_train, y_test = prepare_and_split(df_all, attack_cols)

        accuracies = train_and_score_models(X_train, X_test, y_train, y_test)

        best_model_name, next_day_pred = fit_best_and_predict_next(X, y, accuracies)

        output_text.config(state='normal')
        output_text.insert(tk.END, "\n[ML] Accuracy by model:\n")
        for name in sorted(accuracies.keys(), key=lambda k: accuracies[k], reverse=True):
            output_text.insert(tk.END, f"- {name}: {accuracies[name]*100:.0f}%\n")
        output_text.insert(tk.END, f"[ML] Selected model: {best_model_name} (Accuracy={accuracies[best_model_name]*100:.0f}%)\n")
        output_text.insert(tk.END, f"[ML] Predicted most common attack for tomorrow: {next_day_pred}\n")
        output_text.config(state='disabled')

        save_attack_distribution_chart(df_all, attack_cols, next_day_pred, "Next day")

    except Exception as e:
        output_text.config(state='normal')
        output_text.insert(tk.END, f"\n[ML] Prediction error: {e}\n")
        output_text.config(state='disabled')

def exit_app():
    root.destroy()

root = tk.Tk()
root.title("Log Analyzer with ML Prediction")

frame = tk.Frame(root)
frame.pack(pady=10)

btn_select = tk.Button(frame, text="Select log file(s)", command=run_analysis)
btn_select.grid(row=0, column=0, padx=5)

btn_db = tk.Button(frame, text="Check DB", command=check_db)
btn_db.grid(row=0, column=1, padx=5)

btn_clear = tk.Button(frame, text="Clear DB", command=handle_clear_database)
btn_clear.grid(row=0, column=2, padx=5)

btn_ml = tk.Button(frame, text="ML Prediction", command=run_ml_prediction)
btn_ml.grid(row=0, column=3, padx=5)

btn_exit = tk.Button(frame, text="Exit", command=exit_app)
btn_exit.grid(row=0, column=4, padx=5)

root.rowconfigure(0, weight=1)
root.columnconfigure(0, weight=1)

output_text = tk.Text(root, bg="#1e1e1e", fg="white")
output_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
output_text.config(state='disabled')

root.mainloop()