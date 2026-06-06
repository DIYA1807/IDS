
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.utils import shuffle
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import joblib

# ── Configuration ─────────────────────────────────────────────────────────────
TRAIN_PATH     = r"C:\Users\asd\OneDrive\Desktop\PROJECT\IDS\nsl-kdd\KDDTrain+.txt"
TEST_PATH      = r"C:\Users\asd\OneDrive\Desktop\PROJECT\IDS\nsl-kdd\KDDTest+.txt"
RF_MODEL_PATH  = r"C:\Users\asd\OneDrive\Desktop\PROJECT\IDS\rf_model.pkl"
XGB_MODEL_PATH = r"C:\Users\asd\OneDrive\Desktop\PROJECT\IDS\xgb_model.pkl"

MIN_CLASS_SAMPLES = 30
EXCLUDE_LABELS    = ["warezclient"]
NORMAL_LABEL      = "normal"

COL_NAMES = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label", "difficulty"
]

CAT_COLS = ["protocol_type", "service", "flag"]


# ── Step 1: Load Data ─────────────────────────────────────────────────────────
def load_data():
    print("[1/4] Loading dataset...")
    train_df = pd.read_csv(TRAIN_PATH, names=COL_NAMES)
    test_df  = pd.read_csv(TEST_PATH,  names=COL_NAMES)

    train_df.drop("difficulty", axis=1, inplace=True)
    test_df.drop("difficulty",  axis=1, inplace=True)

    print(f"      Train samples : {len(train_df)}")
    print(f"      Test  samples : {len(test_df)}")
    return train_df, test_df


# ── Step 2: Preprocess ────────────────────────────────────────────────────────
def engineer_features(df):
    df = df.copy()
    df["failed_login_ratio"] = df["num_failed_logins"] / (df["count"] + 1)
    df["root_shell_ratio"]   = df["root_shell"] / (df["duration"] + 1)
    df["suspicion_score"]    = df["num_failed_logins"] + df["root_shell"] + df["su_attempted"]
    return df


def preprocess(train_df, test_df):
    print("[2/4] Preprocessing...")

    all_labels = pd.concat([train_df["label"], test_df["label"]])
    le_label   = LabelEncoder()
    le_label.fit(all_labels)

    train_df["label"] = le_label.transform(train_df["label"])
    test_df["label"]  = le_label.transform(test_df["label"])

    print(f"      Unique classes (raw) : {len(le_label.classes_)}")

    # Drop tiny classes
    class_counts  = train_df["label"].value_counts()
    valid_classes = class_counts[class_counts >= MIN_CLASS_SAMPLES].index
    dropped       = class_counts[class_counts < MIN_CLASS_SAMPLES]

    if len(dropped) > 0:
        print(f"      Dropping {len(dropped)} tiny class(es) with < {MIN_CLASS_SAMPLES} samples:")
        for cls, cnt in dropped.items():
            print(f"        Class {cls} ({le_label.classes_[cls]}) → {cnt} samples")

    # Exclude classes with no test samples
    exclude_encoded = [i for i, name in enumerate(le_label.classes_) if name in EXCLUDE_LABELS]
    if exclude_encoded:
        excluded_names = [le_label.classes_[i] for i in exclude_encoded]
        print(f"      Excluding classes with no test samples: {excluded_names}")
        valid_classes = valid_classes[~valid_classes.isin(exclude_encoded)]

    train_df = train_df[train_df["label"].isin(valid_classes)].copy()
    test_df  = test_df[test_df["label"].isin(valid_classes)].copy()

    # Re-encode labels to be contiguous
    final_le          = LabelEncoder()
    train_df["label"] = final_le.fit_transform(train_df["label"])
    test_df["label"]  = final_le.transform(test_df["label"])

    class_names = {i: le_label.classes_[c] for i, c in enumerate(final_le.classes_)}

    print(f"      Unique classes (after drop) : {len(final_le.classes_)}")
    print(f"      Class mapping : {class_names}")
    print(f"      Train samples (after drop)  : {len(train_df)}")
    print(f"      Test  samples (after drop)  : {len(test_df)}")

    train_df = engineer_features(train_df)
    test_df  = engineer_features(test_df)

    # Encode categorical features
    for col in CAT_COLS:
        le       = LabelEncoder()
        combined = pd.concat([train_df[col], test_df[col]])
        le.fit(combined)
        train_df[col] = le.transform(train_df[col])
        test_df[col]  = le.transform(test_df[col])

    X_train = train_df.drop("label", axis=1)
    y_train = train_df["label"]
    X_test  = test_df.drop("label", axis=1)
    y_test  = test_df["label"]

    feature_names = X_train.columns.tolist()

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, class_names, feature_names, scaler

# ── Step 3: Feature Selection ─────────────────────────────────────────────────
def select_features(X_train, X_test, y_train, feature_names):
    print("[3/4] Selecting important features...")

    FORCE_KEEP = [
        'num_failed_logins', 'num_access_files', 'su_attempted',
        'num_shells', 'root_shell', 'failed_login_ratio',
        'root_shell_ratio', 'suspicion_score'
    ]

    # Train selector model
    selector = SelectFromModel(
        RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1),
        threshold="median"
    )
    selector.fit(X_train, y_train)

    # Initial mask from model
    selected_mask = selector.get_support().copy()

    # Force keep important features
    for i, name in enumerate(feature_names):
        if name in FORCE_KEEP:
            selected_mask[i] = True

    # Apply mask
    X_train_sel = X_train[:, selected_mask]
    X_test_sel  = X_test[:, selected_mask]

    # Get selected feature names
    selected_features = [
        feature_names[i] for i in range(len(feature_names)) if selected_mask[i]
    ]

    print(f"      Features before: {X_train.shape[1]}  |  After: {X_train_sel.shape[1]}")
    print(f"      Selected features ({len(selected_features)}): {selected_features}")

    force_kept = [f for f in FORCE_KEEP if f in selected_features]
    print(f"      ✅ Force-kept ({len(force_kept)}): {force_kept}")

    # 🔥 IMPORTANT: return selected features also
    return X_train_sel, X_test_sel, selected_features

    


# ── Step 4: SMOTE ─────────────────────────────────────────────────────────────
def apply_smote(X_train, y_train):
    print("      Applying SMOTE to balance classes...")

    smote      = SMOTE(sampling_strategy='minority', random_state=42, k_neighbors=3)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    X_res, y_res = shuffle(X_res, y_res, random_state=42)

    print(f"      Samples before SMOTE : {len(y_train)}")
    print(f"      Samples after  SMOTE : {len(y_res)}")
    return X_res, y_res


# ── Two-Stage Predictor ───────────────────────────────────────────────────────
def get_normal_class(class_names):
    for idx, name in class_names.items():
        if name == NORMAL_LABEL:
            return idx
    raise ValueError("'normal' class not found in class_names mapping")


def two_stage_predict(model, X_test, normal_class, guess_passwd_class=None,
                      back_class=None, buffer_overflow_class=None):
    proba        = model.predict_proba(X_test)
    normal_proba = proba[:, normal_class]
    preds        = np.argmax(proba, axis=1).copy()

    # Assign high-confidence normal predictions
    preds[normal_proba >= 0.75] = normal_class

    # Override: back
    if back_class is not None:
        override = (preds == normal_class) & (proba[:, back_class] > 0.05)
        preds[override] = back_class

    # Override: buffer_overflow
    if buffer_overflow_class is not None:
        override = (preds == normal_class) & (proba[:, buffer_overflow_class] > 0.03)
        preds[override] = buffer_overflow_class

    # Override: guess_passwd (last — highest priority)
    if guess_passwd_class is not None:
        override = (preds == normal_class) & (proba[:, guess_passwd_class] > 0.02)
        preds[override] = guess_passwd_class

    return preds


# ── Step 5: Train & Evaluate ──────────────────────────────────────────────────
def train_and_evaluate(X_train, X_test, y_train, y_test, class_names):
    print("[4/4] Training models...\n")

    normal_class          = get_normal_class(class_names)
    target_names          = [class_names[i] for i in sorted(class_names)]
    guess_passwd_class    = next((i for i, l in class_names.items() if l == 'guess_passwd'), None)
    back_class            = next((i for i, l in class_names.items() if l == 'back'), None)
    buffer_overflow_class = next((i for i, l in class_names.items() if l == 'buffer_overflow'), None)

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_split=2,
            max_features='sqrt',
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=350,
            max_depth=8,
            learning_rate=0.1,
            gamma=0.1,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multi:softprob",
            num_class=len(class_names),
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1
        )
    }

    results = {}

    for name, model in models.items():
        print(f"  ── {name} ──")

        # Note: sample_weights less critical now since SMOTE has balanced the training set
        model.fit(X_train, y_train)

        y_pred   = two_stage_predict(model, X_test, normal_class,
                                     guess_passwd_class, back_class,
                                     buffer_overflow_class)
        acc      = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

        print(f"  Accuracy  : {acc * 100:.2f}%")
        print(f"  Macro F1  : {macro_f1:.4f}")
        print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))
        print(confusion_matrix(y_test, y_pred))
        print()

        results[name] = {"accuracy": acc, "macro_f1": macro_f1}

    joblib.dump(models["Random Forest"], RF_MODEL_PATH)
    joblib.dump(models["XGBoost"],       XGB_MODEL_PATH)
    print("  Models saved!")

    best = max(results, key=lambda k: results[k]["macro_f1"])
    print(f"\n  Best Model : {best}")
    print(f"  Macro F1   : {results[best]['macro_f1']:.4f}")
    print(f"  Accuracy   : {results[best]['accuracy'] * 100:.2f}%")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    train_df, test_df = load_data()
    X_train, X_test, y_train, y_test, class_names, feature_names, scaler = preprocess(train_df, test_df)
    
    # Apply SMOTE first (before feature selection) to avoid information leakage
    X_train, y_train = apply_smote(X_train, y_train)
    
    # Then select features on the balanced dataset
    X_train, X_test, selected_features = select_features(
        X_train, X_test, y_train, feature_names
    )

    train_and_evaluate(X_train, X_test, y_train, y_test, class_names)

    
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(selected_features, "selected_features.pkl")
    joblib.dump(feature_names, "feature_names.pkl")
    joblib.dump(class_names, "class_names.pkl")