
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import joblib
import shap
import os
from datetime import datetime
from collections import deque

# ── App ─────────────────────────────────────────────────────────────
app = FastAPI(title="NIDS - Network Intrusion Detection System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load Models & Artifacts ─────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

rf_model  = joblib.load(os.path.join(BASE_DIR, "rf_model.pkl"))
xgb_model = joblib.load(os.path.join(BASE_DIR, "xgb_model.pkl"))

scaler            = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
selected_features = joblib.load(os.path.join(BASE_DIR, "selected_features.pkl"))
CLASS_NAMES       = joblib.load(os.path.join(BASE_DIR, "class_names.pkl"))

# ── Attack Mapping ─────────────────────────────────────────────────
ATTACK_TYPES = {
    "normal": "Normal Traffic",
    "back": "DoS - Back",
    "buffer_overflow": "U2R - Buffer Overflow",
    "guess_passwd": "R2L - Guess Password",
    "ipsweep": "Probe - IP Sweep",
    "neptune": "DoS - Neptune",
    "nmap": "Probe - Nmap",
    "pod": "DoS - Ping of Death",
    "portsweep": "Probe - Port Sweep",
    "satan": "Probe - Satan",
    "smurf": "DoS - Smurf",
    "teardrop": "DoS - Teardrop"
}

# ── Full Feature List (must match training order exactly) ────────────
FEATURE_NAMES = [
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
    # ✅ Engineered features
    "failed_login_ratio",
    "root_shell_ratio",
    "suspicion_score"
]

# ── Two-Stage Predictor ───────────────────────────────────────────
def get_normal_class(class_names_dict):
    """Find the index of the 'normal' class."""
    for idx, name in class_names_dict.items():
        if name == "normal":
            return idx
    raise ValueError("'normal' class not found in class_names mapping")


def two_stage_predict(model, X_test, normal_class, guess_passwd_class=None,
                      back_class=None, buffer_overflow_class=None):
    """Two-stage prediction: high-confidence normal, then attack overrides."""
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


# ── History ────────────────────────────────────────────────────────
history = deque(maxlen=100)

# ── Initialize Attack Detection Classes ────────────────────────────
NORMAL_CLASS = get_normal_class(CLASS_NAMES)
GUESS_PASSWD_CLASS = next((i for i, l in CLASS_NAMES.items() if l == 'guess_passwd'), None)
BACK_CLASS = next((i for i, l in CLASS_NAMES.items() if l == 'back'), None)
BUFFER_OVERFLOW_CLASS = next((i for i, l in CLASS_NAMES.items() if l == 'buffer_overflow'), None)

# ── Input Schema ───────────────────────────────────────────────────
class TrafficInput(BaseModel):
    duration: float = 0.0
    protocol_type: float = 1.0
    service: float = 0.0
    flag: float = 0.0
    src_bytes: float = 0.0
    dst_bytes: float = 0.0
    num_failed_logins: float = 0.0
    logged_in: float = 1.0
    root_shell: float = 1.0
    su_attempted: float = 0.0
    num_shells: float = 0.0
    num_access_files: float = 0.0
    count: float = 1.0
    srv_count: float = 1.0
    serror_rate: float = 0.0
    srv_serror_rate: float = 0.0
    rerror_rate: float = 0.0
    same_srv_rate: float = 1.0
    diff_srv_rate: float = 0.0
    dst_host_count: float = 1.0
    dst_host_srv_count: float = 1.0
    dst_host_same_srv_rate: float = 1.0
    dst_host_diff_srv_rate: float = 0.0
    dst_host_same_src_port_rate: float = 0.0
    dst_host_srv_diff_host_rate: float = 0.0
    dst_host_serror_rate: float = 0.0
    dst_host_srv_serror_rate: float = 0.0
    dst_host_rerror_rate: float = 0.0
    # Note: failed_login_ratio, root_shell_ratio, suspicion_score
    # are computed server-side — not required from frontend

# ── Routes ─────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def root():
    with open(os.path.join(BASE_DIR, "templates", "index.html")) as f:
        return f.read()


@app.post("/predict")
async def predict(data: TrafficInput):
    try:
        # ── Feature Engineering (computed here, not from frontend) ──
        failed_login_ratio = data.num_failed_logins / (data.count + 1)
        root_shell_ratio   = data.root_shell / (data.count + 1)
        suspicion_score    = data.num_failed_logins + data.root_shell + data.su_attempted

        # ── Map engineered features so get_val can find them ──
        engineered = {
            "failed_login_ratio": failed_login_ratio,
            "root_shell_ratio":   root_shell_ratio,
            "suspicion_score":    suspicion_score,
        }

        # ── Build raw input array in exact FEATURE_NAMES order ──
        # Fields not in schema (duration, land, etc.) default to 0.0
        def get_val(f):
            if f in engineered:
                return engineered[f]
            return getattr(data, f, 0.0)

        raw     = np.array([[get_val(f) for f in FEATURE_NAMES]])

        # ── Scale ──
        scaled  = scaler.transform(raw)

        # ── Feature Selection ──
        indices  = [FEATURE_NAMES.index(f) for f in selected_features]
        features = scaled[:, indices]

        # ── Predictions (using two-stage detection for accurate attack identification) ──
        rf_pred_2stage = two_stage_predict(
            rf_model, features, NORMAL_CLASS,
            GUESS_PASSWD_CLASS, BACK_CLASS, BUFFER_OVERFLOW_CLASS
        )[0]
        rf_pred = int(rf_pred_2stage)
        
        rf_proba = rf_model.predict_proba(features)[0]
        rf_conf  = float(rf_proba[rf_pred])

        xgb_proba = xgb_model.predict_proba(features)[0]
        xgb_pred  = int(np.argmax(xgb_proba))
        xgb_conf  = float(xgb_proba[xgb_pred])

        label     = CLASS_NAMES[rf_pred]
        is_attack = label != "normal"

        # ── SHAP ──
        try:
            explainer = shap.TreeExplainer(rf_model)
            shap_vals = explainer.shap_values(features)
            sv        = shap_vals[rf_pred][0]

            shap_data = sorted(
                [
                    {"feature": selected_features[i], "value": float(sv[i])}
                    for i in range(len(selected_features))
                ],
                key=lambda x: abs(x["value"]),
                reverse=True
            )[:10]

        except Exception:
            shap_data = [{"feature": f, "value": 0.0} for f in selected_features[:10]]

        # ── Response ──
        result = {
            "timestamp":   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "prediction":  label,
            "attack_type": ATTACK_TYPES.get(label, label),
            "is_attack":   is_attack,
            "rf":  {"label": CLASS_NAMES[rf_pred],  "confidence": round(rf_conf  * 100, 2)},
            "xgb": {"label": CLASS_NAMES[xgb_pred], "confidence": round(xgb_conf * 100, 2)},
            "all_probabilities": {
                CLASS_NAMES[i]: round(float(p) * 100, 2)
                for i, p in enumerate(rf_proba)
            },
            "shap": shap_data,
        }

        history.appendleft(result)
        return JSONResponse(result)

    except Exception as e:
        import traceback
        traceback.print_exc()   # prints full error in terminal for debugging
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/history")
async def get_history():
    return JSONResponse(list(history))


@app.get("/stats")
async def get_stats():
    total   = len(history)
    attacks = sum(1 for h in history if h["is_attack"])
    normal  = total - attacks

    breakdown = {}
    for h in history:
        breakdown[h["prediction"]] = breakdown.get(h["prediction"], 0) + 1

    return JSONResponse({
        "total":     total,
        "attacks":   attacks,
        "normal":    normal,
        "breakdown": breakdown
    })