from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib, re, pickle
import pandas as pd
from urllib.parse import urlparse

app = Flask(__name__)
CORS(app)

# ── Load all 3 models ────────────────────────────────────────────
models = {
    "Logistic Regression": joblib.load("model_lr.pkl"),
    "Random Forest":       joblib.load("model_rf.pkl"),
    "XGBoost":             joblib.load("model_xgb.pkl"),
}

with open("feature_names.pkl", "rb") as f:
    FEATURE_NAMES = pickle.load(f)

print(f"✅ All 3 models loaded. Expecting {len(FEATURE_NAMES)} features.")

# ── Feature extraction ───────────────────────────────────────────
def get_features(url):
    p = urlparse(url if url.startswith("http") else "https://" + url)
    h = p.hostname or ""

    known = {
        "url_length":      len(url),
        "hostname_length": len(h),
        "path_length":     len(p.path),
        "num_dots":        url.count("."),
        "num_hyphens":     url.count("-"),
        "num_underscores": url.count("_"),
        "num_slashes":     url.count("/"),
        "num_at":          url.count("@"),
        "num_question":    url.count("?"),
        "num_equal":       url.count("="),
        "num_ampersand":   url.count("&"),
        "num_percent":     url.count("%"),
        "num_digits":      sum(c.isdigit() for c in url),
        "num_letters":     sum(c.isalpha() for c in url),
        "uses_https":      int(url.startswith("https")),
        "has_ip":          int(bool(re.search(r"\d+\.\d+\.\d+\.\d+", url))),
        "num_subdomains":  max(0, len(h.split(".")) - 2),
        "has_port":        int(p.port is not None),
        "domain_length":   len(h),
        "tld_length":      len(h.split(".")[-1]) if "." in h else 0,
    }

    row = {col: known.get(col, 0) for col in FEATURE_NAMES}
    return pd.DataFrame([row], columns=FEATURE_NAMES)

# ── Reason generator ─────────────────────────────────────────────
def get_reasons(url):
    reasons = []
    p = urlparse(url if url.startswith("http") else "https://" + url)
    h = p.hostname or ""

    if re.search(r"\d+\.\d+\.\d+\.\d+", url):
        reasons.append({"text": "IP address used instead of domain", "type": "bad"})
    if len(url) > 75:
        reasons.append({"text": f"Very long URL ({len(url)} chars)", "type": "bad"})
    if not url.startswith("https"):
        reasons.append({"text": "No HTTPS encryption", "type": "bad"})
    if re.search(r"login|verify|secure|account|update|paypal|bank", url, re.I):
        reasons.append({"text": "Phishing keywords detected", "type": "bad"})
    if url.count(".") > 4:
        reasons.append({"text": f"Too many dots in URL ({url.count('.')})", "type": "bad"})
    if url.count("-") > 3:
        reasons.append({"text": f"Many hyphens detected ({url.count('-')})", "type": "bad"})
    if h.split(".")[-1] in ["tk", "ml", "cf", "ga", "xyz", "gq"]:
        reasons.append({"text": f"Risky domain ending (.{h.split('.')[-1]})", "type": "bad"})
    if url.startswith("https"):
        reasons.append({"text": "HTTPS is present", "type": "good"})
    if not re.search(r"\d+\.\d+\.\d+\.\d+", url):
        reasons.append({"text": "No raw IP address", "type": "good"})
    if len(url) <= 75:
        reasons.append({"text": "URL length is normal", "type": "good"})

    return reasons

# ── Routes ───────────────────────────────────────────────────────
@app.route("/health")
def health():
    return jsonify({"status": "ok"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        url = request.get_json().get("url", "").strip()
        if not url:
            return jsonify({"error": "No URL provided"}), 400

        features = get_features(url)
        results = {}

        for name, model in models.items():
            prob = float(model.predict_proba(features)[0][1])
            results[name] = {
                "label":      int(prob > 0.5),
                "confidence": round(prob, 4),
            }

        # Majority vote across all 3 models
        votes    = sum(1 for r in results.values() if r["label"] == 1)
        avg_prob = round(sum(r["confidence"] for r in results.values()) / 3, 4)
        consensus = int(votes >= 2)

        return jsonify({
            "url":       url,
            "consensus": {"label": consensus, "confidence": avg_prob, "votes": votes},
            "models":    results,
            "reasons":   get_reasons(url),
        })

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(port=5000, debug=True)