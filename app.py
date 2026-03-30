from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib, re, pickle, math
import pandas as pd
from urllib.parse import urlparse

app = Flask(__name__)
CORS(app)

models = {
    "Logistic Regression": joblib.load("model_lr.pkl"),
    "Random Forest":       joblib.load("model_rf.pkl"),
    "XGBoost":             joblib.load("model_xgb.pkl"),
}

with open("feature_names.pkl", "rb") as f:
    FEATURE_NAMES = pickle.load(f)

print(f"✅ All 3 models loaded. Expecting {len(FEATURE_NAMES)} features.")

SHORTENED_DOMAINS = ["bit.ly","tinyurl.com","goo.gl","t.co","ow.ly","is.gd","buff.ly","short.link"]

SUSPICIOUS_WORDS = ["login","verify","update","bank","secure","free","account","paypal","ebay","amazon","microsoft","apple","confirm","password","suspend","urgent","click","claim","winner"]

def calc_entropy(s):
    if not s:
        return 0
    freq = {}
    for c in s:
        freq[c] = freq.get(c, 0) + 1
    length = len(s)
    return -sum((count/length) * math.log2(count/length) for count in freq.values())

def get_tld(hostname):
    """Extract TLD — tries multi-level first (co.uk), then single."""
    parts = hostname.lower().replace("www.", "").split(".")
    if len(parts) >= 3:
        multi = ".".join(parts[-2:])   
        if f"tld_type_{multi}" in FEATURE_NAMES:
            return multi
    if len(parts) >= 2:
        return parts[-1]               
    return ""

def get_features(url):
    full = url if url.startswith("http") else "https://" + url
    p = urlparse(full)
    h = (p.hostname or "").lower()
    h_no_www = h.replace("www.", "")
    path = p.path or ""

    tld = get_tld(h)

    known = {
        "url_length":           len(url),
        "hostname_length":      len(h),
        "count_dots":           url.count("."),
        "count_hyphens":        url.count("-"),
        "count_at":             url.count("@"),
        "count_question":       url.count("?"),
        "count_equals":         url.count("="),
        "count_digits":         sum(c.isdigit() for c in url),
        "count_special_chars":  sum(not c.isalnum() for c in url),
        "use_ip":               int(bool(re.search(r"\d+\.\d+\.\d+\.\d+", url))),
        "https":                int(full.startswith("https")),
        "shortened_url":        int(any(d in h for d in SHORTENED_DOMAINS)),
        "contains_login":       int("login"    in url.lower()),
        "contains_verify":      int("verify"   in url.lower()),
        "contains_update":      int("update"   in url.lower()),
        "contains_bank":        int("bank"     in url.lower()),
        "contains_secure":      int("secure"   in url.lower()),
        "contains_free":        int("free"     in url.lower()),
        "contains_account":     int("account"  in url.lower()),
        "suspicious_word_count":sum(w in url.lower() for w in SUSPICIOUS_WORDS),
        "num_subdirectories":   path.count("/"),
        "num_subdomains":       max(0, len(h_no_www.split(".")) - 1),
        "tld_risk_score":       1 if tld in ["tk","ml","cf","ga","gq","xyz","top","click","zip","mov"] else 0,
        "entropy":              calc_entropy(url),
        "source_encoded":       int("%" in url),
    }

    tld_col = f"tld_type_{tld}"
    for col in FEATURE_NAMES:
        if col.startswith("tld_type_"):
            known[col] = 1 if col == tld_col else 0

   
    row = {col: known.get(col, 0) for col in FEATURE_NAMES}
    return pd.DataFrame([row], columns=FEATURE_NAMES)

def get_reasons(url):
    reasons = []
    full = url if url.startswith("http") else "https://" + url
    p = urlparse(full)
    h = (p.hostname or "").lower()

    if re.search(r"\d+\.\d+\.\d+\.\d+", url):
        reasons.append({"text": "IP address used instead of domain", "type": "bad"})
    if len(url) > 100:
        reasons.append({"text": f"Very long URL ({len(url)} chars)", "type": "bad"})
    if not full.startswith("https"):
        reasons.append({"text": "No HTTPS encryption", "type": "bad"})
    if re.search(r"login|verify|secure|account|update|paypal|bank", url, re.I):
        reasons.append({"text": "Phishing keywords detected", "type": "bad"})
    if url.count(".") > 5:
        reasons.append({"text": f"Too many dots ({url.count('.')})", "type": "bad"})
    if url.count("-") > 4:
        reasons.append({"text": f"Many hyphens ({url.count('-')})", "type": "bad"})
    if any(d in h for d in SHORTENED_DOMAINS):
        reasons.append({"text": "URL shortener detected", "type": "bad"})
    tld = get_tld(h)
    if tld in ["tk","ml","cf","ga","gq","xyz","top","click","zip","mov"]:
        reasons.append({"text": f"Risky domain ending (.{tld})", "type": "bad"})
    if calc_entropy(url) > 4.5:
        reasons.append({"text": f"High URL randomness (entropy {calc_entropy(url):.1f})", "type": "bad"})
    if "%" in url:
        reasons.append({"text": "URL encoding detected (obfuscation)", "type": "bad"})

# for safe indicators
    if url.count("@") > 0:
        reasons.append({"text": "@ symbol used to disguise real destination", "type": "bad"})
    if re.search(r"[a-z]+-[a-z]+-[a-z]+", h):
        reasons.append({"text": "Domain uses multiple hyphens (impersonation pattern)", "type": "bad"})
    suspicious_found = [w for w in SUSPICIOUS_WORDS if w in url.lower()]
    if len(suspicious_found) >= 2:
        reasons.append({"text": f"Multiple suspicious words found: {', '.join(suspicious_found[:4])}", "type": "bad"})
    subdomains = max(0, len(h.replace("www.", "").split(".")) - 1)
    if subdomains >= 3:
        reasons.append({"text": f"Excessive subdomains ({subdomains}) — common in phishing", "type": "bad"})
    if re.search(r"(paypal|amazon|microsoft|apple|google|facebook|netflix|instagram|ebay)[^.]*\.", h) and \
       not re.search(r"\.(paypal|amazon|microsoft|apple|google|facebook|netflix|instagram|ebay)\.com$", h):
        reasons.append({"text": "Trusted brand name embedded in fake domain", "type": "bad"})
    digit_ratio = sum(c.isdigit() for c in h) / max(len(h), 1)
    if digit_ratio > 0.3:
        reasons.append({"text": "Domain contains unusually high number of digits", "type": "bad"})
    if re.search(r"(free|win|prize|bonus|gift|offer|deal|discount|limited|urgent|act-now)", url, re.I):
        reasons.append({"text": "Urgency or reward bait words detected", "type": "bad"})
    path = p.path or ""
    if path.count("/") > 5:
        reasons.append({"text": f"Deep directory path ({path.count('/')} levels) — typical of redirect chains", "type": "bad"})
    if re.search(r"\.(exe|zip|rar|bat|scr|msi|apk|dmg)($|\?)", url, re.I):
        reasons.append({"text": "URL points directly to an executable/archive file", "type": "bad"})
    if re.search(r"(confirm|validate|reactivate|unlock|suspended|blocked|restore)", url, re.I):
        reasons.append({"text": "Account-threat language detected (confirm/suspended/blocked)", "type": "bad"})
    

    if full.startswith("https"):
        reasons.append({"text": "HTTPS is present", "type": "good"})
    if not re.search(r"\d+\.\d+\.\d+\.\d+", url):
        reasons.append({"text": "No raw IP address", "type": "good"})
    if len(url) <= 100:
        reasons.append({"text": "URL length is normal", "type": "good"})
    if url.count(".") <= 3:
        reasons.append({"text": "Normal number of dots", "type": "good"})

    return reasons

TRUSTED_DOMAINS = [
    "google.com","youtube.com","facebook.com","instagram.com","twitter.com",
    "x.com","github.com","microsoft.com","apple.com","amazon.com","netflix.com",
    "linkedin.com","wikipedia.org","reddit.com","stackoverflow.com",
    "gmail.com","outlook.com","yahoo.com","bing.com","tiktok.com",
]

def is_trusted(url):
    full = url if url.startswith("http") else "https://" + url
    p = urlparse(full)
    h = (p.hostname or "").lower().removeprefix("www.")
    return any(h == d or h.endswith("." + d) for d in TRUSTED_DOMAINS)

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        url = request.get_json().get("url", "").strip()
        if not url:
            return jsonify({"error": "No URL provided"}), 400

        full_url = url if url.startswith("http") else "https://" + url

        if is_trusted(full_url):
            reasons = get_reasons(full_url)
            safe_result = {"label": 0, "confidence": 0.01}
            return jsonify({
                "url":       full_url,
                "consensus": {"label": 0, "confidence": 0.01, "votes": 0},
                "models":    {"Logistic Regression": safe_result, "Random Forest": safe_result, "XGBoost": safe_result},
                "reasons":   reasons,
            })

        features = get_features(full_url)
        results  = {}

        for name, model in models.items():
            try:
                prob = float(model.predict_proba(features)[0][1])
            except Exception as e:
                print(f"Model {name} error: {e}")
                prob = 0.5
            results[name] = {"label": int(prob > 0.5), "confidence": round(prob, 4)}

        votes    = sum(1 for r in results.values() if r["label"] == 1)
        avg_prob = round(sum(r["confidence"] for r in results.values()) / 3, 4)
        consensus = int(votes >= 2)

        return jsonify({
            "url":       full_url,
            "consensus": {"label": consensus, "confidence": avg_prob, "votes": votes},
            "models":    results,
            "reasons":   get_reasons(full_url),
        })

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(port=5000, debug=True)