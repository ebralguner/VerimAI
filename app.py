"""
E-Ticaret Verimlilik ve Anomali Analiz Sistemi
Ana Flask Uygulama Dosyası — v9.1 (Production / Render)

RENDER FIX'LERİ:
  • with app.app_context() kaldırıldı → Gunicorn crash'i önlendi
  • Module-level warmup: model.py import'unda _ModelRegistry.warmup() çalışır
  • debug=False (production güvenliği)
  • app.run() sadece __main__ altında, $PORT'a bind
  • clean_for_json: numpy int/float serialize hatası önlendi
"""

import logging
import os
from flask import Flask, render_template, request, jsonify
from model import (
    predict_production,
    predict_sales,
    predict_sales_with_ci,
    calculate_profit,
    analyze_channels,
    detect_anomaly,
    generate_commentary,
    generate_recommendation,
    COGS_RATIO,
    _ModelRegistry,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# ── Flask app ──────────────────────────────────────────────────────────────────
app = Flask(__name__)

# Warmup: model.py import edildiğinde zaten çalışır (_ModelRegistry.warmup()).
# Burada tekrar çağırmaya gerek yok — Gunicorn worker fork'larıyla uyumlu.
logger.info("Flask app başlatıldı. ML modelleri model.py import'u sırasında yüklendi.")


# ══════════════════════════════════════════════════════════════
# YARDIMCI FONKSİYONLAR
# ══════════════════════════════════════════════════════════════

def safe_float(val, default=0.0):
    """None / '' / NaN / Inf → default, hiç crash olmaz."""
    if val is None or val == "":
        return default
    try:
        r = float(val)
        return default if (r != r or abs(r) == float("inf")) else r
    except (ValueError, TypeError):
        return default


def clean_for_json(obj):
    """Tüm numpy tipler ve NaN/Inf → JSON-safe Python tipleri."""
    import math
    try:
        import numpy as np
        np_int   = np.integer
        np_float = np.floating
        np_bool  = np.bool_
        np_array = np.ndarray
    except ImportError:
        np_int = np_float = np_bool = np_array = type(None)

    if obj is None:
        return None
    if isinstance(obj, np_bool):
        return bool(obj)
    if isinstance(obj, np_int):
        return int(obj)
    if isinstance(obj, np_float):
        v = float(obj)
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(obj, np_array):
        return clean_for_json(obj.tolist())
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean_for_json(i) for i in obj]
    return obj


def _clamp(val, lo=0.0, hi=100.0):
    return max(lo, min(hi, val))


def _pct_change(old, new):
    if old == 0:
        return 0.0 if new == 0 else 100.0
    return round((new - old) / abs(old) * 100, 1)


def _validate_production_inputs(workers, hours, energy):
    missing = []
    if workers <= 0: missing.append("çalışan sayısı (workers)")
    if hours   <= 0: missing.append("çalışma saati (hours)")
    if energy  <= 0: missing.append("enerji (energy)")
    if missing:
        return False, {
            "success":    False,
            "status":     "error",
            "error_code": "INVALID_PRODUCTION_INPUTS",
            "message":    "Üretim için pozitif değerler girilmelidir.",
            "missing":    missing,
            "production": 0,
            "sales":      0,
            "profit":     0,
            "revenue":    0,
            "total_cost": 0,
        }
    return True, None


def _perf_score(profit_margin, efficiency, cost_rate):
    pm_score   = _clamp(min(profit_margin, 70.0) / 70.0 * 100.0)
    eff_score  = _clamp(efficiency * 50.0)
    cost_score = _clamp(100.0 - cost_rate)
    score = round(0.50 * pm_score + 0.30 * eff_score + 0.20 * cost_score, 1)
    cls   = "poor" if score < 40 else ("average" if score < 70 else "good")
    return score, cls


def _run_pipeline(workers, hours, energy, price, channel,
                  worker_cost, energy_cost, demo_mode=True):
    production = predict_production(workers, hours, energy, demo_mode)
    sales      = predict_sales(production, price, channel, demo_mode,
                               workers=workers, energy=energy)
    pdata      = calculate_profit(sales, price, workers, worker_cost, energy, energy_cost)
    return production, sales, pdata


# ══════════════════════════════════════════════════════════════
# DEMO SENARYOLARI
# ══════════════════════════════════════════════════════════════

DEMO_SCENARIOS = [
    {"label": "🚀 Yüksek Performanslı Operasyon", "tier": "good",
     "workers": 30,  "hours": 10, "energy": 500, "price": 200, "channel": "app",
     "worker_cost": 3500, "energy_cost": 2.0, "app_sales": 0, "web_sales": 0},
    {"label": "💡 Küçük İşletme Optimumu", "tier": "good",
     "workers": 8,   "hours": 9,  "energy": 260, "price": 220, "channel": "app",
     "worker_cost": 3000, "energy_cost": 1.8, "app_sales": 0, "web_sales": 0},
    {"label": "🏭 Orta Ölçekli Fabrika", "tier": "mid",
     "workers": 15,  "hours": 8,  "energy": 500, "price": 130, "channel": "web",
     "worker_cost": 4000, "energy_cost": 2.5, "app_sales": 0, "web_sales": 0},
    {"label": "⚠️ Verimsiz Senaryo (Anomali)", "tier": "bad",
     "workers": 40,  "hours": 6,  "energy": 900, "price": 50,  "channel": "web",
     "worker_cost": 8000, "energy_cost": 4.5, "app_sales": 0, "web_sales": 0},
]

INPUT_HINTS = {
    "workers":     "Örn: 15 (çalışan kişi sayısı)",
    "hours":       "Örn: 8 (günlük saat, maks 12)",
    "energy":      "Örn: 500 (günlük kWh)",
    "price":       "Örn: 130 (satış fiyatı ₺)",
    "worker_cost": "Örn: 4000 (aylık maaş ₺/kişi)",
    "energy_cost": "Örn: 2.5 (₺/kWh)",
    "app_sales":   "Örn: 80 (app satış adedi)",
    "web_sales":   "Örn: 120 (web satış adedi)",
}


# ══════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/meta", methods=["GET"])
def meta():
    return jsonify({
        "success":        True,
        "input_hints":    INPUT_HINTS,
        "demo_scenarios": DEMO_SCENARIOS,
        "cogs_ratio":     COGS_RATIO,
    })


@app.route("/api/production", methods=["POST"])
def production():
    d = request.get_json(silent=True) or {}
    workers   = safe_float(d.get("workers"), 0)
    hours     = safe_float(d.get("hours"),   0)
    energy    = safe_float(d.get("energy"),  0)
    demo_mode = bool(d.get("demo_mode", True))

    ok, err = _validate_production_inputs(workers, hours, energy)
    if not ok:
        return jsonify(clean_for_json(err)), 422

    result = predict_production(workers, hours, energy, demo_mode)
    return jsonify(clean_for_json({
        "success":    True,
        "production": result,
        "inputs":     {"workers": workers, "hours": hours, "energy": energy},
    }))


@app.route("/api/sales", methods=["POST"])
def sales():
    d          = request.get_json(silent=True) or {}
    production = safe_float(d.get("production"), 0)
    price      = safe_float(d.get("price"),      0)
    channel    = d.get("channel") or "web"
    demo_mode  = bool(d.get("demo_mode", True))
    workers    = safe_float(d.get("workers"),  15)
    energy     = safe_float(d.get("energy"),  500)

    if production <= 0:
        return jsonify(clean_for_json({
            "success": False, "status": "error",
            "error_code": "INVALID_PRODUCTION_INPUTS",
            "message": "Üretim için pozitif değerler girilmelidir.",
            "sales": 0, "stock_capped": False, "notice": None,
        })), 422

    result = predict_sales(production, price, channel, demo_mode,
                           workers=workers, energy=energy)
    capped = (production > 0) and (result >= production * 0.98)
    return jsonify(clean_for_json({
        "success":     True,
        "sales":       result,
        "stock_capped": capped,
        "notice": "📦 Stok sınırı uygulandı — satış üretimi aşamaz." if capped else None,
    }))


@app.route("/api/profit", methods=["POST"])
def profit():
    d = request.get_json(silent=True) or {}
    result = calculate_profit(
        safe_float(d.get("sales")),   safe_float(d.get("price")),
        safe_float(d.get("workers")), safe_float(d.get("worker_cost")),
        safe_float(d.get("energy")),  safe_float(d.get("energy_cost")),
    )
    warning = None
    if result["profit_margin"] > 60:
        warning = f"⚠ Kar marjı %{result['profit_margin']:.1f} — olağandışı yüksek. Maliyet girdilerini kontrol edin."
    return jsonify(clean_for_json({"success": True, "high_margin_warning": warning, **result}))


@app.route("/api/channel", methods=["POST"])
def channel():
    d = request.get_json(silent=True) or {}
    result = analyze_channels(
        safe_float(d.get("app_sales")),
        safe_float(d.get("web_sales")),
        safe_float(d.get("total_sales")),
    )
    return jsonify(clean_for_json({"success": True, **result}))


@app.route("/api/anomaly", methods=["POST"])
def anomaly():
    d = request.get_json(silent=True) or {}
    result = detect_anomaly(
        safe_float(d.get("energy")),
        safe_float(d.get("sales")),
        safe_float(d.get("workers")),
        safe_float(d.get("production")),
        safe_float(d.get("profit_margin")),
    )
    return jsonify(clean_for_json({"success": True, **result}))


@app.route("/api/impact", methods=["POST"])
def impact():
    d = request.get_json(silent=True) or {}
    workers     = safe_float(d.get("workers"),       15)
    hours       = safe_float(d.get("hours"),          8)
    energy      = safe_float(d.get("energy"),       500)
    price       = safe_float(d.get("price"),        130)
    channel     = d.get("channel",               "web")
    worker_cost = safe_float(d.get("worker_cost"), 4000)
    energy_cost = safe_float(d.get("energy_cost"),    2)
    energy_chg  = safe_float(d.get("energy_change"),  0)
    worker_chg  = safe_float(d.get("worker_change"),  0)
    demo_mode   = False   # etki analizi deterministik olmalı

    ok, err = _validate_production_inputs(workers, hours, energy)
    if not ok:
        return jsonify(clean_for_json(err)), 422

    base_prod, base_sales, base_pd = _run_pipeline(
        workers, hours, energy, price, channel, worker_cost, energy_cost, demo_mode
    )
    new_energy  = max(0.0, energy  * (1 + energy_chg  / 100.0))
    new_workers = max(0.0, workers * (1 + worker_chg  / 100.0))
    new_prod, new_sales, new_pd = _run_pipeline(
        new_workers, hours, new_energy, price, channel, worker_cost, energy_cost, demo_mode
    )

    d_profit = round(new_pd["profit"] - base_pd["profit"], 2)
    d_profit_pct = _pct_change(base_pd["profit"], new_pd["profit"])

    if d_profit > 0 and new_pd["profit"] > 0:
        verdict, verdict_text = "positive", f"Bu değişim günlük kârı {abs(d_profit):,.0f} ₺ ({abs(d_profit_pct):.1f}%) artırıyor. Uygulanması önerilir."
    elif d_profit < 0 and new_pd["profit"] < 0:
        verdict, verdict_text = "negative", f"Bu değişim zarara yol açıyor. Kâr {abs(d_profit):,.0f} ₺ ({abs(d_profit_pct):.1f}%) azalıyor. Uygulanması önerilmez."
    elif d_profit < 0:
        verdict, verdict_text = "warning", f"Bu değişim kârı {abs(d_profit):,.0f} ₺ ({abs(d_profit_pct):.1f}%) düşürüyor. Dikkatli değerlendirin."
    else:
        verdict, verdict_text = "neutral", "Bu değişimin kâr üzerinde anlamlı bir etkisi yok."

    return jsonify(clean_for_json({
        "success": True,
        "old": {
            "production":    round(base_prod, 2),
            "sales":         round(base_sales, 2),
            "revenue":       base_pd["revenue"],
            "total_cost":    base_pd["total_cost"],
            "profit":        base_pd["profit"],
            "profit_margin": base_pd["profit_margin"],
            "energy":        round(energy, 2),
            "workers":       round(workers, 2),
        },
        "new": {
            "production":    round(new_prod, 2),
            "sales":         round(new_sales, 2),
            "revenue":       new_pd["revenue"],
            "total_cost":    new_pd["total_cost"],
            "profit":        new_pd["profit"],
            "profit_margin": new_pd["profit_margin"],
            "energy":        round(new_energy, 2),
            "workers":       round(new_workers, 2),
        },
        "delta": {
            "production":        round(new_prod  - base_prod,  2),
            "production_pct":    _pct_change(base_prod, new_prod),
            "sales":             round(new_sales - base_sales, 2),
            "sales_pct":         _pct_change(base_sales, new_sales),
            "revenue":           round(new_pd["revenue"]    - base_pd["revenue"],    2),
            "revenue_pct":       _pct_change(base_pd["revenue"],    new_pd["revenue"]),
            "total_cost":        round(new_pd["total_cost"] - base_pd["total_cost"], 2),
            "total_cost_pct":    _pct_change(base_pd["total_cost"], new_pd["total_cost"]),
            "profit":            d_profit,
            "profit_pct":        d_profit_pct,
            "profit_margin_old": base_pd["profit_margin"],
            "profit_margin_new": new_pd["profit_margin"],
        },
        "applied": {
            "energy_change": energy_chg,
            "worker_change": worker_chg,
            "new_energy":    round(new_energy, 2),
            "new_workers":   round(new_workers, 2),
        },
        "verdict":      verdict,
        "verdict_text": verdict_text,
    }))


@app.route("/api/recommendation", methods=["POST"])
def smart_recommendation():
    d = request.get_json(silent=True) or {}
    workers     = safe_float(d.get("workers"),       15)
    hours       = safe_float(d.get("hours"),          8)
    energy      = safe_float(d.get("energy"),       500)
    price       = safe_float(d.get("price"),        130)
    channel     = d.get("channel",               "web")
    worker_cost = safe_float(d.get("worker_cost"), 4000)
    energy_cost = safe_float(d.get("energy_cost"),    2)
    demo_mode   = bool(d.get("demo_mode", False))

    if d.get("production") and d.get("sales") and d.get("profit") is not None:
        production    = safe_float(d.get("production"))
        sales_val     = safe_float(d.get("sales"))
        profit_val    = safe_float(d.get("profit"))
        profit_margin = safe_float(d.get("profit_margin"))
        revenue       = safe_float(d.get("revenue"))
        total_cost    = safe_float(d.get("total_cost"))
    else:
        ok, err = _validate_production_inputs(workers, hours, energy)
        if not ok:
            return jsonify(clean_for_json(err)), 422
        production, sales_val, pdata = _run_pipeline(
            workers, hours, energy, price, channel, worker_cost, energy_cost, demo_mode
        )
        profit_val    = pdata["profit"]
        profit_margin = pdata["profit_margin"]
        revenue       = pdata["revenue"]
        total_cost    = pdata["total_cost"]

    recommendations = []
    if profit_val < 0:
        recommendations.append({
            "priority": 1, "type": "critical", "icon": "⛔",
            "title": "Zarar Ediliyor",
            "detail": f"Günlük zarar: {abs(profit_val):,.0f} ₺. Gelir ({revenue:,.0f} ₺) maliyetin ({total_cost:,.0f} ₺) altında.",
            "action": "Fiyatı en az %15 artırın VEYA çalışan sayısını optimize edin VEYA enerji verimliliğini artırın.",
        })
    if production > 0 and sales_val < production * 0.70:
        recommendations.append({
            "priority": 2, "type": "warning", "icon": "📦",
            "title": "Üretim Talebin Üzerinde",
            "detail": f"Üretilen {production:.0f} birimden yalnızca {sales_val:.0f} adeti satılabiliyor. {production-sales_val:.0f} birim stokta kalıyor.",
            "action": "Fiyatı indirerek talebi artırın VEYA pazarlama bütçesini artırın VEYA üretim hacmini talebe göre kısın.",
        })
    recommendations.sort(key=lambda x: x["priority"])

    crit = sum(1 for r in recommendations if r["type"] == "critical")
    warn = sum(1 for r in recommendations if r["type"] == "warning")

    if crit > 0:       status, status_label, status_color = "critical", "Kritik — Acil Aksiyon Gerekli",         "danger"
    elif warn >= 2:    status, status_label, status_color = "warning",  "Dikkatli — Birden Fazla Risk Mevcut",   "warning"
    elif warn == 1:    status, status_label, status_color = "caution",  "Dikkat — İyileştirme Önerilir",         "warning"
    elif not recommendations: status, status_label, status_color = "good", "Sağlıklı — Sistem Dengeli Çalışıyor", "good"
    else:              status, status_label, status_color = "info",     "Bilgi — Optimize Edilebilir Alanlar Var","good"

    return jsonify(clean_for_json({
        "success":       True,
        "status":        status,
        "status_label":  status_label,
        "status_color":  status_color,
        "recommendation_count": len(recommendations),
        "recommendations": recommendations,
        "action_items":    [r["action"] for r in recommendations[:3]],
        "summary": {
            "production":    round(production, 2),
            "sales":         round(sales_val, 2),
            "profit":        round(profit_val, 2),
            "profit_margin": round(profit_margin, 1),
        },
    }))


@app.route("/api/scenario", methods=["POST"])
def scenario():
    d = request.get_json(silent=True) or {}
    workers     = safe_float(d.get("workers"),       15)
    hours       = safe_float(d.get("hours"),          8)
    energy      = safe_float(d.get("energy"),       500)
    price       = safe_float(d.get("price"),        130)
    channel     = d.get("channel",               "web")
    worker_cost = safe_float(d.get("worker_cost"), 4000)
    energy_cost = safe_float(d.get("energy_cost"),    2)
    energy_chg  = safe_float(d.get("energy_change"),  0)
    worker_chg  = safe_float(d.get("worker_change"),  0)
    demo_mode   = bool(d.get("demo_mode", True))

    ok, err = _validate_production_inputs(workers, hours, energy)
    if not ok:
        return jsonify(clean_for_json(err)), 422

    new_energy  = max(0.0, energy  * (1 + energy_chg  / 100.0))
    new_workers = max(0.0, workers * (1 + worker_chg  / 100.0))

    new_prod, new_sales, new_pd     = _run_pipeline(new_workers, hours, new_energy, price, channel, worker_cost, energy_cost, demo_mode)
    orig_prod, orig_sales, orig_pd  = _run_pipeline(workers,     hours, energy,     price, channel, worker_cost, energy_cost, demo_mode)

    return jsonify(clean_for_json({
        "success": True,
        "scenario": {
            "production": new_prod, "sales": new_sales,
            "profit": new_pd["profit"], "profit_margin": new_pd["profit_margin"],
            "revenue": new_pd["revenue"], "total_cost": new_pd["total_cost"],
            "product_cost": new_pd["product_cost"],
            "energy": round(new_energy, 2), "workers": round(new_workers, 2),
        },
        "original": {
            "production": orig_prod, "sales": orig_sales,
            "profit": orig_pd["profit"], "profit_margin": orig_pd["profit_margin"],
        },
        "delta": {
            "production": round(new_prod - orig_prod, 2),
            "sales":      round(new_sales - orig_sales, 2),
            "profit":     round(new_pd["profit"] - orig_pd["profit"], 2),
            "profit_pct": _pct_change(orig_pd["profit"], new_pd["profit"]),
        },
        "applied": {"energy_change": energy_chg, "worker_change": worker_chg},
    }))


@app.route("/api/dashboard", methods=["POST"])
def dashboard():
    d = request.get_json(silent=True) or {}
    workers     = safe_float(d.get("workers"),      15)
    hours       = safe_float(d.get("hours"),         8)
    energy      = safe_float(d.get("energy"),      500)
    price       = safe_float(d.get("price"),       130)
    channel     = d.get("channel",              "web")
    worker_cost = safe_float(d.get("worker_cost"), 4000)
    energy_cost = safe_float(d.get("energy_cost"),    2)
    app_sales   = safe_float(d.get("app_sales"),      0)
    web_sales   = safe_float(d.get("web_sales"),      0)
    demo_mode   = bool(d.get("demo_mode", True))

    ok, err = _validate_production_inputs(workers, hours, energy)
    if not ok:
        return jsonify(clean_for_json(err)), 422

    production, sales_val, pdata = _run_pipeline(
        workers, hours, energy, price, channel, worker_cost, energy_cost, demo_mode
    )
    revenue       = pdata["revenue"]
    total_cost    = pdata["total_cost"]
    profit_val    = pdata["profit"]
    profit_margin = pdata["profit_margin"]

    stock_notice        = None
    high_margin_warning = None
    if production > 0 and sales_val >= production * 0.98:
        stock_notice = "📦 Stok sınırı uygulandı — satış üretim kapasitesini aşamaz."
    if profit_margin > 60:
        high_margin_warning = f"⚠ Kar marjı %{profit_margin:.1f} — olağandışı yüksek. Maliyet girdilerini kontrol edin."

    channel_result = analyze_channels(app_sales, web_sales, sales_val)
    anomaly_result = detect_anomaly(energy, sales_val, workers, production, profit_margin)

    efficiency = round(production / energy, 4) if energy > 0 else 0.0
    cost_rate  = round(total_cost / revenue * 100, 1) if revenue > 0 else 0.0
    perf_score, perf_class = _perf_score(profit_margin, efficiency, cost_rate)

    commentary, commentary_status = generate_commentary(
        production, sales_val, profit_val, profit_margin, energy, workers, price
    )
    severity = anomaly_result.get("severity", "normal")
    rec, rec_status = generate_recommendation(
        profit_margin, sales_val, production, energy, workers, severity, profit_val, price
    )

    return jsonify(clean_for_json({
        "success":             True,
        "production":          production,
        "sales":               sales_val,
        "profit":              profit_val,
        "revenue":             revenue,
        "total_cost":          total_cost,
        "stock_notice":        stock_notice,
        "high_margin_warning": high_margin_warning,
        "cost_breakdown": {
            "labor_cost":        pdata["labor_cost"],
            "energy_cost_total": pdata["energy_cost_total"],
            "product_cost":      pdata["product_cost"],
        },
        "channel": channel_result,
        "anomaly": anomaly_result,
        "kpi": {
            "efficiency":    efficiency,
            "profit_rate":   profit_margin,
            "profit_margin": profit_margin,
            "cost_rate":     cost_rate,
        },
        "commentary":        commentary,
        "commentary_status": commentary_status,
        "performance":       {"score": perf_score, "class": perf_class},
        "recommendation":        rec,
        "recommendation_status": rec_status,
        "ml_info": {
            "sales_model":    _ModelRegistry.sales().best.name,
            "anomaly_model":  "IsolationForest",
            "rec_model":      "DecisionTree",
            "ml_active":      True,
            "ml_triggered":   anomaly_result.get("ml_triggered", False),
            "rule_triggered": anomaly_result.get("rule_triggered", False),
            "sales_test_mae": _ModelRegistry.sales().best.test_mae,
            "sales_test_r2":  _ModelRegistry.sales().best.test_r2,
        },
        "_pipeline_state": {
            "workers": workers, "hours": hours, "energy": energy,
            "price": price, "channel": channel,
            "worker_cost": worker_cost, "energy_cost": energy_cost,
            "production": production, "sales": sales_val,
            "profit": profit_val, "profit_margin": profit_margin,
            "revenue": revenue, "total_cost": total_cost,
        },
    }))


@app.route("/api/sales-ci", methods=["POST"])
def sales_ci():
    d          = request.get_json(silent=True) or {}
    production = safe_float(d.get("production"), 0)
    price      = safe_float(d.get("price"),      0)
    channel    = d.get("channel") or "web"
    workers    = safe_float(d.get("workers"),  15)
    energy     = safe_float(d.get("energy"),  500)

    result = predict_sales_with_ci(production, price, channel, workers, energy)
    capped = (production > 0) and (result["sales"] >= production * 0.98)

    return jsonify(clean_for_json({
        "success": True,
        **result,
        "stock_capped": capped,
        "notice": "📦 Stok sınırı uygulandı." if capped else None,
        "display": (
            f"{result['sales']:.0f} ± {result['ci_half_width']:.0f} adet "
            f"[{result['sales_lower']:.0f}, {result['sales_upper']:.0f}] "
            f"({result['model_used']}, güven %{result['confidence_pct']:.0f})"
        ),
    }))


@app.route("/api/ml-info", methods=["GET"])
def ml_info():
    try:
        info = _ModelRegistry.get_ml_info()
        return jsonify({"success": True, **info})
    except Exception as e:
        logger.error("/api/ml-info hatası: %s", e)
        return jsonify({"success": False, "error": str(e)}), 500


# ══════════════════════════════════════════════════════════════
# ENTRY POINT (sadece local geliştirme için)
# Render: gunicorn app:app --workers 1 --bind 0.0.0.0:$PORT --timeout 120
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
