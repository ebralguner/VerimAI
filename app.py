"""
E-Ticaret Verimlilik ve Anomali Analiz Sistemi
Ana Flask Uygulama Dosyası — v9.0  (Pro ML)

PIPELINE: Üretim → Satış (LR/RF auto-select) → Kar → Kanal → Anomali → Dashboard
ML:  SatışML (LR vs RF, auto-select) + AnomaliML (IsolationForest) + ÖneriML (DecisionTree + proba)
API: /api/impact · /api/recommendation · /api/dashboard · /api/ml-info · /api/sales-ci
"""

import logging
from flask import Flask, render_template, request, jsonify
from model import (
    predict_production,
    predict_sales,
    predict_sales_with_ci,     # YENİ — güven aralıklı satış tahmini
    calculate_profit,
    analyze_channels,
    detect_anomaly,
    generate_commentary,
    generate_recommendation,
    COGS_RATIO,
    _ModelRegistry,            # startup warmup + ml-info
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)


# ══════════════════════════════════════════════════════════════
# ML STARTUP WARMUP
# Modeller ilk request beklenmeden önceden eğitilir.
# ══════════════════════════════════════════════════════════════

def _warmup_ml_models() -> None:
    """Tüm ML modellerini uygulama başlangıcında eğit."""
    logger.info("ML modelleri başlatılıyor...")
    try:
        _ModelRegistry.sales()
        _ModelRegistry.anomaly()
        _ModelRegistry.recommender()
        logger.info("✓ Tüm ML modelleri hazır: SatışML + AnomaliML + ÖneriML")
    except Exception as e:
        logger.error("ML warmup hatası: %s — kural tabanlı fallback aktif.", e)

with app.app_context():
    _warmup_ml_models()


# ══════════════════════════════════════════════════════════════
# YARDIMCI
# ══════════════════════════════════════════════════════════════

def safe_float(val, default=0.0):
    """None / "" / NaN / Inf → default, hiç crash olmaz."""
    if val is None or val == "":
        return default
    try:
        r = float(val)
        return default if (r != r or abs(r) == float("inf")) else r
    except (ValueError, TypeError):
        return default


def clean_for_json(obj):
    """
    Tüm API yanıtları için evrensel JSON temizleyici.

    Dönüştürdüğü tipler:
        numpy.bool_    → bool
        numpy.integer  → int
        numpy.floating → float
        numpy.ndarray  → list
        float NaN/Inf  → None
        dict / list    → özyinelemeli temizleme

    Kullanım:
        return jsonify(clean_for_json({...}))
    """
    import math
    import numpy as np

    if obj is None:
        return None

    # numpy bool (en önce — bool np.integer'dan önce gelmeli)
    if isinstance(obj, np.bool_):
        return bool(obj)

    # numpy tam sayılar (int8, int16, int32, int64, uint* …)
    if isinstance(obj, np.integer):
        return int(obj)

    # numpy kayan noktalı (float16, float32, float64 …)
    if isinstance(obj, np.floating):
        v = float(obj)
        return None if (math.isnan(v) or math.isinf(v)) else v

    # numpy dizi → Python listesi (özyinelemeli)
    if isinstance(obj, np.ndarray):
        return clean_for_json(obj.tolist())

    # Python float — NaN / Inf temizle
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj

    # dict — tüm değerleri temizle
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}

    # list / tuple — öğeleri temizle
    if isinstance(obj, (list, tuple)):
        return [clean_for_json(item) for item in obj]

    # str, int, bool — doğrudan dön
    return obj


def _clamp(val, lo=0.0, hi=100.0):
    return max(lo, min(hi, val))


def _pct_change(old, new):
    """Yüzde değişim. old=0 durumunu güvenli ele alır."""
    if old == 0:
        return 0.0 if new == 0 else 100.0
    return round((new - old) / abs(old) * 100, 1)


def _validate_production_inputs(workers: float, hours: float, energy: float):
    """
    Üretim için zorunlu girdileri kontrol eder.

    Döner: (geçerli_mi: bool, hata_yanıtı: dict | None)

    Kullanım:
        ok, err = _validate_production_inputs(workers, hours, energy)
        if not ok:
            return jsonify(err), 422
    """
    missing = []
    if workers <= 0:
        missing.append("çalışan sayısı (workers)")
    if hours <= 0:
        missing.append("çalışma saati (hours)")
    if energy <= 0:
        missing.append("enerji (energy)")

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


def _perf_score(profit_margin: float, efficiency: float, cost_rate: float):
    """
    Ağırlıklı performans skoru 0-100.
    profit_margin > 70 → kırpılır (şüpheli yüksek)
    """
    pm_score   = _clamp(min(profit_margin, 70.0) / 70.0 * 100.0)
    eff_score  = _clamp(efficiency * 50.0)
    cost_score = _clamp(100.0 - cost_rate)
    score = round(0.50 * pm_score + 0.30 * eff_score + 0.20 * cost_score, 1)
    cls   = "poor" if score < 40 else ("average" if score < 70 else "good")
    return score, cls


def _run_pipeline(workers, hours, energy, price, channel,
                  worker_cost, energy_cost, demo_mode=True):
    """
    Tek noktadan pipeline çalıştırır.
    Döner: (production, sales, pdata dict)

    v9.0: predict_sales artık gerçek workers ve energy alıyor
    → ML modeli production feature'ını doğru kullanır.
    """
    production = predict_production(workers, hours, energy, demo_mode)
    sales      = predict_sales(production, price, channel, demo_mode,
                               workers=workers, energy=energy)
    pdata      = calculate_profit(sales, price, workers, worker_cost, energy, energy_cost)
    return production, sales, pdata


# ══════════════════════════════════════════════════════════════
# DEMO SENARYOLARI  %40 iyi / %40 orta / %20 kötü
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
    {"label": "🏪 Orta Hacimli Mağaza", "tier": "mid",
     "workers": 20,  "hours": 8,  "energy": 600, "price": 110, "channel": "web",
     "worker_cost": 4500, "energy_cost": 3.0, "app_sales": 0, "web_sales": 0},
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


# ── 1. Üretim ────────────────────────────────────────────────
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
    return jsonify(clean_for_json({"success": True, "production": result,
                    "inputs": {"workers": workers, "hours": hours, "energy": energy}}))


# ── 2. Satış ─────────────────────────────────────────────────
@app.route("/api/sales", methods=["POST"])
def sales():
    d          = request.get_json(silent=True) or {}
    production = safe_float(d.get("production"), 0)
    price      = safe_float(d.get("price"),      0)
    channel    = d.get("channel") or "web"
    demo_mode  = bool(d.get("demo_mode", True))
    workers    = safe_float(d.get("workers"),  15)   # YENİ — ML için
    energy     = safe_float(d.get("energy"),  500)   # YENİ — ML için

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


# ── 3. Kar ───────────────────────────────────────────────────
@app.route("/api/profit", methods=["POST"])
def profit():
    d = request.get_json(silent=True) or {}
    result = calculate_profit(
        safe_float(d.get("sales")),   safe_float(d.get("price")),
        safe_float(d.get("workers")), safe_float(d.get("worker_cost")),
        safe_float(d.get("energy")),  safe_float(d.get("energy_cost")),
    )
    high_margin_warning = None
    if result["profit_margin"] > 60:
        high_margin_warning = (
            f"⚠ Kar marjı %{result['profit_margin']:.1f} — olağandışı yüksek. "
            "Maliyet girdilerini kontrol edin."
        )
    return jsonify(clean_for_json({"success": True, "high_margin_warning": high_margin_warning, **result}))


# ── 4. Kanal ─────────────────────────────────────────────────
@app.route("/api/channel", methods=["POST"])
def channel():
    d = request.get_json(silent=True) or {}
    result = analyze_channels(
        safe_float(d.get("app_sales")),
        safe_float(d.get("web_sales")),
        safe_float(d.get("total_sales")),
    )
    return jsonify(clean_for_json({"success": True, **result}))


# ── 5. Anomali ───────────────────────────────────────────────
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


# ── 6. ETKİ ANALİZİ (/api/impact) — YENİ ────────────────────
@app.route("/api/impact", methods=["POST"])
def impact():
    """
    Parametre değişiminin sistem üzerindeki etkisini hesaplar.

    Girdi:
        workers, hours, energy, price, channel, worker_cost, energy_cost
        energy_change  (%)  — enerji değişim yüzdesi
        worker_change  (%)  — çalışan değişim yüzdesi

    Çıktı:
        old  → mevcut değerlerle hesaplanan base sonuçlar
        new  → değiştirilmiş parametrelerle hesaplanan sonuçlar
        delta → new - old (mutlak ve yüzde)
        applied → uygulanan değişimler
        verdict → ne anlama geldiği
    """
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
    demo_mode   = bool(d.get("demo_mode", False))   # etki analizi deterministik olmalı

    logger.info("/api/impact e_chg=%s w_chg=%s", energy_chg, worker_chg)

    # ── Girdi doğrulama ──────────────────────────────────────────
    ok, err = _validate_production_inputs(workers, hours, energy)
    if not ok:
        return jsonify(clean_for_json(err)), 422

    # ── Mevcut (base) ────────────────────────────────────────
    base_prod, base_sales, base_pd = _run_pipeline(
        workers, hours, energy, price, channel, worker_cost, energy_cost, demo_mode
    )

    # ── Yeni (modified) ──────────────────────────────────────
    new_energy  = max(0.0, energy  * (1 + energy_chg  / 100.0))
    new_workers = max(0.0, workers * (1 + worker_chg  / 100.0))

    new_prod, new_sales, new_pd = _run_pipeline(
        new_workers, hours, new_energy, price, channel, worker_cost, energy_cost, demo_mode
    )

    # ── Delta hesapları ──────────────────────────────────────
    d_prod   = round(new_prod            - base_prod,            2)
    d_sales  = round(new_sales           - base_sales,           2)
    d_profit = round(new_pd["profit"]    - base_pd["profit"],    2)
    d_cost   = round(new_pd["total_cost"]- base_pd["total_cost"],2)
    d_rev    = round(new_pd["revenue"]   - base_pd["revenue"],   2)

    d_prod_pct   = _pct_change(base_prod,            new_prod)
    d_sales_pct  = _pct_change(base_sales,           new_sales)
    d_profit_pct = _pct_change(base_pd["profit"],    new_pd["profit"])
    d_cost_pct   = _pct_change(base_pd["total_cost"],new_pd["total_cost"])
    d_rev_pct    = _pct_change(base_pd["revenue"],   new_pd["revenue"])

    # ── Karar yorumu ─────────────────────────────────────────
    if d_profit > 0 and new_pd["profit"] > 0:
        verdict = "positive"
        verdict_text = (
            f"Bu değişim günlük kârı {abs(d_profit):,.0f} ₺ ({abs(d_profit_pct):.1f}%) artırıyor. "
            "Uygulanması önerilir."
        )
    elif d_profit < 0 and new_pd["profit"] < 0:
        verdict = "negative"
        verdict_text = (
            f"Bu değişim zarara yol açıyor. Kâr {abs(d_profit):,.0f} ₺ ({abs(d_profit_pct):.1f}%) "
            "azalıyor. Uygulanması önerilmez."
        )
    elif d_profit < 0:
        verdict = "warning"
        verdict_text = (
            f"Bu değişim kârı {abs(d_profit):,.0f} ₺ ({abs(d_profit_pct):.1f}%) düşürüyor. "
            "Dikkatli değerlendirin."
        )
    else:
        verdict = "neutral"
        verdict_text = "Bu değişimin kâr üzerinde anlamlı bir etkisi yok."

    return jsonify(clean_for_json({
        "success": True,
        "old": {
            "production":    round(base_prod,              2),
            "sales":         round(base_sales,             2),
            "revenue":       base_pd["revenue"],
            "total_cost":    base_pd["total_cost"],
            "profit":        base_pd["profit"],
            "profit_margin": base_pd["profit_margin"],
            "energy":        round(energy,  2),
            "workers":       round(workers, 2),
        },
        "new": {
            "production":    round(new_prod,               2),
            "sales":         round(new_sales,              2),
            "revenue":       new_pd["revenue"],
            "total_cost":    new_pd["total_cost"],
            "profit":        new_pd["profit"],
            "profit_margin": new_pd["profit_margin"],
            "energy":        round(new_energy,  2),
            "workers":       round(new_workers, 2),
        },
        "delta": {
            "production":        d_prod,
            "production_pct":    d_prod_pct,
            "sales":             d_sales,
            "sales_pct":         d_sales_pct,
            "revenue":           d_rev,
            "revenue_pct":       d_rev_pct,
            "total_cost":        d_cost,
            "total_cost_pct":    d_cost_pct,
            "profit":            d_profit,
            "profit_pct":        d_profit_pct,
            "profit_margin_old": base_pd["profit_margin"],
            "profit_margin_new": new_pd["profit_margin"],
        },
        "applied": {
            "energy_change": energy_chg,
            "worker_change": worker_chg,
            "new_energy":    round(new_energy,  2),
            "new_workers":   round(new_workers, 2),
        },
        "verdict":      verdict,
        "verdict_text": verdict_text,
    }))


# ── 7. AKILLI TAVSİYE (/api/recommendation) — YENİ ──────────
@app.route("/api/recommendation", methods=["POST"])
def smart_recommendation():
    """
    Çok koşullu karar destek motoru.
    Tüm durumları birlikte değerlendirerek önceliklendirilmiş
    öneri listesi üretir.

    Girdi: tüm pipeline değerleri (veya dashboard sonucu)
    Çıktı: status, recommendations[], action_items[]
    """
    d = request.get_json(silent=True) or {}

    # Girdi — ya ham değerler ya da dashboard sonucu
    workers       = safe_float(d.get("workers"),       15)
    hours         = safe_float(d.get("hours"),          8)
    energy        = safe_float(d.get("energy"),       500)
    price         = safe_float(d.get("price"),        130)
    channel       = d.get("channel",               "web")
    worker_cost   = safe_float(d.get("worker_cost"), 4000)
    energy_cost   = safe_float(d.get("energy_cost"),    2)
    demo_mode     = bool(d.get("demo_mode", False))

    # Ön-hesaplanmış değerler varsa kullan, yoksa pipeline çalıştır
    if d.get("production") and d.get("sales") and d.get("profit") is not None:
        production    = safe_float(d.get("production"))
        sales_val     = safe_float(d.get("sales"))
        profit_val    = safe_float(d.get("profit"))
        profit_margin = safe_float(d.get("profit_margin"))
        revenue       = safe_float(d.get("revenue"))
        total_cost    = safe_float(d.get("total_cost"))
    else:
        # Pipeline çalıştırılacak — girdi doğrulaması gerekli
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

    logger.info("/api/recommendation profit=%.1f margin=%.1f%% prod=%.1f sales=%.1f",
                profit_val, profit_margin, production, sales_val)

    # ── Kural motoru — öncelik sıralı ────────────────────────
    recommendations = []
    critical_count  = 0
    warning_count   = 0

    # KURAL 1: Zarar
    if profit_val < 0:
        recommendations.append({
            "priority": 1, "type": "critical", "icon": "⛔",
            "title": "Zarar Ediliyor",
            "detail": (
                f"Günlük zarar: {abs(profit_val):,.0f} ₺. "
                f"Gelir ({revenue:,.0f} ₺) maliyetin ({total_cost:,.0f} ₺) altında."
            ),
            "action": (
                "Fiyatı en az %15 artırın VEYA "
                "çalışan sayısını optimize edin VEYA "
                "enerji verimliliğini artırın."
            ),
        })
        critical_count += 1

    # KURAL 2: Satış < Üretim (talep sorunu)
    if production > 0 and sales_val < production * 0.70:
        gap = production - sales_val
        recommendations.append({
            "priority": 2, "type": "warning", "icon": "📦",
            "title": "Üretim Talebin Üzerinde",
            "detail": (
                f"Üretilen {production:.0f} birimden yalnızca {sales_val:.0f} adeti "
                f"satılabiliyor. {gap:.0f} birim stokta kalıyor."
            ),
            "action": (
                "Fiyatı indirerek talebi artırın VEYA "
                "pazarlama bütçesini artırın VEYA "
                "üretim hacmini talebe göre kısın."
            ),
        })
        warning_count += 1

    # KURAL 3: Satış üretimi neredeyse tüketiyor (stok sınırı)
    elif production > 0 and sales_val >= production * 0.96:
        recommendations.append({
            "priority": 3, "type": "info", "icon": "🏭",
            "title": "Üretim Kapasitesi Doluluk Sınırında",
            "detail": (
                f"Satış ({sales_val:.0f}) üretim kapasitesine ({production:.0f}) "
                "çok yakın. Talep daha da artarsa karşılanamayabilir."
            ),
            "action": (
                "Üretim kapasitesini artırmayı planlayın VEYA "
                "fiyatı hafif artırarak talebi dengelyin."
            ),
        })

    # KURAL 4: Kar marjı çok yüksek
    if profit_margin > 60:
        recommendations.append({
            "priority": 4, "type": "warning", "icon": "🔎",
            "title": f"Kar Marjı Olağandışı Yüksek (%{profit_margin:.1f})",
            "detail": (
                "Tipik üretim işletmelerinde %20-45 beklenir. "
                "Bu seviye maliyet girdilerinin eksik girildiğine işaret edebilir."
            ),
            "action": (
                "COGS, işçilik ve enerji maliyetlerini doğrulayın. "
                "Rekabet için fiyatı optimize etmeyi değerlendirin."
            ),
        })
        warning_count += 1

    # KURAL 5: Düşük kar marjı
    elif profit_margin < 15 and profit_val >= 0:
        recommendations.append({
            "priority": 5, "type": "warning", "icon": "📉",
            "title": f"Düşük Kar Marjı (%{profit_margin:.1f})",
            "detail": (
                "Operasyon kârlı ancak sürdürülebilirlik riski var. "
                f"Günlük kâr: {profit_val:,.0f} ₺."
            ),
            "action": (
                "Fiyatı %10-15 artırın VEYA "
                "sabit maliyetleri (işçilik, enerji) optimize edin."
            ),
        })
        warning_count += 1

    # KURAL 6: Enerji verimsizliği
    if energy > 0 and production > 0:
        energy_per_unit = energy / production
        if energy_per_unit > 6:
            recommendations.append({
                "priority": 6, "type": "warning", "icon": "⚡",
                "title": "Enerji Verimsiz Kullanılıyor",
                "detail": (
                    f"Her birim üretim için {energy_per_unit:.1f} kWh enerji harcanıyor. "
                    "Sektör ortalaması 2-4 kWh/birim."
                ),
                "action": (
                    "Makine bakımı yapın, vardiya planını optimize edin, "
                    "enerji yönetim sistemi kurun. %15-20 tasarruf mümkün."
                ),
            })
            warning_count += 1

    # KURAL 7: Yüksek işçilik oranı
    labor_cost = (workers * worker_cost) / 30.0
    if revenue > 0 and labor_cost / revenue > 0.40:
        recommendations.append({
            "priority": 7, "type": "info", "icon": "👥",
            "title": "Yüksek İşçilik/Gelir Oranı",
            "detail": (
                f"İşçilik maliyeti ({labor_cost:,.0f} ₺/gün) gelirin "
                f"%{labor_cost/revenue*100:.0f}'ini oluşturuyor."
            ),
            "action": (
                "Otomasyon yatırımı, vardiya optimizasyonu veya "
                "çalışan başına üretim verimliliğini artırın."
            ),
        })

    # KURAL 8: Fiyat çok düşük veya çok yüksek
    if price > 0 and price < 80:
        recommendations.append({
            "priority": 8, "type": "info", "icon": "💰",
            "title": "Düşük Fiyat — Marj Riski",
            "detail": (
                f"Ürün fiyatı ({price:.0f} ₺) maliyet yapısına göre düşük. "
                "COGS dahil toplam maliyet per ürün yüksek kalabilir."
            ),
            "action": "Minimum kârlı fiyat noktasını hesaplayın. Fiyatı en az break-even'ın %15 üzerine çekin.",
        })
    elif price > 300:
        recommendations.append({
            "priority": 8, "type": "info", "icon": "💰",
            "title": "Yüksek Fiyat — Talep Baskısı",
            "detail": (
                f"Fiyat ({price:.0f} ₺) yüksek. Bu satış miktarını "
                f"baskılıyor olabilir. Satış/Üretim oranı: %{sales_val/production*100:.0f}."
                if production > 0 else f"Fiyat ({price:.0f} ₺) yüksek."
            ),
            "action": "Fiyat esnekliğini test edin. %10 indirim satış hacmini nasıl etkiler simüle edin.",
        })

    # Öncelik sırasına göre sırala
    recommendations.sort(key=lambda x: x["priority"])

    # ── Genel durum ──────────────────────────────────────────
    if critical_count > 0:
        status       = "critical"
        status_label = "Kritik — Acil Aksiyon Gerekli"
        status_color = "danger"
    elif warning_count >= 2:
        status       = "warning"
        status_label = "Dikkatli — Birden Fazla Risk Mevcut"
        status_color = "warning"
    elif warning_count == 1:
        status       = "caution"
        status_label = "Dikkat — İyileştirme Önerilir"
        status_color = "warning"
    elif not recommendations:
        status       = "good"
        status_label = "Sağlıklı — Sistem Dengeli Çalışıyor"
        status_color = "good"
    else:
        status       = "info"
        status_label = "Bilgi — Optimize Edilebilir Alanlar Var"
        status_color = "good"

    # ── Hızlı aksiyon listesi (en önemli 3) ──────────────────
    action_items = [r["action"] for r in recommendations[:3]]

    return jsonify(clean_for_json({
        "success":       True,
        "status":        status,
        "status_label":  status_label,
        "status_color":  status_color,
        "recommendation_count": len(recommendations),
        "recommendations": recommendations,
        "action_items":    action_items,
        "summary": {
            "production":    round(production, 2),
            "sales":         round(sales_val,  2),
            "profit":        round(profit_val, 2),
            "profit_margin": round(profit_margin, 1),
        },
    }))


# ── 8. Senaryo (What-If) ─────────────────────────────────────
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

    new_prod, new_sales, new_pd = _run_pipeline(
        new_workers, hours, new_energy, price, channel, worker_cost, energy_cost, demo_mode
    )
    orig_prod, orig_sales, orig_pd = _run_pipeline(
        workers, hours, energy, price, channel, worker_cost, energy_cost, demo_mode
    )

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


# ── 9. Dashboard — tam pipeline ──────────────────────────────
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

    logger.info("/api/dashboard w=%s h=%s e=%s p=%s ch=%s demo=%s",
                workers, hours, energy, price, channel, demo_mode)

    # ── Girdi doğrulama ──────────────────────────────────────────
    ok, err = _validate_production_inputs(workers, hours, energy)
    if not ok:
        logger.warning("/api/dashboard validation fail: w=%s h=%s e=%s", workers, hours, energy)
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
        high_margin_warning = (
            f"⚠ Kar marjı %{profit_margin:.1f} — olağandışı yüksek. "
            "Maliyet girdilerini kontrol edin."
        )

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
        # ML model durumu (frontend göstergesi için) — v9.0
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
        # Smart recommender için frontend'e hazır veri
        "_pipeline_state": {
            "workers": workers, "hours": hours, "energy": energy,
            "price": price, "channel": channel,
            "worker_cost": worker_cost, "energy_cost": energy_cost,
            "production": production, "sales": sales_val,
            "profit": profit_val, "profit_margin": profit_margin,
            "revenue": revenue, "total_cost": total_cost,
        },
    }))


# ── /api/sales-ci — Güven Aralıklı Satış Tahmini (YENİ) ─────
@app.route("/api/sales-ci", methods=["POST"])
def sales_ci():
    """
    Güven aralıklı satış tahmini.
    Yanıt: sales, sales_lower, sales_upper, ci_half_width,
            model_used, test_mae, confidence_pct, ratio

    Kullanım: Frontend'de "320 ± 25 adet" formatında gösterilebilir.
    """
    d          = request.get_json(silent=True) or {}
    production = safe_float(d.get("production"), 0)
    price      = safe_float(d.get("price"),      0)
    channel    = d.get("channel") or "web"
    workers    = safe_float(d.get("workers"),  15)
    energy     = safe_float(d.get("energy"),  500)

    logger.info("/api/sales-ci production=%s price=%s channel=%s", production, price, channel)

    result = predict_sales_with_ci(production, price, channel, workers, energy)
    capped = (production > 0) and (result["sales"] >= production * 0.98)

    return jsonify(clean_for_json({
        "success": True,
        **result,
        "stock_capped": capped,
        "notice": "📦 Stok sınırı uygulandı." if capped else None,
        "display": (
            f"{result['sales']:.0f} ± {result['ci_half_width']:.0f} adet  "
            f"[{result['sales_lower']:.0f}, {result['sales_upper']:.0f}]  "
            f"({result['model_used']}, güven %{result['confidence_pct']:.0f})"
        ),
    }))


# ── /api/ml-info — Tam ML Metadata (v9.0) ───────────────────
@app.route("/api/ml-info", methods=["GET"])
def ml_info():
    """
    ML model durumu, metrikler ve açıklanabilirlik bilgisi.
    _ModelRegistry.get_ml_info() tüm detayları derler.
    """
    try:
        info = _ModelRegistry.get_ml_info()
        return jsonify({"success": True, **info})
    except Exception as e:
        logger.error("/api/ml-info hatası: %s", e)
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
