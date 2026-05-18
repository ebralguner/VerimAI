"""
E-Ticaret Analiz Sistemi — Model Katmanı v9.1 (Production)
══════════════════════════════════════════════════════════════
RENDER FIX'LERİ:
  • n_jobs=1  (free tier 1 CPU — n_jobs=-1 OOM'a yol açar)
  • RF n_estimators: 120→60 (RAM tasarrufu)
  • IF n_estimators: 200→100 (RAM tasarrufu)
  • Dataset n: 800→600 (startup hızı)
  • Module-level warmup: _ModelRegistry._warmup() import sırasında çalışır,
    gunicorn worker fork'larında tekrar eğitim olmaz.
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════
# SABITLER
# ══════════════════════════════════════════════════════════════

REFERENCE_PRICE = 150.0
COGS_RATIO      = 0.40
MIN_PRICE       = 0.01
RANDOM_STATE    = 42
TEST_SIZE       = 0.20  # %80 train / %20 test


# ══════════════════════════════════════════════════════════════
# SENTETİK VERİ ÜRETECİ
# ══════════════════════════════════════════════════════════════

def _make_dataset(n: int = 600, seed: int = RANDOM_STATE) -> dict:
    """
    Gerçekçi fabrika verisini simüle eden sentetik dataset.
    n=600 (Render free tier startup hızı için optimize edildi).
    """
    rng = np.random.default_rng(seed)

    n_good = int(n * 0.40)
    n_mid  = int(n * 0.35)
    n_bad  = n - n_good - n_mid

    def _seg(w_lo, w_hi, e_lo, e_hi, p_lo, p_hi, h_lo, h_hi, size):
        return {
            "workers": rng.integers(w_lo, w_hi, size).astype(float),
            "energy":  rng.uniform(e_lo, e_hi, size),
            "price":   rng.uniform(p_lo, p_hi, size),
            "hours":   rng.uniform(h_lo, h_hi, size),
        }

    good = _seg(18, 50, 200, 480, 160, 320, 8, 10, n_good)
    mid  = _seg(10, 28, 380, 660,  90, 180, 7,  9, n_mid)
    bad  = _seg( 4, 14, 620, 950,  35,  95, 6,  8, n_bad)

    workers = np.concatenate([good["workers"], mid["workers"], bad["workers"]])
    energy  = np.concatenate([good["energy"],  mid["energy"],  bad["energy"]])
    price   = np.concatenate([good["price"],   mid["price"],   bad["price"]])
    hours   = np.concatenate([good["hours"],   mid["hours"],   bad["hours"]])

    production = (workers * hours * 2.0) + np.sqrt(energy) * 4.5
    production = np.maximum(production + rng.normal(0, 6, n), 5.0)

    price_effect = np.clip((REFERENCE_PRICE / np.maximum(price, MIN_PRICE)) ** 1.3, 0.30, 2.20)
    base_ratio   = rng.uniform(0.60, 0.95, n)
    sales_ratio  = np.clip(base_ratio * price_effect, 0.20, 1.00)

    p98 = np.percentile(sales_ratio, 98)
    sales_ratio = np.clip(sales_ratio, 0.20, p98)

    sales = np.minimum(production * sales_ratio, production)

    energy_ratio = energy / np.maximum(production, 1.0)

    revenue      = sales * price
    labor        = (workers * 4000.0) / 30.0
    e_cost       = energy * 2.5
    cogs         = sales * price * COGS_RATIO
    profit_margin = np.where(
        revenue > 0,
        (revenue - labor - e_cost - cogs) / revenue * 100.0,
        0.0
    )

    return {
        "workers":       workers,
        "energy":        energy,
        "price":         price,
        "hours":         hours,
        "production":    production,
        "sales":         sales,
        "sales_ratio":   sales_ratio,
        "energy_ratio":  energy_ratio,
        "profit_margin": profit_margin,
    }


# ══════════════════════════════════════════════════════════════
# ML MODEL 1: SATIŞ TAHMİNCİSİ
# ══════════════════════════════════════════════════════════════

@dataclass
class _SalesModelResult:
    name:           str
    model:          object
    scaler:         StandardScaler
    train_r2:       float
    test_r2:        float
    train_mae:      float
    test_mae:       float
    residual_std:   float
    feature_names:  list
    coef_or_fi:     list


@dataclass
class _SalesModel:
    lr:           _SalesModelResult
    rf:           _SalesModelResult
    best:         _SalesModelResult
    trained:      bool = True
    feature_names: list = field(default_factory=lambda: [
        "log(price)", "log(energy)", "workers/50", "production/500"
    ])

    @classmethod
    def train(cls, ds: dict) -> "_SalesModel":
        price      = np.maximum(ds["price"], MIN_PRICE)
        energy     = np.maximum(ds["energy"], 1.0)
        workers    = ds["workers"]
        production = ds["production"]
        y          = ds["sales_ratio"]

        feature_names = ["log(price)", "log(energy)", "workers/50", "production/500"]

        X = np.column_stack([
            np.log1p(price),
            np.log1p(energy),
            workers    / 50.0,
            production / 500.0,
        ])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )

        scaler = StandardScaler()
        Xtr_s  = scaler.fit_transform(X_train)
        Xte_s  = scaler.transform(X_test)

        # Linear Regression
        lr_model = LinearRegression()
        lr_model.fit(Xtr_s, y_train)

        lr_train_pred = lr_model.predict(Xtr_s)
        lr_test_pred  = lr_model.predict(Xte_s)
        lr_residuals  = y_train - lr_train_pred
        lr_res_std    = float(np.std(lr_residuals))

        lr_result = _SalesModelResult(
            name          = "LinearRegression",
            model         = lr_model,
            scaler        = scaler,
            train_r2      = round(float(lr_model.score(Xtr_s, y_train)), 4),
            test_r2       = round(float(lr_model.score(Xte_s, y_test)),  4),
            train_mae     = round(float(mean_absolute_error(y_train, lr_train_pred)), 4),
            test_mae      = round(float(mean_absolute_error(y_test,  lr_test_pred)),  4),
            residual_std  = lr_res_std,
            feature_names = feature_names,
            coef_or_fi    = [round(c, 4) for c in lr_model.coef_.tolist()],
        )

        logger.info(
            "[SalesML-LR] train_R²=%.3f test_R²=%.3f test_MAE=%.4f",
            lr_result.train_r2, lr_result.test_r2, lr_result.test_mae,
        )

        # Random Forest — n_jobs=1 (Render free tier uyumu), n_estimators=60 (RAM)
        rf_model = RandomForestRegressor(
            n_estimators=60,      # FIX: 120→60 (RAM tasarrufu)
            max_depth=8,
            min_samples_leaf=5,
            random_state=RANDOM_STATE,
            n_jobs=1,             # FIX: -1→1 (free tier 1 CPU, OOM önleme)
        )
        rf_model.fit(Xtr_s, y_train)

        rf_train_pred = rf_model.predict(Xtr_s)
        rf_test_pred  = rf_model.predict(Xte_s)

        tree_preds  = np.array([t.predict(Xte_s) for t in rf_model.estimators_])
        rf_res_std  = float(np.mean(np.std(tree_preds, axis=0)))

        rf_result = _SalesModelResult(
            name          = "RandomForest",
            model         = rf_model,
            scaler        = scaler,
            train_r2      = round(float(rf_model.score(Xtr_s, y_train)), 4),
            test_r2       = round(float(rf_model.score(Xte_s, y_test)),  4),
            train_mae     = round(float(mean_absolute_error(y_train, rf_train_pred)), 4),
            test_mae      = round(float(mean_absolute_error(y_test,  rf_test_pred)),  4),
            residual_std  = rf_res_std,
            feature_names = feature_names,
            coef_or_fi    = [round(fi, 4) for fi in rf_model.feature_importances_.tolist()],
        )

        logger.info(
            "[SalesML-RF] train_R²=%.3f test_R²=%.3f test_MAE=%.4f",
            rf_result.train_r2, rf_result.test_r2, rf_result.test_mae,
        )

        best = lr_result if lr_result.test_mae <= rf_result.test_mae else rf_result
        logger.info(
            "[SalesML] Seçilen: %s (MAE=%.4f)", best.name, best.test_mae
        )

        return cls(lr=lr_result, rf=rf_result, best=best, trained=True,
                   feature_names=feature_names)

    def predict_ratio_with_ci(
        self,
        workers:    float,
        energy:     float,
        price:      float,
        production: float,
    ) -> tuple:
        price      = max(MIN_PRICE, price)
        energy     = max(1.0, energy)
        production = max(1.0, production)

        X   = np.array([[np.log1p(price), np.log1p(energy),
                          workers / 50.0,  production / 500.0]])
        X_s = self.best.scaler.transform(X)
        raw = float(self.best.model.predict(X_s)[0])
        ratio = float(np.clip(raw, 0.20, 1.00))

        half_ci = 1.96 * self.best.residual_std
        lo = float(np.clip(ratio - half_ci, 0.10, 1.00))
        hi = float(np.clip(ratio + half_ci, 0.10, 1.00))

        return ratio, lo, hi


# ══════════════════════════════════════════════════════════════
# ML MODEL 2: ANOMALİ TESPİTÇİSİ
# ══════════════════════════════════════════════════════════════

ANOMALY_THRESHOLDS = {
    "critical": -0.20,
    "risky":     0.00,
    "normal":    0.00,
}


@dataclass
class _AnomalyModel:
    model:   IsolationForest
    scaler:  StandardScaler
    trained: bool = False

    @classmethod
    def train(cls, ds: dict) -> "_AnomalyModel":
        prod = np.maximum(ds["production"], 1.0)
        X    = np.column_stack([
            ds["energy"] / prod,
            np.clip(ds["sales"] / prod, 0.0, 1.5),
            ds["workers"],
            np.log1p(ds["energy"]),
        ])
        scaler = StandardScaler()
        X_s    = scaler.fit_transform(X)
        model  = IsolationForest(
            contamination=0.08,
            random_state=RANDOM_STATE,
            n_estimators=100,     # FIX: 200→100 (RAM tasarrufu)
        )
        model.fit(X_s)
        logger.info("[AnomalyML] Eğitim tamamlandı. n=%d", len(prod))
        return cls(model=model, scaler=scaler, trained=True)

    def score_sample(
        self, workers: float, energy: float, sales: float, production: float
    ) -> tuple:
        production = max(production, 1.0)
        X   = np.array([[
            energy / production,
            min(sales / production, 1.5),
            workers,
            np.log1p(energy),
        ]])
        X_s        = self.scaler.transform(X)
        pred       = self.model.predict(X_s)[0]
        score      = round(float(self.model.decision_function(X_s)[0]), 4)
        is_anomaly = pred == -1

        if score < ANOMALY_THRESHOLDS["critical"]:
            interp = "Kritik Bölge (score < −0.20)"
        elif score < ANOMALY_THRESHOLDS["normal"]:
            interp = "Sınır Bölgesi (−0.20 ≤ score < 0)"
        else:
            interp = "Normal Bölge (score ≥ 0)"

        return is_anomaly, score, interp


# ══════════════════════════════════════════════════════════════
# ML MODEL 3: ÖNERİ SINIFLANDIRICI
# ══════════════════════════════════════════════════════════════

REC_LABELS = {
    0: ("fiyat_artir",    "💡 Kar marjı düşük — fiyatı %10-15 artırın veya maliyeti düşürün."),
    1: ("talep_artir",    "📣 Üretim talebin çok üzerinde — pazarlama güçlendirin veya fiyatı indirin."),
    2: ("enerji_optim",   "⚡ Enerji verimsizliği tespit edildi — bakım ve verimlilik yatırımı yapın."),
    3: ("kapasite_artir", "🏭 Kapasite doluluk sınırında — üretim kapasitesini artırmayı planlayın."),
    4: ("dengeli_byu",    "✅ Sistem dengeli — büyüme için kapasite genişlemesi değerlendirin."),
    5: ("zarar_acil",     "⛔ Acil: Zarar ediliyor — fiyat artırın ve maliyetleri denetleyin."),
}

REC_FEATURE_NAMES = ["profit_margin", "sales_ratio", "energy_ratio"]


@dataclass
class _RecommenderModel:
    model:              DecisionTreeClassifier
    feature_importance: dict
    trained:            bool = False

    @classmethod
    def train(cls, ds: dict) -> "_RecommenderModel":
        pm = ds["profit_margin"]
        sr = ds["sales_ratio"]
        er = ds["energy_ratio"]

        labels = np.full(len(pm), 4, dtype=int)

        labels[pm < -5]                                                      = 5
        labels[(pm >= -5)  & (pm < 22)]                                      = 0
        labels[(pm >= 22)  & (sr < 0.65)]                                    = 1
        labels[(pm >= 22)  & (sr >= 0.65) & (er > 5.5)]                     = 2
        labels[(pm >= 30)  & (sr > 0.92) & (er <= 5.5)]                     = 3
        labels[(pm >= 22) & (pm < 30) & (sr >= 0.65) & (er <= 5.5) & (labels == 4)] = 0

        X     = np.column_stack([pm, sr, er])
        model = DecisionTreeClassifier(
            max_depth=4, min_samples_leaf=10, random_state=RANDOM_STATE
        )
        model.fit(X, labels)

        fi   = model.feature_importances_
        fi_d = {name: round(float(v), 4) for name, v in zip(REC_FEATURE_NAMES, fi)}

        logger.info(
            "[RecommenderML] Eğitim tamamlandı. acc=%.3f fi=%s",
            model.score(X, labels), fi_d
        )
        return cls(model=model, feature_importance=fi_d, trained=True)

    def predict_with_confidence(
        self, profit_margin: float, sales_ratio: float, energy_ratio: float
    ) -> tuple:
        X     = np.array([[profit_margin, sales_ratio, energy_ratio]])
        pred  = int(self.model.predict(X)[0])
        proba = self.model.predict_proba(X)[0]
        conf  = round(float(np.max(proba)) * 100, 1)
        key, text = REC_LABELS.get(pred, REC_LABELS[4])
        return key, text, conf


# ══════════════════════════════════════════════════════════════
# SINGLETON MODEL REGISTRY
# Module-level warmup: Gunicorn pre-fork modeliyle uyumlu.
# Worker fork'lamadan ÖNCE _ModelRegistry._warmup() çalışır,
# fork sonrası her worker hafızada hazır modeli taşır.
# ══════════════════════════════════════════════════════════════

class _ModelRegistry:
    _sales:       "_SalesModel | None"       = None
    _anomaly:     "_AnomalyModel | None"     = None
    _recommender: "_RecommenderModel | None" = None
    _ds:          "dict | None"              = None
    _initialized: bool                       = False

    @classmethod
    def _get_dataset(cls) -> dict:
        if cls._ds is None:
            cls._ds = _make_dataset(n=600, seed=RANDOM_STATE)
        return cls._ds

    @classmethod
    def sales(cls) -> _SalesModel:
        if cls._sales is None:
            logger.info("[ModelRegistry] SatışML eğitiliyor...")
            cls._sales = _SalesModel.train(cls._get_dataset())
        return cls._sales

    @classmethod
    def anomaly(cls) -> _AnomalyModel:
        if cls._anomaly is None:
            logger.info("[ModelRegistry] AnomaliML eğitiliyor...")
            cls._anomaly = _AnomalyModel.train(cls._get_dataset())
        return cls._anomaly

    @classmethod
    def recommender(cls) -> _RecommenderModel:
        if cls._recommender is None:
            logger.info("[ModelRegistry] ÖneriML eğitiliyor...")
            cls._recommender = _RecommenderModel.train(cls._get_dataset())
        return cls._recommender

    @classmethod
    def warmup(cls) -> None:
        """Tüm modelleri eğit. Gunicorn preload_app=True ile çağrılır."""
        if cls._initialized:
            return
        logger.info("[ModelRegistry] Warmup başlıyor...")
        try:
            cls.sales()
            cls.anomaly()
            cls.recommender()
            cls._initialized = True
            logger.info("[ModelRegistry] ✓ Tüm modeller hazır.")
        except Exception as e:
            logger.error("[ModelRegistry] Warmup hatası: %s", e)

    @classmethod
    def get_ml_info(cls) -> dict:
        sm  = cls.sales()
        am  = cls.anomaly()
        rm  = cls.recommender()

        return {
            "ml_active": True,
            "architecture": "Hibrit — Deterministik (üretim, kar) + ML (satış, anomali, öneri)",
            "dataset": {
                "size": 600, "seed": RANDOM_STATE,
                "test_size": f"{int(TEST_SIZE * 100)}%",
            },
            "models": {
                "sales": {
                    "selected_model": sm.best.name,
                    "features":       sm.feature_names,
                    "lr": {
                        "train_r2":  sm.lr.train_r2, "test_r2": sm.lr.test_r2,
                        "test_mae":  sm.lr.test_mae,
                    },
                    "rf": {
                        "train_r2":  sm.rf.train_r2, "test_r2": sm.rf.test_r2,
                        "test_mae":  sm.rf.test_mae,
                        "n_estimators": 60,
                    },
                },
                "anomaly": {
                    "model": "IsolationForest",
                    "contamination": 0.08,
                    "n_estimators": 100,
                    "thresholds": ANOMALY_THRESHOLDS,
                },
                "recommender": {
                    "model": "DecisionTreeClassifier",
                    "feature_importance": rm.feature_importance,
                },
            },
        }


# ══════════════════════════════════════════════════════════════
# 1. ÜRETİM
# ══════════════════════════════════════════════════════════════

def predict_production(
    workers: float, hours: float, energy: float, demo_mode: bool = True
) -> float:
    workers = max(0.0, float(workers))
    hours   = max(0.0, float(hours))
    energy  = max(0.0, float(energy))
    if workers <= 0 or hours <= 0:
        return 0.0
    base = (workers * hours * 2.0) + (math.sqrt(energy) * 4.5)
    nf   = np.random.uniform(0.90, 1.10) if demo_mode else 1.0
    return max(0.0, round(base * nf, 2))


# ══════════════════════════════════════════════════════════════
# 2. SATIŞ
# ══════════════════════════════════════════════════════════════

def predict_sales(
    production: float,
    price:      float,
    channel:    str,
    demo_mode:  bool  = True,
    workers:    float = 15.0,
    energy:     float = 500.0,
) -> float:
    production = max(0.0, float(production))
    price      = max(MIN_PRICE, float(price))
    workers    = max(1.0, float(workers))
    energy     = max(1.0, float(energy))

    if production <= 0:
        return 0.0

    try:
        ratio, _lo, _hi = _ModelRegistry.sales().predict_ratio_with_ci(
            workers=workers, energy=energy, price=price, production=production
        )
    except Exception as e:
        logger.warning("[predict_sales] ML hatası (%s) — fallback.", e)
        pe    = float(np.clip((REFERENCE_PRICE / price) ** 1.3, 0.30, 2.20))
        lo_r  = 0.60 if demo_mode else 0.75
        hi_r  = 0.95
        ratio = float(np.clip(
            (np.random.uniform(lo_r, hi_r) if demo_mode else (lo_r + hi_r) / 2) * pe,
            0.20, 1.00
        ))

    ch_effect = 1.10 if str(channel).lower() == "web" else 0.90
    ratio     = float(np.clip(ratio * ch_effect, 0.15, 1.00))

    if demo_mode:
        ratio = float(np.clip(ratio * np.random.uniform(0.95, 1.05), 0.15, 1.00))

    return max(0.0, round(min(production * ratio, production), 2))


def predict_sales_with_ci(
    production: float,
    price:      float,
    channel:    str,
    workers:    float = 15.0,
    energy:     float = 500.0,
) -> dict:
    production = max(0.0, float(production))
    price      = max(MIN_PRICE, float(price))
    workers    = max(1.0, float(workers))
    energy     = max(1.0, float(energy))

    if production <= 0:
        return {
            "sales": 0.0, "sales_lower": 0.0, "sales_upper": 0.0,
            "ratio": 0.0, "ci_half_width": 0.0,
            "model_used": "N/A", "test_mae": 0.0, "confidence_pct": 0.0,
        }

    sm                = _ModelRegistry.sales()
    ratio, lo_r, hi_r = sm.predict_ratio_with_ci(workers, energy, price, production)

    ch_effect = 1.10 if str(channel).lower() == "web" else 0.90
    ratio     = float(np.clip(ratio * ch_effect, 0.15, 1.00))
    lo_r      = float(np.clip(lo_r  * ch_effect, 0.10, 1.00))
    hi_r      = float(np.clip(hi_r  * ch_effect, 0.10, 1.00))

    sales      = round(min(production * ratio, production), 2)
    sales_lo   = round(min(production * lo_r,  production), 2)
    sales_hi   = round(min(production * hi_r,  production), 2)
    half_width = round((sales_hi - sales_lo) / 2, 2)

    mae      = sm.best.test_mae
    conf_pct = round(max(0.0, min(100.0, (1.0 - mae / max(ratio, 0.01)) * 100)), 1)

    return {
        "sales":          sales,
        "sales_lower":    sales_lo,
        "sales_upper":    sales_hi,
        "ratio":          round(ratio, 4),
        "ci_half_width":  half_width,
        "model_used":     sm.best.name,
        "test_mae":       mae,
        "confidence_pct": conf_pct,
    }


# ══════════════════════════════════════════════════════════════
# 3. KAR
# ══════════════════════════════════════════════════════════════

def calculate_profit(
    sales: float, price: float, workers: float,
    worker_cost: float, energy: float, energy_cost: float,
) -> dict:
    sales       = max(0.0, float(sales));       price       = max(0.0, float(price))
    workers     = max(0.0, float(workers));     worker_cost = max(0.0, float(worker_cost))
    energy      = max(0.0, float(energy));      energy_cost = max(0.0, float(energy_cost))

    revenue           = sales * price
    labor_cost        = (workers * worker_cost) / 30.0
    energy_cost_total = energy * energy_cost
    product_cost      = sales * price * COGS_RATIO
    total_cost        = labor_cost + energy_cost_total + product_cost
    profit            = revenue - total_cost
    profit_margin     = (profit / revenue * 100.0) if revenue > 0 else 0.0

    return {
        "revenue":           round(revenue,           2),
        "labor_cost":        round(labor_cost,        2),
        "energy_cost_total": round(energy_cost_total, 2),
        "product_cost":      round(product_cost,      2),
        "total_cost":        round(total_cost,        2),
        "profit":            round(profit,            2),
        "profit_margin":     round(profit_margin,     1),
    }


# ══════════════════════════════════════════════════════════════
# 4. KANAL ANALİZİ
# ══════════════════════════════════════════════════════════════

def analyze_channels(
    app_sales: float, web_sales: float, total_sales: float = 0.0
) -> dict:
    app_sales   = max(0.0, float(app_sales))
    web_sales   = max(0.0, float(web_sales))
    total_sales = max(0.0, float(total_sales))

    if app_sales <= 0 and web_sales <= 0:
        wr        = np.random.uniform(0.50, 0.70)
        web_sales = total_sales * wr
        app_sales = total_sales - web_sales

    total   = app_sales + web_sales
    app_pct = (app_sales / total * 100.0) if total > 0 else 50.0
    web_pct = (web_sales / total * 100.0) if total > 0 else 50.0

    if app_sales > web_sales * 1.10:
        b, r, s = "App",     "📱 App kanalı daha verimli. App kampanyalarına yatırım artırılabilir.", "good"
    elif web_sales > app_sales * 1.10:
        b, r, s = "Web",     "🌐 Web kanalı daha verimli. SEO ve web reklamcılığı güçlendirilebilir.", "good"
    else:
        b, r, s = "Dengeli", "⚖️ Kanallar dengeli. Her iki kanala eşit yatırım sürdürülebilir.", "warning"

    return {
        "app_sales": round(app_sales, 2), "web_sales": round(web_sales, 2),
        "app_percentage": round(app_pct, 1), "web_percentage": round(web_pct, 1),
        "better_channel": b, "recommendation": r, "status": s,
    }


# ══════════════════════════════════════════════════════════════
# 5. ANOMALİ TESPİTİ
# ══════════════════════════════════════════════════════════════

def detect_anomaly(
    energy: float, sales: float, workers: float,
    production: float, profit_margin: float = 0.0,
) -> dict:
    energy     = max(0.0, float(energy))
    sales      = max(0.0, float(sales))
    workers    = max(0.0, float(workers))
    production = max(0.0, float(production))

    rule_hit, rule_sev, rule_reason = False, "normal", ""

    if production > 0 and sales > production * 1.02:
        rule_hit, rule_sev = True, "kritik"
        rule_reason = "🚨 Satış üretimi aşıyor — fiziksel stok sınırı ihlali."
    elif profit_margin > 80:
        rule_hit, rule_sev = True, "kritik"
        rule_reason = f"⚠ Kar marjı %{profit_margin:.1f} — kritik. Maliyet girdileri eksik."
    elif profit_margin > 60:
        rule_hit, rule_sev = True, "orta"
        rule_reason = f"🔎 Kar marjı %{profit_margin:.1f} — olağandışı yüksek."
    elif energy > 0 and production > 0 and (energy / production) > 8:
        rule_hit, rule_sev = True, "kritik"
        rule_reason = f"⚡ Enerji/üretim kritik: {energy:.0f} kWh → {production:.0f} birim."
    elif energy > 850:
        rule_hit, rule_sev = True, "orta"
        rule_reason = f"⚡ Enerji {energy:.0f} kWh — yüksek."

    ml_anomaly, score, score_interp = False, 0.0, "Normal Bölge (score ≥ 0)"
    try:
        ml_anomaly, score, score_interp = _ModelRegistry.anomaly().score_sample(
            workers, energy, sales, production
        )
    except Exception as e:
        logger.warning("[detect_anomaly] ML hatası: %s", e)

    is_anomaly = rule_hit or ml_anomaly

    if rule_hit and rule_sev == "kritik":
        severity, label, reason, status = "kritik", "Kritik Anomali!", rule_reason, "danger"
    elif rule_hit:
        severity, label, reason, status = "orta", "Riskli Durum", rule_reason, "warning"
    elif ml_anomaly and score < ANOMALY_THRESHOLDS["critical"]:
        severity = "kritik"
        label    = "Kritik Anomali!"
        reason   = f"ML ciddi sapma tespit etti. {score_interp}."
        status   = "danger"
    elif ml_anomaly:
        severity = "orta"
        label    = "Riskli Durum"
        reason   = f"ML sınır dışı değerler tespit etti. {score_interp}."
        status   = "warning"
    else:
        severity = "normal"
        label    = "Sistem Normal"
        reason   = f"Tüm göstergeler beklenen aralıkta. {score_interp}."
        status   = "good"

    return {
        "is_anomaly":           is_anomaly,
        "label":                label,
        "reason":               reason,
        "severity":             severity,
        "status":               status,
        "score":                score,
        "score_interpretation": score_interp,
        "ml_triggered":         ml_anomaly,
        "rule_triggered":       rule_hit,
    }


# ══════════════════════════════════════════════════════════════
# 6. AI YORUM
# ══════════════════════════════════════════════════════════════

def generate_commentary(
    production: float, sales: float, profit: float,
    profit_margin: float, energy: float, workers: float, price: float = 0.0,
) -> tuple:
    def _fmt(d, n, o, c): return f"DURUM: {d}\nNEDEN: {n}\nÖNERİ: {o}", c

    ml_key = _get_ml_rec_key(profit_margin, sales, production, energy)

    if profit < 0:
        return _fmt(
            "⛔ Zarar Tespit Edildi",
            f"Maliyet geliri {abs(profit):,.0f} ₺ aşıyor.",
            "Fiyatı artırın veya maliyet kalemlerini düşürün.", "danger"
        )
    if production > 0 and sales > production * 1.02:
        return _fmt(
            "🚨 Veri Tutarsızlığı",
            "Satış üretimi aşıyor — fiziksel olarak mümkün değil.",
            "Üretim ve satış girdilerini kontrol edin.", "danger"
        )
    if profit_margin > 80:
        return _fmt(
            "⚠ Anormal Yüksek Kar Marjı",
            f"Kar marjı %{profit_margin:.1f} — sektör normunun çok üzerinde.",
            "COGS, işçilik ve enerji maliyetlerini doğrulayın.", "danger"
        )
    if profit_margin > 60:
        return _fmt(
            "🔎 Yüksek Kar Marjı — Doğrulama Gerekli",
            f"Kar marjı %{profit_margin:.1f}. Tipik aralık: %20-45.",
            "COGS ve maliyet girdilerini eksiksiz girin.", "warning"
        )
    if energy > 0 and production > 0 and (energy / production) > 6:
        ratio = energy / production
        return _fmt(
            "⚡ Enerji Verimsizliği",
            f"Birim başına {ratio:.1f} kWh enerji — sektör ortalamasının üzerinde.",
            "Makine bakımı ve enerji yönetim sistemi değerlendirin.", "warning"
        )
    if production > 0 and sales > 0 and (sales / production) < 0.65:
        ml_note = " (ML de talep artışı öneriyor.)" if ml_key == "talep_artir" else ""
        return _fmt(
            "📦 Düşük Satış/Üretim Oranı",
            f"Üretimin %{sales/production*100:.0f}'i satılabiliyor.{ml_note}",
            "Pazarlama kampanyası veya fiyat optimizasyonu deneyin.", "warning"
        )
    if profit_margin < 10:
        return _fmt(
            "📉 Düşük Kar Marjı",
            f"Kar marjı %{profit_margin:.1f} — sürdürülebilirlik riski.",
            "Fiyatı %10-15 artırın veya sabit maliyetleri optimize edin.", "warning"
        )
    if profit_margin < 25:
        return _fmt(
            "📊 Gelişime Açık Performans",
            f"Kar marjı %{profit_margin:.1f} — kabul edilebilir ama iyileştirilebilir.",
            "Enerji verimliliğini artırın veya daha yüksek marjlı segmente geçin.", "warning"
        )

    sell_r = (sales / production * 100) if production > 0 else 0
    return _fmt(
        "✅ Sistem Dengeli ve Sağlıklı",
        f"Kar marjı %{profit_margin:.1f}, satış/üretim %{sell_r:.0f}.",
        "Strateji sürdürülebilir. Büyüme için kapasite genişlemesi değerlendirin.", "good"
    )


def _get_ml_rec_key(profit_margin: float, sales: float, production: float, energy: float) -> str:
    try:
        prod      = max(production, 1.0)
        sr        = float(np.clip(sales / prod, 0.0, 1.5))
        er        = float(energy / prod)
        key, _, _ = _ModelRegistry.recommender().predict_with_confidence(profit_margin, sr, er)
        return key
    except Exception:
        return "dengeli_byu"


def generate_recommendation(
    profit_margin: float, sales: float, production: float,
    energy: float, workers: float, severity: str,
    profit: float, price: float = 0.0,
) -> tuple:
    if severity == "kritik":
        return "🚨 Kritik anomali — sistemi durdurun ve tüm girdileri doğrulayın.", "danger"
    if profit < 0:
        return (
            "💡 Acil: (1) Fiyatı en az %15 artırın. "
            "(2) Enerji tüketimini audit edin. (3) Çalışan sayısını optimize edin.", "danger"
        )
    if profit_margin > 60:
        return "🔎 Maliyet girdilerini doğrulayın — COGS, işçilik ve enerji eksiksiz girildi mi?", "warning"

    try:
        prod        = max(production, 1.0)
        sr          = float(np.clip(sales / prod, 0.0, 1.5))
        er          = float(energy / prod)
        key, text, conf = _ModelRegistry.recommender().predict_with_confidence(profit_margin, sr, er)

        if profit_margin < 15 and key not in ("fiyat_artir", "zarar_acil"):
            return "📈 Marjı artırmak için: Fiyatı %10-15 artırın veya enerji optimizasyonu yapın.", "warning"

        cls    = "danger" if key == "zarar_acil" else ("warning" if key != "dengeli_byu" else "good")
        suffix = f" (Model güveni: %{conf:.0f})" if conf < 70 else ""
        return text + suffix, cls

    except Exception as e:
        logger.warning("[generate_recommendation] ML hatası: %s", e)
        if profit_margin < 15:
            return "📈 Marjı artırmak için: Fiyatı %10-15 artırın.", "warning"
        if production > 0 and sales < production * 0.70:
            return "📣 Satış/üretim düşük — pazarlama güçlendirin.", "warning"
        if energy > 700:
            return "⚡ Enerji optimizasyonu ile %15-20 tasarruf mümkün.", "warning"
        return "✅ Sistem sağlıklı. Büyüme için kapasite artışını değerlendirin.", "good"


# ══════════════════════════════════════════════════════════════
# MODULE-LEVEL WARMUP
# Gunicorn'un --preload seçeneği ile fork öncesi çalışır.
# Bu satır model.py import edildiğinde warmup'ı tetikler.
# ══════════════════════════════════════════════════════════════
_ModelRegistry.warmup()
