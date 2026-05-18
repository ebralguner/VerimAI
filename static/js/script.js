/**
 * E-Ticaret Analiz Sistemi — script.js v8.0
 * ══════════════════════════════════════════
 * YENİ:
 *  1. /api/impact    → Etki Analizi (renk kodlu delta tablosu)
 *  2. /api/recommendation → Akıllı Tavsiye (öncelikli öneri listesi)
 *  3. Çift grafik: Gelir/Maliyet/Kar + Üretim/Satış
 *  4. Profesyonel PDF — tüm bölümler dahil
 *  5. STATE tam senkronize — her API sonrası sıfırdan yazar
 *  6. Edge case: price=0, workers=0 → API'ye gitmeden uyarı
 */

"use strict";

// ═══════════════════════════════════════════════════════════════
// GLOBAL STATE
// ═══════════════════════════════════════════════════════════════
const STATE = {
  // Girdi
  workers: 0, hours: 0, energy: 0, price: 0, channel: "web",
  workerCost: 0, energyCost: 0, appSales: 0, webSales: 0,

  // Pipeline çıktıları (her API sonrası taze yazılır)
  production: 0, sales: 0, profit: 0, profitMargin: 0,
  revenue: 0, totalCost: 0,

  // Kontrol
  demoMode:   false,
  lastResult: null,
  history:    [],
  chartFinancial: null,   // Gelir/Maliyet/Kar grafiği
  chartVolume:    null,   // Üretim/Satış grafiği
};

// ═══════════════════════════════════════════════════════════════
// DEMO SENARYOLARI
// ═══════════════════════════════════════════════════════════════
let DEMO_SCENARIOS = [
  { label: "🚀 Yüksek Performanslı Operasyon", tier: "good",
    workers: 30, hours: 10, energy: 500, price: 200, channel: "app",
    workerCost: 3500, energyCost: 2.0, appSales: 0, webSales: 0 },
  { label: "💡 Küçük İşletme Optimumu", tier: "good",
    workers: 8, hours: 9, energy: 260, price: 220, channel: "app",
    workerCost: 3000, energyCost: 1.8, appSales: 0, webSales: 0 },
  { label: "🏭 Orta Ölçekli Fabrika", tier: "mid",
    workers: 15, hours: 8, energy: 500, price: 130, channel: "web",
    workerCost: 4000, energyCost: 2.5, appSales: 0, webSales: 0 },
  { label: "🏪 Orta Hacimli Mağaza", tier: "mid",
    workers: 20, hours: 8, energy: 600, price: 110, channel: "web",
    workerCost: 4500, energyCost: 3.0, appSales: 0, webSales: 0 },
  { label: "⚠️ Verimsiz Senaryo (Anomali)", tier: "bad",
    workers: 40, hours: 6, energy: 900, price: 50, channel: "web",
    workerCost: 8000, energyCost: 4.5, appSales: 0, webSales: 0 },
];
let _scenarioIdx = 0;

function _randomScenario() {
  const t = Math.random();
  let s;
  if (t < 0.40) {
    s = { label: "✨ Rastgele İyi", tier: "good",
      workers: _ri(20,45), hours: _rf(8,10), energy: _ri(200,450), price: _ri(160,280),
      channel: Math.random()>.5?"app":"web", workerCost: _ri(2500,4000), energyCost: _rf(1.5,2.5) };
  } else if (t < 0.80) {
    s = { label: "⚖️ Rastgele Orta", tier: "mid",
      workers: _ri(10,25), hours: _rf(7,9), energy: _ri(350,650), price: _ri(100,160),
      channel: Math.random()>.5?"app":"web", workerCost: _ri(3500,5500), energyCost: _rf(2.0,3.5) };
  } else {
    s = { label: "🔴 Rastgele Riskli", tier: "bad",
      workers: _ri(5,15), hours: _rf(6,8), energy: _ri(700,950), price: _ri(40,90),
      channel: "web", workerCost: _ri(5000,9000), energyCost: _rf(3.5,5.5) };
  }
  s.appSales = 0; s.webSales = 0;
  return s;
}
const _ri = (lo,hi) => Math.round(Math.random()*(hi-lo)+lo);
const _rf = (lo,hi) => parseFloat((Math.random()*(hi-lo)+lo).toFixed(1));


// ═══════════════════════════════════════════════════════════════
// YARDIMCI FONKSİYONLAR
// ═══════════════════════════════════════════════════════════════

function fmt(n, dec=1) {
  if (n===null||n===undefined||isNaN(n)) return "—";
  const a = Math.abs(n);
  if (a>=1_000_000) return (n/1_000_000).toFixed(dec)+"M";
  if (a>=1_000)     return (n/1_000).toFixed(dec)+"K";
  return parseFloat(n.toFixed(dec)).toLocaleString("tr-TR");
}
const fmtTL  = n => fmt(n)+" ₺";
const fmtPct = n => "%"+fmt(n,1);

/** Delta renk kodlaması */
function deltaHTML(val, pct, opts={}) {
  const {suffix=""} = opts;
  const absVal = Math.abs(val);
  const absPct = Math.abs(pct||0);
  if (val > 0.01) return `<span class="delta-pos">▲ +${fmt(absVal)}${suffix} <small>(+${fmt(absPct,1)}%)</small></span>`;
  if (val < -0.01) return `<span class="delta-neg">▼ −${fmt(absVal)}${suffix} <small>(−${fmt(absPct,1)}%)</small></span>`;
  return `<span class="delta-neu">→ Değişmedi</span>`;
}

// ═══════════════════════════════════════════════════════════════
// EDGE-CASE: VALIDATION ERROR HANDLER
// ═══════════════════════════════════════════════════════════════

/**
 * API yanıtının bir validation hatası olup olmadığını kontrol eder.
 * Backend 422 ile { status:"error", error_code:"INVALID_PRODUCTION_INPUTS" } döner.
 */
function isValidationError(res) {
  return res && (res.status === "error" || res.success === false) &&
         (res.error_code === "INVALID_PRODUCTION_INPUTS" || res.production === 0 && res.sales === 0);
}

/**
 * Dashboard/pipeline kartlarını boş + anlamlı uyarıyla sıfırlar.
 * Eski sonuçların kullanıcıya yanlış bilgi vermesini önler.
 */
function _showEmptyDashboard(message) {
  const DASH_IDS = [
    "dashProduction","dashSales","dashRevenue","dashProfit",
    "dashAnomaly","dashChannel",
  ];
  const placeholder = "—";
  DASH_IDS.forEach(id => {
    const el = document.getElementById(id);
    if (el) { el.textContent = placeholder; el.className = "metric-value"; }
  });

  // Grafikleri temizle
  if (STATE.chartFinancial) { STATE.chartFinancial.destroy(); STATE.chartFinancial = null; }
  if (STATE.chartVolume)    { STATE.chartVolume.destroy();    STATE.chartVolume    = null; }

  // KPI'ları sıfırla
  ["kpiEfficiency","kpiProfitRate","kpiCostRate"].forEach(id => {
    const el = document.getElementById(id); if (el) el.textContent = "—";
  });

  // Performans skoru gizle
  const ps = document.getElementById("perfScoreSection");
  if (ps) ps.style.display = "none";

  // Commentary — uyarı mesajı göster
  const box = document.getElementById("commentaryBox");
  const txt = document.getElementById("commentaryText");
  if (box && txt) {
    box.className = "commentary-box warning";
    box.style.display = "flex";
    txt.innerHTML = `<strong>⚠ Veri Yetersiz</strong><br>${message}`;
  }

  // Öneri kutusunu gizle
  const recBox = document.getElementById("recommendationBox");
  if (recBox) recBox.style.display = "none";

  // Banner temizle (stok / marj uyarısı kalmasın)
  showStockBanner(null);
  showMarginWarning(null);

  // Dashboard sonuçlarını göster ama uyarıyla
  const dr = document.getElementById("dashboardResults");
  if (dr) dr.style.display = "block";
  setDashboardEmpty(true);

  // STATE'i sıfırla — eski değerler hesaplamada kullanılmasın
  _patchState({ production:0, sales:0, profit:0, profitMargin:0, revenue:0, totalCost:0 });
}

/**
 * Validation hatasını yakala, kullanıcıya göster ve dashboard'u temizle.
 * @param {object} res - API yanıtı (res.message içermeli)
 */
function handleValidationError(res) {
  const msg = res?.message || "Üretim için pozitif değerler girilmelidir.";
  toast(msg, "error");
  _showEmptyDashboard(msg);
}

/**
 * Üretim girdilerini gönderilmeden önce frontend'de kontrol eder.
 * @returns {boolean} geçerli mi
 */
function _validateProductionInputs() {
  const w = getVal("workers", 0);
  const h = getVal("hours",   0);
  const e = getVal("energy",  0);
  const missing = [];
  if (w <= 0) missing.push("Çalışan (workers)");
  if (h <= 0) missing.push("Çalışma Saati (hours)");
  if (e <= 0) missing.push("Enerji (energy)");
  if (missing.length) {
    const msg = `Üretim için pozitif değerler girilmelidir: ${missing.join(", ")}`;
    toast(msg, "error");
    _showEmptyDashboard(msg);
    return false;
  }
  return true;
}

function toast(msg, type="success") {
  const c = document.getElementById("toastContainer");
  if (!c) return;
  const icons = {success:"✓",info:"ℹ",error:"✕",warning:"⚠"};
  const el = document.createElement("div");
  el.className = `toast ${type}`;
  el.innerHTML = `<span>${icons[type]||"ℹ"}</span><span>${msg}</span>`;
  c.appendChild(el);
  setTimeout(()=>{
    el.style.cssText+="opacity:0;transform:translateX(20px);transition:all .3s";
    setTimeout(()=>el.remove(),300);
  },3500);
}

function setLoading(btnId, loading) {
  const btn = document.getElementById(btnId);
  if (!btn) return;
  btn.disabled = loading;
  btn.classList.toggle("is-loading", loading);
  const sp = btn.querySelector(".spinner");
  const tx = btn.querySelector(".btn-text");
  if (sp) sp.style.display = loading?"inline-block":"none";
  if (tx) {
    if (loading) { tx.dataset.orig=tx.dataset.orig||tx.textContent; tx.textContent="AI analiz yapıyor..."; }
    else { tx.textContent=tx.dataset.orig||tx.textContent; }
  }
}

function setDashboardEmpty(show) {
  const el = document.getElementById("dashboardEmpty");
  if (!el) return;
  el.style.display = show ? "block" : "none";
}

function _parseAnimatedNumber(text) {
  if (!text) return NaN;
  const cleaned = String(text)
    .replace(/[^\d,.-]/g, "")
    .replace(/\.(?=\d{3}\b)/g, "")
    .replace(",", ".");
  const n = parseFloat(cleaned);
  return Number.isFinite(n) ? n : NaN;
}

function animateValue(el, target, render, duration=850) {
  if (!el) return;
  if (!Number.isFinite(target)) {
    el.textContent = "—";
    return;
  }
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (reduceMotion) {
    el.textContent = render(target);
    return;
  }
  const from = _parseAnimatedNumber(el.textContent);
  const startValue = Number.isFinite(from) ? from : 0;
  const diff = target - startValue;
  const t0 = performance.now();
  const easeOut = t => 1 - Math.pow(1 - t, 3);
  const step = now => {
    const p = Math.min((now - t0) / duration, 1);
    const v = startValue + diff * easeOut(p);
    el.textContent = render(v);
    if (p < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

function getVal(id, fb=0) {
  const el=document.getElementById(id); if(!el) return fb;
  const v=parseFloat(el.value); return isNaN(v)?fb:v;
}
function setVal(id, value) {
  const el=document.getElementById(id); if(!el) return;
  el.value=value;
  el.classList.add("field-filled");
  setTimeout(()=>el.classList.remove("field-filled"),600);
}

async function api(endpoint, data) {
  const res = await fetch(endpoint, {
    method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(data)
  });
  if (!res.ok) throw new Error(`API ${res.status}`);
  return res.json();
}

/** Banner helper — stock/margin uyarıları */
function _setBanner(id, html, colorVar) {
  let el=document.getElementById(id);
  if (!el) {
    el=document.createElement("div"); el.id=id;
    el.style.cssText="border-radius:8px;padding:10px 16px;font-size:13px;margin:10px 0;display:none;align-items:center;gap:8px;";
    const hero=document.querySelector(".page-hero"); if(hero) hero.after(el);
  }
  if (html) {
    el.style.cssText+=`;background:var(${colorVar}-dim);border:1px solid var(${colorVar});color:var(${colorVar});display:flex`;
    el.innerHTML=html;
  } else { el.style.display="none"; }
}
const showStockBanner   = h => _setBanner("stockNoticeBar",  h, "--red");
const showMarginWarning = h => _setBanner("marginWarningBar",h, "--yellow");

function severityBadge(severity) {
  const m={kritik:["var(--red-dim)","var(--red)","KRİTİK"],orta:["var(--yellow-dim)","var(--yellow)","RİSKLİ"],normal:["var(--green-dim)","var(--green)","NORMAL"]};
  const [bg,color,label]=m[severity]||m.normal;
  return `<span class="severity-badge" style="background:${bg};color:${color};">${label}</span>`;
}

function _validateInputs(fields) {
  const missing=[];
  fields.forEach(({id,label})=>{ if(!getVal(id)) missing.push(label); });
  if (missing.length) { toast(`Eksik alan: ${missing.join(", ")}`, "error"); return false; }
  return true;
}

function _updateLogoForTheme(theme) {
  const el = document.getElementById("logoIcon");
  if (!el) return;
  el.src = theme === "light"
    ? "/static/assets/logo-icon.png"
    : "/static/assets/logo-icon-dark.png";
}

function initThemeToggle() {
  const btn = document.getElementById("themeToggle");
  if (!btn) return;
  const icon = btn.querySelector(".theme-toggle-icon");
  const text = btn.querySelector(".theme-toggle-text");
  const apply = theme => {
    document.body.setAttribute("data-theme", theme);
    btn.setAttribute("aria-pressed", String(theme === "light"));
    if (icon) icon.textContent = theme === "light" ? "☀" : "◐";
    if (text) text.textContent = theme === "light" ? "Light" : "Dark";
    _updateLogoForTheme(theme);
  };
  apply("dark");
  btn.addEventListener("click", () => {
    const current = document.body.getAttribute("data-theme") || "dark";
    apply(current === "light" ? "dark" : "light");
  });
}

function initTooltips() {
  const controls = Array.from(document.querySelectorAll("input.form-control, select.form-control, textarea.form-control"));
  if (!controls.length) return;
  controls.forEach(el => {
    if (el.dataset.tooltip) return;
    const label = el.closest(".form-group")?.querySelector("label");
    if (!label) return;
    const tooltip = label.textContent?.replace(/\s+/g, " ").trim();
    if (!tooltip) return;
    el.dataset.tooltip = tooltip;
    if (!el.title) el.title = tooltip;
  });
  const tip = document.createElement("div");
  tip.className = "ui-tooltip";
  document.body.appendChild(tip);

  const moveTip = (x, y) => {
    const pad = 14;
    const width = tip.offsetWidth || 160;
    const height = tip.offsetHeight || 28;
    const left = Math.min(Math.max(pad, x + 12), window.innerWidth - width - pad);
    const top = Math.min(Math.max(pad, y - height - 10), window.innerHeight - height - pad);
    tip.style.left = `${left}px`;
    tip.style.top = `${top}px`;
  };

  const showTip = (el, x, y) => {
    const text = el.dataset.tooltip || el.title || "";
    if (!text) return;
    if (!el.dataset.nativeTitle && el.title) el.dataset.nativeTitle = el.title;
    el.removeAttribute("title");
    tip.textContent = text;
    tip.classList.add("show");
    moveTip(x, y);
  };

  const hideTip = el => {
    tip.classList.remove("show");
    if (el?.dataset?.nativeTitle) el.title = el.dataset.nativeTitle;
  };

  controls.forEach(el => {
    el.addEventListener("mouseenter", e => showTip(el, e.clientX, e.clientY));
    el.addEventListener("mousemove", e => moveTip(e.clientX, e.clientY));
    el.addEventListener("mouseleave", () => hideTip(el));
    el.addEventListener("focus", () => {
      const rect = el.getBoundingClientRect();
      showTip(el, rect.left + rect.width / 2, rect.top);
    });
    el.addEventListener("blur", () => hideTip(el));
  });
}

function initSectionReveal() {
  const targets = Array.from(document.querySelectorAll(
    ".page-hero, .section-title, .card, .metric-card, .kpi-card, .chart-card, .history-panel, .run-section"
  ));
  if (!targets.length) return;
  targets.forEach(el => el.classList.add("reveal-section"));
  if (!("IntersectionObserver" in window)) {
    targets.forEach(el => el.classList.add("is-visible"));
    return;
  }
  const io = new IntersectionObserver(entries => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add("is-visible");
        io.unobserve(entry.target);
      }
    });
  }, { threshold: 0.14, rootMargin: "0px 0px -6% 0px" });
  targets.forEach(el => io.observe(el));
}

// ═══════════════════════════════════════════════════════════════
// STATE SENKRONIZASYONU
// ═══════════════════════════════════════════════════════════════

function readInputs() {
  const ch=document.getElementById("channel");
  return {
    workers:    getVal("workers",15),   hours:      getVal("hours",8),
    energy:     getVal("energy",500),   price:      getVal("price",130),
    channel:    ch?ch.value:"web",
    workerCost: getVal("workerCost",4000), energyCost: getVal("energyCost",2),
    appSales:   getVal("appSales",0),   webSales:   getVal("webSales",0),
  };
}

function syncState() { Object.assign(STATE, readInputs()); }

/** Pipeline çıktılarını STATE'e yaz — eski veri kalmaz */
function _patchState(patch) {
  const keys=["production","sales","profit","profitMargin","revenue","totalCost"];
  keys.forEach(k=>{ if(patch[k]!==undefined) STATE[k]=patch[k]; });
}

/** Dashboard sonucundan tam STATE yenileme */
function _applyDashboardResult(res) {
  _patchState({
    production:   res.production,
    sales:        res.sales,
    profit:       res.profit,
    profitMargin: res.kpi.profit_margin,
    revenue:      res.revenue,
    totalCost:    res.total_cost,
  });
  STATE.lastResult = res;
  setVal("productionInput", res.production.toFixed(1));
}


// ═══════════════════════════════════════════════════════════════
// MOD SİSTEMİ
// ═══════════════════════════════════════════════════════════════

function activateDemoMode() {
  _scenarioIdx=0;
  applyScenario(DEMO_SCENARIOS[0]);
  STATE.demoMode=true;
  _updateModeUI(true);
  toast(`Demo: "${DEMO_SCENARIOS[0].label}"`, "info");
  runDashboard();
}

let _nextClick=0;
function nextDemoScenario() {
  _nextClick++;
  const sc = _nextClick%5===0 ? _randomScenario() : DEMO_SCENARIOS[(_scenarioIdx=((_scenarioIdx+1)%DEMO_SCENARIOS.length))];
  applyScenario(sc);
  toast(`Senaryo: "${sc.label}"`, "info");
  runDashboard();
}

function applyScenario(sc) {
  const map={workers:"workers",hours:"hours",energy:"energy",price:"price",
    workerCost:"workerCost",energyCost:"energyCost",appSales:"appSales",webSales:"webSales"};
  Object.entries(map).forEach(([k,id])=>{ if(sc[k]!==undefined) setVal(id,sc[k]); });
  const ch=document.getElementById("channel"); if(ch&&sc.channel) ch.value=sc.channel;
  syncState();
}

function activateRealMode() {
  ["workers","hours","energy","price","workerCost","energyCost","appSales","webSales"]
    .forEach(id=>{ const el=document.getElementById(id); if(el) el.value=""; });
  const ch=document.getElementById("channel"); if(ch) ch.value="web";
  _patchState({production:0,sales:0,profit:0,profitMargin:0,revenue:0,totalCost:0});
  STATE.demoMode=false;
  showStockBanner(null); showMarginWarning(null);
  _updateModeUI(false);
  toast("Gerçek mod aktif — değerleri girin.", "info");
}

function _updateModeUI(isDemo) {
  document.getElementById("modeBtnDemo")?.classList.toggle("mode-active",isDemo);
  document.getElementById("modeBtnReal")?.classList.toggle("mode-active",!isDemo);
  const nb=document.getElementById("btnNextScenario");
  if(nb) nb.style.display=isDemo?"inline-flex":"none";
  const lb=document.getElementById("modeLabel");
  if(lb){ lb.textContent=isDemo?"🎭 Demo Mod Aktif":"⚙ Gerçek Mod Aktif"; lb.style.color=isDemo?"var(--yellow)":"var(--green)"; }
}


// ═══════════════════════════════════════════════════════════════
// 1. ÜRETİM
// ═══════════════════════════════════════════════════════════════

async function calculateProduction() {
  syncState();
  if (!_validateProductionInputs()) return;
  setLoading("btnProduction",true);
  try {
    const res = await api("/api/production",{workers:STATE.workers,hours:STATE.hours,energy:STATE.energy,demo_mode:STATE.demoMode});
    if (isValidationError(res)) { handleValidationError(res); return; }
    _patchState({production:res.production});
    document.getElementById("resultProduction").textContent=fmt(res.production);
    document.getElementById("resultProductionCard").style.display="block";
    setVal("productionInput",res.production.toFixed(1));
    toast("✓ Üretim hesaplandı → Satış modülüne aktarıldı.","success");
  } catch(e){toast("Hata: "+e.message,"error");}
  finally{setLoading("btnProduction",false);}
}


// ═══════════════════════════════════════════════════════════════
// 2. SATIŞ
// ═══════════════════════════════════════════════════════════════

async function calculateSales() {
  syncState();
  const production=getVal("productionInput",STATE.production);
  if(!_validateInputs([{id:"price",label:"Fiyat"}])||!production){
    toast("Önce üretim hesaplayın.","error"); return;
  }
  setLoading("btnSales",true);
  try {
    const res=await api("/api/sales",{production,price:STATE.price,channel:STATE.channel,demo_mode:STATE.demoMode});
    _patchState({sales:res.sales});
    document.getElementById("resultSales").textContent=fmt(res.sales);
    document.getElementById("resultSalesCard").style.display="block";
    if(res.stock_capped&&res.notice){
      showStockBanner(`<span>🚫</span><span>${res.notice}</span>`);
      toast(res.notice,"error");
    } else { showStockBanner(null); }
    if(!getVal("appSales")&&!getVal("webSales")){
      setVal("appSales",Math.round(res.sales*0.40));
      setVal("webSales",Math.round(res.sales*0.60));
    }
    toast("✓ Satış hesaplandı.","success");
  } catch(e){toast("Hata: "+e.message,"error");}
  finally{setLoading("btnSales",false);}
}


// ═══════════════════════════════════════════════════════════════
// 3. KAR
// ═══════════════════════════════════════════════════════════════

async function calculateProfit() {
  syncState();
  const salesVal=STATE.sales||getVal("salesInput",0);
  if(!salesVal){toast("Önce satış hesaplayın.","error");return;}
  if(!_validateInputs([{id:"price",label:"Fiyat"},{id:"workers",label:"Çalışan"},{id:"workerCost",label:"Maaş"}])) return;
  setLoading("btnProfit",true);
  try {
    const res=await api("/api/profit",{sales:salesVal,price:STATE.price,workers:STATE.workers,
      worker_cost:STATE.workerCost,energy:STATE.energy,energy_cost:STATE.energyCost});
    _patchState({profit:res.profit,profitMargin:res.profit_margin,revenue:res.revenue,totalCost:res.total_cost});

    const el=document.getElementById("resultProfit");
    el.textContent=fmtTL(res.profit);
    el.className="metric-value "+(res.profit>=0?"green":"red");
    document.getElementById("resultRevenue").textContent=fmtTL(res.revenue);
    document.getElementById("resultCost").textContent=fmtTL(res.total_cost);
    const prEl=document.getElementById("resultProfitRate");
    if(prEl){prEl.textContent=fmtPct(res.profit_margin);prEl.style.color=res.profit_margin>=0?"var(--green)":"var(--red)";}
    _renderCostBreak("resultCostBreak",res);
    document.getElementById("resultProfitCard").style.display="block";

    if(res.high_margin_warning){
      showMarginWarning(`<span>⚠</span><span>${res.high_margin_warning}</span>`);
      toast(res.high_margin_warning,"warning");
    } else {
      showMarginWarning(null);
      toast(res.profit>=0?"✓ Karlı operasyon!":"⚠ Zarar!",res.profit>=0?"success":"error");
    }
  } catch(e){toast("Hata: "+e.message,"error");}
  finally{setLoading("btnProfit",false);}
}

function _renderCostBreak(elId, res) {
  const el=document.getElementById(elId); if(!el) return;
  const parts=[];
  if(res.labor_cost!==undefined)        parts.push(`İşçilik: <strong>${fmtTL(res.labor_cost)}</strong>`);
  if(res.energy_cost_total!==undefined) parts.push(`Enerji: <strong>${fmtTL(res.energy_cost_total)}</strong>`);
  if(res.product_cost!==undefined)      parts.push(`COGS: <strong>${fmtTL(res.product_cost)}</strong>`);
  el.innerHTML=parts.join(" · ");
  el.style.display="block";
}


// ═══════════════════════════════════════════════════════════════
// 4. KANAL
// ═══════════════════════════════════════════════════════════════

async function calculateChannel() {
  syncState();
  setLoading("btnChannel",true);
  try {
    const res=await api("/api/channel",{app_sales:STATE.appSales||getVal("appSales",0),
      web_sales:STATE.webSales||getVal("webSales",0),total_sales:STATE.sales});
    updateChannelPanel(res);
    toast("Kanal analizi tamamlandı.","success");
  } catch(e){toast("Hata: "+e.message,"error");}
  finally{setLoading("btnChannel",false);}
}


// ═══════════════════════════════════════════════════════════════
// 5. ANOMALİ
// ═══════════════════════════════════════════════════════════════

async function calculateAnomaly() {
  syncState();
  if(!STATE.energy||!STATE.workers){toast("Enerji ve çalışan zorunlu.","error");return;}
  setLoading("btnAnomaly",true);
  try {
    const res=await api("/api/anomaly",{energy:STATE.energy,sales:STATE.sales,
      workers:STATE.workers,production:STATE.production,profit_margin:STATE.profitMargin});
    updateAnomalyPanel(res);
    document.getElementById("resultAnomalyCard").style.display="block";
    toast(res.is_anomaly?"⚠ Anomali!":"✓ Normal.","success");
  } catch(e){toast("Hata: "+e.message,"error");}
  finally{setLoading("btnAnomaly",false);}
}


// ═══════════════════════════════════════════════════════════════
// 6. ETKİ ANALİZİ (/api/impact) — YENİ
// ═══════════════════════════════════════════════════════════════

async function runImpactAnalysis() {
  syncState();
  if (!_validateProductionInputs()) return;
  if (!STATE.price) {
    toast("Ürün fiyatı zorunlu.", "error"); return;
  }

  const energyChange = getVal("impactEnergyChange", 0);
  const workerChange = getVal("impactWorkerChange", 0);

  if (energyChange === 0 && workerChange === 0) {
    toast("En az bir değişim parametresi girin.", "error"); return;
  }

  setLoading("btnImpact", true);
  try {
    const res = await api("/api/impact", {
      workers:       STATE.workers,   hours:       STATE.hours,
      energy:        STATE.energy,    price:       STATE.price,
      channel:       STATE.channel,
      worker_cost:   STATE.workerCost, energy_cost: STATE.energyCost,
      energy_change: energyChange,    worker_change: workerChange,
      demo_mode:     false,   // etki analizi deterministik
    });
    if (isValidationError(res)) { handleValidationError(res); return; }

    _renderImpactResult(res);
    document.getElementById("impactResultCard").style.display = "block";
    document.getElementById("impactResultCard").scrollIntoView({behavior:"smooth",block:"nearest"});
    toast("✓ Etki analizi tamamlandı.", "success");
  } catch(e) { toast("Hata: "+e.message, "error"); }
  finally    { setLoading("btnImpact", false); }
}

function _renderImpactResult(res) {
  const o=res.old, n=res.new, d=res.delta;

  // Özet başlık
  const summaryEl = document.getElementById("impactSummary");
  if (summaryEl) {
    summaryEl.className = `impact-summary ${res.verdict}`;
    summaryEl.innerHTML =
      `<span class="impact-verdict-icon">${
        res.verdict==="positive"?"✅":res.verdict==="negative"?"⛔":res.verdict==="warning"?"⚠️":"→"
      }</span>
       <span>${res.verdict_text}</span>`;
  }

  // Uygulanan değişimler
  const appliedEl = document.getElementById("impactApplied");
  if (appliedEl) {
    const parts=[];
    if(res.applied.energy_change!==0) parts.push(`Enerji ${res.applied.energy_change>0?"+":""}${res.applied.energy_change}% (${res.applied.new_energy} kWh)`);
    if(res.applied.worker_change!==0) parts.push(`Çalışan ${res.applied.worker_change>0?"+":""}${res.applied.worker_change}% (${res.applied.new_workers} kişi)`);
    appliedEl.textContent = parts.join(" · ");
  }

  // Delta tablosu
  const rows = [
    { label:"Üretim",      old:o.production, new:n.production, delta:d.production, pct:d.production_pct, suffix:" birim" },
    { label:"Satış",       old:o.sales,      new:n.sales,      delta:d.sales,      pct:d.sales_pct,      suffix:" adet"  },
    { label:"Gelir",       old:o.revenue,    new:n.revenue,    delta:d.revenue,    pct:d.revenue_pct,    suffix:" ₺"    },
    { label:"Toplam Maliyet", old:o.total_cost, new:n.total_cost, delta:d.total_cost, pct:d.total_cost_pct, suffix:" ₺" },
    { label:"Kar",         old:o.profit,     new:n.profit,     delta:d.profit,     pct:d.profit_pct,     suffix:" ₺"    },
    { label:"Kar Marjı",   old:o.profit_margin, new:n.profit_margin,
      delta:n.profit_margin-o.profit_margin, pct:null, suffix:"%", isMargin:true },
  ];

  const tbody = document.getElementById("impactTableBody");
  if (tbody) {
    tbody.innerHTML = rows.map(r => {
      const oldFmt  = r.isMargin ? fmtPct(r.old)   : (r.suffix===" ₺" ? fmtTL(r.old)   : fmt(r.old)+r.suffix);
      const newFmt  = r.isMargin ? fmtPct(r.new)   : (r.suffix===" ₺" ? fmtTL(r.new)   : fmt(r.new)+r.suffix);
      const dHTML   = r.isMargin
        ? (r.delta>0.05?`<span class="delta-pos">▲ +${fmt(r.delta,1)}%</span>`
          :r.delta<-0.05?`<span class="delta-neg">▼ ${fmt(r.delta,1)}%</span>`
          :`<span class="delta-neu">→</span>`)
        : deltaHTML(r.delta, r.pct, {suffix: r.suffix===" ₺"?" ₺":r.suffix==="%" ? "%" : ""});
      // Kâr satırını vurgula
      const isProfit = r.label === "Kar";
      return `<tr class="${isProfit?"impact-row-profit":""}">
        <td class="impact-label">${r.label}</td>
        <td class="impact-old">${oldFmt}</td>
        <td class="impact-new">${newFmt}</td>
        <td class="impact-delta">${dHTML}</td>
      </tr>`;
    }).join("");
  }
}


// ═══════════════════════════════════════════════════════════════
// 7. AKILLI TAVSİYE (/api/recommendation) — YENİ
// ═══════════════════════════════════════════════════════════════

async function runSmartRecommendation() {
  syncState();

  // Önce mevcut STATE'den gönder; pipeline çalıştırılmamışsa backend çalıştırır
  const payload = {
    workers:       STATE.workers,    hours:       STATE.hours,
    energy:        STATE.energy,     price:       STATE.price,
    channel:       STATE.channel,
    worker_cost:   STATE.workerCost, energy_cost: STATE.energyCost,
    // Ön-hesaplanmış değerler varsa ekle
    ...(STATE.production && {
      production:    STATE.production,
      sales:         STATE.sales,
      profit:        STATE.profit,
      profit_margin: STATE.profitMargin,
      revenue:       STATE.revenue,
      total_cost:    STATE.totalCost,
    }),
    demo_mode: false,
  };

  setLoading("btnRecommendation", true);
  try {
    const res = await api("/api/recommendation", payload);
    if (isValidationError(res)) { handleValidationError(res); return; }
    _renderRecommendationResult(res);
    document.getElementById("recommendationResultCard").style.display = "block";
    document.getElementById("recommendationResultCard").scrollIntoView({behavior:"smooth",block:"nearest"});
    toast(`✓ ${res.recommendation_count} öneri üretildi.`, "success");
  } catch(e) { toast("Hata: "+e.message, "error"); }
  finally    { setLoading("btnRecommendation", false); }
}

function _renderRecommendationResult(res) {
  // Durum etiketi
  const statusEl = document.getElementById("recStatusLabel");
  if (statusEl) {
    const colorMap = {critical:"var(--red)",warning:"var(--yellow)",caution:"var(--yellow)",good:"var(--green)",info:"var(--blue)"};
    statusEl.textContent = res.status_label;
    statusEl.style.color = colorMap[res.status] || "var(--blue)";
    statusEl.style.background = colorMap[res.status]
      ? colorMap[res.status].replace(")","-dim)").replace("var(","var(") : "var(--blue-dim)";
    statusEl.style.cssText += ";padding:6px 14px;border-radius:20px;font-size:13px;font-weight:600;display:inline-block;margin-bottom:16px;";
  }

  // Özet sayılar
  const s = res.summary;
  const sumEl = document.getElementById("recSummaryRow");
  if (sumEl && s) {
    sumEl.innerHTML = `
      <div class="rec-sum-item"><span class="rec-sum-label">Üretim</span><span class="rec-sum-val" style="color:var(--blue)">${fmt(s.production)} birim</span></div>
      <div class="rec-sum-item"><span class="rec-sum-label">Satış</span><span class="rec-sum-val" style="color:var(--purple)">${fmt(s.sales)} adet</span></div>
      <div class="rec-sum-item"><span class="rec-sum-label">Kar</span><span class="rec-sum-val" style="color:${s.profit>=0?"var(--green)":"var(--red)"}">${fmtTL(s.profit)}</span></div>
      <div class="rec-sum-item"><span class="rec-sum-label">Kar Marjı</span><span class="rec-sum-val" style="color:${s.profit_margin>=15?"var(--green)":"var(--red)"}">${fmtPct(s.profit_margin)}</span></div>`;
  }

  // Öneri listesi
  const listEl = document.getElementById("recList");
  if (!listEl) return;
  if (!res.recommendations.length) {
    listEl.innerHTML = `<div class="rec-empty">✅ Sistemde aktif iyileştirme alanı tespit edilmedi.</div>`;
    return;
  }

  const typeColors = {
    critical: {bg:"var(--red-dim)",border:"var(--red)",icon_bg:"var(--red)"},
    warning:  {bg:"var(--yellow-dim)",border:"var(--yellow)",icon_bg:"var(--yellow)"},
    info:     {bg:"var(--blue-dim)",border:"var(--blue)",icon_bg:"var(--blue)"},
  };

  listEl.innerHTML = res.recommendations.map((r,i)=>{
    const c=typeColors[r.type]||typeColors.info;
    return `<div class="rec-item" style="border-left:3px solid ${c.border};background:${c.bg};border-radius:8px;padding:14px 16px;margin-bottom:10px;">
      <div class="rec-item-header" style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">
        <span style="font-size:20px;">${r.icon}</span>
        <span class="rec-item-title" style="font-weight:600;color:var(--fg-primary);font-size:14px;">${r.title}</span>
        <span style="margin-left:auto;font-size:11px;color:var(--fg-muted);background:var(--bg-overlay);padding:2px 8px;border-radius:10px;">Öncelik #${i+1}</span>
      </div>
      <p class="rec-item-detail" style="color:var(--fg-secondary);font-size:13px;margin:0 0 8px 30px;">${r.detail}</p>
      <p class="rec-item-action" style="color:var(--fg-primary);font-size:13px;margin:0 0 0 30px;font-style:italic;">
        → ${r.action}
      </p>
    </div>`;
  }).join("");
}


// ═══════════════════════════════════════════════════════════════
// 8. SENARYO (What-If)
// ═══════════════════════════════════════════════════════════════

async function runScenario() {
  syncState();
  const energyChange=getVal("scenarioEnergyChange",0);
  const workerChange=getVal("scenarioWorkerChange",0);
  setLoading("btnScenario",true);
  try {
    const res=await api("/api/scenario",{
      workers:STATE.workers,hours:STATE.hours,energy:STATE.energy,price:STATE.price,
      channel:STATE.channel,worker_cost:STATE.workerCost,energy_cost:STATE.energyCost,
      energy_change:energyChange,worker_change:workerChange,demo_mode:STATE.demoMode
    });
    if (isValidationError(res)) { handleValidationError(res); return; }
    const s=res.scenario, d=res.delta;

    document.getElementById("scProduction").textContent=fmt(s.production)+" birim";
    document.getElementById("scSales").textContent=fmt(s.sales)+" adet";
    const scP=document.getElementById("scProfit");
    scP.textContent=fmtTL(s.profit);
    scP.className="scenario-metric-val "+(s.profit>=0?"green":"red");
    document.getElementById("scProductionDelta").innerHTML=deltaHTML(d.production,d.production_pct||null,{suffix:" birim"});
    document.getElementById("scSalesDelta").innerHTML=deltaHTML(d.sales,null,{suffix:" adet"});
    document.getElementById("scProfitDelta").innerHTML=deltaHTML(d.profit,d.profit_pct||null,{suffix:" ₺"});
    document.getElementById("scenarioApplied").textContent=
      `Enerji ${energyChange>=0?"+":""}${energyChange}% · Çalışan ${workerChange>=0?"+":""}${workerChange}%`;
    const mEl=document.getElementById("scMargin");
    if(mEl){mEl.textContent=fmtPct(s.profit_margin||0);mEl.style.color=(s.profit_margin||0)>=0?"var(--green)":"var(--red)";}
    _renderCostBreak("scCostBreak",{product_cost:s.product_cost});
    const cEl=document.getElementById("scenarioCommentary");
    if(d.profit>0){cEl.textContent=`✓ Kâr ${fmt(d.profit)} ₺ artıyor.`;cEl.className="scenario-commentary good";}
    else if(d.profit<0){cEl.textContent=`✕ Kâr ${fmt(Math.abs(d.profit))} ₺ azalıyor.`;cEl.className="scenario-commentary danger";}
    else{cEl.textContent="→ Kâr değişmiyor.";cEl.className="scenario-commentary neutral";}
    document.getElementById("scenarioResultCard").style.display="block";
    toast("Senaryo hesaplandı.","success");
  } catch(e){toast("Hata: "+e.message,"error");}
  finally{setLoading("btnScenario",false);}
}


// ═══════════════════════════════════════════════════════════════
// 9. DASHBOARD — tam pipeline + grafik güncelleme
// ═══════════════════════════════════════════════════════════════

async function runDashboard() {
  syncState();
  // ── Frontend ön doğrulama — API'ye gitmeden yakala ───────────
  if (!_validateProductionInputs()) return;
  if (!STATE.price) {
    toast("Ürün fiyatı zorunlu.", "error"); return;
  }
  setLoading("btnDashboard",true);
  try {
    const res=await api("/api/dashboard",{
      workers:STATE.workers,hours:STATE.hours,energy:STATE.energy,price:STATE.price,
      channel:STATE.channel,worker_cost:STATE.workerCost,energy_cost:STATE.energyCost,
      app_sales:STATE.appSales,web_sales:STATE.webSales,demo_mode:STATE.demoMode
    });

    // ── Backend validation error (422 / success:false) ────────
    if (isValidationError(res)) { handleValidationError(res); return; }

    // ── STATE güncelle ────────────────────────────────────
    _applyDashboardResult(res);

    // ── Bannerlar ─────────────────────────────────────────
    // production=0 iken stok sınırı uyarısı gösterme (edge-case)
    const showStock = res.stock_notice && res.production > 0;
    showStockBanner(showStock ? `<span>🚫</span><span>${res.stock_notice}</span>` : null);
    showMarginWarning(res.high_margin_warning?`<span>⚠</span><span>${res.high_margin_warning}</span>`:null);

    // ── Özet metrikler ────────────────────────────────────
    document.getElementById("dashProduction").textContent=fmt(res.production);
    document.getElementById("dashSales").textContent=fmt(res.sales);
    document.getElementById("dashRevenue").textContent=fmtTL(res.revenue);
    const pEl=document.getElementById("dashProfit");
    pEl.textContent=fmtTL(res.profit);
    pEl.className="metric-value "+(res.profit>=0?"green":"red");
    const aEl=document.getElementById("dashAnomaly");
    aEl.textContent=res.anomaly.label;
    aEl.className="metric-value "+(res.anomaly.is_anomaly?"red":"green");
    const svEl=document.getElementById("dashAnomalySeverity");
    if(svEl) svEl.innerHTML=severityBadge(res.anomaly.severity);
    document.getElementById("dashChannel").textContent=res.channel.better_channel;

    // ── KPI ──────────────────────────────────────────────
    const eff=res.kpi.efficiency;
    const kpiEffEl=document.getElementById("kpiEfficiency");
    if(kpiEffEl){
      kpiEffEl.style.color=eff>0.6?"var(--green)":"var(--yellow)";
      animateValue(kpiEffEl, eff, v => Math.max(0, v).toFixed(3));
    }
    const pm=res.kpi.profit_margin;
    const kpiPmEl=document.getElementById("kpiProfitRate");
    if(kpiPmEl){
      kpiPmEl.style.color=pm>60?"var(--yellow)":pm>=15?"var(--green)":"var(--red)";
      animateValue(kpiPmEl, pm, v => fmtPct(Math.max(-999, v)));
    }
    const cr=res.kpi.cost_rate;
    const kpiCrEl=document.getElementById("kpiCostRate");
    if(kpiCrEl){
      kpiCrEl.style.color=cr<80?"var(--green)":"var(--red)";
      animateValue(kpiCrEl, cr, v => fmtPct(Math.max(-999, v)));
    }

    // COGS maliyet kırılımı
    _renderCostBreak("dashCostBreak",{
      labor_cost:res.cost_breakdown?.labor_cost,
      energy_cost_total:res.cost_breakdown?.energy_cost_total,
      product_cost:res.cost_breakdown?.product_cost
    });

    // ── AI Yorum ──────────────────────────────────────────
    _renderCommentary(res.commentary,res.commentary_status);

    // ── Performans skoru ──────────────────────────────────
    renderPerfScore(res.performance,res.recommendation,res.recommendation_status);

    // ── Çift grafik ───────────────────────────────────────
    updateCharts(res);

    // ── Kanal + Anomali panelleri ─────────────────────────
    updateChannelPanel(res.channel);
    updateAnomalyPanel(res.anomaly);

    // ── Pipeline adımları ─────────────────────────────────
    [1,2,3,4,5].forEach(i=>document.getElementById("pStep"+i)?.classList.add("active"));

    // ── Geçmiş ───────────────────────────────────────────
    _saveHistory(res);

    document.getElementById("dashboardResults").style.display="block";
    document.getElementById("pdfExportSection").style.display="block";
    setDashboardEmpty(false);
    document.getElementById("dashboardResults").scrollIntoView({behavior:"smooth",block:"start"});
    toast("✓ Dashboard analizi tamamlandı!","success");
  } catch(e){toast("Hata: "+e.message,"error");}
  finally{setLoading("btnDashboard",false);}
}

function _renderCommentary(text,status) {
  const box=document.getElementById("commentaryBox");
  const el=document.getElementById("commentaryText");
  if(!box||!el) return;
  box.className=`commentary-box ${status}`;
  box.style.display="flex";
  const formatted=(text||"")
    .replace(/^DURUM:/m,"<strong>DURUM:</strong>")
    .replace(/^NEDEN:/m,"<br><strong>NEDEN:</strong>")
    .replace(/^ÖNERİ:/m,"<br><strong>ÖNERİ:</strong>");
  el.innerHTML=formatted;
}


// ═══════════════════════════════════════════════════════════════
// ÇİFT GRAFİK — Finansal + Hacim
// ═══════════════════════════════════════════════════════════════

function updateCharts(res) {
  _updateFinancialChart(res.revenue, res.total_cost, res.profit);
  _updateVolumeChart(res.production, res.sales);
}

const CHART_DEFAULTS = {
  responsive:true, maintainAspectRatio:true,
  animation:{duration:700,easing:"easeOutQuart"},
  plugins:{
    legend:{display:true,labels:{color:"#8b949e",font:{family:"JetBrains Mono",size:11}}},
    tooltip:{backgroundColor:"#161b22",borderColor:"#21262d",borderWidth:1,
      titleColor:"#e6edf3",bodyColor:"#8b949e",
      callbacks:{label:c=>` ${c.parsed.y.toLocaleString("tr-TR")}`}},
  },
  scales:{
    x:{grid:{color:"#21262d"},ticks:{color:"#8b949e",font:{family:"JetBrains Mono",size:11}}},
    y:{grid:{color:"#21262d"},ticks:{color:"#8b949e",font:{family:"JetBrains Mono",size:11}}},
  },
};

function _updateFinancialChart(revenue, cost, profit) {
  const ctx=document.getElementById("chartFinancial"); if(!ctx) return;
  if(STATE.chartFinancial) STATE.chartFinancial.destroy();
  STATE.chartFinancial=new Chart(ctx.getContext("2d"),{
    type:"bar",
    data:{
      labels:["Gelir","Maliyet","Kar"],
      datasets:[{
        label:"Tutar (₺)",
        data:[revenue,cost,profit],
        backgroundColor:["rgba(63,185,80,.75)","rgba(248,81,73,.75)",profit>=0?"rgba(56,139,253,.85)":"rgba(248,81,73,.85)"],
        borderColor:["#3fb950","#f85149",profit>=0?"#388bfd":"#f85149"],
        borderWidth:1.5,borderRadius:8,
      }],
    },
    options:{...CHART_DEFAULTS,plugins:{...CHART_DEFAULTS.plugins,
      legend:{display:false},
      title:{display:true,text:"💰 Finansal Özet (₺)",color:"#8b949e",font:{size:12,family:"JetBrains Mono"}}}},
  });
}

function _updateVolumeChart(production, sales) {
  const ctx=document.getElementById("chartVolume"); if(!ctx) return;
  if(STATE.chartVolume) STATE.chartVolume.destroy();
  const stockUsed=production>0?(sales/production*100):0;
  STATE.chartVolume=new Chart(ctx.getContext("2d"),{
    type:"bar",
    data:{
      labels:["Üretim (birim)","Satış (adet)"],
      datasets:[
        {label:"Üretim",data:[production,null],backgroundColor:"rgba(56,139,253,.75)",borderColor:"#388bfd",borderWidth:1.5,borderRadius:8},
        {label:"Satış",data:[null,sales],backgroundColor:"rgba(163,113,247,.75)",borderColor:"#a371f7",borderWidth:1.5,borderRadius:8},
      ],
    },
    options:{...CHART_DEFAULTS,plugins:{...CHART_DEFAULTS.plugins,
      title:{display:true,text:`📦 Üretim vs Satış — Kapasite Kullanım: %${stockUsed.toFixed(0)}`,color:"#8b949e",font:{size:12,family:"JetBrains Mono"}},
    }},
  });
}


// ═══════════════════════════════════════════════════════════════
// PANEL GÜNCELLEYİCİLER
// ═══════════════════════════════════════════════════════════════

function updateChannelPanel(ch) {
  const $=id=>document.getElementById(id), q=s=>document.querySelector(s);
  const bl=$("channelBetterLabel");    if(bl) bl.textContent=ch.better_channel+" Kanalı";
  const rc=$("channelRecommendation"); if(rc) rc.textContent=ch.recommendation;
  const ba=q(".channel-bar-app");      if(ba) ba.style.width=ch.app_percentage+"%";
  const bw=q(".channel-bar-web");      if(bw) bw.style.width=ch.web_percentage+"%";
  const ap=$("appPct"); if(ap) ap.textContent=`App: ${fmt(ch.app_sales)} (${ch.app_percentage}%)`;
  const wp=$("webPct"); if(wp) wp.textContent=`Web: ${fmt(ch.web_sales)} (${ch.web_percentage}%)`;
  const cd=$("resultChannelCard"); if(cd) cd.style.display="block";
}

function updateAnomalyPanel(an) {
  const $=id=>document.getElementById(id);
  const box=$("anomalyBox"); if(box) box.className=`anomaly-box ${an.status}`;
  const ic=$("anomalyIcon");   if(ic) ic.textContent=an.severity==="kritik"?"🚨":an.is_anomaly?"⚠️":"✅";
  const lb=$("anomalyLabel");  if(lb) lb.textContent=an.label;
  const rs=$("anomalyReason"); if(rs) rs.textContent=an.reason;
  const bd=$("anomalySeverityBadge"); if(bd) bd.innerHTML=severityBadge(an.severity);
  const cd=$("resultAnomalyCard"); if(cd) cd.style.display="block";
}


// ═══════════════════════════════════════════════════════════════
// PERFORMANS SKORU
// ═══════════════════════════════════════════════════════════════

function renderPerfScore(perf,rec,recSt) {
  const C=2*Math.PI*50;
  const pct=Math.min(100,Math.max(0,perf.score))/100;
  const colors={poor:"var(--red)",average:"var(--yellow)",good:"var(--green)"};
  const labels={poor:"Kötü Performans",average:"Orta Performans",good:"İyi Performans"};
  const color=colors[perf.class]||"var(--blue)";
  const arc=document.getElementById("perfArcFill");
  if(arc){arc.setAttribute("stroke",color);arc.setAttribute("stroke-dasharray",`${(C*pct).toFixed(1)} ${C.toFixed(1)}`);}
  const nEl=document.getElementById("perfScoreNumber"); if(nEl){nEl.textContent=perf.score;nEl.style.color=color;}
  const cEl=document.getElementById("perfScoreClass");  if(cEl){cEl.textContent=labels[perf.class]||"—";cEl.style.color=color;}
  document.getElementById("perfScoreSection").style.display="block";
  const rBox=document.getElementById("recommendationBox");
  if(rBox){rBox.className=`recommendation-box ${recSt}`;rBox.style.display="flex";
    const rt=document.getElementById("recommendationText"); if(rt) rt.textContent=rec;}
}


// ═══════════════════════════════════════════════════════════════
// GEÇMİŞ
// ═══════════════════════════════════════════════════════════════

function _saveHistory(res) {
  STATE.history=[{id:Date.now(),ts:new Date().toLocaleString("tr-TR"),
    production:res.production,sales:res.sales,profit:res.profit,
    profitRate:res.kpi?.profit_margin??0,anomaly:res.anomaly.is_anomaly?"Anomali":"Normal",
    severity:res.anomaly.severity??"normal",channel:res.channel.better_channel,
    perfScore:res.performance?.score??0},...STATE.history].slice(0,5);
  renderHistory();
}
function clearHistory(){STATE.history=[];renderHistory();toast("Geçmiş temizlendi.","info");}

function renderHistory() {
  const list=document.getElementById("historyList");
  const empty=document.getElementById("historyEmpty");
  if(!list) return;
  list.innerHTML="";
  if(!STATE.history.length){if(empty) empty.style.display="block";return;}
  if(empty) empty.style.display="none";
  const sevC={kritik:["var(--red-dim)","var(--red)"],orta:["var(--yellow-dim)","var(--yellow)"],normal:["var(--green-dim)","var(--green)"]};
  STATE.history.forEach((r,i)=>{
    const pc=r.profit>=0?"var(--green)":"var(--red)";
    const ac=r.anomaly==="Normal"?"var(--green)":"var(--red)";
    const [sbg,sc]=sevC[r.severity]||sevC.normal;
    const row=document.createElement("div");
    row.className="history-row"; row.style.animationDelay=`${i*0.05}s`;
    row.innerHTML=`
      <div class="history-date">
        <span class="history-index">#${STATE.history.length-i}</span>${r.ts}
        ${r.perfScore?`<span class="perf-mini-badge">${r.perfScore}/100</span>`:""}
      </div>
      <div class="history-metrics">
        <div class="history-metric"><span class="history-mlabel">Üretim</span><span class="history-mval" style="color:var(--blue)">${fmt(r.production)} birim</span></div>
        <div class="history-metric"><span class="history-mlabel">Satış</span><span class="history-mval" style="color:var(--purple)">${fmt(r.sales)} adet</span></div>
        <div class="history-metric"><span class="history-mlabel">Kar</span><span class="history-mval" style="color:${pc}">${fmt(r.profit)} ₺</span></div>
        <div class="history-metric"><span class="history-mlabel">Marj</span><span class="history-mval" style="color:${pc}">${fmtPct(r.profitRate)}</span></div>
        <div class="history-metric"><span class="history-mlabel">Anomali</span><span class="history-mval" style="color:${ac}">${r.anomaly}</span></div>
        <div class="history-metric"><span class="history-mlabel">Şiddet</span><span class="severity-badge" style="background:${sbg};color:${sc};">${r.severity.toUpperCase()}</span></div>
      </div>`;
    list.appendChild(row);
  });
}


// ═══════════════════════════════════════════════════════════════
// PDF EXPORT — profesyonel, tüm bölümler dahil
// ═══════════════════════════════════════════════════════════════

function downloadPDF() {
  const r=STATE.lastResult;
  if(!r){toast("Önce analizi çalıştırın.","error");return;}
  const {jsPDF}=window.jspdf;
  const doc=new jsPDF({orientation:"portrait",unit:"mm",format:"a4"});
  const pw=doc.internal.pageSize.getWidth();
  const mg=20; let y=20;

  // TR karakter dönüşümü
  const tr=s=>String(s)
    .replace(/ğ/g,"g").replace(/Ğ/g,"G").replace(/ü/g,"u").replace(/Ü/g,"U")
    .replace(/ş/g,"s").replace(/Ş/g,"S").replace(/ı/g,"i").replace(/İ/g,"I")
    .replace(/ö/g,"o").replace(/Ö/g,"O").replace(/ç/g,"c").replace(/Ç/g,"C")
    .replace(/⚠|✓|✅|⚡|📦|🚨|⛔|📉|📈|📊|💡|🔎|👥|💰|⚙|🏭|🚀|💡|⚖️|🏪|⭐/g,"");

  const section=(title)=>{
    y+=3;
    doc.setFillColor(22,27,34); doc.rect(mg,y-5,pw-mg*2,8,"F");
    doc.setFontSize(9);doc.setTextColor(139,148,158);doc.setFont("helvetica","bold");
    doc.text(tr(title.toUpperCase()),mg+2,y); y+=9;
    doc.setFont("helvetica","normal");
  };

  const line=(label,value,col)=>{
    doc.setFontSize(10);doc.setTextColor(100,100,100);doc.text(tr(label),mg,y);
    doc.setFontSize(11);doc.setTextColor(...(col||[230,237,243]));
    doc.text(tr(String(value)),mg+78,y); y+=8;
  };

  const para=(text,col)=>{
    doc.setFontSize(10);doc.setTextColor(...(col||[170,180,190]));
    const lines=doc.splitTextToSize(tr(text),pw-mg*2);
    doc.text(lines,mg,y); y+=lines.length*5+4;
  };

  const newPageIfNeeded=()=>{ if(y>265){doc.addPage();y=20;} };

  // ── Başlık sayfası ─────────────────────────────────────
  doc.setFillColor(13,17,23); doc.rect(0,0,pw,38,"F");
  doc.setFontSize(20);doc.setTextColor(56,139,253);doc.setFont("helvetica","bold");
  doc.text("E-Ticaret Analiz Raporu",mg,16);
  doc.setFontSize(9);doc.setTextColor(139,148,158);doc.setFont("helvetica","normal");
  doc.text("Flask + scikit-learn  |  AI Destekli Karar Destek Sistemi v8.0",mg,24);
  doc.text(`Rapor Tarihi: ${new Date().toLocaleString("tr-TR")}`,mg,30);
  y=46;

  // ── Girdi Parametreleri ────────────────────────────────
  section("Girdi Parametreleri");
  line("Calisanlar",      `${STATE.workers} kisi`,            [200,200,200]);
  line("Calisma Saati",   `${STATE.hours} saat/gun`,          [200,200,200]);
  line("Enerji Tuketimi", `${STATE.energy} kWh`,              [200,200,200]);
  line("Urun Fiyati",     `${STATE.price} TL`,                [200,200,200]);
  line("Kanal",           STATE.channel.toUpperCase(),        [200,200,200]);
  line("Aylik Maas",      `${STATE.workerCost} TL/kisi`,      [200,200,200]);
  line("Enerji Birim Mal",`${STATE.energyCost} TL/kWh`,       [200,200,200]);
  newPageIfNeeded();

  // ── Üretim & Satış ─────────────────────────────────────
  section("Uretim & Satis");
  line("Gunluk Uretim", `${fmt(r.production)} birim`, [56,139,253]);
  line("Gunluk Satis",  `${fmt(r.sales)} adet`,       [163,113,247]);
  if(r.production>0) line("Kapasite Kullanimi", `%${(r.sales/r.production*100).toFixed(0)}`,[200,200,200]);
  newPageIfNeeded();

  // ── Maliyet Kırılımı ───────────────────────────────────
  section("Maliyet Kirilimi (Gunluk)");
  line("Iscilik (Aylik/30)", fmtTL(r.cost_breakdown?.labor_cost),        [200,150,80]);
  line("Enerji",             fmtTL(r.cost_breakdown?.energy_cost_total), [200,150,80]);
  line("COGS (Urun Mal.)",   fmtTL(r.cost_breakdown?.product_cost),      [200,150,80]);
  line("TOPLAM MALIYET",     fmtTL(r.total_cost),                        [248,81,73]);
  newPageIfNeeded();

  // ── Kar Analizi ────────────────────────────────────────
  section("Kar Analizi");
  const pc=r.profit>=0?[63,185,80]:[248,81,73];
  line("Toplam Gelir",   fmtTL(r.revenue),              [63,185,80]);
  line("Net Kar",        fmtTL(r.profit),                pc);
  line("Kar Marji",      fmtPct(r.kpi?.profit_margin??0),pc);
  line("Maliyet Orani",  fmtPct(r.kpi?.cost_rate??0),   [200,150,80]);
  newPageIfNeeded();

  // ── Performans ─────────────────────────────────────────
  section("Performans Skoru");
  line("Genel Skor",  `${r.performance?.score??"-"} / 100`, [56,139,253]);
  line("Siniflandirma",r.performance?.class??"—",           [56,139,253]);
  line("Verimlilik",  (r.kpi?.efficiency??0).toFixed(3),    [56,139,253]);
  newPageIfNeeded();

  // ── Anomali ────────────────────────────────────────────
  section("Anomali Tespiti");
  const ac=r.anomaly.is_anomaly?[248,81,73]:[63,185,80];
  line("Sonuc",   r.anomaly.label,                  ac);
  line("Siddet",  r.anomaly.severity.toUpperCase(), ac);
  para(r.anomaly.reason,[170,180,190]);
  newPageIfNeeded();

  // ── Kanal ──────────────────────────────────────────────
  section("Kanal Analizi");
  line("Ustun Kanal", r.channel.better_channel,         [56,139,253]);
  line("App Satisi",  `${fmtTL(r.channel.app_sales)} (%${r.channel.app_percentage})`,[56,139,253]);
  line("Web Satisi",  `${fmtTL(r.channel.web_sales)} (%${r.channel.web_percentage})`,[63,185,80]);
  para(r.channel.recommendation,[170,180,190]);
  newPageIfNeeded();

  // ── AI Yorum ───────────────────────────────────────────
  section("AI Yorumu (DURUM / NEDEN / ONERI)");
  para((r.commentary||"").replace(/\n/g," | "),[200,210,220]);
  newPageIfNeeded();

  // ── Footer ─────────────────────────────────────────────
  y+=4;
  doc.setDrawColor(33,38,45); doc.line(mg,y,pw-mg,y); y+=6;
  doc.setFontSize(8);doc.setTextColor(72,79,88);
  doc.text("Bu rapor otomatik uretilmistir. Gercek is kararlari icin uzman gorusu aliniz.",mg,y);
  doc.text(`v8.0 — ${new Date().toLocaleDateString("tr-TR")}`,pw-mg,y,{align:"right"});

  doc.save(`analiz-raporu-${Date.now()}.pdf`);
  toast("✓ PDF rapor indirildi.","success");
}


// ═══════════════════════════════════════════════════════════════
// SAYFA YÜKLENME
// ═══════════════════════════════════════════════════════════════

document.addEventListener("DOMContentLoaded", async ()=>{
  // Buton orijinal metinleri
  document.querySelectorAll(".btn-text").forEach(el=>{el.dataset.orig=el.textContent;});
  initThemeToggle();
  initTooltips();
  initSectionReveal();
  setDashboardEmpty(true);

  document.getElementById("btnNextScenario")?.addEventListener("click",nextDemoScenario);

  _updateModeUI(false);
  renderHistory();

  // productionInput → STATE
  document.getElementById("productionInput")?.addEventListener("input",e=>{
    STATE.production=parseFloat(e.target.value)||0;
  });

  // Girdi değişiklikleri → STATE
  ["workers","hours","energy","price","workerCost","energyCost","appSales","webSales"]
    .forEach(id=>document.getElementById(id)?.addEventListener("change",syncState));

  // /api/meta — placeholder ve senaryolar
  try {
    const meta=await fetch("/api/meta").then(r=>r.json());
    if(meta.input_hints){
      const idMap={workers:"workers",hours:"hours",energy:"energy",price:"price",
        worker_cost:"workerCost",energy_cost:"energyCost",app_sales:"appSales",web_sales:"webSales"};
      Object.entries(meta.input_hints).forEach(([k,hint])=>{
        const el=document.getElementById(idMap[k]||k); if(el) el.placeholder=hint;
      });
    }
    if(meta.demo_scenarios) DEMO_SCENARIOS=meta.demo_scenarios.map(s=>({
      ...s,workerCost:s.worker_cost,energyCost:s.energy_cost,
      appSales:s.app_sales,webSales:s.web_sales
    }));
  } catch(_) {
    // Statik fallback placeholder
    const ph={workers:"Örn: 15",hours:"Örn: 8",energy:"Örn: 500",price:"Örn: 130 ₺",
      workerCost:"Örn: 4000 ₺/ay",energyCost:"Örn: 2.5 ₺/kWh",appSales:"Örn: 80",webSales:"Örn: 120"};
    Object.entries(ph).forEach(([id,p])=>{const el=document.getElementById(id);if(el) el.placeholder=p;});
  }

  console.log("✓ E-Ticaret Analiz Sistemi v8.0 — Impact + SmartRecommender + DualChart Aktif");
});
