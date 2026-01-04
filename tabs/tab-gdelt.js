    loadMappingComparisonData();
// Per-tab initializer for GDELT partial
(function(){
    function safeAddListener(el, ev, fn) {
        if (!el) return;
        el.addEventListener(ev, fn);
    }

    window.initTabGdelt = function initTabGdelt() {
        try {
            // Remove problematic inline handlers and attach safe listeners
            const byId = id => document.getElementById(id);

            const bindButton = (id, fnName) => {
                const el = byId(id);
                if (!el) return;
                // remove inline onclick to avoid ReferenceError when fn missing
                if (el.getAttribute('onclick')) el.removeAttribute('onclick');
                if (typeof window[fnName] === 'function') {
                    el.addEventListener('click', window[fnName]);
                    el.disabled = false;
                } else {
                    el.disabled = true;
                    el.title = `${fnName} not available yet`;
                }
            };

            bindButton('load-gdelt-data-btn', 'loadGDELTData');
            bindButton('analyze-themes-btn', 'analyzeThemes');
            bindButton('generate-matrix-btn', 'generateCountryThemeMatrix');
            bindButton('export-selected-btn', 'exportSelectedCategoriesAnalysis');

            // Replace inline onchange handlers similarly
            const bindChange = (id, fnName) => {
                const el = byId(id);
                if (!el) return;
                if (el.getAttribute('onchange')) el.removeAttribute('onchange');
                if (typeof window[fnName] === 'function') {
                    el.addEventListener('change', window[fnName]);
                }
            };

            bindChange('gdelt-country-select', 'updateTotalDocsTable');
            bindChange('gdelt-quality-filter', 'updateMonthlyQualityTable');
            bindChange('monthly-detail-country', 'updateMonthlyDetailTable');
            bindChange('monthly-detail-theme', 'updateMonthlyDetailTable');
            bindChange('monthly-detail-year', 'updateMonthlyDetailTable');

            // Initialize pipeline/export helpers if available
            if (typeof initPipelineButtons === 'function') initPipelineButtons();
            if (typeof initExportFilters === 'function') initExportFilters();

            // Wire import file input
            const importInput = byId('import-file-input');
            if (importInput) {
                if (importInput.getAttribute('onchange')) importInput.removeAttribute('onchange');
                if (typeof handleGDELTImportFile === 'function') {
                    importInput.addEventListener('change', handleGDELTImportFile);
                }
            }

            if (typeof updateExportInfo === 'function') updateExportInfo();
            if (typeof autoLoadLastAnalysis === 'function') setTimeout(autoLoadLastAnalysis, 50);

        } catch (err) {
            console.warn('initTabGdelt error', err);
        }
    };

})();

// --- Minimal GDELT data/load/table helpers moved from previous inline script ---
// Ensure globals exist
window.gdeltTotalDocs = window.gdeltTotalDocs || [];
window.gdeltMonthlyQuality = window.gdeltMonthlyQuality || [];
window.gdeltMonthlyDetail = window.gdeltMonthlyDetail || [];
window.gdeltTrendAnalysis = window.gdeltTrendAnalysis || [];
window.gdeltDataLoaded = window.gdeltDataLoaded || false;
window.selectedIPTCCategories = window.selectedIPTCCategories || new Set();
window.categoryCountryChart = window.categoryCountryChart || null;
window.monthlyDetailSort = window.monthlyDetailSort || { field: 'ym', dir: 'asc' };
window.IPTC_COLORS = window.IPTC_COLORS || {
    'economy, business and finance': '#FF9800',
    'conflict, war and peace': '#F44336',
    'politics and government': '#2196F3',
    'health': '#4CAF50',
    'education': '#9C27B0',
    'disaster, accident and emergency incident': '#E91E63',
    'environment': '#8BC34A',
    'society': '#607D8B',
    'crime, law and justice': '#3F51B5',
    'science and technology': '#009688',
    'labour': '#FFC107',
    'lifestyle and leisure': '#00BCD4',
    'religion': '#795548',
    'human interest': '#FF5722',
    'sport': '#CDDC39',
    'arts, culture, entertainment and media': '#673AB7',
    'weather': '#03A9F4',
    'unmapped': '#9E9E9E'
};
window.mappingResults = window.mappingResults || { v1: null, v2: null, v3: null };

function hexToRgba(hex, alpha=0.25) {
    if (!hex) return `rgba(33,150,243,${alpha})`;
    let normalized = hex.replace('#','');
    if (normalized.length === 3) normalized = normalized.split('').map(c=>c+c).join('');
    const num = parseInt(normalized,16);
    if (Number.isNaN(num)) return `rgba(33,150,243,${alpha})`;
    const r = (num >> 16) & 255;
    const g = (num >> 8) & 255;
    const b = num & 255;
    return `rgba(${r},${g},${b},${alpha})`;
}

function colorForCategory(cat){
    return window.IPTC_COLORS[cat] || '#667eea';
}

// Simple CSV parser (comma-separated, no quoted commas handling)
function parseCSV(text) {
    const lines = text.trim().split('\n').map(l => l.trim()).filter(Boolean);
    if (lines.length === 0) return [];
    const headers = lines[0].split(',').map(h => h.trim());
    const out = [];
    for (let i = 1; i < lines.length; i++) {
        const cols = lines[i].split(',').map(c => c.trim());
        const obj = {};
        for (let j = 0; j < headers.length; j++) {
            const key = headers[j] || `col${j}`;
            let val = cols[j] === undefined ? '' : cols[j];
            // try numbers
            if (val !== '' && !isNaN(val)) val = Number(val);
            obj[key] = val;
        }
        out.push(obj);
    }
    return out;
}

// Show status inside GDELT partial
function showGDELTStatus(msg, type='info') {
    const el = document.getElementById('gdelt-status') || document.getElementById('config-status');
    if (!el) return;
    el.textContent = msg;
    el.className = `status-message ${type}`;
}

// Load CSV files used by GDELT tab
window.loadGDELTData = async function loadGDELTData() {
    const btn = document.getElementById('load-gdelt-data-btn');
    if (btn) { btn.disabled = true; btn.innerHTML = '<span class="loading-spinner"></span>Yükleniyor...'; }
    showGDELTStatus('📥 CSV dosyaları yükleniyor...', 'info');
    try {
        const resp1 = await fetch('/data/gdelt_top15_themes_by_country_2022_2024.csv');
        if (!resp1.ok) throw new Error('Top 15 tema CSV bulunamadı');
        const text1 = await resp1.text();
        window.gdeltTotalDocs = parseCSV(text1);

        const resp2 = await fetch('/data/gdelt_monthly_quality_metrics.csv');
        if (!resp2.ok) throw new Error('Aylık kalite CSV bulunamadı');
        const text2 = await resp2.text();
        window.gdeltMonthlyQuality = parseCSV(text2);

        const resp3 = await fetch('/data/gdelt_monthly_docs_per_theme_country_2022_2024.csv');
        if (!resp3.ok) throw new Error('Aylık detay CSV bulunamadı');
        const text3 = await resp3.text();
        window.gdeltMonthlyDetail = parseCSV(text3);

        try {
            const resp4 = await fetch('/data/gdelt_theme_trend_analysis_2022_2024.csv');
            if (resp4.ok) {
                const text4 = await resp4.text();
                window.gdeltTrendAnalysis = parseCSV(text4);
                // annotate with current mapping if available
                if (window.iptcMappingLookup) {
                    window.gdeltTrendAnalysis.forEach(r => {
                        const code = r.theme_code || r.theme;
                        if (code && window.iptcMappingLookup[code]) r.iptc_category = window.iptcMappingLookup[code];
                    });
                }
            }
        } catch (e) {
            console.log('Trend analiz dosyası yüklenemedi (opsiyonel)');
        }

        // mirror to more generic names used elsewhere
        window.totalDocsData = window.gdeltTotalDocs;
        window.monthlyQualityData = window.gdeltMonthlyQuality;
        window.monthlyDetailData = window.gdeltMonthlyDetail;

        window.gdeltDataLoaded = true;

        updateTotalDocsTable();
        updateMonthlyQualityTable();
        updateMonthlyDetailTable();
    updateTrendAnalysisTable();

        const analyzeBtn = document.getElementById('analyze-themes-btn');
        const matrixBtn = document.getElementById('generate-matrix-btn');
        if (analyzeBtn) analyzeBtn.disabled = false;
        if (matrixBtn) matrixBtn.disabled = false;

        showGDELTStatus(`✅ Veriler yüklendi! Top15: ${window.gdeltTotalDocs.length}, Quality: ${window.gdeltMonthlyQuality.length}, Detail: ${window.gdeltMonthlyDetail.length}`, 'success');

    } catch (err) {
        console.error('GDELT load error', err);
        showGDELTStatus(`❌ Yükleme hatası: ${err.message}`, 'error');
    } finally {
        if (btn) { btn.disabled = false; btn.innerHTML = '📥 CSV Verilerini Yükle'; }
    }
};

async function loadMappingComparisonData() {
    const sources = {
        v1: '/results/gdelt_iptc_mapping_v1_gkg.json',
        v2: '/results/gdelt_iptc_mapping_v2_gkg.json',
        v3: '/results/gdelt_iptc_mapping_v3_combined.json'
    };
    await Promise.all(Object.entries(sources).map(async ([version, url]) => {
        try {
            const resp = await fetch(url);
            if (!resp.ok) throw new Error(`${url} yüklenemedi (${resp.status})`);
            const data = await resp.json();
            const counts = {};
            const categories = data.iptc_categories || {};
            Object.entries(categories).forEach(([cat, info]) => {
                counts[cat] = info.theme_count || (info.themes ? info.themes.length : 0);
            });
            const fallback = (data.summary && data.summary.themes_per_iptc) || {};
            Object.entries(fallback).forEach(([cat, value]) => {
                if (!counts[cat] && typeof value === 'number') counts[cat] = value;
            });
            window.mappingResults[version] = { counts, metadata: data.metadata || {} };
            updateMappingComparisonTable();
        } catch (err) {
            console.warn('mapping comparison load failed', version, err);
        }
    }));
}

// Update table functions (basic rendering)
function updateTotalDocsTable() {
    const tbody = document.getElementById('total-docs-tbody');
    if (!tbody) return;
    const data = window.gdeltTotalDocs || [];
    if (!data || data.length === 0) {
        tbody.innerHTML = '<tr><td colspan="4" style="text-align:center;padding:30px;color:#999;">CSV yüklenmedi. Yukarıdaki butona tıklayın.</td></tr>';
        return;
    }
    const rows = data.map(r => {
        const country = r.country || '';
        const theme = r.theme_code || r.theme || '';
        const cat = r.iptc_category || getIPTCCategory(theme) || '';
        const color = cat ? colorForCategory(cat) : '#9e9e9e';
        const total = r.total_docs || r.n_docs || 0;
        const catBadge = cat ? `<span class="theme-pill tiny" style="background:${color}14;border-color:${color};color:${color}">${cat}</span>` : '';
        return `<tr><td style="padding:8px">${country}</td><td style="padding:8px">${theme}</td><td style="padding:8px;text-align:right">${total}</td><td style="padding:8px">${catBadge}</td></tr>`;
    }).join('');
    tbody.innerHTML = rows;
}

function updateMonthlyQualityTable() {
    const tbody = document.getElementById('monthly-quality-tbody');
    if (!tbody) return;
    const data = window.gdeltMonthlyQuality || [];
    if (!data || data.length === 0) {
        tbody.innerHTML = '<tr><td colspan="6" style="text-align:center;padding:30px;color:#999;">CSV yüklenmedi. Yukarıdaki butona tıklayın.</td></tr>';
        return;
    }
    const filter = (document.getElementById('gdelt-quality-filter') || {}).value || 'all';
    const filtered = data.filter(r => {
        const months_total = r.months_total || 0;
        const months_ok = r.months_ok || 0;
        const ratio = months_total > 0 ? (months_ok / months_total) : 0;
        if (filter === 'good') return ratio >= 0.7;
        if (filter === 'marginal') return ratio >= 0.4 && ratio < 0.7;
        if (filter === 'poor') return ratio < 0.4;
        return true;
    });
    const rows = filtered.map(r => {
        const country = r.country || '';
        const theme = r.theme_code || r.theme || '';
        const cat = r.iptc_category || getIPTCCategory(theme) || '';
        const color = cat ? colorForCategory(cat) : '#9e9e9e';
        const months_total = r.months_total || 0;
        const months_ok = r.months_ok || 0;
        const minDocs = r.min_docs || r.median_docs || 0;
        const median = r.median_docs || 0;
        return `<tr>
            <td style="padding:8px">${country}</td>
            <td style="padding:8px">${theme}${cat ? `<div style='font-size:10px;color:${color};margin-top:2px'>${cat}</div>`:''}</td>
            <td style="padding:8px;text-align:right">${months_total}</td>
            <td style="padding:8px;text-align:right">${months_ok}</td>
            <td style="padding:8px;text-align:right">${minDocs}</td>
            <td style="padding:8px;text-align:right">${median}</td>
        </tr>`;
    }).join('');
    tbody.innerHTML = rows;
}

function updateMonthlyDetailTable() {
    const tbody = document.getElementById('monthly-detail-tbody');
    if (!tbody) return;
    const data = window.gdeltMonthlyDetail || [];
    if (!data || data.length === 0) {
        tbody.innerHTML = '<tr><td colspan="5" style="text-align:center;padding:30px;color:#999;">CSV yüklenmedi. "CSV Verilerini Yükle" butonuna tıklayın.</td></tr>';
        updateMonthlyDetailStats([]);
        return;
    }
    const selCountry = (document.getElementById('monthly-detail-country') || {}).value || 'ALL';
    const selTheme = (document.getElementById('monthly-detail-theme') || {}).value || 'ALL';
    const selYear = (document.getElementById('monthly-detail-year') || {}).value || 'ALL';
    const minDocs = Number((document.getElementById('monthly-detail-min-docs') || {}).value || 0);

    // populate theme dropdown once data is available
    const themeSelect = document.getElementById('monthly-detail-theme');
    if (themeSelect && themeSelect.options.length <= 1) {
        const themes = [...new Set(data.map(r => r.theme_code || r.theme).filter(Boolean))].sort();
        const opts = themes.map(t => `<option value="${t}">${t}</option>`).join('');
        themeSelect.insertAdjacentHTML('beforeend', opts);
    }

    let filtered = data.filter(r => {
        const year = (r.ym || '').toString().slice(0,4) || r.year;
        if (selCountry !== 'ALL' && r.country !== selCountry) return false;
        if (selTheme !== 'ALL' && (r.theme_code || r.theme) !== selTheme) return false;
        if (selYear !== 'ALL' && String(year) !== selYear) return false;
        if (Number(r.n_docs || 0) < minDocs) return false;
        return true;
    });

    // sort
    const { field, dir } = monthlyDetailSort;
    filtered = filtered.sort((a,b)=>{
        const av = a[field] || 0; const bv = b[field] || 0;
        if (av < bv) return dir === 'asc' ? -1 : 1;
        if (av > bv) return dir === 'asc' ? 1 : -1;
        return 0;
    });

    const rows = filtered.slice(0,400).map(r => {
        const ymLabel = r.ym || (r.year && r.month ? `${r.year}-${String(r.month).padStart(2,'0')}` : '');
        const cat = r.iptc_category || getIPTCCategory(r.theme_code) || '';
        const color = cat ? colorForCategory(cat) : '#9e9e9e';
        const catBadge = cat ? `<span class="theme-pill tiny" style="background:${color}12;border-color:${color};color:${color}">${cat}</span>` : '';
        return `<tr><td style="padding:8px">${ymLabel}</td><td style="padding:8px">${r.country||''}</td><td style="padding:8px">${r.theme_code||''}</td><td style="padding:8px">${catBadge}</td><td style="padding:8px;text-align:right">${r.n_docs||0}</td></tr>`;
    }).join('');
    tbody.innerHTML = rows || '<tr><td colspan="5" style="text-align:center;padding:20px;color:#999;">Kriterlere uyan kayıt yok</td></tr>';

    const countEl = document.getElementById('monthly-detail-count');
    if (countEl) countEl.textContent = `${filtered.length} kayıt gösteriliyor`;
    updateMonthlyDetailStats(filtered);
}

// Trend analysis table rendering
let trendSortState = { field: 'theme_code', dir: 'asc' };
function updateTrendAnalysisTable(){
    const tbody = document.getElementById('trend-analysis-tbody');
    const countEl = document.getElementById('trend-count');
    if (!tbody) return;
    const data = window.gdeltTrendAnalysis || [];
    if (!data || data.length === 0){
        tbody.innerHTML = '<tr><td colspan="6" style="text-align:center;padding:20px;color:#999;">Trend CSV yüklenmedi</td></tr>';
        if (countEl) countEl.textContent = '';
        updateTrendSummary([]);
        return;
    }
    const catFilter = (document.getElementById('trend-category-filter')||{}).value || 'ALL';
    const minSlope = parseFloat((document.getElementById('trend-min-slope')||{}).value || '-1');
    const maxSlope = parseFloat((document.getElementById('trend-max-slope')||{}).value || '1');
    const filtered = data.filter(r => {
        const slope = Number(r.trend_slope || r.slope || 0);
        const cat = r.trend_category || r.category || '';
        if (catFilter !== 'ALL' && cat !== catFilter) return false;
        if (slope < minSlope || slope > maxSlope) return false;
        return true;
    });
    // sort
    const { field, dir } = trendSortState;
    filtered.sort((a,b)=>{
        const av = a[field] || 0; const bv = b[field] || 0;
        if (av < bv) return dir === 'asc' ? -1 : 1;
        if (av > bv) return dir === 'asc' ? 1 : -1;
        return 0;
    });
        const rows = filtered.slice(0,400).map(r=>{
                const slope = Number(r.trend_slope || r.slope || 0);
                const avg = Number(r.avg_docs || r.avg || 0);
                    const cat = r.trend_category || '';
                    const iptc = r.iptc_category || getIPTCCategory(r.theme_code) || '';
                    const iptcColor = iptc ? colorForCategory(iptc) : '#9e9e9e';
                const totalDocs = r.total_docs || (avg && r.months_count ? avg * Number(r.months_count) : '');
        return `<tr>
          <td style="padding:8px">${r.theme_code||''}</td>
                <td style="padding:8px">${iptc ? `<span class="theme-pill tiny" style="background:${iptcColor}14;border-color:${iptcColor};color:${iptcColor}">${iptc}</span>` : ''}</td>
          <td style="padding:8px;text-align:center">${slope.toFixed(2)}</td>
          <td style="padding:8px;text-align:center">${cat}</td>
          <td style="padding:8px;text-align:right">${avg.toFixed(1)}</td>
                    <td style="padding:8px;text-align:right">${totalDocs || r.months_count || ''}</td>
        </tr>`;
    }).join('');
    tbody.innerHTML = rows || '<tr><td colspan="6" style="text-align:center;padding:20px;color:#999;">Kriterlere uyan kayıt yok</td></tr>';
        if (countEl) countEl.textContent = `${filtered.length} kayıt`;
        updateTrendSummary(filtered);
}

function sortTrendTable(field){
    if (trendSortState.field === field){
        trendSortState.dir = trendSortState.dir === 'asc' ? 'desc' : 'asc';
    } else {
        trendSortState.field = field; trendSortState.dir = 'asc';
    }
    updateTrendAnalysisTable();
}

// expose helpers used by other parts
window.parseCSV = parseCSV;
window.updateTotalDocsTable = updateTotalDocsTable;
window.updateMonthlyQualityTable = updateMonthlyQualityTable;
window.updateMonthlyDetailTable = updateMonthlyDetailTable;
window.showGDELTStatus = showGDELTStatus;

// Minimal IPTC category definitions and helpers (extracted from legacy inline script)
const IPTC_CATEGORIES = {
    'arts, culture, entertainment and media': ['MEDIA', 'CULTURE'],
    'conflict, war and peace': ['CRISISLEX', 'ARMEDCONFLICT', 'KILL', 'TERROR', 'SECURITY', 'MILITARY'],
    'crime, law and justice': ['CRIME', 'JUSTICE', 'LAW'],
    'disaster, accident and emergency incident': ['MANMADE', 'DISASTER', 'NATURAL'],
    'economy, business and finance': ['ECON', 'EPU', 'EPU_ECONOMY', 'TAX', 'AGRICULTURE'],
    'education': ['EDUCATION'],
    'environment': ['ENV', 'CLIMATE'],
    'health': ['HEALTH', 'MEDICAL', 'GENERAL_HEALTH'],
    'human interest': ['AFFECT', 'GENERAL'],
    'labour': ['LABOR', 'EMPLOYMENT', 'WORKFORCE'],
    'lifestyle and leisure': ['TOURISM', 'TRAVEL', 'LEISURE'],
    'politics and government': ['LEADER', 'ELECTION', 'DEMOCRACY', 'LEGISLATION', 'GOVERNMENT', 'GENERAL_GOVERNMENT', 'GOV', 'UNGP'],
    'religion': ['RELIGION'],
    'science and technology': ['SCIENCE', 'TECHNOLOGY', 'RESEARCH'],
    'society': ['SOC', 'SOCIAL', 'COMMUNITY'],
    'sport': ['SPORT', 'SPORTS'],
    'weather': ['WEATHER', 'CLIMATE_WEATHER']
};

const METADATA_PREFIXES = ['TAX_', 'WB_', 'USPEC_', 'UNGP_', 'SOC_', 'SLFID_'];

function getIPTCCategory(themeCode) {
    if (!themeCode) return null;
    // Prefer dynamic mapping produced by the clustering/mapping modules
    try{
        if (window.iptcMappingLookup && window.iptcMappingLookup[themeCode]) return window.iptcMappingLookup[themeCode];
        // check versioned lookup if activeMappingInfo exists
        if (window.activeMappingInfo && window.iptcMappingLookupByVersion) {
            const v = window.activeMappingInfo.version;
            const s = window.activeMappingInfo.source;
            if (window.iptcMappingLookupByVersion[v] && window.iptcMappingLookupByVersion[v][s] && window.iptcMappingLookupByVersion[v][s][themeCode]) {
                return window.iptcMappingLookupByVersion[v][s][themeCode];
            }
        }
    }catch(e){ /* ignore and fallback */ }

    // Fallback to static IPTC_CATEGORIES mapping
    for (const [label, themes] of Object.entries(IPTC_CATEGORIES)) {
        if (themes.includes(themeCode)) return label;
    }
    return null;
}

function getThemeType(themeCode) {
    if (!themeCode) return 'other';
    for (const p of METADATA_PREFIXES) if (themeCode.startsWith(p)) return 'metadata';
    if (getIPTCCategory(themeCode)) return 'iptc';
    return 'other';
}

// Analyze themes and build lightweight IPTC summaries used by the GDELT UI
window.analyzeThemes = function analyzeThemes() {
    if (!window.gdeltDataLoaded) {
        showGDELTStatus('⚠️ CSV yüklenmedi. Lütfen önce verileri yükleyin.', 'error');
        return;
    }

    const themeCountryDocs = {};
    const addDoc = (country, theme, docs) => {
        if (!country || !theme) return;
        const key = `${country}||${theme}`;
        if (!themeCountryDocs[key]) themeCountryDocs[key] = 0;
        themeCountryDocs[key] += Number(docs || 0);
    };

    (window.gdeltTotalDocs || []).forEach(r => addDoc(r.country, r.theme_code || r.theme, r.total_docs || r.n_docs));
    (window.gdeltMonthlyDetail || []).forEach(r => addDoc(r.country, r.theme_code || r.theme, r.n_docs));
    // ensure quality-only themes are represented even if doc totals missing
    (window.gdeltMonthlyQuality || []).forEach(r => addDoc(r.country, r.theme_code || r.theme, 0));

    const themes = Object.entries(themeCountryDocs).map(([key, docs]) => {
        const [country, theme] = key.split('||');
        return { theme_code: theme, country, total_docs: docs };
    });

    const themesByIPTC = {};
    const allMapped = {};

    themes.forEach(t => {
        const cat = getIPTCCategory(t.theme_code) || 'unmapped';
        themesByIPTC[cat] = themesByIPTC[cat] || { themes: [], totalDocs: 0 };
        themesByIPTC[cat].themes.push(t);
        themesByIPTC[cat].totalDocs += t.total_docs || 0;

        const key = t.theme_code;
        if (!allMapped[key]) allMapped[key] = { theme_code: key, totalDocs: 0, countries: new Set() };
        allMapped[key].totalDocs += t.total_docs || 0;
        if (t.country) allMapped[key].countries.add(t.country);
    });

    const allCategories = Object.keys(window.IPTC_COLORS).filter(c => c !== 'unmapped');
    allCategories.forEach(cat => {
        if (!themesByIPTC[cat]) themesByIPTC[cat] = { themes: [], totalDocs: 0 };
    });

    const allMappedThemes = Object.values(allMapped).map(m => ({
        theme_code: m.theme_code,
        totalDocs: m.totalDocs,
        countryCount: m.countries.size
    })).sort((a,b) => b.totalDocs - a.totalDocs);

    window.themesByIPTC = themesByIPTC;
    window.allMappedThemes = allMappedThemes;
    window.themeCountryDocs = themeCountryDocs;

    const metrics = buildCategoryMetrics(allCategories);
    renderIPTCSummaryCards(metrics.map(m => [m.cat, themesByIPTC[m.cat]]));
    renderIPTCSelectionPanel(allCategories);
    generateCountryThemeMatrix();

    showGDELTStatus(`✅ Analiz tamamlandı: ${allMappedThemes.length} tema işleme alındı (17 kategori kapsandı).`, 'success');
};

// Render simple IPTC summary cards
function renderIPTCSummaryCards(sortedIPTCCategories) {
    const grid = document.getElementById('iptc-summary-cards-grid');
    if (!grid) return;
    const html = sortedIPTCCategories.map(([cat, info]) => {
        const count = (info && info.themes) ? info.themes.length : 0;
        const docs = (info && info.totalDocs) ? info.totalDocs : 0;
        const themeTags = (info && info.themes ? info.themes : []).slice(0, 12).map(t=>{
            const code = t.theme_code || t.theme || t;
            const color = colorForCategory(cat);
            return `<span class="theme-pill" style="border-color:${color};color:${color};font-size:9px;padding:2px 6px;">${code}</span>`;
        }).join('');
        return `<div class="stat-card minimal" style="border:1px solid ${colorForCategory(cat)}33;box-shadow:0 8px 20px -12px ${colorForCategory(cat)};"><div class="stat-card__header" style="color:${colorForCategory(cat)};font-size:15px;font-weight:700;margin-bottom:6px;">${cat}</div>
            <div class="stat-card__body" style="font-size:8px;line-height:1.5">${themeTags || '<span class="theme-pill muted" style="font-size:9px;">Eşleşme yok</span>'}</div>
            <div class="stat-card__footer" style="font-size:8px;text-transform:uppercase;letter-spacing:0.6px;color:#555;margin-top:10px;">${count} tema • ${docs.toLocaleString()} dok.</div>
        </div>`;
    }).join('');
    grid.innerHTML = html;
}

function buildCategoryMetrics(allCategories) {
    const metrics = [];
    const source = window.themesByIPTC || {};
    const cats = (allCategories && allCategories.length ? allCategories : Object.keys(window.IPTC_COLORS || {})).filter(c => c !== 'unmapped');
    cats.forEach(cat => {
        const info = source[cat] || { themes: [], totalDocs: 0 };
        const themes = info.themes || [];
        const totalDocs = themes.reduce((s,t)=>s+Number(t.total_docs||t.totalDocs||0),0);
        metrics.push({ cat, themeCount: themes.length, totalDocs });
    });
    metrics.sort((a,b)=> b.totalDocs - a.totalDocs || b.themeCount - a.themeCount);
    window.categoryRecommendationOrder = metrics.map(m=>m.cat);
    return metrics;
}

// Render selection panel with recommendations and matched themes
function renderIPTCSelectionPanel(categories) {
    const listEl = document.getElementById('iptc-selection-list');
    if (!listEl) return;
    const metrics = buildCategoryMetrics(categories);
    const orderedCats = metrics.map(m=>m.cat);
    const recSet = new Set(metrics.slice(0,5).map(m=>m.cat));
    const altSet = new Set(metrics.slice(5,7).map(m=>m.cat));
    const rows = orderedCats.map(cat => {
        const info = (window.themesByIPTC && window.themesByIPTC[cat]) || { themes: [] };
        const themeCount = info.themes.length;
        const docs = info.themes.reduce((s,t)=>s+Number(t.total_docs||t.totalDocs||0),0);
        const badge = recSet.has(cat) ? '<span class="pill pill-rec">Önerilen</span>' : (altSet.has(cat) ? '<span class="pill pill-alt">Alternatif</span>' : '');
        const color = colorForCategory(cat);
        const softBg = `${color}12`;
        const themes = info.themes.slice(0,8).map(t=>`<span class="theme-pill small" style="border-color:${color};color:${color};font-size:9px;padding:2px 6px;">${t.theme_code||t.theme||''}</span>`).join('') || '<span class="theme-pill muted">Tema yok</span>';
        return `<label class="iptc-select-row" title="${themeCount} tema, ${docs.toLocaleString()} doküman" style="background:${softBg};border:1px solid ${color}26;">
            <input type="checkbox" class="iptc-cat-checkbox" value="${cat}" ${themeCount ? '' : ''}>
            <div class="iptc-select-content" style="border-left:4px solid ${color};">
                <div class="iptc-select-title" style="font-size:14px;font-weight:700;color:${color};">${cat}${badge ? ' '+badge : ''}</div>
                <div class="iptc-select-sub" style="font-size:11px;color:#334155;">${themeCount} tema • ${docs.toLocaleString()} dok.</div>
                <div class="iptc-select-themes" style="font-size:9px;">${themes}</div>
            </div>
        </label>`;
    }).join('');
    listEl.innerHTML = rows || '<div style="text-align:center;padding:12px;color:#999;">Kategori bulunamadı</div>';

    listEl.querySelectorAll('.iptc-cat-checkbox').forEach(cb => cb.addEventListener('change', updateCategorySelection));
}

// Generate a basic country-category matrix placeholder
function generateCountryThemeMatrix() {
    const container = document.getElementById('country-theme-matrix');
    if (!container) return;
    if (!window.gdeltDataLoaded) {
        showGDELTStatus('❌ Önce CSV verilerini yükleyin!', 'error');
        return;
    }
    if (!window.themesByIPTC || Object.keys(window.themesByIPTC).length === 0) {
        if (typeof analyzeThemes === 'function') analyzeThemes();
    }

    const countries = [...new Set((window.gdeltTotalDocs || []).map(r => r.country).filter(Boolean))];
    const metrics = buildCategoryMetrics(Object.keys(window.IPTC_COLORS));
    const categories = metrics.map(m=>m.cat);

    // Build lookups
    const qualityByCT = {};
    (window.gdeltMonthlyQuality||[]).forEach(r => {
        const key = `${r.country}||${r.theme_code}`;
        qualityByCT[key] = { months_ok: Number(r.months_ok||0), months_total: Number(r.months_total||0) };
    });
    const docsByCT = {};
    const docSource = window.themeCountryDocs || {};
    Object.entries(docSource).forEach(([key, val]) => { docsByCT[key] = Number(val||0); });
    if (!Object.keys(docsByCT).length) {
        (window.gdeltTotalDocs||[]).forEach(r => {
            const key = `${r.country}||${r.theme_code}`;
            docsByCT[key] = Number(r.total_docs||0);
        });
    }

    const matrix = {};
    categories.forEach(cat => {
        const themes = (window.themesByIPTC && window.themesByIPTC[cat] && window.themesByIPTC[cat].themes) || [];
        countries.forEach(country => {
            let monthsOk=0, monthsTotal=0, docs=0, themesWithData=0;
            themes.forEach(t => {
                const key = `${country}||${t.theme_code||t.theme}`;
                const q = qualityByCT[key];
                if (q) { monthsOk += q.months_ok; monthsTotal += q.months_total; themesWithData += 1; }
                if (docsByCT[key]) docs += docsByCT[key];
            });
            const ratio = monthsTotal>0 ? monthsOk/monthsTotal : 0;
            const base = colorForCategory(cat);
            let decision='✖', bg=`${base}0D`, color='#c62828';
            if (!themes.length) { decision='-'; bg='#f5f5f5'; color='#999'; }
            else if (ratio>=0.7) { decision='✔'; bg=`${base}24`; color=base; }
            else if (ratio>=0.4) { decision='⚠'; bg=`${base}14`; color='#ed6c02'; }
            matrix[`${country}||${cat}`] = { ratio, decision, bg, color, docs, themesWithData, totalThemes: themes.length, baseColor: base };
        });
    });

    let html = '<table class="data-table" style="width:100%;border-collapse:collapse;font-size:0.8em;">';
    html += '<thead><tr><th style="padding:8px;position:sticky;left:0;background:#E91E63;color:white;">Ülke</th>' + categories.map(cat => {
        const color = colorForCategory(cat);
        const shortLabel = cat.split(',')[0].slice(0,14);
        return `<th style="padding:6px 4px;text-align:center;background:${color};color:white;writing-mode:vertical-lr;text-orientation:mixed;">${shortLabel}</th>`;
    }).join('') + '</tr></thead>';
    html += '<tbody>';
    countries.forEach(country => {
        html += `<tr><td style="padding:6px 8px;font-weight:600;position:sticky;left:0;background:white;">${country}</td>`;
        categories.forEach(cat => {
            const cell = matrix[`${country}||${cat}`];
            const tooltip = cell ? `${(cell.ratio*100).toFixed(0)}% kalite\n${cell.themesWithData}/${cell.totalThemes} tema\n${(cell.docs/1000).toFixed(0)}K dok.` : 'Veri yok';
            html += `<td style="padding:4px;text-align:center;background:${cell?.bg||'#f5f5f5'};color:${cell?.color||'#666'};border:1px solid ${(cell&&cell.baseColor)||'#e0e0e0'}26;" title="${tooltip}">${cell?.decision||'-'}${cell&&cell.totalThemes?`<div style='font-size:0.65em;color:${cell.color};'>${(cell.ratio*100).toFixed(0)}%</div>`:''}</td>`;
        });
        html += '</tr>';
    });
    html += '</tbody></table>';
    container.innerHTML = html;

    showGDELTStatus(`✅ Matris oluşturuldu! ${categories.length} kategori (seçili/sıralı) × ${countries.length} ülke`, 'success');
}

// Simple chart/updater stubs
function updateCategoryCountryChart() {
    const selected = Array.from(window.selectedIPTCCategories);
    const canvas = document.getElementById('category-country-bar-chart');
    if (!canvas || typeof Chart === 'undefined') return;
    if (categoryCountryChart) { categoryCountryChart.destroy(); categoryCountryChart = null; }
    if (!selected.length) return;

    const countries = [...new Set((window.gdeltTotalDocs||[]).map(r=>r.country).filter(Boolean))];
    const countryNames = { CE:'Sri Lanka', HO:'Honduras', HR:'Croatia', KG:'Kyrgyzstan', LO:'Slovakia', SA:'Saudi Arabia' };
    const datasets = [];

    selected.forEach(cat => {
        const info = (window.themesByIPTC && window.themesByIPTC[cat]) || { themes: [] };
        const data = countries.map(c => info.themes.filter(t=>t.country===c).reduce((s,t)=>s+Number(t.total_docs||0),0));
        datasets.push({
            label: cat.split(',')[0],
            data,
            backgroundColor: hexToRgba(colorForCategory(cat), 0.35),
            borderColor: colorForCategory(cat),
            borderWidth: 1
        });
    });

    categoryCountryChart = new Chart(canvas.getContext('2d'), {
        type: 'bar',
        data: { labels: countries.map(c=>countryNames[c]||c), datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { position: 'bottom' } },
            scales: { y: { beginAtZero: true, title: { display: true, text: 'Doküman' } } }
        }
    });
}

function updateMappingComparisonTable() {
    const tbody = document.getElementById('mapping-diff-tbody');
    if (!tbody) return;
    const categories = Object.keys(window.IPTC_COLORS || {}).filter(c => c !== 'unmapped');
    if (!window.mappingResults.v1 || !window.mappingResults.v2 || !window.mappingResults.v3) {
        tbody.innerHTML = '<tr><td colspan="6" style="text-align:center;padding:15px;color:#999;">V1/V2/V3 sonuçları yükleniyor...</td></tr>';
        return;
    }
    const rows = categories.map(cat => {
        const v1 = window.mappingResults.v1.counts[cat] || 0;
        const v2 = window.mappingResults.v2.counts[cat] || 0;
        const v3 = window.mappingResults.v3.counts[cat] || 0;
        const diffV1 = v3 - v1;
        const diffV2 = v3 - v2;
        const diffColor1 = diffV1 > 0 ? '#4CAF50' : (diffV1 < 0 ? '#f44336' : '#999');
        const diffColor2 = diffV2 > 0 ? '#4CAF50' : (diffV2 < 0 ? '#f44336' : '#999');
        const baseColor = colorForCategory(cat);
        return `<tr style="border-bottom:1px solid #ffe0b2;">
            <td style="padding:6px 8px;text-transform:capitalize;color:${baseColor};">${cat.split(',')[0]}</td>
            <td style="padding:6px 8px;text-align:center;">${v1}</td>
            <td style="padding:6px 8px;text-align:center;">${v2}</td>
            <td style="padding:6px 8px;text-align:center;">${v3}</td>
            <td style="padding:6px 8px;text-align:center;color:${diffColor1};font-weight:600;">${diffV1>0?'+':''}${diffV1}</td>
            <td style="padding:6px 8px;text-align:center;color:${diffColor2};font-weight:600;">${diffV2>0?'+':''}${diffV2}</td>
        </tr>`;
    }).join('');
    tbody.innerHTML = rows;
}

function updateCategorySelection() {
    const checked = Array.from(document.querySelectorAll('.iptc-cat-checkbox:checked')).map(cb => cb.value);
    window.selectedIPTCCategories = new Set(checked);
    updateCategorySelectionCount();
    updateSelectedCategoriesSummary();
    updateCategoryCountryChart();
    updateMappingComparisonTable();
}

function updateCategorySelectionCount() {
    const countEl = document.getElementById('selected-category-count');
    if (!countEl) return;
    const totalCats = (Object.keys(window.IPTC_COLORS || {}).filter(c => c !== 'unmapped').length) || 17;
    countEl.textContent = `${window.selectedIPTCCategories.size} / ${totalCats}`;
    countEl.style.background = window.selectedIPTCCategories.size >= 5 ? '#4CAF50' : '#2196F3';
}

function selectRecommendedCategories() {
    const order = window.categoryRecommendationOrder || buildCategoryMetrics().map(m=>m.cat);
    const picks = order.slice(0,5);
    document.querySelectorAll('.iptc-cat-checkbox').forEach(cb => {
        if (picks.includes(cb.value)) cb.checked = true;
    });
    updateCategorySelection();
}

function selectAlternativeCategories() {
    const order = window.categoryRecommendationOrder || buildCategoryMetrics().map(m=>m.cat);
    const picks = order.slice(5,7);
    document.querySelectorAll('.iptc-cat-checkbox').forEach(cb => {
        if (picks.includes(cb.value)) cb.checked = true;
    });
    updateCategorySelection();
}

function clearCategorySelection() {
    document.querySelectorAll('.iptc-cat-checkbox').forEach(cb => { cb.checked = false; });
    window.selectedIPTCCategories = new Set();
    updateCategorySelection();
}

function updateSelectedCategoriesSummary() {
    const summaryEl = document.getElementById('selected-categories-summary');
    if (!summaryEl) return;
    if (!window.selectedIPTCCategories.size) {
        summaryEl.innerHTML = '<div style="text-align:center;color:#999;padding:20px;grid-column:1/-1;">Kategori seçin...</div>';
        return;
    }
    let html = '';
    let totalThemes = 0;
    let totalDocs = 0;
    window.selectedIPTCCategories.forEach(cat => {
        const info = (window.themesByIPTC && window.themesByIPTC[cat]) || { themes: [] };
        const themeCount = info.themes.length;
        const docs = info.themes.reduce((s,t)=>s+Number(t.total_docs||t.totalDocs||0),0);
        totalThemes += themeCount;
        totalDocs += docs;
        const baseColor = colorForCategory(cat);
        html += `<div style="background:${hexToRgba(baseColor,0.12)};border:1px solid ${hexToRgba(baseColor,0.4)};border-radius:8px;padding:8px;">
            <div style="font-weight:700;color:${baseColor};font-size:0.9em;">${cat.split(',')[0]}</div>
            <div style="font-size:0.8em;color:${hexToRgba(baseColor,0.8)};">${themeCount} tema • ${docs.toLocaleString()} dok.</div>
        </div>`;
    });
    html += `<div style="background:#e3f2fd;border:1px solid #bbdefb;border-radius:8px;padding:8px;grid-column:1/-1;">
        <div style="font-weight:700;color:#0d47a1;">Toplam</div>
        <div style="font-size:0.85em;color:#2c3e50;">${totalThemes} tema • ${totalDocs.toLocaleString()} dok.</div>
    </div>`;
    summaryEl.innerHTML = html;
}

function updateMonthlyDetailStats(rows){
    const totalRows = rows.length;
    const totalDocs = rows.reduce((s,r)=>s+Number(r.n_docs||0),0);
    const uniqueThemes = new Set(rows.map(r=>r.theme_code||r.theme)).size;
    const avgDocs = totalRows ? totalDocs / totalRows : 0;
    const setText = (id, val) => { const el = document.getElementById(id); if (el) el.textContent = val; };
    setText('detail-total-rows', totalRows.toLocaleString());
    setText('detail-total-docs', totalDocs.toLocaleString());
    setText('detail-unique-themes', uniqueThemes.toLocaleString());
    setText('detail-avg-docs', avgDocs.toFixed(1));
}

function updateTrendSummary(rows){
    const totals = { artan:0, azalan:0, stabil:0 };
    rows.forEach(r => {
        const cat = (r.trend_category || r.category || '').toLowerCase();
        if (cat.includes('art')) totals.artan += 1;
        else if (cat.includes('aza') || cat.includes('decline')) totals.azalan += 1;
        else totals.stabil += 1;
    });
    const setText = (id, val) => { const el = document.getElementById(id); if (el) el.textContent = val; };
    setText('trend-total-themes', rows.length.toLocaleString());
    setText('trend-artan-count', totals.artan.toLocaleString());
    setText('trend-azalan-count', totals.azalan.toLocaleString());
    setText('trend-stabil-count', totals.stabil.toLocaleString());
}

function sortMonthlyDetailTable(field){
    if (monthlyDetailSort.field === field) {
        monthlyDetailSort.dir = monthlyDetailSort.dir === 'asc' ? 'desc' : 'asc';
    } else {
        monthlyDetailSort.field = field;
        monthlyDetailSort.dir = 'asc';
    }
    updateMonthlyDetailTable();
}

function exportSelectedCategoriesAnalysis() {
    if (!selectedIPTCCategories.size) { showGDELTStatus('⚠️ Lütfen en az bir kategori seçin.', 'error'); return; }
    const selected = Array.from(selectedIPTCCategories);
    const payload = {
        generated_at: new Date().toISOString(),
        categories: selected,
        summary: selected.map(cat => ({ category: cat, themes: (window.themesByIPTC[cat]||{themes:[]}).themes.length }))
    };
    const blob = new Blob([JSON.stringify(payload, null, 2)], {type:'application/json'});
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `selected_iptc_analysis_${new Date().toISOString().slice(0,10)}.json`;
    a.click();
    showGDELTStatus('✅ Seçili kategoriler JSON olarak indirildi.', 'success');
}

// Render IPTC mapping tables when clustering results are loaded
window.onIPTCMappingLoaded = function(mappingJson, version, source){
    if (!mappingJson || !Array.isArray(mappingJson.themes)) return;
    renderIPTCThemesTable(mappingJson);
    renderIPTCSummaryCardsFromMapping(mappingJson);
    // re-run GDELT analysis if data exists
    if (window.gdeltDataLoaded && typeof analyzeThemes === 'function') analyzeThemes();
    updateTrendAnalysisTable();
};

function renderIPTCThemesTable(mappingJson){
    const tbody = document.getElementById('iptc-themes-tbody');
    if (!tbody) return;
    const themes = mappingJson.themes || [];
    const rows = themes.map(t => {
        const code = t.theme_code || t.theme || '';
        const cat = t.iptc_final_label || t.iptc_label || t.iptc_category || t.iptc_nn_label || t.iptc_rule_label || (t.iptc && t.iptc.label) || '';
        const sim = (t.nn_score || t.similarity || '').toString();
        const conf = (t.rule_score || t.confidence || '').toString();
        const second = t.second_best_label || t.second_best || '';
        return `<tr><td style="padding:8px">${code}</td><td style="padding:8px">${cat}</td><td style="padding:8px;text-align:center">${sim}</td><td style="padding:8px;text-align:center">${conf}</td><td style="padding:8px">${second}</td></tr>`;
    }).join('');
    tbody.innerHTML = rows || '<tr><td colspan="5" style="text-align:center;padding:20px;color:#999;">Tema bulunamadı</td></tr>';
}

function renderIPTCSummaryCardsFromMapping(mappingJson){
    const grid = document.getElementById('iptc-summary-cards-grid');
    if (!grid) return;
    const byCat = {};
    // seed all 17 categories so they are always visible
    Object.keys(IPTC_CATEGORIES).forEach(cat => { byCat[cat] = { themes: [], totalDocs: 0 }; });
    (mappingJson.themes||[]).forEach(t=>{
        const cat = t.iptc_final_label || t.iptc_label || t.iptc_category || t.iptc_nn_label || t.iptc_rule_label || 'unmapped';
        byCat[cat] = byCat[cat] || { themes: [], totalDocs: 0 };
        byCat[cat].themes.push(t.theme_code || t.theme || '');
    });
    const cards = Object.entries(byCat).map(([cat, info]) => {
        const list = (info.themes||[]).slice(0,18).map(c=>`<span class="theme-pill">${c}</span>`).join('') || '<span class="theme-pill muted">Eşleşme yok</span>';
        return `<div class="stat-card minimal">
            <div class="stat-card__header">${cat}</div>
            <div class="stat-card__body">${list}</div>
            <div class="stat-card__footer">${info.themes.length} tema eşleşti</div>
        </div>`;
    }).join('');
    grid.innerHTML = cards;
}

// Expose some functions globally for other tabs
window.generateCountryThemeMatrix = generateCountryThemeMatrix;
window.exportSelectedCategoriesAnalysis = exportSelectedCategoriesAnalysis;
window.renderIPTCSummaryCards = renderIPTCSummaryCards;
window.renderIPTCSelectionPanel = renderIPTCSelectionPanel;
window.updateMappingComparisonTable = updateMappingComparisonTable;
window.updateTrendAnalysisTable = updateTrendAnalysisTable;
window.sortTrendTable = sortTrendTable;
window.sortMonthlyDetailTable = sortMonthlyDetailTable;
window.selectRecommendedCategories = selectRecommendedCategories;
window.selectAlternativeCategories = selectAlternativeCategories;
window.clearCategorySelection = clearCategorySelection;
window.updateCategorySelection = updateCategorySelection;


