// tabs/tab-clustering.js
// Clean implementation for the Clustering tab. Provides safe init, run/load/apply functions
// and preserves backwards-compatible global function names expected by HTML inline handlers.

(function(){
  // Per-version/source storage (local to this module)
  const results = {
    v1: { vargo: null, gkg: null },
    v2: { vargo: null, gkg: null },
    v3: { vargo: null, gkg: null }
  };

  const mappingVariants = [
    ['v1', 'vargo'],
    ['v1', 'gkg'],
    ['v2', 'vargo'],
    ['v2', 'gkg'],
    ['v3', 'vargo'],
    ['v3', 'combined']
  ];

  const comparisonSelections = { v1: 'vargo', v2: 'gkg', v3: 'combined' };

  // Active pointer used by other modules via window.iptcMappingResults
  function setActiveResults(obj, version, source){
    try {
      window.iptcMappingResults = obj || null;
      // also set version-specific global refs for compatibility
      if (version === 'v1') window.iptcMappingResultsV1 = obj || null;
      if (version === 'v2') window.iptcMappingResultsV2 = obj || null;
      if (version === 'v3') window.iptcMappingResultsV3 = obj || null;
      // store in local cache
      if (version && source) results[version][source] = obj || null;
    } catch(e){ console.warn('setActiveResults error', e); }
  }

  // Utility: safe element getter
  const $ = id => document.getElementById(id);

  // Utility: fetch JSON with friendly error
  async function fetchJson(path){
    const r = await fetch(path);
    if (!r.ok) throw new Error(`${path} returned ${r.status}`);
    return await r.json();
  }

  // Apply mapping results into module, update globals and UI
  function applyMappingResults(json, version, source){
    if (!json) return;
    // normalize possible nested iptc_categories
    if (!json.iptc_categories && json.metadata && json.metadata.iptc_categories) json.iptc_categories = json.metadata.iptc_categories;

    // store
    if (version && source) results[version][source] = json;
    // choose active: prefer the just-applied
    setActiveResults(json, version, source);
    // build and expose lookup for use by other tabs
    try{ buildLookupAndExpose(json, version, source); }catch(e){}
    // attempt to annotate GDELT data immediately when results are loaded
    try{ if (typeof annotateGDELTWithMapping === 'function') annotateGDELTWithMapping(); }catch(e){}
    // notify GDELT tab renderers to refresh tables/cards with the new mapping
    try{ if (typeof window.onIPTCMappingLoaded === 'function') window.onIPTCMappingLoaded(json, version, source); }catch(e){}

    // render results using page-provided renderers if available
    let rendered = false;
    try {
      if (typeof displayIPTCMappingResultsForSource === 'function') {
        displayIPTCMappingResultsForSource(version, source, json);
        rendered = true;
      } else if (typeof displayIPTCMappingResults === 'function') {
        displayIPTCMappingResults(json);
        rendered = true;
      } else if (typeof displayIPTCThemesTable === 'function') {
        displayIPTCThemesTable(json.themes || []);
        rendered = true;
      }
    } catch(e){ console.warn('display render failed', e); }

    if (!rendered){
      try { renderMappingResultsCard(version, source, json); }catch(e){ console.warn('fallback render failed', e); }
    }

    // attempt to build charts if helpers are available
    try { if (typeof createTSNEScatterChart === 'function') createTSNEScatterChart(); } catch(e){ console.warn(e); }
    try { if (typeof generateClusteringCharts === 'function') generateClusteringCharts(); } catch(e){ console.warn(e); }
    try { if (typeof updateSharedMappingStats === 'function') updateSharedMappingStats(); } catch(e){}
    // If GDELT data already loaded, re-run analyze to reflect new categories
    try{ if (typeof analyzeThemes === 'function' && window.gdeltDataLoaded) analyzeThemes(); }catch(e){}

    try{ renderMappingComparisonTable(); } catch(e){ console.warn('comparison table refresh', e); }
  }

  // Build a quick lookup theme_code -> iptc_label for frontend grouping
  function buildLookupAndExpose(json, version, source){
    try{
      const lookup = {};
      if (json && Array.isArray(json.themes)){
        json.themes.forEach(t => {
          const code = t.theme_code || t.theme || t.code;
          const label = t.iptc_final_label || t.iptc_label || t.iptc_category || t.iptc_nn_label || t.iptc_rule_label || (t.iptc && t.iptc.label) || null;
          if (code && label) lookup[code] = label;
        });
      }
      // expose both a generic and versioned lookup
      window.iptcMappingLookup = lookup;
      window.iptcMappingLookupByVersion = window.iptcMappingLookupByVersion || {};
      if (!window.iptcMappingLookupByVersion[version]) window.iptcMappingLookupByVersion[version] = {};
      window.iptcMappingLookupByVersion[version][source] = lookup;
      // set an active info object
      window.activeMappingInfo = { version: version, source: source };
    }catch(e){ console.warn('buildLookupAndExpose', e); }
  }

  // Annotate loaded GDELT data structures with iptc_category for UI tables
  function annotateGDELTWithMapping(){
    try{
      const lookup = window.iptcMappingLookup || {};
      // allow either global or window-scoped flag (some code paths use gdeltDataLoaded var)
      const dataLoadedFlag = (typeof window.gdeltDataLoaded !== 'undefined')
        ? window.gdeltDataLoaded
        : (typeof gdeltDataLoaded !== 'undefined' ? gdeltDataLoaded : false);
      if (!dataLoadedFlag) return;
      ['gdeltTotalDocs','gdeltMonthlyQuality','gdeltMonthlyDetail'].forEach(key=>{
        const arr = window[key];
        if (!Array.isArray(arr)) return;
        arr.forEach(r=>{
          const code = r.theme_code || r.theme || r.themeCode || r.theme_code;
          const mapped = lookup && code ? (lookup[code] || null) : null;
          if (mapped) r.iptc_category = mapped;
          else if (r.iptc_category) {/* keep existing */}
          else r.iptc_category = r.iptc_label || null;
        });
      });
      // trend analysis table annotations
      const trendArr = window.gdeltTrendAnalysis;
      if (Array.isArray(trendArr)){
        trendArr.forEach(r=>{
          const code = r.theme_code || r.theme;
          const mapped = lookup && code ? (lookup[code] || null) : null;
          if (mapped) r.iptc_category = mapped;
        });
      }
      // refresh tables/charts in GDELT tab if functions available
      try{ if (typeof updateTotalDocsTable === 'function') updateTotalDocsTable(); }catch(e){}
      try{ if (typeof updateMonthlyQualityTable === 'function') updateMonthlyQualityTable(); }catch(e){}
      try{ if (typeof updateMonthlyDetailTable === 'function') updateMonthlyDetailTable(); }catch(e){}
      try{ if (typeof generateCountryThemeMatrix === 'function') generateCountryThemeMatrix(); }catch(e){}
      try{ if (typeof updateTrendAnalysisTable === 'function') updateTrendAnalysisTable(); }catch(e){}
    }catch(e){ console.warn('annotateGDELTWithMapping', e); }
  }

  function formatSourceLabel(source){
    const map = { vargo: 'Vargo', gkg: 'GKG', combined: 'Combined' };
    if (!source) return '';
    return map[source] || source.toUpperCase();
  }

  function ensureMappingResultsTemplate(version, source){
    try{
      const id = `${version}-${source}-results`;
      const container = document.getElementById(id);
      if (!container) return null;
      if (container.dataset.hydrated === 'true') return container;
      const template = document.getElementById('mapping-results-template');
      if (!template || !template.content) return container;
      const clone = template.content.cloneNode(true);
      const labelEl = clone.querySelector('[data-role="results-label"]');
      if (labelEl) labelEl.textContent = `${version.toUpperCase()} · ${formatSourceLabel(source)}`;
      container.appendChild(clone);
      container.dataset.hydrated = 'true';
      return container;
    }catch(err){ console.warn('ensureMappingResultsTemplate', err); return null; }
  }

  function hydrateMappingResultsShells(){
    const template = document.getElementById('mapping-results-template');
    if (!template) return;
    mappingVariants.forEach(([version, source]) => ensureMappingResultsTemplate(version, source));
  }

  function formatTimestamp(value){
    if (!value) return 'Oluşturulma zamanı bilinmiyor';
    const date = new Date(value);
    if (!Number.isNaN(date.getTime())) return `Oluşturuldu: ${date.toLocaleString('tr-TR')}`;
    return `Oluşturuldu: ${value}`;
  }

  function summarizeCategories(themes){
    const stats = {};
    if (!Array.isArray(themes)) return stats;
    themes.forEach(theme => {
      const label = theme.iptc_final_label || theme.iptc_label || 'Belirtilmemiş';
      stats[label] = (stats[label] || 0) + 1;
    });
    return stats;
  }

  function buildTopCategoryBadges(stats){
    const entries = Object.entries(stats || {}).filter(([label]) => !!label);
    if (!entries.length) return '<span class="mapping-tag neutral">Kategori verisi yok</span>';
    entries.sort((a,b)=> b[1] - a[1]);
    return entries.slice(0,4)
      .map(([label,count]) => `<span class="mapping-tag">${escapeHtml(label)} (${count})</span>`)
      .join('');
  }

  function decisionShortLabel(key){
    const map = {
      agreement: 'Uyum',
      rule_preferred: 'Kural',
      nn_confident: 'NN Güven',
      nn_override: 'NN Override',
      nn_weak: 'NN Zayıf',
      ambiguous: 'Belirsiz',
      fallback: 'Fallback'
    };
    return map[key] || key;
  }

  function buildDecisionSummary(stats){
    if (!stats) return 'Karar verisi yok';
    const entries = Object.entries(stats)
      .filter(([,count]) => count > 0)
      .sort((a,b)=> b[1] - a[1])
      .slice(0,3);
    if (!entries.length) return 'Karar verisi yok';
    return entries.map(([key,count]) => `${decisionShortLabel(key)}: ${count}`).join(' · ');
  }

  function formatPercent(value){
    if (typeof value !== 'number' || !isFinite(value)) return '-';
    return `%${(value * 100).toFixed(1)}`;
  }

  function confidenceClass(value){
    if (typeof value !== 'number' || !isFinite(value)) return 'neutral';
    if (value >= 0.6) return 'confidence-high';
    if (value >= 0.4) return 'confidence-medium';
    return 'confidence-low';
  }

  function decisionBadge(source){
    if (!source) return '';
    const map = {
      agreement: { label: '✓ Uyum', cls: 'badge-agreement' },
      rule_preferred: { label: 'Kural', cls: 'badge-rule' },
      nn_confident: { label: 'NN', cls: 'badge-nn' },
      nn_override: { label: 'NN+', cls: 'badge-nn' },
      nn_weak: { label: 'NN zayıf', cls: 'badge-ambiguous' },
      ambiguous: { label: 'Belirsiz', cls: 'badge-ambiguous' },
      fallback: { label: 'Fallback', cls: 'badge-fallback' }
    };
    const meta = map[source] || null;
    if (!meta) return '';
    return `<span class="mapping-badge-chip ${meta.cls}">${meta.label}</span>`;
  }

  function escapeHtml(value){
    if (value === null || value === undefined) return '';
    return String(value)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  function buildThemeRows(themes){
    if (!Array.isArray(themes) || !themes.length){
      return '<tr><td colspan="5" class="mapping-placeholder">IPTC eşleştirme sonuçları yüklenmedi</td></tr>';
    }
    const sorted = themes.slice().sort((a,b)=>{
      const bn = (typeof b.nn_score === 'number' ? b.nn_score : (typeof b.similarity === 'number' ? b.similarity : 0));
      const an = (typeof a.nn_score === 'number' ? a.nn_score : (typeof a.similarity === 'number' ? a.similarity : 0));
      return bn - an;
    });
    return sorted.map(theme => {
      const themeCode = escapeHtml(theme.theme_code || theme.theme || '-');
      const iptcLabel = escapeHtml(theme.iptc_final_label || theme.iptc_label || 'Belirsiz');
      const similarityVal = (typeof theme.nn_score === 'number') ? theme.nn_score : (typeof theme.similarity === 'number' ? theme.similarity : null);
      const confidenceVal = (typeof theme.final_confidence === 'number') ? theme.final_confidence
                          : ((typeof theme.confidence === 'number') ? theme.confidence
                          : ((typeof theme.rule_confidence === 'number') ? theme.rule_confidence : null));
      const secondLabel = escapeHtml(theme.iptc_second_label || theme.iptc_label_2nd || '-');
      const confidenceCls = confidenceClass(confidenceVal);
      const decision = decisionBadge(theme.decision_source || '');
      return `
        <tr>
          <td class="code-cell">${themeCode}</td>
          <td>
            <span class="mapping-tag neutral">${iptcLabel}</span>
            ${decision}
          </td>
          <td><span class="mapping-tag similarity">${formatPercent(similarityVal)}</span></td>
          <td><span class="mapping-tag ${confidenceCls}">${formatPercent(confidenceVal)}</span></td>
          <td class="second-label">${secondLabel}</td>
        </tr>`;
    }).join('');
  }

  function renderMappingResultsCard(version, source, data){
    const container = ensureMappingResultsTemplate(version, source);
    if (!container) return;
    const themes = Array.isArray(data) ? data : (Array.isArray(data && data.themes) ? data.themes : (Array.isArray(data && data.results) ? data.results : []));
    const metadata = (data && data.metadata) ? data.metadata : {};
    const labelEl = container.querySelector('[data-role="results-label"]');
    if (labelEl) labelEl.textContent = `${version.toUpperCase()} · ${formatSourceLabel(source)}`;
    const generated = metadata.generated_at || metadata.created_at || (data && (data.generated_at || data.created_at)) || null;
    const metaEl = container.querySelector('[data-role="generated-at"]');
    if (metaEl) metaEl.textContent = formatTimestamp(generated);
    const themeBadge = container.querySelector('[data-role="theme-total"]');
    if (themeBadge) themeBadge.textContent = `${themes.length} tema`;
    const categoryStats = summarizeCategories(themes);
    const categoryBadge = container.querySelector('[data-role="category-count"]');
    if (categoryBadge) categoryBadge.textContent = `${Math.max(0, Object.keys(categoryStats).filter(Boolean).length)} IPTC kategorisi`;
    const decisionSummaryEl = container.querySelector('[data-role="decision-summary"]');
    if (decisionSummaryEl) decisionSummaryEl.textContent = buildDecisionSummary(metadata.decision_stats);
    const topCatEl = container.querySelector('[data-role="top-categories"]');
    if (topCatEl) topCatEl.innerHTML = buildTopCategoryBadges(categoryStats);
    const tbody = container.querySelector('[data-role="mapping-tbody"]');
    if (tbody) tbody.innerHTML = buildThemeRows(themes);
  }

  if (typeof document !== 'undefined'){
    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', hydrateMappingResultsShells);
    else hydrateMappingResultsShells();
  }


  // --- Balancing & Export Helpers (frontend) ---
  // Generate a balancing plan per selected IPTC category and country.
  // targetStrategy: 'min' (default) or 'median'
  window.generateBalancingPlan = function(targetStrategy = 'min'){
    const categories = Array.from(selectedIPTCCategories || []);
    if (!categories.length) { showStatus('Önce IPTC kategorilerini seçin', 'error'); return null; }
    const detail = window.gdeltMonthlyDetail || monthlyDetailData || [];
    if (!detail || detail.length === 0) { showStatus('Monthly detail verisi yok', 'error'); return null; }

    const totals = {}; // totals[category][country] = count
    detail.forEach(r => {
      const country = r.country;
      const cat = r.iptc_category || (typeof getIPTCCategory === 'function' ? getIPTCCategory(r.theme_code) : null);
      if (!cat || !categories.includes(cat)) return;
      const n = r.n_docs || r.total_docs || 0;
      totals[cat] = totals[cat] || {};
      totals[cat][country] = (totals[cat][country] || 0) + (Number(n) || 0);
    });

    const plan = [];
    categories.forEach(cat => {
      const byCountry = totals[cat] || {};
      const countries = Object.keys(byCountry);
      if (!countries.length) return;
      let target = 0;
      const counts = countries.map(c => byCountry[c] || 0).sort((a,b)=>a-b);
      if (targetStrategy === 'median') target = counts[Math.floor(counts.length/2)];
      else target = counts[0]; // 'min' default
      countries.forEach(country => {
        plan.push({ category: cat, country, current: byCountry[country]||0, target });
      });
    });

    window.balancingPlan = plan;
    showStatus('Dengeleme planı oluşturuldu', 'success');
    return plan;
  };

  // Export balancing plan as CSV
  window.exportBalancingPlanCSV = function(){
    if (!window.balancingPlan) { showStatus('Önce dengeleme planı oluşturun', 'error'); return; }
    const csv = ['category,country,current,target']
      .concat(window.balancingPlan.map(r => `${r.category},${r.country},${r.current},${r.target}`))
      .join('\n');
    const blob = new Blob([csv], { type: 'text/csv' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'balancing_plan.csv';
    a.click();
    showStatus('Dengeleme planı indirildi', 'success');
  };

  // Generate example BigQuery sampling SQL per plan entry and download as .sql
  window.generateBigQuerySQL = function(){
    if (!window.balancingPlan) { showStatus('Önce dengeleme planı oluşturun', 'error'); return; }
    const totalDocs = window.gdeltTotalDocs || totalDocsData || [];
    const sqlFragments = [];
    window.balancingPlan.forEach(row => {
      const cat = row.category, country = row.country, limit = row.target;
      const codes = totalDocs.filter(r => (r.iptc_category || (typeof getIPTCCategory === 'function' ? getIPTCCategory(r.theme_code) : null)) === cat)
                             .map(r => r.theme_code)
                             .filter(Boolean);
      if (!codes.length) return;
      const codeList = codes.map(c => `'${c}'`).join(', ');
      const sql = `-- ${cat} | ${country} | sample ${limit}\nSELECT * FROM \`bigquery-public-data.gdelt.gkg\` \nWHERE SourceCountry='${country}' AND theme IN (${codeList})\nORDER BY RAND()\nLIMIT ${limit};\n`;
      sqlFragments.push(sql);
    });
    const blob = new Blob([sqlFragments.join('\n')], { type: 'text/sql' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'bigquery_sampling.sql';
    a.click();
    showStatus('BigQuery SQL örnekleri indirildi', 'success');
    return sqlFragments;
  };

  // Load saved results for a version (tries vargo and gkg)
  async function loadResultsFor(version){
    let sources = ['vargo','gkg'];
    if (version === 'v3') sources = ['vargo','gkg','combined'];
        for (const s of sources){
      try{
        const path = `/results/gdelt_iptc_mapping_${version}_${s}.json`;
        const j = await fetchJson(path);
        applyMappingResults(j, version, s);
      }catch(e){ /* ignore missing */ }
    }
  }

  // Run mapping on server
  async function runMapping(version, source){
    configureProgressSteps(version);
    showIPTCProgress();
    showIPTCMappingStatus(`🚀 ${version.toUpperCase()} (${source}) çalıştırılıyor...`, 'info');
    try{
      updateIPTCProgressStep(1,'active');
      const resp = await fetch('/api/run-iptc-mapping', {
        method: 'POST', headers: {'Content-Type':'application/json'},
        body: JSON.stringify({ algorithm: version, mapping_source: source })
      });
      if (!resp.ok){
        const txt = await resp.text().catch(()=>null);
        throw new Error(txt || `API ${resp.status}`);
      }
      const json = await resp.json();
      // mark progress complete
      for (let i=1;i<=5;i++) updateIPTCProgressStep(i,'completed');
      applyMappingResults(json, version, source);
      showIPTCMappingStatus(`✅ ${version.toUpperCase()}-${source} tamamlandı (${(json.themes||[]).length} tema)`, 'success');
      setTimeout(hideIPTCProgress, 800);
      return json;
    }catch(err){
      hideIPTCProgress();
      showIPTCMappingStatus(`❌ ${version.toUpperCase()}-${source} hatası: ${err.message}`, 'error');
      throw err;
    }
  }

  // Configure progress step labels based on algorithm version
  function configureProgressSteps(version){
    const defaults = {
      v1: ['CSV yükle', 'Embedding', 'En yakın IPTC', 'Karar/etiket', 'JSON yaz'],
      v2: ['CSV yükle', 'Embedding+kurallar', 'Katmanlı eşleme', 'Güven puanı', 'JSON yaz'],
      v3: ['CSV yükle', 'Alt-kümeleme', 'Hiyerarşik eşleme', 'Son etiket', 'JSON yaz']
    };
    const labels = defaults[version] || defaults.v2;
    const steps = document.querySelectorAll('.progress-step');
    steps.forEach((step, idx) => {
      const text = step.querySelector('.step-text');
      if (text && labels[idx]) text.textContent = labels[idx];
    });
    const bar = document.getElementById('iptc-progress-bar');
    if (bar) bar.style.width = '0%';
  }

  // Backwards-compatible global functions expected by inline HTML
  window.runIPTCMappingV1 = async function(){
    const sel = document.querySelector('input[name="v1-mapping-source"]:checked');
    if (!sel) return showIPTCMappingStatus('Lütfen V1 için kaynak seçin', 'error');
    return runMapping('v1', sel.value);
  };

  window.runIPTCMappingV2 = async function(){
    const sel = document.querySelector('input[name="v2-mapping-source"]:checked');
    if (!sel) return showIPTCMappingStatus('Lütfen V2 için kaynak seçin', 'error');
    return runMapping('v2', sel.value);
  };

  window.runIPTCMappingV3 = async function(){
    const sel = document.querySelector('input[name="v3-mapping-source"]:checked');
    if (!sel) return showIPTCMappingStatus('Lütfen V3 için kaynak seçin', 'error');
    return runMapping('v3', sel.value);
  };

  window.loadIPTCMappingResultsV1 = function(){ return loadResultsFor('v1'); };
  window.loadIPTCMappingResultsV2 = function(){ return loadResultsFor('v2'); };
  window.loadIPTCMappingResultsV3 = function(){ return loadResultsFor('v3'); };

  // TSNE wrappers keep old function names available and set active result
  window.createTSNEScatterChartV1Vargo = function(){ if (results.v1.vargo){ setActiveResults(results.v1.vargo,'v1','vargo'); try{ if (typeof createTSNEScatterChart==='function') createTSNEScatterChart(); }catch(e){} } };
  window.createTSNEScatterChartV1GKG = function(){ if (results.v1.gkg){ setActiveResults(results.v1.gkg,'v1','gkg'); try{ if (typeof createTSNEScatterChart==='function') createTSNEScatterChart(); }catch(e){} } };
  window.createTSNEScatterChartV2Vargo = function(){ if (results.v2.vargo){ setActiveResults(results.v2.vargo,'v2','vargo'); try{ if (typeof createTSNEScatterChart==='function') createTSNEScatterChart(); }catch(e){} } };
  window.createTSNEScatterChartV2GKG = function(){ if (results.v2.gkg){ setActiveResults(results.v2.gkg,'v2','gkg'); try{ if (typeof createTSNEScatterChart==='function') createTSNEScatterChart(); }catch(e){} } };
  window.createTSNEScatterChartV3Vargo = function(){ if (results.v3.vargo){ setActiveResults(results.v3.vargo,'v3','vargo'); try{ if (typeof createTSNEScatterChart==='function') createTSNEScatterChart(); }catch(e){} } };
  window.createTSNEScatterChartV3GKG = function(){ if (results.v3.gkg){ setActiveResults(results.v3.gkg,'v3','gkg'); try{ if (typeof createTSNEScatterChart==='function') createTSNEScatterChart(); }catch(e){} } };

  // Initialize tab: bind buttons and try to pre-load v2 results
  function initTabClustering(){
    try{
      const rv1 = $('run-v1-btn'); if (rv1) rv1.addEventListener('click', window.runIPTCMappingV1);
      const rv2 = $('run-v2-btn'); if (rv2) rv2.addEventListener('click', window.runIPTCMappingV2);
      const rv3 = $('run-v3-btn'); if (rv3) rv3.addEventListener('click', window.runIPTCMappingV3);
      const lv1 = $('load-v1-btn'); if (lv1) lv1.addEventListener('click', window.loadIPTCMappingResultsV1);
      const lv2 = $('load-v2-btn'); if (lv2) lv2.addEventListener('click', window.loadIPTCMappingResultsV2);
      const lv3 = $('load-v3-btn'); if (lv3) lv3.addEventListener('click', window.loadIPTCMappingResultsV3);
      const activeSelect = $('active-mapping-select'); if (activeSelect) activeSelect.addEventListener('change', ()=>{ window.activeMapping = activeSelect.value; try{ if (typeof updateSharedMappingStats==='function') updateSharedMappingStats(); }catch(e){} });
      // Ensure a global switchActiveMapping exists for inline HTML handlers
      window.switchActiveMapping = function(){
        try{
          const sel = $('active-mapping-select');
          if (!sel) return;
          const val = sel.value; // e.g. 'v2-vargo'
          window.activeMapping = val;
          // update info text
          const info = $('active-mapping-info');
          if (info) info.textContent = val.replace('-', ' ').toUpperCase() + ' sonuçları kullanılıyor';
          // show/hide result containers by exact id (val + '-results')
          const targetId = `${val}-results`;
          document.querySelectorAll('.mapping-results-container').forEach(el=>{
            if (!el.id) return;
            if (el.id === targetId) el.style.display = '';
            else el.style.display = 'none';
          });
          // if we have a cached result for this version/source, activate it
          const parts = val.split('-');
          const version = parts[0];
          const source = parts.slice(1).join('-');
          if (results[version] && results[version][source]){
            setActiveResults(results[version][source], version, source);
            buildLookupAndExpose(results[version][source], version, source);
            // annotate GDELT data if present so tables immediately reflect mapping
            try{ annotateGDELTWithMapping(); }catch(e){}
          } else {
            // try to load it from results folder
            loadResultsFor(version).then(()=>{
              // after load, if this specific source is present, activate it
              if (results[version] && results[version][source]){
                setActiveResults(results[version][source], version, source);
                buildLookupAndExpose(results[version][source], version, source);
                try{ annotateGDELTWithMapping(); }catch(e){}
              }
            }).catch(()=>{});
          }
          // If GDELT data is loaded, trigger re-analysis so GDELT UI groups update
          try{ if (typeof analyzeThemes === 'function' && window.gdeltDataLoaded) analyzeThemes(); }catch(e){}
          // trigger optional shared updates
          try{ if (typeof updateSharedMappingStats==='function') updateSharedMappingStats(); }catch(e){}
          try{ if (typeof createTSNEScatterChart==='function') createTSNEScatterChart(); }catch(e){}
        }catch(e){ console.warn('switchActiveMapping', e); }
      };
      hydrateMappingResultsShells();
      registerComparisonSelectors();
      renderMappingComparisonTable();
      // Sync initial value
      try{ window.switchActiveMapping(); }catch(e){}
      // pre-load all saved outputs
      loadResultsFor('v1').catch(()=>{});
      loadResultsFor('v2').catch(()=>{});
      loadResultsFor('v3').catch(()=>{});
    }catch(e){ console.warn('initTabClustering error', e); }
  }

  window.displayIPTCMappingResultsForSource = function(version, source, data){
    try { renderMappingResultsCard(version, source, data); } catch(err){ console.warn('displayIPTCMappingResultsForSource', err); }
  };

  window.displayIPTCMappingResults = function(data){
    const info = window.activeMappingInfo || { version: 'v2', source: 'vargo' };
    try { renderMappingResultsCard(info.version, info.source, data); } catch(err){ console.warn('displayIPTCMappingResults', err); }
  };

  window.displayIPTCThemesTable = function(themes){
    const info = window.activeMappingInfo || { version: 'v2', source: 'vargo' };
    const payload = { themes: Array.isArray(themes) ? themes : [], metadata: {} };
    try { renderMappingResultsCard(info.version, info.source, payload); } catch(err){ console.warn('displayIPTCThemesTable', err); }
  };

  function gatherThemeCodes(){
    const codes = new Set();
    Object.values(results).forEach(versionSet => {
      Object.values(versionSet).forEach(payload => {
        if (!payload) return;
        const themes = Array.isArray(payload) ? payload : (Array.isArray(payload.themes) ? payload.themes : (Array.isArray(payload.results) ? payload.results : []));
        themes.forEach(theme => {
          const code = theme.theme_code || theme.theme || theme.code;
          if (code) codes.add(code);
        });
      });
    });
    return Array.from(codes).sort();
  }

  function findThemeEntry(payload, themeCode){
    if (!payload || !themeCode) return null;
    const themes = Array.isArray(payload) ? payload : (Array.isArray(payload.themes) ? payload.themes : (Array.isArray(payload.results) ? payload.results : []));
    return themes.find(theme => (theme.theme_code || theme.theme || theme.code) === themeCode) || null;
  }

  function extractThemeDescription(theme){
    if (!theme) return '';
    return theme.text_repr || theme.description || theme.issue_category || theme.text || theme.note || theme.definition || theme.iptc_definition || '';
  }

  function getPrimaryMatchScore(theme){
    if (!theme) return null;
    return theme.similarity ?? theme.nn_score ?? theme.final_confidence ?? theme.confidence ?? theme.rule_confidence ?? theme.similarity_2nd ?? theme.second_score ?? null;
  }

  function renderMatchCell(version, themeCode){
    const source = comparisonSelections[version];
    if (!source) return '<div class="comparison-empty">Seçim yok</div>';
    const payload = results[version] && results[version][source];
    const theme = findThemeEntry(payload, themeCode);
    if (!theme) return '<div class="comparison-empty">Veri yok</div>';
    const label = theme.iptc_final_label || theme.iptc_label || theme.category || theme.iptc_nn_label || 'Belirsiz';
    const score = getPrimaryMatchScore(theme);
    const percent = formatPercent(score);
    const helperText = theme.iptc_second_label ? `${escapeHtml(theme.iptc_second_label)} (${formatPercent(theme.second_score ?? theme.similarity_2nd)})` : '';
    return `<div style="display:flex;flex-direction:column;gap:4px;padding:6px 0;">
        <span style="font-weight:600;color:#1f2560">${escapeHtml(label)}</span>
        <span style="font-size:0.85em;color:#4f5d85">${percent}</span>
        ${helperText ? `<span style="font-size:0.78em;color:#6d7aa3">${helperText}</span>` : ''}
    </div>`;
  }

  function renderMappingComparisonTable(){
    const tbody = document.getElementById('mapping-comparison-tbody');
    if (!tbody) return;
    const codes = gatherThemeCodes();
    if (!codes.length){
      tbody.innerHTML = '<tr><td colspan="5" style="padding:32px 10px; text-align:center; color:#8691a8;">Henüz karşılaştırma verisi yok.</td></tr>';
      return;
    }
    const rows = codes.map(code => {
      const desc = escapeHtml(getPrimaryDescription(code));
      return `<tr>
        <td style="padding:10px;border-bottom:1px solid #edf1fb;font-weight:600;color:#1b2437;">${escapeHtml(code)}</td>
        <td style="padding:10px;border-bottom:1px solid #edf1fb;color:#3a4155;">${desc || '<span style="color:#9ba5c4;">Tanım yok</span>'}</td>
        <td style="padding:10px;border-bottom:1px solid #edf1fb;vertical-align:top;">${renderMatchCell('v1', code)}</td>
        <td style="padding:10px;border-bottom:1px solid #edf1fb;vertical-align:top;">${renderMatchCell('v2', code)}</td>
        <td style="padding:10px;border-bottom:1px solid #edf1fb;vertical-align:top;">${renderMatchCell('v3', code)}</td>
      </tr>`;
    }).join('');
    tbody.innerHTML = rows;
  }

  function getPrimaryDescription(code){
    for (const version of Object.keys(results)){
      for (const source of Object.keys(results[version])){
        const entry = findThemeEntry(results[version][source], code);
        if (entry){
          const desc = extractThemeDescription(entry);
          if (desc) return desc;
        }
      }
    }
    return '';
  }

  function registerComparisonSelectors(){
    document.querySelectorAll('[data-comparison-version]').forEach(select => {
      const version = select.getAttribute('data-comparison-version');
      if (!version || !comparisonSelections[version]) return;
      select.value = comparisonSelections[version];
      select.addEventListener('change', () => {
        comparisonSelections[version] = select.value;
        renderMappingComparisonTable();
      });
    });
  }

  // expose init (loader calls window.initTabClustering)
  window.initTabClustering = initTabClustering;

})();
