// Per-tab JS for tab-export
function initTabExport() {
    console.log('initTabExport called');

    // Column checkbox listeners
    const checkboxes = document.querySelectorAll('.export-col-checkbox');
    checkboxes.forEach(cb => cb.addEventListener('change', updateSelectedColumnCount));
    updateSelectedColumnCount();

    // Select / Deselect buttons
    const selectAllBtn = document.getElementById('select-all-export-cols-btn');
    const deselectAllBtn = document.getElementById('deselect-all-export-cols-btn');
    if (selectAllBtn) selectAllBtn.addEventListener('click', selectAllExportCols);
    if (deselectAllBtn) deselectAllBtn.addEventListener('click', deselectAllExportCols);

    // Export buttons (call global implementations if present)
    const exportXLSX = document.getElementById('export-xlsx-btn');
    const exportCSV = document.getElementById('export-csv-btn');
    const exportLatex = document.getElementById('export-latex-btn');
    if (exportXLSX) exportXLSX.addEventListener('click', () => callIfExists('exportGDELTDataXLSX'));
    if (exportCSV) exportCSV.addEventListener('click', () => callIfExists('exportGDELTDataCSV'));
    if (exportLatex) exportLatex.addEventListener('click', () => callIfExists('showGDELTLatexTable'));

    // LaTeX copy
    const copyLatexBtn = document.getElementById('copy-latex-table-btn');
    if (copyLatexBtn) copyLatexBtn.addEventListener('click', copyLatexTable);

    // IPTC export buttons
    const exportIptcJson = document.getElementById('export-iptc-json-btn');
    const exportIptcCsv = document.getElementById('export-iptc-csv-btn');
    if (exportIptcJson) exportIptcJson.addEventListener('click', () => callIfExists('exportIPTCMappingJSON'));
    if (exportIptcCsv) exportIptcCsv.addEventListener('click', () => callIfExists('exportIPTCMappingCSV'));

    // Chart export buttons
    const exportChartPNGBtn = document.getElementById('export-chart-png-btn');
    const exportChartSVGBtn = document.getElementById('export-chart-svg-btn');
    if (exportChartPNGBtn) exportChartPNGBtn.addEventListener('click', () => callIfExists('exportChartPNG'));
    if (exportChartSVGBtn) exportChartSVGBtn.addEventListener('click', () => callIfExists('exportChartSVG'));

    // IPTC treemap and copy
    const copyIptcLatexBtn = document.getElementById('copy-iptc-latex-btn');
    if (copyIptcLatexBtn) copyIptcLatexBtn.addEventListener('click', () => callIfExists('copyIPTCLatexCode'));

    // Bulk export buttons
    const exportAllXlsxBtn = document.getElementById('export-all-xlsx-btn');
    const exportFullReportBtn = document.getElementById('export-full-report-btn');
    const exportAllChartsPngBtn = document.getElementById('export-all-charts-png-btn');
    if (exportAllXlsxBtn) exportAllXlsxBtn.addEventListener('click', () => callIfExists('exportAllGDELTData'));
    if (exportFullReportBtn) exportFullReportBtn.addEventListener('click', () => callIfExists('exportFullReport'));
    if (exportAllChartsPngBtn) exportAllChartsPngBtn.addEventListener('click', () => callIfExists('exportAllChartsPNG'));
}

// Helper: call global function by name if it exists, otherwise log a warning
function callIfExists(fnName) {
    try {
        const fn = window[fnName];
        if (typeof fn === 'function') return fn();
        console.warn(`${fnName} is not defined`);
        if (typeof window.showStatus === 'function') window.showStatus(`${fnName} bulunamadı.`, 'error');
    } catch (e) {
        console.error('callIfExists error', e);
    }
}

// Column selection helpers
function updateSelectedColumnCount() {
    const checked = document.querySelectorAll('.export-col-checkbox:checked').length;
    const total = document.querySelectorAll('.export-col-checkbox').length;
    const countEl = document.getElementById('selected-columns-count');
    if (countEl) countEl.textContent = `${checked}/${total} sütun seçili`;
}

function selectAllColumns() {
    document.querySelectorAll('.export-col-checkbox').forEach(cb => cb.checked = true);
    updateSelectedColumnCount();
}

function deselectAllColumns() {
    document.querySelectorAll('.export-col-checkbox').forEach(cb => cb.checked = false);
    updateSelectedColumnCount();
}

function selectAllExportCols() { selectAllColumns(); }
function deselectAllExportCols() { deselectAllColumns(); }

function getSelectedGDELTColumns() {
    const checkboxes = document.querySelectorAll('.export-col-checkbox:checked');
    return Array.from(checkboxes).map(cb => cb.value);
}

// Copy LaTeX table code to clipboard
function copyLatexTable() {
    const ta = document.getElementById('latex-table-code');
    if (!ta) return;
    const text = ta.value || '';
    if (!navigator.clipboard) {
        // fallback
        ta.select();
        document.execCommand('copy');
        if (typeof window.showStatus === 'function') window.showStatus('LaTeX kodu kopyalandı', 'success');
        return;
    }
    navigator.clipboard.writeText(text).then(() => {
        if (typeof window.showStatus === 'function') window.showStatus('LaTeX kodu kopyalandı', 'success');
    }).catch(err => console.error('copyLatexTable failed', err));
}

// If copyIPTCLatexCode not implemented elsewhere, provide a simple implementation
function copyIPTCLatexCode() {
    const preview = document.getElementById('iptc-latex-preview');
    if (!preview) return;
    const text = preview.innerText || preview.textContent || '';
    if (navigator.clipboard) {
        navigator.clipboard.writeText(text).then(() => {
            if (typeof window.showStatus === 'function') window.showStatus('IPTC LaTeX kodu kopyalandı', 'success');
        }).catch(err => console.error('copyIPTCLatexCode failed', err));
    } else {
        const ta = document.createElement('textarea');
        ta.value = text; document.body.appendChild(ta); ta.select(); document.execCommand('copy'); ta.remove();
        if (typeof window.showStatus === 'function') window.showStatus('IPTC LaTeX kodu kopyalandı', 'success');
    }
}

// --- Export tab helpers: provide implementations for missing functions referenced by tab HTML ---
// These try to call existing global implementations first; otherwise provide safe fallbacks

function generateIPTCTreemap() {
    if (typeof window.generateIPTCTreemap === 'function' && window.generateIPTCTreemap !== generateIPTCTreemap) {
        return window.generateIPTCTreemap();
    }
    const canvasId = 'iptc-treemap-canvas';
    const canvas = document.getElementById(canvasId);
    if (!canvas) return console.warn('No canvas for IPTC treemap');
    if (typeof Chart === 'undefined') {
        if (typeof window.showStatus === 'function') window.showStatus('Chart.js bulunamadı; treemap çizilemiyor', 'error');
        return;
    }

    // Build category volumes from available data
    const totals = {};
    const rows = window.gdeltTotalDocs || [];
    const getCat = window.getIPTCCategory || ((t) => t);
    rows.forEach(r => {
        const cat = getCat(r.theme_code) || 'unknown';
        totals[cat] = (totals[cat] || 0) + (parseInt(r.total_docs) || 0);
    });

    const labels = Object.keys(totals).sort((a,b)=>totals[b]-totals[a]);
    const data = labels.map(l => totals[l]);
    // destroy existing instance on same canvas
    if (canvas._chartInstance) canvas._chartInstance.destroy();
    canvas._chartInstance = new Chart(canvas.getContext('2d'), {
        type: 'pie',
        data: { labels, datasets: [{ data, backgroundColor: labels.map((_,i)=>getColor(i)) }] },
        options: { responsive: true, plugins: { title: { display: true, text: 'IPTC Kategori Hacmi (treemap fallback: pie)' } } }
    });
    if (typeof window.showStatus === 'function') window.showStatus('IPTC treemap oluşturuldu (pie fallback)', 'success');
}

function generateCountryCategoryChart() {
    if (typeof window.generateCountryCategoryChart === 'function' && window.generateCountryCategoryChart !== generateCountryCategoryChart) {
        return window.generateCountryCategoryChart();
    }
    const canvasId = 'country-category-canvas';
    const canvas = document.getElementById(canvasId);
    if (!canvas) return console.warn('No canvas for country-category');
    if (typeof Chart === 'undefined') {
        if (typeof window.showStatus === 'function') window.showStatus('Chart.js bulunamadı; grafik çizilemiyor', 'error');
        return;
    }

    // Build top categories by country from gdeltTotalDocs
    const rows = window.gdeltTotalDocs || [];
    // rows: { country, theme_code, total_docs }
    const byCountry = {};
    const getCat = window.getIPTCCategory || ((t) => t);
    rows.forEach(r => {
        const c = r.country || 'UNK';
        const cat = getCat(r.theme_code) || 'unknown';
        byCountry[c] = byCountry[c] || {};
        byCountry[c][cat] = (byCountry[c][cat] || 0) + (parseInt(r.total_docs) || 0);
    });

    const countries = Object.keys(byCountry).sort((a,b)=>{
        const sa = Object.values(byCountry[a]).reduce((s,x)=>s+x,0);
        const sb = Object.values(byCountry[b]).reduce((s,x)=>s+x,0);
        return sb - sa;
    }).slice(0,6);
    const categories = Array.from(new Set([].concat(...countries.map(c=>Object.keys(byCountry[c]||{}))))).slice(0,8);

    const datasets = categories.map((cat, idx) => ({
        label: cat,
        data: countries.map(c => byCountry[c][cat] || 0),
        backgroundColor: getColor(idx, 0.7)
    }));

    if (canvas._chartInstance) canvas._chartInstance.destroy();
    canvas._chartInstance = new Chart(canvas.getContext('2d'), {
        type: 'bar',
        data: { labels: countries, datasets },
        options: { responsive: true, scales: { x: { stacked: true }, y: { stacked: true } }, plugins: { title: { display: true, text: 'Ülke - Kategori Dağılımı' } } }
    });
    if (typeof window.showStatus === 'function') window.showStatus('Ülke-kategori grafiği oluşturuldu', 'success');
}

function exportCountryCategoryLaTeX() {
    // Create a simple LaTeX table from current country-category aggregation
    const rows = window.gdeltTotalDocs || [];
    const getCat = window.getIPTCCategory || ((t) => t);
    const byCountry = {};
    rows.forEach(r => {
        const c = r.country || 'UNK';
        const cat = getCat(r.theme_code) || 'unknown';
        byCountry[c] = byCountry[c] || {};
        byCountry[c][cat] = (byCountry[c][cat] || 0) + (parseInt(r.total_docs) || 0);
    });

    const countries = Object.keys(byCountry).sort();
    const categories = Array.from(new Set([].concat(...countries.map(c=>Object.keys(byCountry[c]||{}))))).sort();

    let latex = '\\begin{tabular}{l' + 'r'.repeat(categories.length) + '}\\\n';
    latex += categories.map(c=>escapeLaTeX(c)).join(' & ') + ' \\\\ \hline\\\n';
    countries.forEach(country => {
        const line = [escapeLaTeX(country)].concat(categories.map(cat => byCountry[country][cat] || 0));
        latex += line.join(' & ') + ' \\\\ \\\\n+';
    });
    latex += '\\end{tabular}';

    const ta = document.getElementById('latex-table-code');
    if (ta) { ta.value = latex; ta.style.display = 'block'; ta.scrollIntoView(); }
    if (typeof navigator.clipboard !== 'undefined') navigator.clipboard.writeText(latex).catch(()=>{});
    if (typeof window.showStatus === 'function') window.showStatus('Ülke-kategori LaTeX kodu hazırlandı ve panoya kopyalandı', 'success');
}

// small helpers
function getColor(index, alpha=1) {
    const palette = [ '#4CAF50','#FF9800','#9C27B0','#2196F3','#E91E63','#00BCD4','#FFC107','#8BC34A','#607D8B' ];
    const c = palette[index % palette.length] || '#777';
    if (alpha === 1) return c;
    // add alpha
    const hex = c.replace('#','');
    const r = parseInt(hex.substring(0,2),16);
    const g = parseInt(hex.substring(2,4),16);
    const b = parseInt(hex.substring(4,6),16);
    return `rgba(${r},${g},${b},${alpha})`;
}

function escapeLaTeX(s) {
    if (!s) return '';
    return String(s).replace(/([\\%$#_{}~^\\])/g, '\\$1');
}

// ------------------ Additional charts: Category Volume, V1-V2 diff, Time Series ------------------

function generateIPTCCategoryVolumeChart() {
    if (typeof window.generateIPTCCategoryVolumeChart === 'function' && window.generateIPTCCategoryVolumeChart !== generateIPTCCategoryVolumeChart) {
        return window.generateIPTCCategoryVolumeChart();
    }
    const canvasId = 'iptc-category-volume-canvas';
    const canvas = document.getElementById(canvasId);
    if (!canvas) return console.warn('No canvas for IPTC category volume');
    if (typeof Chart === 'undefined') return window.showStatus?.('Chart.js bulunamadı; grafik çizilemiyor', 'error');

    const rows = window.gdeltTotalDocs || [];
    const getCat = window.getIPTCCategory || ((t) => t);
    const totals = {};
    rows.forEach(r => {
        const cat = getCat(r.theme_code) || 'unknown';
        totals[cat] = (totals[cat] || 0) + (parseInt(r.total_docs) || 0);
    });
    const labels = Object.keys(totals).sort((a,b)=>totals[b]-totals[a]);
    const data = labels.map(l => totals[l]);

    if (canvas._chartInstance) canvas._chartInstance.destroy();
    canvas._chartInstance = new Chart(canvas.getContext('2d'), {
        type: 'bar',
        data: { labels, datasets:[{ label: 'Doküman Hacmi', data, backgroundColor: labels.map((_,i)=>getColor(i)) }] },
        options: { indexAxis: 'y', responsive: true, plugins: { title: { display: true, text: 'IPTC Kategori Hacmi' } } }
    });
    window.showStatus?.('IPTC kategori hacmi grafiği oluşturuldu','success');
}

function generateV1V2DifferenceChart() {
    if (typeof window.generateV1V2DifferenceChart === 'function' && window.generateV1V2DifferenceChart !== generateV1V2DifferenceChart) {
        return window.generateV1V2DifferenceChart();
    }
    const canvasId = 'v1v2-difference-canvas';
    const canvas = document.getElementById(canvasId);
    if (!canvas) return console.warn('No canvas for V1V2 difference');
    if (typeof Chart === 'undefined') return window.showStatus?.('Chart.js bulunamadı; grafik çizilemiyor', 'error');

    const v1 = window.iptcMappingResultsV1?.mappings || [];
    const v2 = window.iptcMappingResultsV2?.mappings || [];
    const countByCat = {};
    v1.forEach(m => { countByCat[m.iptc_category] = countByCat[m.iptc_category] || {v1:0,v2:0}; countByCat[m.iptc_category].v1 += 1; });
    v2.forEach(m => { countByCat[m.iptc_category] = countByCat[m.iptc_category] || {v1:0,v2:0}; countByCat[m.iptc_category].v2 += 1; });

    const labels = Object.keys(countByCat).sort((a,b)=> (countByCat[b].v2 - countByCat[b].v1) - (countByCat[a].v2 - countByCat[a].v1)).slice(0,20);
    const v1data = labels.map(l => countByCat[l].v1 || 0);
    const v2data = labels.map(l => countByCat[l].v2 || 0);

    if (canvas._chartInstance) canvas._chartInstance.destroy();
    canvas._chartInstance = new Chart(canvas.getContext('2d'), {
        type: 'bar',
        data: { labels, datasets: [ { label: 'V1', data: v1data, backgroundColor: getColor(0,0.6) }, { label: 'V2', data: v2data, backgroundColor: getColor(1,0.6) } ] },
        options: { responsive: true, plugins: { title: { display: true, text: 'V1 vs V2 Tema Sayısı (kategori bazında)' } }, scales: { x: { stacked: false }, y: { beginAtZero: true } } }
    });
    window.showStatus?.('V1-V2 karşılaştırma grafiği oluşturuldu','success');
}

function generateTimeSeriesChart() {
    if (typeof window.generateTimeSeriesChart === 'function' && window.generateTimeSeriesChart !== generateTimeSeriesChart) {
        return window.generateTimeSeriesChart();
    }
    const canvasId = 'time-series-canvas';
    const canvas = document.getElementById(canvasId);
    if (!canvas) return console.warn('No canvas for time series');
    if (typeof Chart === 'undefined') return window.showStatus?.('Chart.js bulunamadı; grafik çizilemiyor', 'error');

    const rows = window.gdeltMonthlyDetail || [];
    // rows: { ym, country, theme_code, n_docs }
    const byMonth = {};
    rows.forEach(r => { const m = r.ym || r.ymd || r.month || 'unk'; byMonth[m] = (byMonth[m]||0) + (parseInt(r.n_docs)||0); });
    const labels = Object.keys(byMonth).sort();
    const data = labels.map(l => byMonth[l]);

    if (canvas._chartInstance) canvas._chartInstance.destroy();
    canvas._chartInstance = new Chart(canvas.getContext('2d'), {
        type: 'line',
        data: { labels, datasets: [{ label: 'Toplam Doküman', data, borderColor: getColor(2), backgroundColor: getColor(2,0.2), fill:true }] },
        options: { responsive: true, plugins: { title: { display: true, text: 'Zaman Serileri (Toplam Doküman/ay)' } }, scales: { x: { display: true }, y: { beginAtZero:true } } }
    });
    window.showStatus?.('Zaman serisi grafiği oluşturuldu','success');
}

// Simple canvas export helpers (PNG, fallback for SVG)
function exportCanvasPNG(canvasId, filename) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return window.showStatus?.('İlgili grafik bulunamadı','error');
    try {
        const url = canvas.toDataURL('image/png');
        const a = document.createElement('a'); a.href = url; a.download = filename || 'chart.png'; a.click();
        window.showStatus?.('Grafik PNG olarak indirildi','success');
    } catch (e) { console.error(e); window.showStatus?.('PNG oluşturulamadı','error'); }
}

function exportCanvasSVG(canvasId, filename) {
    // Chart.js uses canvas; produce PNG and warn about SVG fallback
    window.showStatus?.('SVG export desteklenmiyor; PNG indiriliyor yerine','info');
    exportCanvasPNG(canvasId, filename?.replace(/\.svg$/, '.png'));
}

// Expose export button handlers used in HTML
function exportIPTCCategoryVolumePNG() { exportCanvasPNG('iptc-category-volume-canvas','iptc_category_volume.png'); }
function exportIPTCCategoryVolumeSVG() { exportCanvasSVG('iptc-category-volume-canvas','iptc_category_volume.svg'); }
function exportV1V2DifferencePNG() { exportCanvasPNG('v1v2-difference-canvas','v1v2_difference.png'); }
function exportV1V2DifferenceSVG() { exportCanvasSVG('v1v2-difference-canvas','v1v2_difference.svg'); }
function exportTimeSeriesPNG() { exportCanvasPNG('time-series-canvas','time_series.png'); }
function exportTimeSeriesSVG() { exportCanvasSVG('time-series-canvas','time_series.svg'); }


