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
    
    // IPTC volume version selector
    const volumeSel = document.getElementById('iptc-volume-version-select');
    if (volumeSel) volumeSel.addEventListener('change', () => selectIptcVolumeVersion(volumeSel.value));
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

// Set active mapping version/source for volume chart and downstream lookups
function selectIptcVolumeVersion(val) {
    if (!val) return;
    const [version, source] = val.split('-');
    // update global active mapping info used by getIPTCCategory
    window.activeMappingInfo = { version, source };
    window.activeMapping = `${version}-${source}`;
    // attempt to use existing switchActiveMapping for UI consistency
    try {
        if (typeof window.switchActiveMapping === 'function') {
            const sel = document.getElementById('active-mapping-select');
            if (sel) {
                sel.value = `${version}-${source}`;
                window.switchActiveMapping();
                return;
            }
        }
    } catch (e) { /* ignore and fall back */ }
    // fallback: ensure lookups are built if cached results exist
    try {
        if (window.iptcMappingLookupByVersion && window.iptcMappingLookupByVersion[version] && window.iptcMappingLookupByVersion[version][source]) {
            const lookup = window.iptcMappingLookupByVersion[version][source];
            window.iptcMappingLookup = lookup;
        }
    } catch (e) { /* ignore */ }
    try { generateIPTCCategoryVolumeChart(); } catch (e) { console.warn('selectIptcVolumeVersion regenerate', e); }
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
    // Build stacked ybar PGFPlots figure for country-category distribution
    const rows = window.gdeltTotalDocs || [];
    if (!rows.length) return window.showStatus?.('Ülke-kategori verisi bulunamadı', 'error');
    const getCat = window.getIPTCCategory || ((t) => t);
    const byCountry = {};
    rows.forEach(r => {
        const c = (r.country || 'UNK').trim();
        const cat = getCat(r.theme_code) || 'unknown';
        byCountry[c] = byCountry[c] || {};
        byCountry[c][cat] = (byCountry[c][cat] || 0) + (parseInt(r.total_docs) || 0);
    });

    // choose top 6 countries by total docs
    const countryTotals = Object.entries(byCountry).map(([c, cats]) => [c, Object.values(cats).reduce((s,x)=>s+x,0)]);
    const topCountries = countryTotals.sort((a,b)=>b[1]-a[1]).slice(0,6).map(([c])=>c);
    const categories = Array.from(new Set([].concat(...topCountries.map(c=>Object.keys(byCountry[c]||{})))));

    const palette = [ 'green!60', 'orange!70', 'purple!60', 'blue!60', 'red!60', 'cyan!60', 'yellow!70', 'lime!60', 'teal!60', 'magenta!50' ];
    const catSorted = categories.sort((a,b)=>a.localeCompare(b));

    const coordsFor = (cat) => topCountries.map(c => `(${escapeLaTeX(c)},${byCountry[c][cat] || 0})`).join(' ');
    const xcoords = topCountries.map(c => escapeLaTeX(c)).join(', ');

    let latex = '% Country-Category stacked bar (pgfplots)\n';
    latex += '% Requires: \\usepackage{pgfplots}\n';
    latex += '% \\pgfplotsset{compat=1.18}\n\n';
    latex += '\\begin{figure}[t]\n';
    latex += '\\centering\n';
    latex += '\\begin{tikzpicture}\n';
    latex += '\\begin{axis}[\n';
    latex += '    ybar stacked,\n';
    latex += '    bar width=0.28cm,\n';
    latex += '    width=\\columnwidth,\n';
    latex += '    height=0.85\\columnwidth,\n';
    latex += '    xlabel={Countries},\n';
    latex += '    ylabel={Documents},\n';
    latex += '    symbolic x coords={' + xcoords + '},\n';
    latex += '    xtick=data,\n';
    latex += '    ymin=0,\n';
    latex += '    enlarge x limits=0.15,\n';
    latex += '    title={Country--Category Distribution},\n';
    latex += '    title style={font=\\small},\n';
    latex += '    legend style={\n';
    latex += '        at={(0.5,1.05)},\n';
    latex += '        anchor=south,\n';
    latex += '        legend columns=2,\n';
    latex += '        font=\\scriptsize,\n';
    latex += '        draw=none,\n';
    latex += '        /tikz/every even column/.append style={column sep=6pt}\n';
    latex += '    },\n';
    latex += '    tick label style={font=\\scriptsize},\n';
    latex += '    y tick label style={font=\\scriptsize, /pgf/number format/fixed}\n';
    latex += ']\n\n';

    catSorted.forEach((cat, idx) => {
        const color = palette[idx % palette.length];
        latex += `% ${escapeLaTeX(cat)}\n`;
        latex += `\\addplot[fill=${color}] coordinates {\n${coordsFor(cat)}\n};\n`;
        latex += `\\addlegendentry{${escapeLaTeX(cat)}}\n\n`;
    });

    latex += '\\end{axis}\n';
    latex += '\\end{tikzpicture}\n';
    latex += '\\caption{Country-level IPTC category distribution (stacked).}\n';
    latex += '\\label{fig:country-category-stacked}\n';
    latex += '\\end{figure}\n';

    const ta = document.getElementById('latex-table-code');
    const preview = document.getElementById('latex-table-preview');
    if (ta && preview) {
        ta.value = latex;
        preview.style.display = 'block';
        ta.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
    if (typeof navigator.clipboard !== 'undefined') navigator.clipboard.writeText(latex).catch(()=>{});
    window.showStatus?.('Ülke-kategori LaTeX grafiği hazırlandı', 'success');
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
    return String(s).replace(/([\\%$#_{}~^&])/g, '\\$1');
}

// Shared helper to aggregate per-category counts across V1/V2/V3 mapping results
function collectVersionDiffData(limit = 20) {
    const toArray = (payload) => {
        if (!payload) return [];
        if (Array.isArray(payload)) return payload;
        if (Array.isArray(payload.mappings)) return payload.mappings;
        if (Array.isArray(payload.themes)) return payload.themes;
        if (Array.isArray(payload.results)) return payload.results;
        return [];
    };

    const v1Raw = toArray(window.iptcMappingResultsV1);
    const v2Raw = toArray(window.iptcMappingResultsV2);
    const v3Raw = toArray(window.iptcMappingResultsV3);

    const pickCat = (m) => m?.iptc_category || m?.iptc_label || m?.iptc_final_label || m?.iptc_final || m?.category || 'unknown';

    const countByCat = {};
    const bump = (arr, key) => arr.forEach(m => {
        const cat = pickCat(m);
        countByCat[cat] = countByCat[cat] || { v1: 0, v2: 0, v3: 0 };
        countByCat[cat][key] += 1;
    });

    bump(v1Raw, 'v1');
    bump(v2Raw, 'v2');
    bump(v3Raw, 'v3');

    const labels = Object.keys(countByCat).sort((a, b) => {
        const aVals = countByCat[a];
        const bVals = countByCat[b];
        const aRange = Math.max(aVals.v1, aVals.v2, aVals.v3) - Math.min(aVals.v1, aVals.v2, aVals.v3);
        const bRange = Math.max(bVals.v1, bVals.v2, bVals.v3) - Math.min(bVals.v1, bVals.v2, bVals.v3);
        if (bRange !== aRange) return bRange - aRange;
        const aTotal = aVals.v1 + aVals.v2 + aVals.v3;
        const bTotal = bVals.v1 + bVals.v2 + bVals.v3;
        return bTotal - aTotal;
    }).slice(0, limit);

    return { labels, countByCat, sourceSizes: { v1: v1Raw.length, v2: v2Raw.length, v3: v3Raw.length } };
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

    const { labels, countByCat, sourceSizes } = collectVersionDiffData();
    if (!labels.length) {
        return window.showStatus?.('V1/V2/V3 sonuçları bulunamadı', 'error');
    }

    const v1data = labels.map(l => countByCat[l].v1 || 0);
    const v2data = labels.map(l => countByCat[l].v2 || 0);
    const v3data = labels.map(l => countByCat[l].v3 || 0);

    if (canvas._chartInstance) canvas._chartInstance.destroy();
    canvas._chartInstance = new Chart(canvas.getContext('2d'), {
        type: 'bar',
        data: {
            labels,
            datasets: [
                { label: 'V1', data: v1data, backgroundColor: getColor(0, 0.6) },
                { label: 'V2', data: v2data, backgroundColor: getColor(1, 0.6) },
                { label: 'V3', data: v3data, backgroundColor: getColor(2, 0.6) }
            ]
        },
        options: {
            responsive: true,
            plugins: { title: { display: true, text: 'V1 vs V2 vs V3 Tema Sayısı (kategori bazında)' } },
            scales: { x: { stacked: false }, y: { beginAtZero: true } }
        }
    });

    const loadedVersions = ['v1','v2','v3'].filter(v => sourceSizes[v]);
    const statusMsg = loadedVersions.length ? `V1-V2-V3 karşılaştırma grafiği oluşturuldu (${loadedVersions.join(', ')} yüklü)` : 'V1-V2-V3 karşılaştırma grafiği oluşturuldu';
    window.showStatus?.(statusMsg, 'success');
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

// Export LaTeX for IPTC treemap (category totals)
function exportIPTCTreemapLaTeX() {
    const rows = window.gdeltTotalDocs || [];
    if (!rows.length) return window.showStatus?.('IPTC verisi bulunamadı', 'error');
    const getCat = window.getIPTCCategory || ((t) => t);
    const totals = {};
    rows.forEach(r => {
        const cat = getCat(r.theme_code) || 'unknown';
        totals[cat] = (totals[cat] || 0) + (parseInt(r.total_docs) || 0);
    });
    const labels = Object.keys(totals).sort((a, b) => totals[b] - totals[a]);

    let latex = '\\begin{tabular}{lr}\\\nKategori & Dokuman \\\\ \\hline\\\n';
    labels.forEach(cat => { latex += `${escapeLaTeX(cat)} & ${totals[cat]} \\\\ \n`; });
    latex += '\\end{tabular}';

    const ta = document.getElementById('latex-table-code');
    const preview = document.getElementById('latex-table-preview');
    if (ta && preview) {
        ta.value = latex;
        preview.style.display = 'block';
        ta.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
    if (typeof navigator.clipboard !== 'undefined') navigator.clipboard.writeText(latex).catch(() => {});
    window.showStatus?.('IPTC treemap LaTeX tablosu hazırlandı', 'success');
}

// Export LaTeX (pgfplots area + line) for time series chart
function exportTimeSeriesLaTeX() {
    const rows = window.gdeltMonthlyDetail || [];
    if (!rows.length) return window.showStatus?.('Zaman serisi verisi bulunamadı', 'error');
    const byMonth = {};
    rows.forEach(r => { const m = r.ym || r.ymd || r.month || 'unk'; byMonth[m] = (byMonth[m]||0) + (parseInt(r.n_docs)||0); });
    const labels = Object.keys(byMonth).sort();
    const coords = labels.map((m, idx) => ({ x: idx + 1, y: (byMonth[m] || 0) / 1e6 }));

    const xTickLabels = labels.map(l => escapeLaTeX(l)).join(',');
    const coordStr = coords.map(c => `(${c.x},${c.y.toFixed(2)})`).join('\n');

    let latex = '% Monthly time series of total news documents (pgfplots)\n';
    latex += '% Requires: \\usepackage{pgfplots}\n';
    latex += '% \\pgfplotsset{compat=1.18}\n\n';
    latex += '\\begin{figure}[t]\n';
    latex += '\\centering\n';
    latex += '\\begin{tikzpicture}\n';
    latex += '\\begin{axis}[\n';
    latex += '    width=\\columnwidth,\n';
    latex += '    height=0.6\\columnwidth,\n';
    latex += '    xlabel={Month},\n';
    latex += '    ylabel={Documents (millions)},\n';
    latex += '    ymin=0,\n';
    latex += '    xtick=data,\n';
    latex += '    xticklabels={' + xTickLabels + '},\n';
    latex += '    x tick label style={\n        rotate=45,\n        anchor=east,\n        font=\\scriptsize\n    },\n';
    latex += '    yticklabel style={/pgf/number format/fixed, /pgf/number format/precision=1},\n';
    latex += '    grid=both,\n';
    latex += '    grid style={line width=.1pt, draw=gray!30},\n';
    latex += '    legend style={at={(0.5,1.05)}, anchor=south, font=\\scriptsize, draw=none}\n';
    latex += ']\n\n';

    latex += '% Filled area\n';
    latex += '\\addplot[fill=purple!25, draw=none] coordinates {\n' + coordStr + '\n} \\closedcycle;\n\n';

    latex += '% Line + markers\n';
    latex += '\\addplot[thick, color=purple, mark=*, mark size=1.8pt] coordinates {\n' + coordStr + '\n};\n';
    latex += '\\addlegendentry{Total Documents}\n\n';

    latex += '\\end{axis}\n';
    latex += '\\end{tikzpicture}\n';
    latex += '\\caption{Monthly time series of total news documents (2022--2024).}\n';
    latex += '\\label{fig:monthly-documents}\n';
    latex += '\\end{figure}\n';

    const ta = document.getElementById('latex-table-code');
    const preview = document.getElementById('latex-table-preview');
    if (ta && preview) {
        ta.value = latex;
        preview.style.display = 'block';
        ta.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
    if (typeof navigator.clipboard !== 'undefined') navigator.clipboard.writeText(latex).catch(() => {});
    window.showStatus?.('Zaman serisi LaTeX grafiği hazırlandı', 'success');
}

// Export LaTeX (pgfplots) for IPTC category volume bar chart
function exportIPTCCategoryVolumeLaTeX() {
    const rows = window.gdeltTotalDocs || [];
    if (!rows.length) return window.showStatus?.('IPTC verisi bulunamadı', 'error');

    const getCat = window.getIPTCCategory || ((t) => t);
    const totals = {};
    rows.forEach(r => {
        const cat = getCat(r.theme_code) || 'unknown';
        totals[cat] = (totals[cat] || 0) + (parseInt(r.total_docs) || 0);
    });
    const labels = Object.keys(totals).sort((a, b) => totals[b] - totals[a]);

    const coord = labels.map(cat => `(${escapeLaTeX(cat)}, ${totals[cat]})`).join('\n');
    const xcoords = labels.map(c => escapeLaTeX(c)).join(', \n        ');

    let latex = '% IPTC Category-Level Document Volume\n';
    latex += '% Requires: \\usepackage{pgfplots}\n';
    latex += '% \\pgfplotsset{compat=1.18}\n\n';
    latex += '\\begin{figure*}[t]\n';
    latex += '\\centering\n';
    latex += '\\begin{tikzpicture}\n';
    latex += '\\begin{axis}[\n';
    latex += '    ybar,\n';
    latex += '    bar width=0.65cm,\n';
    latex += '    width=\\textwidth,\n';
    latex += '    height=0.45\\textwidth,\n';
    latex += '    xlabel={IPTC Categories},\n';
    latex += '    ylabel={Total Number of Documents},\n';
    latex += '    symbolic x coords={\n        ' + xcoords + '\n    },\n';
    latex += '    xtick=data,\n';
    latex += '    ymin=0,\n';
    latex += '    enlarge x limits=0.1,\n';
    latex += '    title={Document Volume by IPTC Category},\n';
    latex += '    x tick label style={\n        rotate=30,\n        anchor=east,\n        font=\\scriptsize\n    }\n';
    latex += ']\n\n';

    latex += '\\addplot[fill=blue!60] coordinates {\n' + coord + '\n};\n\n';
    latex += '\\end{axis}\n';
    latex += '\\end{tikzpicture}\n';
    latex += '\\caption{Document Volume by IPTC Category}\n';
    latex += '\\label{fig:iptc-category-volume}\n';
    latex += '\\end{figure*}\n';

    const ta = document.getElementById('latex-table-code');
    const preview = document.getElementById('latex-table-preview');
    if (ta && preview) {
        ta.value = latex;
        preview.style.display = 'block';
        ta.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
    if (typeof navigator.clipboard !== 'undefined') navigator.clipboard.writeText(latex).catch(() => {});
    window.showStatus?.('IPTC hacim LaTeX grafiği hazırlandı', 'success');
}

// Export LaTeX table for V1-V2-V3 comparison
function exportV1V2DifferenceLaTeX() {
    const { labels, countByCat, sourceSizes } = collectVersionDiffData();
    if (!labels.length) return window.showStatus?.('V1/V2/V3 sonuçları bulunamadı', 'error');
    // Build pgfplots bar chart similar to legacy V1-V2 export
    const coord = (k, arr) => arr.map(cat => `(${escapeLaTeX(cat)}, ${countByCat[cat][k] || 0})`).join('\n');
    const xcoords = labels.map(c => escapeLaTeX(c)).join(',\n        ');

    let latex = '% V1 vs. V2 vs. V3 Theme Coverage Comparison\n';
    latex += '% Requires: \\usepackage{pgfplots}\n';
    latex += '% \\pgfplotsset{compat=1.18}\n\n';
    latex += '\\begin{figure*}[t]\n';
    latex += '\\centering\n';
    latex += '\\begin{tikzpicture}\n';
    latex += '\\begin{axis}[\n';
    latex += '    ybar,\n';
    latex += '    bar width=0.32cm,\n';
    latex += '    width=\\textwidth,\n';
    latex += '    height=0.45\\textwidth,\n';
    latex += '    xlabel={IPTC Categories},\n';
    latex += '    ylabel={Number of Themes},\n';
    latex += '    symbolic x coords={\n        ' + xcoords + '\n    },\n';
    latex += '    xtick=data,\n';
    latex += '    ymin=0,\n';
    latex += '    enlarge x limits=0.12,\n';
    latex += '    title={Comparison of Theme Coverage Between V1, V2 and V3},\n';
    latex += '    legend style={\n        at={(0.5,-0.25)},\n        anchor=north,\n        legend columns=3,\n        font=\\scriptsize,\n        draw=none\n    },\n';
    latex += '    x tick label style={\n        rotate=30,\n        anchor=east,\n        font=\\scriptsize\n    }\n';
    latex += ']\n\n';

    latex += '% V1\n';
    latex += '\\addplot[fill=blue!60] coordinates {\n' + coord('v1', labels) + '\n};\n';
    latex += '\\addlegendentry{V1}\n\n';

    latex += '% V2\n';
    latex += '\\addplot[fill=red!60] coordinates {\n' + coord('v2', labels) + '\n};\n';
    latex += '\\addlegendentry{V2}\n\n';

    latex += '% V3\n';
    latex += '\\addplot[fill=green!60] coordinates {\n' + coord('v3', labels) + '\n};\n';
    latex += '\\addlegendentry{V3}\n\n';

    latex += '\\end{axis}\n';
    latex += '\\end{tikzpicture}\n';
    latex += '\\caption{Comparison of Theme Coverage Between V1, V2 and V3 Algorithms}\n';
    latex += '\\label{fig:v1v2v3-difference}\n';
    latex += '\\end{figure*}\n';

    const ta = document.getElementById('latex-table-code');
    const preview = document.getElementById('latex-table-preview');
    if (ta && preview) {
        ta.value = latex;
        preview.style.display = 'block';
        ta.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
    if (typeof navigator.clipboard !== 'undefined') navigator.clipboard.writeText(latex).catch(() => {});

    const loadedVersions = ['v1','v2','v3'].filter(v => sourceSizes[v]);
    const statusMsg = loadedVersions.length ? `V1-V2-V3 LaTeX grafiği hazırlandı (${loadedVersions.join(', ')} yüklü)` : 'V1-V2-V3 LaTeX grafiği hazırlandı';
    window.showStatus?.(statusMsg, 'success');
}


