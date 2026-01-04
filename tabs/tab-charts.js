// Per-tab JS for tab-charts
function initTabCharts() {
    console.log('initTabCharts called');
    // Place tab-specific initialization here if needed
}

// Provide chart implementations and expose generateGDELTCharts
(function(){
    // ensure global chart arrays exist
    window.gdeltChartInstances = window.gdeltChartInstances || [];
    window.clusteringChartInstances = window.clusteringChartInstances || [];

    // IPTC renk haritas (basic fallback)
    const iptcColors = {
        'economy, business and finance': '#2196F3',
        'conflict, war and peace': '#f44336',
        'politics and government': '#0f3460',
        'health': '#4CAF50',
        'education': '#9C27B0',
        'disaster, accident and emergency incident': '#FF9800',
        'environment': '#00BCD4',
        'human interest': '#FF5722',
        'sport': '#795548',
        'science and technology': '#3F51B5',
        'society': '#607D8B',
        'lifestyle and leisure': '#8BC34A',
        'labour': '#FFC107',
        'religion': '#673AB7',
        'arts, culture, entertainment and media': '#E91E63',
        'weather': '#03A9F4'
    };

    function getIPTCCategory(themeCode) {
        if (typeof window.getIPTCCategory === 'function') return window.getIPTCCategory(themeCode);
        return null;
    }

    window.generateGDELTCharts = function generateGDELTCharts() {
        const gdeltTotalDocs = window.gdeltTotalDocs || [];
        if (!gdeltTotalDocs || gdeltTotalDocs.length === 0) {
            const statusEl = document.getElementById('status-message') || document.getElementById('config-status');
            if (statusEl) statusEl.textContent = '❌ Grafik oluşturmak için önce GDELT Tema sekmesinden CSV verilerini yükleyin!';
            return;
        }

        // destroy existing
        window.gdeltChartInstances.forEach(c => { try { c.destroy(); } catch(e){} });
        window.gdeltChartInstances = [];

        // create charts
        createIPTCDistributionChart();
        createCountryThemeChart();
        createTopThemesChart();
        createIPTCSimilarityChart();

        const statusEl = document.getElementById('status-message') || document.getElementById('config-status');
        if (statusEl) statusEl.textContent = '✅ GDELT grafikleri oluşturuldu!';
    };

    function createIPTCDistributionChart() {
        const canvas = document.getElementById('iptc-distribution-chart');
        if (!canvas || typeof Chart === 'undefined') return;
        const ctx = canvas.getContext('2d');
        const gdeltTotalDocs = window.gdeltTotalDocs || [];

        const iptcTotals = {};
        gdeltTotalDocs.forEach(item => {
            const iptc = getIPTCCategory(item.theme_code);
            if (iptc) {
                iptcTotals[iptc] = (iptcTotals[iptc] || 0) + (item.total_docs || 0);
            }
        });

        const sorted = Object.entries(iptcTotals).sort((a,b)=>b[1]-a[1]).slice(0,10);
        const labels = sorted.map(([cat]) => cat.split(',')[0].trim());
        const data = sorted.map(([,val])=>val);
        const colors = sorted.map(([cat])=>iptcColors[cat] || '#999');

        const chart = new Chart(ctx, {
            type: 'bar',
            data: { labels, datasets: [{ label: 'Toplam Doküman Sayısı', data, backgroundColor: colors.map(c=>c+'CC'), borderColor: colors, borderWidth:2 }] },
            options: { responsive:true, maintainAspectRatio:false, plugins:{ title:{ display:true, text:'IPTC Kategori Bazlı Doküman Dağılımı', font:{size:14}}, legend:{display:false}}, scales:{ y:{ beginAtZero:true, title:{display:true, text:'Doküman Sayısı'}}, x:{ ticks:{ maxRotation:45, minRotation:45 }, title:{ display:true, text:'IPTC Kategorisi'} } } }
        });
        window.gdeltChartInstances.push(chart);
    }

    function createCountryThemeChart() {
        const canvas = document.getElementById('country-theme-chart');
        if (!canvas || typeof Chart === 'undefined') return;
        const ctx = canvas.getContext('2d');
        const gdeltTotalDocs = window.gdeltTotalDocs || [];

        const countryTotals = {};
        gdeltTotalDocs.forEach(item => {
            const country = item.country || 'Unknown';
            countryTotals[country] = (countryTotals[country] || 0) + (item.total_docs || 0);
        });

        const sorted = Object.entries(countryTotals).sort((a,b)=>b[1]-a[1]).slice(0,15);
        const labels = sorted.map(([country])=>country);
        const data = sorted.map(([,val])=>val);

        const chart = new Chart(ctx, {
            type: 'bar',
            data: { labels, datasets: [{ label: 'Toplam Doküman', data, backgroundColor: 'rgba(33,150,243,0.7)', borderColor:'rgba(33,150,243,1)', borderWidth:1 }] },
            options: { indexAxis:'y', responsive:true, maintainAspectRatio:false, plugins:{ title:{ display:true, text:'Ülke Bazlı Toplam Tema Hacmi', font:{size:14}}, legend:{display:false}}, scales:{ x:{ beginAtZero:true, title:{ display:true, text:'Toplam Doküman' } }, y:{ title:{ display:true, text:'Ülke' } } } }
        });
        window.gdeltChartInstances.push(chart);
    }

    function createTopThemesChart() {
        const canvas = document.getElementById('top-themes-chart');
        if (!canvas || typeof Chart === 'undefined') return;
        const ctx = canvas.getContext('2d');
        const gdeltTotalDocs = window.gdeltTotalDocs || [];

        const themeTotals = {};
        gdeltTotalDocs.forEach(item => {
            const theme = item.theme_code || item.theme || 'Unknown';
            themeTotals[theme] = (themeTotals[theme] || 0) + (item.total_docs || 0);
        });
        const sorted = Object.entries(themeTotals).sort((a,b)=>b[1]-a[1]).slice(0,20);
        const labels = sorted.map(([t])=>t);
        const data = sorted.map(([,v])=>v);

        const chart = new Chart(ctx, {
            type: 'bar',
            data: { labels, datasets: [{ label:'Toplam Doküman', data, backgroundColor:'rgba(255,87,34,0.7)', borderColor:'rgba(255,87,34,1)', borderWidth:1 }] },
            options: { indexAxis:'y', responsive:true, maintainAspectRatio:false, plugins:{ title:{ display:true, text:'En Yüksek Hacimli Temalar', font:{size:14}}, legend:{display:false}}, scales:{ x:{ beginAtZero:true }, y:{ ticks:{ autoSkip:false } } } }
        });
        window.gdeltChartInstances.push(chart);
    }

    function createIPTCSimilarityChart() {
        const canvas = document.getElementById('iptc-similarity-chart');
        if (!canvas || typeof Chart === 'undefined') return;
        const ctx = canvas.getContext('2d');
        const chart = new Chart(ctx, { type:'bar', data:{ labels:['No data'], datasets:[{ label:'Similarity', data:[0], backgroundColor:['#ccc'] }] }, options:{ responsive:true, maintainAspectRatio:false, plugins:{ legend:{display:false}} } });
        window.gdeltChartInstances.push(chart);
    }

    // expose
    window.createIPTCDistributionChart = createIPTCDistributionChart;
    window.createCountryThemeChart = createCountryThemeChart;
    window.createTopThemesChart = createTopThemesChart;
    window.createIPTCSimilarityChart = createIPTCSimilarityChart;

    window.initTabCharts = function initTabCharts() { console.log('initTabCharts called'); };

})();
