/**
 * Chart.js visualizations for benchmark results.
 * Expects `metricsData` to be set globally before this script loads.
 */
(function() {
    if (typeof metricsData === 'undefined' || !metricsData.pipelines) return;

    const slugs = metricsData.pipeline_slugs;
    const COLORS = [
        '#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4'
    ];

    // --- Accuracy bar chart ---
    const accCtx = document.getElementById('accuracy-chart');
    if (accCtx) {
        new Chart(accCtx, {
            type: 'bar',
            data: {
                labels: slugs,
                datasets: [
                    {
                        label: 'String Accuracy %',
                        data: slugs.map(s => metricsData.pipelines[s].string_accuracy),
                        backgroundColor: slugs.map((_, i) => COLORS[i % COLORS.length] + 'cc'),
                        borderColor: slugs.map((_, i) => COLORS[i % COLORS.length]),
                        borderWidth: 1,
                    },
                    {
                        label: 'Char Accuracy %',
                        data: slugs.map(s => metricsData.pipelines[s].char_accuracy),
                        backgroundColor: slugs.map((_, i) => COLORS[i % COLORS.length] + '66'),
                        borderColor: slugs.map((_, i) => COLORS[i % COLORS.length]),
                        borderWidth: 1,
                    }
                ]
            },
            options: {
                responsive: true,
                scales: { y: { beginAtZero: true, max: 100, title: { display: true, text: '%' } } },
                plugins: { legend: { labels: { color: '#e2e8f0' } } },
            }
        });
    }

    // --- Latency bar chart ---
    const latCtx = document.getElementById('latency-chart');
    if (latCtx) {
        new Chart(latCtx, {
            type: 'bar',
            data: {
                labels: slugs,
                datasets: [
                    {
                        label: 'Mean (ms)',
                        data: slugs.map(s => metricsData.pipelines[s].latency_mean),
                        backgroundColor: COLORS.map(c => c + 'cc'),
                        borderColor: COLORS,
                        borderWidth: 1,
                    },
                    {
                        label: 'P95 (ms)',
                        data: slugs.map(s => metricsData.pipelines[s].latency_p95),
                        backgroundColor: COLORS.map(c => c + '44'),
                        borderColor: COLORS,
                        borderWidth: 1,
                    }
                ]
            },
            options: {
                responsive: true,
                scales: { y: { beginAtZero: true, title: { display: true, text: 'ms' } } },
                plugins: { legend: { labels: { color: '#e2e8f0' } } },
            }
        });
    }

    // --- Accuracy vs Latency scatter ---
    const scatterCtx = document.getElementById('scatter-chart');
    if (scatterCtx) {
        new Chart(scatterCtx, {
            type: 'scatter',
            data: {
                datasets: slugs.map((s, i) => ({
                    label: s,
                    data: [{ x: metricsData.pipelines[s].latency_mean, y: metricsData.pipelines[s].string_accuracy }],
                    backgroundColor: COLORS[i % COLORS.length],
                    pointRadius: 8,
                    pointHoverRadius: 12,
                }))
            },
            options: {
                responsive: true,
                scales: {
                    x: { title: { display: true, text: 'Mean Latency (ms)' } },
                    y: { title: { display: true, text: 'String Accuracy %' }, beginAtZero: true, max: 100 }
                },
                plugins: { legend: { labels: { color: '#e2e8f0' } } },
            }
        });
    }

    // --- Confusion matrix heatmap ---
    const confCtx = document.getElementById('confusion-chart');
    const confSelect = document.getElementById('confusion-pipeline');
    if (confCtx && confSelect) {
        let confChart = null;

        function renderConfusion(slug) {
            const matrix = metricsData.pipelines[slug].confusion_matrix;
            if (!matrix) return;

            const data = [];
            for (let i = 0; i < 10; i++) {
                for (let j = 0; j < 10; j++) {
                    data.push({ x: j, y: i, v: matrix[i][j] });
                }
            }

            const maxVal = Math.max(...data.map(d => d.v), 1);

            if (confChart) confChart.destroy();
            confChart = new Chart(confCtx, {
                type: 'scatter',
                data: {
                    datasets: [{
                        data: data.map(d => ({ x: d.x, y: d.y })),
                        pointRadius: data.map(d => Math.max(3, (d.v / maxVal) * 18)),
                        backgroundColor: data.map(d => d.x === d.y ?
                            `rgba(16,185,129,${Math.min(1, d.v / maxVal + 0.2)})` :
                            `rgba(239,68,68,${Math.min(1, d.v / maxVal + 0.1)})`),
                        pointStyle: 'rect',
                    }]
                },
                options: {
                    responsive: true,
                    scales: {
                        x: { title: { display: true, text: 'Predicted' }, min: -0.5, max: 9.5, ticks: { stepSize: 1 } },
                        y: { title: { display: true, text: 'Actual' }, min: -0.5, max: 9.5, ticks: { stepSize: 1 }, reverse: true }
                    },
                    plugins: {
                        legend: { display: false },
                        tooltip: {
                            callbacks: {
                                label: (ctx) => {
                                    const d = data[ctx.dataIndex];
                                    return `Actual ${d.y} -> Predicted ${d.x}: ${d.v}`;
                                }
                            }
                        }
                    }
                }
            });
        }

        confSelect.addEventListener('change', () => renderConfusion(confSelect.value));
        renderConfusion(slugs[0]);
    }
})();
