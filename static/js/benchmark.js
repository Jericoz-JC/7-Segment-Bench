/**
 * Benchmark runner — starts benchmark and listens for SSE progress events.
 */
(function() {
    const form = document.getElementById('bench-form');
    const startBtn = document.getElementById('start-btn');
    const progressCard = document.getElementById('progress-card');
    const benchFill = document.getElementById('bench-fill');
    const benchPct = document.getElementById('bench-pct');
    const benchLog = document.getElementById('bench-log');

    form.addEventListener('submit', async e => {
        e.preventDefault();

        const datasetId = document.getElementById('bench-dataset').value;
        const name = document.getElementById('bench-name').value;
        const checked = document.querySelectorAll('input[name="pipelines"]:checked');
        const slugs = Array.from(checked).map(cb => cb.value);

        if (!datasetId || slugs.length === 0) return;

        startBtn.disabled = true;
        startBtn.textContent = 'Starting...';
        progressCard.style.display = 'block';
        benchLog.innerHTML = '';

        const resp = await fetch('/benchmark/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ dataset_id: parseInt(datasetId), pipeline_slugs: slugs, name })
        });
        const data = await resp.json();

        if (data.error) {
            logMsg(data.error, 'error');
            startBtn.disabled = false;
            startBtn.textContent = 'Start Benchmark';
            return;
        }

        // Connect SSE
        const runId = data.run_id;
        const source = new EventSource(`/benchmark/progress/${runId}`);

        source.onmessage = (event) => {
            const msg = JSON.parse(event.data);

            switch (msg.type) {
                case 'started':
                    logMsg(`Started: ${msg.images} images x ${msg.pipelines.length} pipelines = ${msg.total} predictions`);
                    break;
                case 'pipeline_start':
                    logMsg(`Running pipeline: ${msg.pipeline}`, 'info');
                    break;
                case 'progress':
                    const pct = ((msg.processed / msg.total) * 100).toFixed(1);
                    benchFill.style.width = pct + '%';
                    benchPct.textContent = pct + '%';
                    const icon = msg.correct ? '\u2713' : '\u2717';
                    logMsg(`${icon} ${msg.pipeline} | ${msg.image}: predicted="${msg.predicted}" gt="${msg.ground_truth}"`,
                        msg.correct ? 'success' : 'warn');
                    break;
                case 'pipeline_done':
                    logMsg(`Completed: ${msg.pipeline}`, 'info');
                    break;
                case 'pipeline_error':
                    logMsg(`Error in ${msg.pipeline}: ${msg.error}`, 'error');
                    break;
                case 'completed':
                    logMsg('Benchmark completed!', 'success');
                    benchFill.style.width = '100%';
                    benchPct.textContent = '100%';
                    source.close();
                    startBtn.disabled = false;
                    startBtn.textContent = 'Start Benchmark';
                    // Add link to results
                    logMsg(`<a href="/results/${msg.run_id}" class="btn btn-primary">View Results</a>`, 'html');
                    break;
                case 'error':
                    logMsg(msg.message, 'error');
                    source.close();
                    startBtn.disabled = false;
                    startBtn.textContent = 'Start Benchmark';
                    break;
            }
        };

        source.onerror = () => {
            source.close();
            startBtn.disabled = false;
            startBtn.textContent = 'Start Benchmark';
        };
    });

    function logMsg(text, type = '') {
        const div = document.createElement('div');
        div.className = 'log-entry log-' + type;
        if (type === 'html') {
            div.innerHTML = text;
        } else {
            div.textContent = text;
        }
        benchLog.appendChild(div);
        benchLog.scrollTop = benchLog.scrollHeight;
    }
})();
