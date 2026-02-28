(function() {
    // ----------------------------
    // Batch quick-test flow
    // ----------------------------
    const batchForm = document.getElementById('batch-test-form');
    const batchBtn = document.getElementById('batch-start-btn');
    const batchFill = document.getElementById('quick-batch-fill');
    const batchPct = document.getElementById('quick-batch-pct');
    const batchLog = document.getElementById('quick-batch-log');
    const batchSummary = document.getElementById('quick-batch-summary');

    let batchEventSource = null;

    function logBatch(text, type) {
        const div = document.createElement('div');
        div.className = 'log-entry log-' + (type || 'info');
        if (type === 'html') {
            div.innerHTML = text;
        } else {
            div.textContent = text;
        }
        batchLog.appendChild(div);
        batchLog.scrollTop = batchLog.scrollHeight;
    }

    function setBatchProgress(processed, total) {
        const pct = total > 0 ? ((processed / total) * 100) : 0;
        const pctText = pct.toFixed(1) + '%';
        batchFill.style.width = pctText;
        batchPct.textContent = pctText;
    }

    if (batchForm) {
        batchForm.addEventListener('submit', async (e) => {
            e.preventDefault();

            const datasetId = document.getElementById('batch-dataset').value;
            const sampleSize = parseInt(document.getElementById('batch-size').value, 10);
            const runName = document.getElementById('batch-name').value.trim();
            const slugs = Array.from(document.querySelectorAll('input[name="batch_pipelines"]:checked'))
                .map((el) => el.value);

            if (!datasetId || !slugs.length) {
                logBatch('Select a dataset and at least one pipeline.', 'error');
                return;
            }
            if (Number.isNaN(sampleSize) || sampleSize < 10 || sampleSize > 50) {
                logBatch('Sample size must be between 10 and 50.', 'error');
                return;
            }

            if (batchEventSource) {
                batchEventSource.close();
                batchEventSource = null;
            }

            batchBtn.disabled = true;
            batchBtn.textContent = 'Starting...';
            batchLog.innerHTML = '';
            setBatchProgress(0, 0);
            batchSummary.textContent = 'Starting batch run...';

            try {
                const resp = await fetch('/test/run-batch', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        dataset_id: parseInt(datasetId, 10),
                        pipeline_slugs: slugs,
                        sample_size: sampleSize,
                        name: runName || undefined,
                    }),
                });
                const data = await resp.json();
                if (!resp.ok || data.error) {
                    logBatch(data.error || 'Could not start batch run.', 'error');
                    batchSummary.textContent = 'Batch run failed to start.';
                    batchBtn.disabled = false;
                    batchBtn.textContent = 'Start Batch Quick Test';
                    return;
                }

                const runId = data.run_id;
                logBatch(
                    `Run #${runId} started with ${data.selected_count} sampled image(s).`,
                    'success'
                );
                batchSummary.textContent = `Running benchmark stream for run #${runId}.`;

                batchEventSource = new EventSource(`/benchmark/progress/${runId}`);
                batchEventSource.onmessage = (event) => {
                    const msg = JSON.parse(event.data);
                    switch (msg.type) {
                        case 'keepalive':
                            break;
                        case 'started':
                            logBatch(
                                `Started: ${msg.images} image(s) x ${msg.pipelines.length} pipeline(s) = ${msg.total} predictions.`,
                                'info'
                            );
                            break;
                        case 'pipeline_start':
                            logBatch(`Pipeline start: ${msg.pipeline}`, 'info');
                            break;
                        case 'progress':
                            setBatchProgress(msg.processed, msg.total);
                            logBatch(
                                `${msg.pipeline} | ${msg.image} | pred=\"${msg.predicted}\" gt=\"${msg.ground_truth}\"`,
                                msg.correct ? 'success' : 'warn'
                            );
                            break;
                        case 'pipeline_done':
                            logBatch(`Pipeline done: ${msg.pipeline}`, 'info');
                            break;
                        case 'pipeline_error':
                            logBatch(`Pipeline error (${msg.pipeline}): ${msg.error}`, 'error');
                            break;
                        case 'completed':
                            setBatchProgress(1, 1);
                            batchSummary.textContent = `Completed run #${msg.run_id}.`;
                            logBatch(
                                `<a href=\"/results/${msg.run_id}\" class=\"btn btn-primary\">Open Results</a>`,
                                'html'
                            );
                            batchEventSource.close();
                            batchEventSource = null;
                            batchBtn.disabled = false;
                            batchBtn.textContent = 'Start Batch Quick Test';
                            break;
                        case 'error':
                            logBatch(msg.message || 'Stream error', 'error');
                            batchSummary.textContent = 'Batch run failed.';
                            if (batchEventSource) {
                                batchEventSource.close();
                                batchEventSource = null;
                            }
                            batchBtn.disabled = false;
                            batchBtn.textContent = 'Start Batch Quick Test';
                            break;
                        default:
                            if (msg.status) {
                                batchSummary.textContent = `Run status: ${msg.status}`;
                            }
                    }
                };

                batchEventSource.onerror = () => {
                    logBatch('Progress stream disconnected.', 'warn');
                    if (batchEventSource) {
                        batchEventSource.close();
                        batchEventSource = null;
                    }
                    batchBtn.disabled = false;
                    batchBtn.textContent = 'Start Batch Quick Test';
                };
            } catch (err) {
                logBatch(String(err), 'error');
                batchSummary.textContent = 'Batch run failed.';
                batchBtn.disabled = false;
                batchBtn.textContent = 'Start Batch Quick Test';
            }
        });
    }

    // ----------------------------
    // Single-image flow
    // ----------------------------
    const testDrop = document.getElementById('test-drop');
    const testFile = document.getElementById('test-file');
    const testCanvas = document.getElementById('test-canvas');
    const singleBtn = document.getElementById('single-test-btn');

    let testImage = null;
    let testROI = null;
    let drawing = false;
    let startX = 0;
    let startY = 0;

    function loadTestImage(file) {
        testImage = file;
        singleBtn.disabled = false;
        testCanvas.style.display = 'block';
        const img = new Image();
        img.onload = () => {
            const maxW = testCanvas.parentElement.clientWidth - 20;
            const scale = Math.min(1, maxW / img.width);
            testCanvas.width = img.width * scale;
            testCanvas.height = img.height * scale;
            testCanvas._scale = scale;
            testCanvas._img = img;
            const ctx = testCanvas.getContext('2d');
            ctx.drawImage(img, 0, 0, testCanvas.width, testCanvas.height);
        };
        img.src = URL.createObjectURL(file);
    }

    if (testDrop && testFile && testCanvas && singleBtn) {
        testDrop.addEventListener('click', () => testFile.click());
        testDrop.addEventListener('dragover', (e) => {
            e.preventDefault();
            testDrop.classList.add('dragover');
        });
        testDrop.addEventListener('dragleave', () => testDrop.classList.remove('dragover'));
        testDrop.addEventListener('drop', (e) => {
            e.preventDefault();
            testDrop.classList.remove('dragover');
            if (e.dataTransfer.files.length) {
                loadTestImage(e.dataTransfer.files[0]);
            }
        });
        testFile.addEventListener('change', (e) => {
            if (e.target.files.length) {
                loadTestImage(e.target.files[0]);
            }
        });

        testCanvas.addEventListener('mousedown', (e) => {
            drawing = true;
            const r = testCanvas.getBoundingClientRect();
            startX = e.clientX - r.left;
            startY = e.clientY - r.top;
        });
        testCanvas.addEventListener('mousemove', (e) => {
            if (!drawing) return;
            const r = testCanvas.getBoundingClientRect();
            const x = e.clientX - r.left;
            const y = e.clientY - r.top;
            const ctx = testCanvas.getContext('2d');
            ctx.drawImage(testCanvas._img, 0, 0, testCanvas.width, testCanvas.height);
            ctx.strokeStyle = '#00ff88';
            ctx.lineWidth = 2;
            ctx.strokeRect(startX, startY, x - startX, y - startY);
        });
        testCanvas.addEventListener('mouseup', (e) => {
            drawing = false;
            const r = testCanvas.getBoundingClientRect();
            const x = e.clientX - r.left;
            const y = e.clientY - r.top;
            const s = testCanvas._scale;
            testROI = {
                x: Math.round(Math.min(startX, x) / s),
                y: Math.round(Math.min(startY, y) / s),
                width: Math.round(Math.abs(x - startX) / s),
                height: Math.round(Math.abs(y - startY) / s),
            };
            document.getElementById('test-roi').value = JSON.stringify(testROI);
        });

        document.getElementById('single-test-form').addEventListener('submit', async (e) => {
            e.preventDefault();
            if (!testImage) return;
            singleBtn.disabled = true;
            singleBtn.textContent = 'Running...';

            const form = new FormData();
            form.append('image', testImage);
            if (testROI) form.append('roi', JSON.stringify(testROI));
            document.querySelectorAll('input[name="single_pipelines"]:checked').forEach((cb) => {
                form.append('pipelines', cb.value);
            });

            const resp = await fetch('/test/run', { method: 'POST', body: form });
            const data = await resp.json();

            const grid = document.getElementById('single-results-grid');
            grid.innerHTML = '';
            document.getElementById('single-results-panel').style.display = 'block';

            Object.entries(data).forEach(([slug, result]) => {
                const card = document.createElement('div');
                card.className = 'result-card';
                let debugHtml = '';
                if (result.debug_images) {
                    Object.entries(result.debug_images).forEach(([name, b64]) => {
                        debugHtml +=
                            `<div class=\"debug-img\"><img src=\"data:image/png;base64,${b64}\" alt=\"${name}\"><span>${name}</span></div>`;
                    });
                }
                card.innerHTML = `
                    <h3>${slug}</h3>
                    <div class=\"result-predicted\">${result.predicted || '(none)'}</div>
                    <div class=\"result-meta\">
                        <span>Latency: ${result.latency_ms} ms</span>
                        <span>Confidence: ${(result.confidence * 100).toFixed(1)}%</span>
                    </div>
                    ${result.error ? `<div class=\"error-msg\">${result.error}</div>` : ''}
                    ${debugHtml ? `<div class=\"debug-images\">${debugHtml}</div>` : ''}
                `;
                grid.appendChild(card);
            });

            singleBtn.disabled = false;
            singleBtn.textContent = 'Run Single Image';
        });
    }
})();
