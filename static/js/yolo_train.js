(function() {
    const form = document.getElementById('yolo-train-form');
    const trainBtn = document.getElementById('yolo-train-btn');
    const statusEl = document.getElementById('yolo-train-status');
    const logEl = document.getElementById('yolo-train-log');
    const checkBtn = document.getElementById('check-tesseract-btn');
    const checkResult = document.getElementById('check-tesseract-result');

    let pollTimer = null;

    function log(msg, type) {
        const div = document.createElement('div');
        div.className = 'log-entry log-' + (type || 'info');
        div.textContent = msg;
        logEl.appendChild(div);
        logEl.scrollTop = logEl.scrollHeight;
    }

    async function pollJob(jobId) {
        if (pollTimer) {
            clearInterval(pollTimer);
        }
        pollTimer = setInterval(async () => {
            try {
                const resp = await fetch(`/train/yolo/status/${jobId}`);
                const data = await resp.json();
                if (!resp.ok || data.error) {
                    log(data.error || 'Could not fetch job status.', 'error');
                    clearInterval(pollTimer);
                    pollTimer = null;
                    trainBtn.disabled = false;
                    trainBtn.textContent = 'Start YOLO Training';
                    statusEl.textContent = 'Training status unavailable.';
                    return;
                }

                statusEl.textContent = `${data.status} | ${data.phase} | ${data.message}`;
                if (data.status === 'completed') {
                    log('Training completed. Model registered.', 'success');
                    clearInterval(pollTimer);
                    pollTimer = null;
                    trainBtn.disabled = false;
                    trainBtn.textContent = 'Start YOLO Training';
                    setTimeout(() => window.location.reload(), 600);
                } else if (data.status === 'failed') {
                    log(`Training failed: ${data.error || 'unknown error'}`, 'error');
                    clearInterval(pollTimer);
                    pollTimer = null;
                    trainBtn.disabled = false;
                    trainBtn.textContent = 'Start YOLO Training';
                }
            } catch (err) {
                log(String(err), 'error');
                clearInterval(pollTimer);
                pollTimer = null;
                trainBtn.disabled = false;
                trainBtn.textContent = 'Start YOLO Training';
            }
        }, 2000);
    }

    if (form) {
        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            trainBtn.disabled = true;
            trainBtn.textContent = 'Queueing...';
            logEl.innerHTML = '';
            statusEl.textContent = 'Submitting training job...';

            const payload = {
                dataset_id: parseInt(document.getElementById('train-dataset').value, 10),
                epochs: parseInt(document.getElementById('train-epochs').value, 10),
                imgsz: parseInt(document.getElementById('train-imgsz').value, 10),
                batch: parseInt(document.getElementById('train-batch').value, 10),
                val_ratio: parseFloat(document.getElementById('train-val-ratio').value),
                seed: parseInt(document.getElementById('train-seed').value, 10),
                device: document.getElementById('train-device').value.trim(),
                run_name: document.getElementById('train-run-name').value.trim(),
                activate: true,
            };

            if (!payload.dataset_id) {
                log('Select a dataset before training.', 'error');
                trainBtn.disabled = false;
                trainBtn.textContent = 'Start YOLO Training';
                return;
            }

            try {
                const resp = await fetch('/train/yolo/start', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload),
                });
                const data = await resp.json();
                if (!resp.ok || data.error) {
                    log(data.error || 'Could not start training.', 'error');
                    trainBtn.disabled = false;
                    trainBtn.textContent = 'Start YOLO Training';
                    statusEl.textContent = 'Training not started.';
                    return;
                }

                log(`Training job queued: ${data.job_id}`, 'info');
                statusEl.textContent = `Job ${data.job_id} is running.`;
                trainBtn.textContent = 'Running...';
                pollJob(data.job_id);
            } catch (err) {
                log(String(err), 'error');
                trainBtn.disabled = false;
                trainBtn.textContent = 'Start YOLO Training';
                statusEl.textContent = 'Training not started.';
            }
        });
    }

    if (checkBtn && checkResult) {
        checkBtn.addEventListener('click', async () => {
            checkResult.className = 'status status-running';
            checkResult.textContent = 'checking';
            try {
                const resp = await fetch('/api/pipelines/p05/preflight');
                const data = await resp.json();
                if (resp.ok && data.ok) {
                    checkResult.className = 'status status-completed';
                    checkResult.textContent = data.version || 'ok';
                } else {
                    checkResult.className = 'status status-failed';
                    checkResult.textContent = data.error || 'unavailable';
                }
            } catch (err) {
                checkResult.className = 'status status-failed';
                checkResult.textContent = String(err);
            }
        });
    }

    document.querySelectorAll('.activate-model-btn').forEach((btn) => {
        btn.addEventListener('click', async () => {
            const id = btn.dataset.modelId;
            btn.disabled = true;
            btn.textContent = 'Activating...';
            try {
                const resp = await fetch(`/train/yolo/models/${id}/activate`, { method: 'POST' });
                const data = await resp.json();
                if (!resp.ok || data.error) {
                    alert(data.error || 'Could not activate model.');
                    btn.disabled = false;
                    btn.textContent = 'Activate';
                    return;
                }
                window.location.reload();
            } catch (err) {
                alert(String(err));
                btn.disabled = false;
                btn.textContent = 'Activate';
            }
        });
    });
})();
