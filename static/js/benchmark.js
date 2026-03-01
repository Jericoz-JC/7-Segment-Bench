/**
 * Benchmark runner - starts benchmark and listens for SSE progress events.
 * Supports stepper, per-pipeline progress cards, ETA, and elapsed timer.
 */
(function() {
    const form = document.getElementById('bench-form');
    if (!form) return;

    const startBtn = document.getElementById('start-btn');
    const progressCard = document.getElementById('progress-card');
    const benchFill = document.getElementById('bench-fill');
    const benchPct = document.getElementById('bench-pct');
    const benchLog = document.getElementById('bench-log');
    const llmPanel = document.getElementById('bench-llm-config');
    const llmProvider = document.getElementById('bench-llm-provider');
    const llmModel = document.getElementById('bench-llm-model');
    const llmApiKey = document.getElementById('bench-llm-key');
    const llmBaseUrlWrap = document.getElementById('bench-llm-baseurl-wrap');
    const llmBaseUrl = document.getElementById('bench-llm-baseurl');

    // Stepper elements
    const stepInit = document.getElementById('step-init');
    const stepPipelines = document.getElementById('step-pipelines');
    const stepMetrics = document.getElementById('step-metrics');
    const stepDone = document.getElementById('step-done');
    const conn12 = document.getElementById('conn-1-2');
    const conn23 = document.getElementById('conn-2-3');
    const conn34 = document.getElementById('conn-3-4');

    // Meta row elements
    const metaCount = document.getElementById('bench-meta-count');
    const metaElapsed = document.getElementById('bench-meta-elapsed');
    const metaEta = document.getElementById('bench-meta-eta');

    // Pipeline progress container
    const pipelineCardsContainer = document.getElementById('pipeline-progress-cards');

    let elapsedTimer = null;
    let startTime = null;

    const defaultModels = {
        anthropic: 'claude-sonnet-4-20250514',
        openai: 'gpt-4o',
        openrouter: 'openrouter/auto',
        ollama: 'llava',
    };

    function selectedPipelines() {
        const checked = document.querySelectorAll('input[name="pipelines"]:checked');
        return Array.from(checked).map((cb) => cb.value);
    }

    function isLlmSelected() {
        return selectedPipelines().includes('p07_llm_vision');
    }

    function updateLlmUi() {
        if (!llmPanel || !llmProvider || !llmModel || !llmBaseUrlWrap) return;
        const show = isLlmSelected();
        llmPanel.style.display = show ? 'block' : 'none';
        const provider = (llmProvider.value || 'anthropic').trim().toLowerCase();
        llmBaseUrlWrap.style.display = provider === 'ollama' ? 'block' : 'none';
    }

    async function saveProviderKey(provider, keyValue) {
        const resp = await fetch('/api/keys', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ provider, key_value: keyValue }),
        });
        const data = await resp.json();
        if (!resp.ok || data.error) {
            throw new Error(data.error || 'Could not save API key.');
        }
    }

    async function hasActiveProviderKey(provider) {
        const resp = await fetch('/api/keys');
        const data = await resp.json();
        if (!resp.ok || !Array.isArray(data)) {
            throw new Error('Could not verify API key status.');
        }
        return data.some((item) => item.provider === provider && item.is_active);
    }

    async function ensureLlmProviderReady(provider, keyValue) {
        if (keyValue) {
            await saveProviderKey(provider, keyValue);
            if (llmApiKey) llmApiKey.value = '';
            return;
        }
        if (provider === 'ollama') {
            return;
        }
        const hasActive = await hasActiveProviderKey(provider);
        if (!hasActive) {
            throw new Error(`No active API key found for ${provider}. Enter one in the LLM settings.`);
        }
    }

    function buildPipelineConfigs(slugs) {
        const configs = {};
        if (!slugs.includes('p07_llm_vision') || !llmProvider || !llmModel) {
            return configs;
        }

        const provider = (llmProvider.value || 'anthropic').trim().toLowerCase();
        const model = llmModel.value.trim() || defaultModels[provider] || defaultModels.openai;
        const cfg = { provider, model };

        if (provider === 'ollama' && llmBaseUrl) {
            cfg.base_url = llmBaseUrl.value.trim() || 'http://localhost:11434/v1';
        }

        configs.p07_llm_vision = cfg;
        return configs;
    }

    // --- Stepper ---
    function updateStepper(phase) {
        const steps = [stepInit, stepPipelines, stepMetrics, stepDone];
        const connectors = [conn12, conn23, conn34];

        // Reset all
        steps.forEach(s => { if (s) { s.className = 'bench-step'; } });
        connectors.forEach(c => { if (c) { c.className = 'bench-step-connector'; } });

        if (phase === 'initializing' || phase === 'loading_data') {
            if (stepInit) stepInit.classList.add('active');
        } else if (phase && phase.startsWith('pipeline:')) {
            if (stepInit) stepInit.classList.add('completed');
            if (conn12) conn12.classList.add('completed');
            if (stepPipelines) stepPipelines.classList.add('active');
        } else if (phase === 'computing_metrics') {
            if (stepInit) stepInit.classList.add('completed');
            if (conn12) conn12.classList.add('completed');
            if (stepPipelines) stepPipelines.classList.add('completed');
            if (conn23) conn23.classList.add('completed');
            if (stepMetrics) stepMetrics.classList.add('active');
        } else if (phase === 'finalizing' || phase === 'completed') {
            steps.forEach(s => { if (s) s.classList.add('completed'); });
            connectors.forEach(c => { if (c) c.classList.add('completed'); });
        }
    }

    // --- Pipeline Progress Cards ---
    function initPipelineCards(slugs) {
        if (!pipelineCardsContainer) return;
        pipelineCardsContainer.innerHTML = slugs.map(slug =>
            `<div class="pipeline-progress-card" id="pp-card-${slug}">
                <div class="pipeline-progress-name">
                    <span>${slug}</span>
                    <span class="status status-pending" id="pp-status-${slug}">pending</span>
                </div>
                <div class="pipeline-progress-bar">
                    <div class="pipeline-progress-fill" id="pp-fill-${slug}"></div>
                </div>
                <div class="pipeline-progress-count" id="pp-count-${slug}">0 / 0</div>
            </div>`
        ).join('');
    }

    function updatePipelineCard(slug, processed, total) {
        const fill = document.getElementById(`pp-fill-${slug}`);
        const count = document.getElementById(`pp-count-${slug}`);
        if (fill && total > 0) {
            fill.style.width = ((processed / total) * 100).toFixed(1) + '%';
        }
        if (count) {
            count.textContent = `${processed} / ${total}`;
        }
    }

    function setPipelineCardStatus(slug, status) {
        const card = document.getElementById(`pp-card-${slug}`);
        const statusEl = document.getElementById(`pp-status-${slug}`);
        if (card) {
            card.className = 'pipeline-progress-card';
            if (status === 'running') card.classList.add('active');
            else if (status === 'completed') card.classList.add('completed');
            else if (status === 'failed') card.classList.add('failed');
        }
        if (statusEl) {
            statusEl.className = 'status';
            if (status === 'running') {
                statusEl.classList.add('status-running');
                statusEl.textContent = 'running';
            } else if (status === 'completed') {
                statusEl.classList.add('status-completed');
                statusEl.textContent = 'completed';
            } else if (status === 'failed') {
                statusEl.classList.add('status-failed');
                statusEl.textContent = 'failed';
            } else {
                statusEl.classList.add('status-pending');
                statusEl.textContent = 'pending';
            }
        }
    }

    // --- ETA / Elapsed ---
    function formatETA(seconds) {
        if (!seconds || seconds < 0 || !isFinite(seconds)) return '--';
        const m = Math.floor(seconds / 60);
        const s = Math.floor(seconds % 60);
        return `${m}:${String(s).padStart(2, '0')}`;
    }

    function formatElapsed(ms) {
        const totalSec = Math.floor(ms / 1000);
        const m = Math.floor(totalSec / 60);
        const s = totalSec % 60;
        return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
    }

    function startElapsedTimer() {
        startTime = Date.now();
        if (elapsedTimer) clearInterval(elapsedTimer);
        elapsedTimer = setInterval(() => {
            if (metaElapsed) {
                metaElapsed.textContent = formatElapsed(Date.now() - startTime);
            }
        }, 1000);
    }

    function stopElapsedTimer() {
        if (elapsedTimer) {
            clearInterval(elapsedTimer);
            elapsedTimer = null;
        }
    }

    // --- Track per-pipeline image counts ---
    let pipelineImageCounts = {};
    let totalImages = 0;

    document.querySelectorAll('input[name="pipelines"]').forEach((cb) => {
        cb.addEventListener('change', updateLlmUi);
    });

    if (llmProvider) {
        llmProvider.addEventListener('change', () => {
            const provider = (llmProvider.value || 'anthropic').trim().toLowerCase();
            if (llmModel) {
                llmModel.value = defaultModels[provider] || defaultModels.openai;
            }
            updateLlmUi();
        });
    }

    updateLlmUi();

    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        const datasetId = document.getElementById('bench-dataset').value;
        const name = document.getElementById('bench-name').value;
        const slugs = selectedPipelines();

        if (!datasetId || slugs.length === 0) return;

        startBtn.disabled = true;
        startBtn.textContent = 'Starting...';
        progressCard.style.display = 'block';
        benchLog.innerHTML = '';
        pipelineImageCounts = {};
        totalImages = 0;

        // Reset stepper and cards
        updateStepper('initializing');
        initPipelineCards(slugs);
        if (metaCount) metaCount.textContent = '0 / 0';
        if (metaEta) metaEta.textContent = '--';
        startElapsedTimer();

        try {
            const pipelineConfigs = buildPipelineConfigs(slugs);
            if (pipelineConfigs.p07_llm_vision) {
                const provider = pipelineConfigs.p07_llm_vision.provider;
                const keyValue = llmApiKey ? llmApiKey.value.trim() : '';
                await ensureLlmProviderReady(provider, keyValue);
            }

            const resp = await fetch('/benchmark/start', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    dataset_id: parseInt(datasetId, 10),
                    pipeline_slugs: slugs,
                    pipeline_configs: pipelineConfigs,
                    name,
                }),
            });
            const data = await resp.json();

            if (!resp.ok || data.error) {
                logMsg(data.error || 'Could not start benchmark.', 'error');
                startBtn.disabled = false;
                startBtn.textContent = 'Start Benchmark';
                stopElapsedTimer();
                return;
            }

            const runId = data.run_id;
            const source = new EventSource(`/benchmark/progress/${runId}`);

            source.onmessage = (event) => {
                const msg = JSON.parse(event.data);

                switch (msg.type) {
                    case 'started':
                        totalImages = msg.images;
                        logMsg(
                            `Started: ${msg.images} images x ${msg.pipelines.length} pipelines = ${msg.total} predictions`
                        );
                        if (metaCount) metaCount.textContent = `0 / ${msg.total}`;
                        updateStepper('loading_data');
                        break;

                    case 'phase_change':
                        updateStepper(msg.phase);
                        break;

                    case 'pipeline_start':
                        logMsg(`Running pipeline: ${msg.pipeline}`, 'info');
                        setPipelineCardStatus(msg.pipeline, 'running');
                        pipelineImageCounts[msg.pipeline] = 0;
                        updateStepper(`pipeline:${msg.pipeline}`);
                        break;

                    case 'progress': {
                        const pct = ((msg.processed / msg.total) * 100).toFixed(1);
                        benchFill.style.width = pct + '%';
                        benchPct.textContent = pct + '%';
                        if (metaCount) metaCount.textContent = `${msg.processed} / ${msg.total}`;

                        // Update per-pipeline count
                        if (msg.pipeline) {
                            pipelineImageCounts[msg.pipeline] = (pipelineImageCounts[msg.pipeline] || 0) + 1;
                            updatePipelineCard(msg.pipeline, pipelineImageCounts[msg.pipeline], totalImages);
                        }

                        const icon = msg.correct ? '\u2713' : '\u2717';
                        logMsg(
                            `${icon} ${msg.pipeline} | ${msg.image}: predicted="${msg.predicted}" gt="${msg.ground_truth}"`,
                            msg.correct ? 'success' : 'warn'
                        );
                        break;
                    }

                    case 'pipeline_progress':
                        if (msg.pipeline) {
                            updatePipelineCard(msg.pipeline, msg.pipeline_processed, msg.pipeline_total);
                        }
                        break;

                    case 'eta':
                        if (metaEta && msg.seconds_remaining != null) {
                            metaEta.textContent = formatETA(msg.seconds_remaining);
                        }
                        break;

                    case 'pipeline_done':
                        logMsg(`Completed: ${msg.pipeline}`, 'info');
                        setPipelineCardStatus(msg.pipeline, 'completed');
                        updatePipelineCard(msg.pipeline, totalImages, totalImages);
                        break;

                    case 'pipeline_error':
                        logMsg(`Error in ${msg.pipeline}: ${msg.error}`, 'error');
                        setPipelineCardStatus(msg.pipeline, 'failed');
                        break;

                    case 'completed':
                        logMsg('Benchmark completed!', 'success');
                        benchFill.style.width = '100%';
                        benchPct.textContent = '100%';
                        if (metaEta) metaEta.textContent = 'done';
                        updateStepper('completed');
                        stopElapsedTimer();
                        source.close();
                        startBtn.disabled = false;
                        startBtn.textContent = 'Start Benchmark';
                        logMsg(`<a href="/results/${msg.run_id}" class="btn btn-primary">View Results</a>`, 'html');
                        break;

                    case 'error':
                        logMsg(msg.message, 'error');
                        stopElapsedTimer();
                        source.close();
                        startBtn.disabled = false;
                        startBtn.textContent = 'Start Benchmark';
                        break;
                }
            };

            source.onerror = () => {
                source.close();
                stopElapsedTimer();
                startBtn.disabled = false;
                startBtn.textContent = 'Start Benchmark';
            };
        } catch (err) {
            logMsg(String(err), 'error');
            stopElapsedTimer();
            startBtn.disabled = false;
            startBtn.textContent = 'Start Benchmark';
        }
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
