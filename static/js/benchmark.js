/**
 * Benchmark runner - starts benchmark and listens for SSE progress events.
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
                return;
            }

            const runId = data.run_id;
            const source = new EventSource(`/benchmark/progress/${runId}`);

            source.onmessage = (event) => {
                const msg = JSON.parse(event.data);

                switch (msg.type) {
                    case 'started':
                        logMsg(
                            `Started: ${msg.images} images x ${msg.pipelines.length} pipelines = ${msg.total} predictions`
                        );
                        break;
                    case 'pipeline_start':
                        logMsg(`Running pipeline: ${msg.pipeline}`, 'info');
                        break;
                    case 'progress': {
                        const pct = ((msg.processed / msg.total) * 100).toFixed(1);
                        benchFill.style.width = pct + '%';
                        benchPct.textContent = pct + '%';
                        const icon = msg.correct ? '\u2713' : '\u2717';
                        logMsg(
                            `${icon} ${msg.pipeline} | ${msg.image}: predicted="${msg.predicted}" gt="${msg.ground_truth}"`,
                            msg.correct ? 'success' : 'warn'
                        );
                        break;
                    }
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
        } catch (err) {
            logMsg(String(err), 'error');
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
