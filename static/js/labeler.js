/**
 * Canvas-based ROI labeling tool for 7-segment display images.
 * Supports raw GT symbols (- and .), benchmark-target preview, auto-suggest,
 * and save-and-next review flow.
 */
(function() {
    const canvas = document.getElementById('labeler-canvas');
    const ctx = canvas.getContext('2d');
    const gtInput = document.getElementById('gt-input');
    const benchmarkPreview = document.getElementById('benchmark-target');
    const displayType = document.getElementById('display-type');
    const autoSuggestToggle = document.getElementById('auto-suggest');
    const saveBtn = document.getElementById('save-btn');
    const deleteBtn = document.getElementById('delete-btn');
    const labelMsg = document.getElementById('label-msg');
    const labelInfo = document.getElementById('label-info');
    const counter = document.getElementById('img-counter');

    const thumbItems = Array.from(document.querySelectorAll('.thumb-item'));
    const labelStateByImageId = new Map();

    let currentIndex = 0;
    let currentImage = null;
    let imgElement = new Image();
    let scale = 1;
    let roi = null;
    let drawing = false;
    let startX = 0;
    let startY = 0;
    let existingLabels = [];
    let currentDraft = null;

    function sanitizeRaw(value) {
        return String(value || '')
            .replace(/\s+/g, '')
            .replace(/[^0-9.-]/g, '')
            .slice(0, 50);
    }

    function toBenchmarkTarget(value) {
        return sanitizeRaw(value).replace(/[^0-9]/g, '');
    }

    function isValidGroundTruth(value) {
        const raw = sanitizeRaw(value);
        return raw.length > 0 && /\d/.test(raw);
    }

    function escapeHtml(value) {
        return String(value || '')
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#039;');
    }

    function setBenchmarkPreview(value) {
        if (!benchmarkPreview) return;
        const target = toBenchmarkTarget(value);
        benchmarkPreview.textContent = target || '-';
    }

    function choosePrimaryLabel(labels) {
        if (!labels || labels.length === 0) return null;
        const manual = labels.find(l => l.verified);
        return manual || labels[0];
    }

    function setThumbStatus(imageId, status) {
        const statusEl = document.getElementById(`status-${imageId}`);
        if (!statusEl) return;
        const resolved = ['verified', 'suggested', 'unlabeled'].includes(status) ? status : 'unlabeled';
        statusEl.dataset.status = resolved;
        statusEl.className = `thumb-status ${resolved}`;
        statusEl.textContent = resolved === 'verified' ? '\u2713' : (resolved === 'suggested' ? '~' : '');
        labelStateByImageId.set(imageId, resolved);
    }

    function getThumbStatus(index) {
        const item = thumbItems[index];
        if (!item) return 'unlabeled';
        const id = parseInt(item.dataset.id, 10);
        const el = document.getElementById(`status-${id}`);
        if (el && el.dataset && el.dataset.status) {
            return el.dataset.status;
        }
        return labelStateByImageId.get(id) || 'unlabeled';
    }

    function nextIndexPreferUnresolved() {
        if (thumbItems.length === 0) return -1;
        let fallback = -1;
        for (let offset = 1; offset <= thumbItems.length; offset += 1) {
            const idx = (currentIndex + offset) % thumbItems.length;
            if (fallback === -1) fallback = idx;
            if (getThumbStatus(idx) !== 'verified') {
                return idx;
            }
        }
        return fallback;
    }

    function firstUnresolvedIndex() {
        if (thumbItems.length === 0) return -1;
        for (let i = 0; i < thumbItems.length; i += 1) {
            if (getThumbStatus(i) !== 'verified') {
                return i;
            }
        }
        return 0;
    }

    function drawCanvas() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(imgElement, 0, 0, canvas.width, canvas.height);

        existingLabels.forEach(l => {
            ctx.strokeStyle = '#34c77b';
            ctx.lineWidth = 2;
            ctx.strokeRect(l.roi_x * scale, l.roi_y * scale, l.roi_width * scale, l.roi_height * scale);
            ctx.fillStyle = 'rgba(52,199,123,0.15)';
            ctx.fillRect(l.roi_x * scale, l.roi_y * scale, l.roi_width * scale, l.roi_height * scale);
            ctx.fillStyle = '#34c77b';
            ctx.font = '14px monospace';
            ctx.fillText(l.ground_truth_raw || l.ground_truth, l.roi_x * scale + 4, l.roi_y * scale - 4);
        });

        if (roi) {
            ctx.strokeStyle = '#d94f7a';
            ctx.lineWidth = 2;
            ctx.setLineDash([5, 5]);
            ctx.strokeRect(roi.x * scale, roi.y * scale, roi.width * scale, roi.height * scale);
            ctx.setLineDash([]);
        }
    }

    function updateLabelInfo() {
        if (existingLabels.length === 0) {
            labelInfo.innerHTML = '<p class="empty-state">No saved labels for this image.</p>';
            return;
        }
        labelInfo.innerHTML = existingLabels.map(l => `
            <div class="label-entry">
                <span class="label-gt">${escapeHtml(l.ground_truth_raw || l.ground_truth)}</span>
                <span class="label-roi">[${l.roi_x},${l.roi_y} ${l.roi_width}x${l.roi_height}]</span>
                <span class="label-type">${escapeHtml(l.display_type)} / ${escapeHtml(l.label_source || 'manual')}</span>
                <button class="btn btn-sm btn-danger" onclick="deleteLabel(${l.id})">x</button>
            </div>
        `).join('');
    }

    async function requestSuggestion() {
        if (!currentImage) return;
        const body = { image_id: currentImage.id };
        if (roi) {
            body.roi_x = roi.x;
            body.roi_y = roi.y;
            body.roi_width = roi.width;
            body.roi_height = roi.height;
        }

        labelMsg.textContent = 'Suggesting with p05...';
        labelMsg.className = 'form-msg';
        try {
            const resp = await fetch('/label/suggest', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body),
            });
            const data = await resp.json();
            if (!resp.ok || data.error) {
                labelMsg.textContent = data.error || 'Suggestion failed';
                labelMsg.className = 'form-msg error';
                return;
            }
            if (!data.suggested_raw) {
                labelMsg.textContent = 'Suggestion found no digits. Enter label manually.';
                labelMsg.className = 'form-msg';
                return;
            }
            gtInput.value = data.suggested_raw;
            setBenchmarkPreview(gtInput.value);
            labelMsg.textContent = `Suggested: ${data.suggested_raw}`;
            labelMsg.className = 'form-msg success';
        } catch (err) {
            labelMsg.textContent = `Suggestion failed: ${err}`;
            labelMsg.className = 'form-msg error';
        }
    }

    async function loadLabels(imageId, options) {
        const opts = options || {};
        const allowSuggest = opts.allowSuggest !== false;
        const resp = await fetch(`/label/image/${imageId}/labels`);
        existingLabels = await resp.json();
        currentDraft = null;

        updateLabelInfo();

        if (existingLabels.length > 0) {
            const active = choosePrimaryLabel(existingLabels);
            if (active) {
                roi = {
                    x: active.roi_x,
                    y: active.roi_y,
                    width: active.roi_width,
                    height: active.roi_height,
                };
                gtInput.value = active.ground_truth_raw || active.ground_truth || '';
                displayType.value = active.display_type || 'led';
                setBenchmarkPreview(gtInput.value);
                setThumbStatus(imageId, active.verified ? 'verified' : 'suggested');
            }
        } else {
            roi = null;
            gtInput.value = '';
            setBenchmarkPreview('');
            setThumbStatus(imageId, 'unlabeled');
            try {
                const draftResp = await fetch(`/label/image/${imageId}/draft`);
                currentDraft = await draftResp.json();
            } catch (err) {
                currentDraft = null;
            }
            if (currentDraft) {
                roi = {
                    x: currentDraft.roi_x,
                    y: currentDraft.roi_y,
                    width: currentDraft.roi_width,
                    height: currentDraft.roi_height,
                };
                displayType.value = currentDraft.display_type || 'led';
            } else {
                displayType.value = 'led';
            }
            if (allowSuggest && autoSuggestToggle && autoSuggestToggle.checked) {
                await requestSuggestion();
            }
        }

        drawCanvas();
    }

    function loadImage(index) {
        if (index < 0 || index >= thumbItems.length) return;
        currentIndex = index;
        const item = thumbItems[index];
        const src = item.dataset.src;
        const id = parseInt(item.dataset.id, 10);
        const origW = parseInt(item.dataset.width, 10);
        const origH = parseInt(item.dataset.height, 10);

        thumbItems.forEach(t => t.classList.remove('active'));
        item.classList.add('active');
        item.scrollIntoView({ block: 'nearest' });
        counter.textContent = `${index + 1} / ${thumbItems.length}`;

        imgElement = new Image();
        imgElement.onload = async () => {
            const maxW = canvas.parentElement.clientWidth - 20;
            const maxH = window.innerHeight - 220;
            scale = Math.min(1, maxW / imgElement.width, maxH / imgElement.height);
            canvas.width = imgElement.width * scale;
            canvas.height = imgElement.height * scale;
            drawCanvas();
            await loadLabels(id, { allowSuggest: true });
        };

        currentImage = { id, origW, origH };
        existingLabels = [];
        roi = null;
        gtInput.value = '';
        setBenchmarkPreview('');
        labelMsg.textContent = '';
        labelMsg.className = 'form-msg';
        imgElement.src = src;
    }

    async function saveLabel(options) {
        const opts = options || {};
        const advance = opts.advance !== false;
        if (!currentImage) return;

        const groundTruthRaw = sanitizeRaw(gtInput.value);
        gtInput.value = groundTruthRaw;
        setBenchmarkPreview(groundTruthRaw);

        if (!isValidGroundTruth(groundTruthRaw)) {
            labelMsg.textContent = 'Ground truth must contain at least one digit (allowed: 0-9, -, .)';
            labelMsg.className = 'form-msg error';
            return;
        }

        const body = {
            image_id: currentImage.id,
            ground_truth: groundTruthRaw,
            display_type: displayType.value,
            replace: true,
        };

        if (roi) {
            body.roi_x = roi.x;
            body.roi_y = roi.y;
            body.roi_width = roi.width;
            body.roi_height = roi.height;
        } else {
            body.roi_x = 0;
            body.roi_y = 0;
            body.roi_width = currentImage.origW;
            body.roi_height = currentImage.origH;
        }

        try {
            const resp = await fetch('/label/save', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body),
            });
            const data = await resp.json();
            if (!resp.ok || data.error) {
                labelMsg.textContent = data.error || 'Save failed';
                labelMsg.className = 'form-msg error';
                return;
            }

            labelMsg.textContent = `Saved: ${data.ground_truth_raw || data.ground_truth}`;
            labelMsg.className = 'form-msg success';
            await loadLabels(currentImage.id, { allowSuggest: false });

            if (advance) {
                const nextIdx = nextIndexPreferUnresolved();
                if (nextIdx >= 0 && nextIdx !== currentIndex) {
                    loadImage(nextIdx);
                }
            }
        } catch (err) {
            labelMsg.textContent = `Save failed: ${err}`;
            labelMsg.className = 'form-msg error';
        }
    }

    canvas.addEventListener('mousedown', e => {
        drawing = true;
        const rect = canvas.getBoundingClientRect();
        startX = e.clientX - rect.left;
        startY = e.clientY - rect.top;
    });

    canvas.addEventListener('mousemove', e => {
        if (!drawing) return;
        const rect = canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        drawCanvas();
        ctx.strokeStyle = '#d94f7a';
        ctx.lineWidth = 2;
        ctx.setLineDash([5, 5]);
        ctx.strokeRect(startX, startY, x - startX, y - startY);
        ctx.setLineDash([]);
    });

    canvas.addEventListener('mouseup', e => {
        if (!drawing) return;
        drawing = false;
        const rect = canvas.getBoundingClientRect();
        const endX = e.clientX - rect.left;
        const endY = e.clientY - rect.top;

        const x1 = Math.min(startX, endX) / scale;
        const y1 = Math.min(startY, endY) / scale;
        const w = Math.abs(endX - startX) / scale;
        const h = Math.abs(endY - startY) / scale;
        if (w > 5 && h > 5) {
            roi = {
                x: Math.round(x1),
                y: Math.round(y1),
                width: Math.round(w),
                height: Math.round(h),
            };
        }
        drawCanvas();
        gtInput.focus();
    });

    gtInput.addEventListener('input', () => {
        setBenchmarkPreview(gtInput.value);
    });

    saveBtn.addEventListener('click', () => {
        saveLabel({ advance: true });
    });

    window.deleteLabel = async function(labelId) {
        await fetch(`/label/delete/${labelId}`, { method: 'DELETE' });
        if (currentImage) {
            await loadLabels(currentImage.id, { allowSuggest: true });
        }
    };

    deleteBtn.addEventListener('click', async () => {
        if (existingLabels.length > 0) {
            const active = choosePrimaryLabel(existingLabels);
            if (active) {
                await window.deleteLabel(active.id);
            }
        } else {
            if (currentImage) {
                setThumbStatus(currentImage.id, 'unlabeled');
            }
        }
    });

    document.addEventListener('keydown', e => {
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') {
            if (e.key === 'Enter') {
                e.preventDefault();
                saveLabel({ advance: !e.shiftKey });
            }
            return;
        }

        if (e.key === 'ArrowRight' || e.key === 'ArrowDown') {
            e.preventDefault();
            loadImage(currentIndex + 1);
        } else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') {
            e.preventDefault();
            loadImage(currentIndex - 1);
        } else if (e.key === 'Delete') {
            e.preventDefault();
            deleteBtn.click();
        } else if (e.key === 'Enter') {
            e.preventDefault();
            saveLabel({ advance: !e.shiftKey });
        }
    });

    thumbItems.forEach((item, i) => {
        const id = parseInt(item.dataset.id, 10);
        const statusEl = document.getElementById(`status-${id}`);
        const initialStatus = statusEl && statusEl.dataset ? statusEl.dataset.status : 'unlabeled';
        labelStateByImageId.set(id, initialStatus || 'unlabeled');
        item.addEventListener('click', () => loadImage(i));
    });

    if (thumbItems.length > 0) {
        const idx = firstUnresolvedIndex();
        loadImage(idx >= 0 ? idx : 0);
    }
})();
