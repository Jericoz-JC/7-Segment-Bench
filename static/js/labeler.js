/**
 * Canvas-based ROI labeling tool for 7-segment display images.
 * Draws bounding boxes, captures coordinates scaled to original resolution.
 */
(function() {
    const canvas = document.getElementById('labeler-canvas');
    const ctx = canvas.getContext('2d');
    const gtInput = document.getElementById('gt-input');
    const displayType = document.getElementById('display-type');
    const saveBtn = document.getElementById('save-btn');
    const deleteBtn = document.getElementById('delete-btn');
    const labelMsg = document.getElementById('label-msg');
    const labelInfo = document.getElementById('label-info');
    const counter = document.getElementById('img-counter');
    const thumbList = document.getElementById('thumb-list');

    const thumbItems = Array.from(document.querySelectorAll('.thumb-item'));
    let currentIndex = 0;
    let currentImage = null;
    let imgElement = new Image();
    let scale = 1;
    let roi = null;
    let drawing = false;
    let startX = 0, startY = 0;
    let existingLabels = [];

    function loadImage(index) {
        if (index < 0 || index >= thumbItems.length) return;
        currentIndex = index;
        const item = thumbItems[index];
        const src = item.dataset.src;
        const id = parseInt(item.dataset.id);
        const origW = parseInt(item.dataset.width);
        const origH = parseInt(item.dataset.height);

        // Highlight current thumbnail
        thumbItems.forEach(t => t.classList.remove('active'));
        item.classList.add('active');
        item.scrollIntoView({ block: 'nearest' });
        counter.textContent = `${index + 1} / ${thumbItems.length}`;

        imgElement = new Image();
        imgElement.onload = () => {
            const maxW = canvas.parentElement.clientWidth - 20;
            const maxH = window.innerHeight - 200;
            scale = Math.min(1, maxW / imgElement.width, maxH / imgElement.height);
            canvas.width = imgElement.width * scale;
            canvas.height = imgElement.height * scale;
            drawCanvas();
            loadLabels(id);
        };
        imgElement.src = src;
        currentImage = { id, origW, origH };
        roi = null;
        gtInput.value = '';
        labelMsg.textContent = '';
    }

    function drawCanvas() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(imgElement, 0, 0, canvas.width, canvas.height);

        // Draw existing labels
        existingLabels.forEach(l => {
            ctx.strokeStyle = '#00ff88';
            ctx.lineWidth = 2;
            ctx.strokeRect(l.roi_x * scale, l.roi_y * scale, l.roi_width * scale, l.roi_height * scale);
            ctx.fillStyle = 'rgba(0,255,136,0.15)';
            ctx.fillRect(l.roi_x * scale, l.roi_y * scale, l.roi_width * scale, l.roi_height * scale);
            ctx.fillStyle = '#00ff88';
            ctx.font = '14px monospace';
            ctx.fillText(l.ground_truth, l.roi_x * scale + 4, l.roi_y * scale - 4);
        });

        // Draw current ROI
        if (roi) {
            ctx.strokeStyle = '#ffaa00';
            ctx.lineWidth = 2;
            ctx.setLineDash([5, 5]);
            ctx.strokeRect(roi.x * scale, roi.y * scale, roi.width * scale, roi.height * scale);
            ctx.setLineDash([]);
        }
    }

    async function loadLabels(imageId) {
        const resp = await fetch(`/label/image/${imageId}/labels`);
        existingLabels = await resp.json();
        updateLabelInfo();
        drawCanvas();

        // Update thumbnail status
        const statusEl = document.getElementById(`status-${imageId}`);
        if (statusEl) {
            statusEl.textContent = existingLabels.length > 0 ? '\u2713' : '';
            statusEl.className = 'thumb-status' + (existingLabels.length > 0 ? ' labeled' : '');
        }

        // Pre-fill if single label exists
        if (existingLabels.length === 1) {
            const l = existingLabels[0];
            roi = { x: l.roi_x, y: l.roi_y, width: l.roi_width, height: l.roi_height };
            gtInput.value = l.ground_truth;
            displayType.value = l.display_type;
        }
    }

    function updateLabelInfo() {
        if (existingLabels.length === 0) {
            labelInfo.innerHTML = '<p class="empty-state">No labels for this image.</p>';
            return;
        }
        labelInfo.innerHTML = existingLabels.map(l =>
            `<div class="label-entry">
                <span class="label-gt">${l.ground_truth}</span>
                <span class="label-roi">[${l.roi_x},${l.roi_y} ${l.roi_width}x${l.roi_height}]</span>
                <span class="label-type">${l.display_type}</span>
                <button class="btn btn-sm btn-danger" onclick="deleteLabel(${l.id})">x</button>
            </div>`
        ).join('');
    }

    // Canvas mouse events for ROI drawing
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
        ctx.strokeStyle = '#ffaa00';
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
            roi = { x: Math.round(x1), y: Math.round(y1), width: Math.round(w), height: Math.round(h) };
        }
        drawCanvas();
        gtInput.focus();
    });

    // Save label
    async function saveLabel() {
        if (!currentImage) return;
        const gt = gtInput.value.trim();
        if (!gt) {
            labelMsg.textContent = 'Enter ground truth digits';
            labelMsg.className = 'form-msg error';
            return;
        }
        if (!/^\d+$/.test(gt)) {
            labelMsg.textContent = 'Ground truth must be digits only';
            labelMsg.className = 'form-msg error';
            return;
        }

        const body = {
            image_id: currentImage.id,
            ground_truth: gt,
            display_type: displayType.value,
            replace: false,
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

        const resp = await fetch('/label/save', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body)
        });
        const data = await resp.json();
        if (data.error) {
            labelMsg.textContent = data.error;
            labelMsg.className = 'form-msg error';
        } else {
            labelMsg.textContent = `Saved: ${data.ground_truth}`;
            labelMsg.className = 'form-msg success';
            loadLabels(currentImage.id);
        }
    }

    saveBtn.addEventListener('click', saveLabel);

    // Delete label
    window.deleteLabel = async function(labelId) {
        await fetch(`/label/delete/${labelId}`, { method: 'DELETE' });
        if (currentImage) loadLabels(currentImage.id);
    };

    deleteBtn.addEventListener('click', async () => {
        if (existingLabels.length > 0) {
            const lastLabel = existingLabels[existingLabels.length - 1];
            await window.deleteLabel(lastLabel.id);
        }
    });

    // Keyboard navigation
    document.addEventListener('keydown', e => {
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') {
            if (e.key === 'Enter') {
                e.preventDefault();
                saveLabel();
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
        }
    });

    // Click on thumbnail
    thumbItems.forEach((item, i) => {
        item.addEventListener('click', () => loadImage(i));
    });

    // Initial load
    if (thumbItems.length > 0) {
        loadImage(0);
    }
})();
