// Application State
const state = {
  currentImageUrl: null,
  currentSampleName: 'sample_1.jpg',
  uploadedFile: null,
  imageObj: new Image(),
  detections: [],
  selectedId: null,
  hoveredId: null,
  counts: { fish: 0, reefs: 0, human_divers: 0 },
  filters: {
    fish: true,
    reefs: false,
    human_divers: false
  },
  styles: {
    outlines: true,
    masks: true,
    badges: true
  }
};

// Styling Palette per Class
const PALETTES = {
  fish: {
    stroke: '#00ffb3',
    fill: 'rgba(0, 255, 179, 0.24)',
    glow: 'rgba(0, 255, 179, 0.65)',
    labelBg: 'rgba(5, 25, 20, 0.92)',
    badgeColor: '#00ffb3'
  },
  reefs: {
    stroke: '#ffb703',
    fill: 'rgba(255, 183, 3, 0.22)',
    glow: 'rgba(255, 183, 3, 0.55)',
    labelBg: 'rgba(30, 22, 5, 0.92)',
    badgeColor: '#ffb703'
  },
  human_divers: {
    stroke: '#e056fd',
    fill: 'rgba(224, 86, 253, 0.26)',
    glow: 'rgba(224, 86, 253, 0.65)',
    labelBg: 'rgba(28, 8, 32, 0.92)',
    badgeColor: '#e056fd'
  }
};

// DOM References
const canvas = document.getElementById('mainCanvas');
const ctx = canvas.getContext('2d');
const canvasLoader = document.getElementById('canvasLoader');
const viewStatus = document.getElementById('viewStatus');

const confSlider = document.getElementById('confSlider');
const confValue = document.getElementById('confValue');
const showFish = document.getElementById('showFish');
const showReefs = document.getElementById('showReefs');
const showDivers = document.getElementById('showDivers');

const toggleOutlines = document.getElementById('toggleOutlines');
const toggleMasks = document.getElementById('toggleMasks');
const toggleBadges = document.getElementById('toggleBadges');

const runBtn = document.getElementById('runBtn');
const btnSpinner = document.getElementById('btnSpinner');
const btnText = document.getElementById('btnText');

const fishCountVal = document.getElementById('fishCountVal');
const reefCountVal = document.getElementById('reefCountVal');
const diverCountVal = document.getElementById('diverCountVal');
const latencyVal = document.getElementById('latencyVal');
const totalBadge = document.getElementById('totalBadge');

const fishList = document.getElementById('fishList');
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const presetGrid = document.getElementById('presetGrid');
const downloadImgBtn = document.getElementById('downloadImgBtn');

// Initialize
window.addEventListener('DOMContentLoaded', () => {
  setupEventListeners();
  loadSampleThumbnails();
});

function setupEventListeners() {
  confSlider.addEventListener('input', (e) => {
    confValue.textContent = parseFloat(e.target.value).toFixed(2);
  });
  confSlider.addEventListener('change', () => runSegmentation());

  // Filter Toggles
  showFish.addEventListener('change', (e) => { state.filters.fish = e.target.checked; render(); updateSidebar(); });
  showReefs.addEventListener('change', (e) => { state.filters.reefs = e.target.checked; render(); updateSidebar(); });
  showDivers.addEventListener('change', (e) => { state.filters.human_divers = e.target.checked; render(); updateSidebar(); });

  // Style Toggles
  toggleOutlines.addEventListener('change', (e) => { state.styles.outlines = e.target.checked; render(); });
  toggleMasks.addEventListener('change', (e) => { state.styles.masks = e.target.checked; render(); });
  toggleBadges.addEventListener('change', (e) => { state.styles.badges = e.target.checked; render(); });

  runBtn.addEventListener('click', runSegmentation);

  // File Upload
  dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.classList.add('dragover'); });
  dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
  dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0]);
    }
  });

  // Canvas Mouse Events
  canvas.addEventListener('mousemove', handleCanvasHover);
  canvas.addEventListener('mouseleave', () => {
    if (state.hoveredId !== null) {
      state.hoveredId = null;
      render();
    }
  });
  canvas.addEventListener('click', handleCanvasClick);

  // Download Annotated Image
  downloadImgBtn.addEventListener('click', downloadCanvasImage);
}

async function loadSampleThumbnails() {
  try {
    const res = await fetch('/api/samples');
    const samples = await res.json();
    presetGrid.innerHTML = '';

    samples.forEach((s, idx) => {
      const thumb = document.createElement('div');
      thumb.className = `preset-thumb ${idx === 0 ? 'active' : ''}`;
      thumb.dataset.filename = s.filename;
      thumb.innerHTML = `<img src="${s.url}" alt="Sample ${idx + 1}" loading="lazy">`;

      thumb.addEventListener('click', () => {
        document.querySelectorAll('.preset-thumb').forEach(t => t.classList.remove('active'));
        thumb.classList.add('active');
        state.currentSampleName = s.filename;
        state.uploadedFile = null;
        loadImageAndSegment(s.url);
      });

      presetGrid.appendChild(thumb);
    });

    if (samples.length > 0) {
      state.currentSampleName = samples[0].filename;
      loadImageAndSegment(samples[0].url);
    }
  } catch (err) {
    console.error('Failed to load samples:', err);
  }
}

function handleFile(file) {
  state.uploadedFile = file;
  state.currentSampleName = null;
  document.querySelectorAll('.preset-thumb').forEach(t => t.classList.remove('active'));

  const reader = new FileReader();
  reader.onload = (e) => {
    loadImageAndSegment(e.target.result);
  };
  reader.readAsDataURL(file);
}

function loadImageAndSegment(url) {
  state.imageObj.onload = () => {
    canvas.width = state.imageObj.naturalWidth;
    canvas.height = state.imageObj.naturalHeight;
    render();
    runSegmentation();
  };
  state.imageObj.src = url;
}

async function runSegmentation() {
  const startTime = performance.now();
  setLoading(true);
  viewStatus.textContent = 'Running custom YOLO11s-seg GPU inference...';

  try {
    const formData = new FormData();
    if (state.uploadedFile) {
      formData.append('file', state.uploadedFile);
    } else {
      formData.append('sample_name', state.currentSampleName);
    }
    formData.append('conf_threshold', confSlider.value);

    const res = await fetch('/api/segment', {
      method: 'POST',
      body: formData
    });

    if (!res.ok) throw new Error(`Inference error: ${res.statusText}`);

    const data = await res.json();
    const elapsed = Math.round(performance.now() - startTime);

    state.detections = data.detections || [];
    state.counts = data.counts || { fish: 0, reefs: 0, human_divers: 0 };
    latencyVal.textContent = `${elapsed} ms`;

    fishCountVal.textContent = state.counts.fish || 0;
    reefCountVal.textContent = state.counts.reefs || 0;
    diverCountVal.textContent = state.counts.human_divers || 0;

    viewStatus.textContent = `Completed: ${state.detections.length} total segments identified`;

    render();
    updateSidebar();

  } catch (err) {
    console.error(err);
    viewStatus.textContent = `Error: ${err.message}`;
  } finally {
    setLoading(false);
  }
}

function setLoading(isLoading) {
  canvasLoader.style.display = isLoading ? 'flex' : 'none';
  btnSpinner.style.display = isLoading ? 'inline-block' : 'none';
  btnText.textContent = isLoading ? 'Segmenting...' : '⚡ Run Instance Segmentation';
  runBtn.disabled = isLoading;
}

function getVisibleDetections() {
  return state.detections.filter(d => state.filters[d.class_name]);
}

function render() {
  if (!state.imageObj.complete || state.imageObj.naturalWidth === 0) return;

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(state.imageObj, 0, 0);

  const visible = getVisibleDetections();

  visible.forEach((det) => {
    const isSelected = det.id === state.selectedId;
    const isHovered = det.id === state.hoveredId;
    const palette = PALETTES[det.class_name] || PALETTES.fish;

    if (det.polygon && det.polygon.length > 2) {
      // 1. Translucent Mask Fill
      if (state.styles.masks) {
        ctx.beginPath();
        ctx.moveTo(det.polygon[0][0], det.polygon[0][1]);
        for (let i = 1; i < det.polygon.length; i++) {
          ctx.lineTo(det.polygon[i][0], det.polygon[i][1]);
        }
        ctx.closePath();
        ctx.fillStyle = isSelected || isHovered ? 'rgba(0, 255, 179, 0.45)' : palette.fill;
        ctx.fill();
      }

      // 2. Neon Contour Glow
      if (state.styles.outlines) {
        ctx.save();
        ctx.beginPath();
        ctx.moveTo(det.polygon[0][0], det.polygon[0][1]);
        for (let i = 1; i < det.polygon.length; i++) {
          ctx.lineTo(det.polygon[i][0], det.polygon[i][1]);
        }
        ctx.closePath();
        ctx.lineWidth = isSelected || isHovered ? 3.5 : 2.2;
        ctx.strokeStyle = isSelected || isHovered ? '#ffffff' : palette.stroke;
        ctx.shadowColor = isSelected || isHovered ? '#00e5ff' : palette.glow;
        ctx.shadowBlur = isSelected || isHovered ? 18 : 10;
        ctx.stroke();
        ctx.restore();
      }

      // 3. Label Badge
      if (state.styles.badges) {
        drawBadge(det, palette, isSelected || isHovered);
      }
    }
  });
}

function drawBadge(det, palette, isActive) {
  let topY = Infinity;
  let topX = 0;
  det.polygon.forEach(pt => {
    if (pt[1] < topY) {
      topY = pt[1];
      topX = pt[0];
    }
  });

  const label = `${det.class_name.toUpperCase()} (${Math.round(det.confidence * 100)}%)`;
  const subText = `${Math.round(det.oriented_length_px)}px length | ${Math.round(det.area_px)}px²`;

  ctx.save();
  ctx.font = 'bold 11px Outfit, sans-serif';
  const width = Math.max(ctx.measureText(label).width, ctx.measureText(subText).width) + 16;
  const height = 30;
  const x = Math.max(8, topX - width / 2);
  const y = Math.max(8, topY - height - 6);

  ctx.fillStyle = isActive ? 'rgba(5, 15, 25, 0.95)' : palette.labelBg;
  ctx.strokeStyle = isActive ? '#ffffff' : palette.stroke;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.roundRect(x, y, width, height, 5);
  ctx.fill();
  ctx.stroke();

  ctx.fillStyle = '#ffffff';
  ctx.fillText(label, x + 8, y + 13);
  ctx.font = '10px JetBrains Mono, monospace';
  ctx.fillStyle = palette.stroke;
  ctx.fillText(subText, x + 8, y + 25);
  ctx.restore();
}

function updateSidebar() {
  const visible = getVisibleDetections();
  totalBadge.textContent = visible.length;

  if (visible.length === 0) {
    fishList.innerHTML = `<div class="empty-state"><span>No visible instances with current class filters and confidence ≥ ${confSlider.value}</span></div>`;
    return;
  }

  fishList.innerHTML = '';
  visible.forEach((det) => {
    const card = document.createElement('div');
    card.className = `instance-card ${det.id === state.selectedId ? 'selected' : ''}`;
    card.id = `card-${det.id}`;

    const palette = PALETTES[det.class_name] || PALETTES.fish;

    card.innerHTML = `
      <div class="instance-header">
        <span class="instance-class" style="color: ${palette.stroke}">#${det.id} ${det.class_name}</span>
        <span class="instance-conf" style="background: ${palette.fill}; color: ${palette.stroke}">
          ${Math.round(det.confidence * 100)}%
        </span>
      </div>
      <div class="instance-stats">
        <div><small>Oriented Length</small><strong>${det.oriented_length_px} px</strong></div>
        <div><small>Body Depth</small><strong>${det.body_depth_px} px</strong></div>
        <div><small>Mask Area</small><strong>${det.area_px} px²</strong></div>
        <div><small>Polygon Points</small><strong>${det.polygon.length} pts</strong></div>
      </div>
    `;

    card.addEventListener('mouseenter', () => { state.hoveredId = det.id; render(); });
    card.addEventListener('mouseleave', () => { state.hoveredId = null; render(); });
    card.addEventListener('click', () => { state.selectedId = det.id; highlightCard(det.id); render(); });

    fishList.appendChild(card);
  });
}

function highlightCard(id) {
  document.querySelectorAll('.instance-card').forEach(c => c.classList.remove('selected'));
  const el = document.getElementById(`card-${id}`);
  if (el) {
    el.classList.add('selected');
    el.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }
}

function handleCanvasHover(e) {
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  const mouseX = (e.clientX - rect.left) * scaleX;
  const mouseY = (e.clientY - rect.top) * scaleY;

  const visible = getVisibleDetections();
  const hovered = visible.find(det => isPointInPoly([mouseX, mouseY], det.polygon));
  const newId = hovered ? hovered.id : null;

  if (state.hoveredId !== newId) {
    state.hoveredId = newId;
    render();
  }
}

function handleCanvasClick(e) {
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  const clickX = (e.clientX - rect.left) * scaleX;
  const clickY = (e.clientY - rect.top) * scaleY;

  const visible = getVisibleDetections();
  const clicked = visible.find(det => isPointInPoly([clickX, clickY], det.polygon));
  state.selectedId = clicked ? clicked.id : null;
  if (state.selectedId) highlightCard(state.selectedId);
  render();
}

function isPointInPoly(pt, poly) {
  const x = pt[0], y = pt[1];
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const xi = poly[i][0], yi = poly[i][1];
    const xj = poly[j][0], yj = poly[j][1];
    const intersect = ((yi > y) !== (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi) + xi);
    if (intersect) inside = !inside;
  }
  return inside;
}

function downloadCanvasImage() {
  const link = document.createElement('a');
  link.download = `KAUST_Segmentation_${Date.now()}.png`;
  link.href = canvas.toDataURL('image/png');
  link.click();
}
