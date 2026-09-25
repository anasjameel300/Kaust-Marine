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
const modelSelect = document.getElementById('modelSelect');
const modelStatusBadge = document.getElementById('modelStatusBadge');
const subBrandModel = document.getElementById('subBrandModel');
const showFish = document.getElementById('showFish');

// Stage 2 & 3 Controls
const classifierSelect = document.getElementById('classifierSelect');
const presetSelect = document.getElementById('presetSelect');
const scaleSlider = document.getElementById('scaleSlider');
const scaleValue = document.getElementById('scaleValue');

const showReefs = document.getElementById('showReefs');
const showDivers = document.getElementById('showDivers');

const toggleOutlines = document.getElementById('toggleOutlines');
const toggleMasks = document.getElementById('toggleMasks');
const toggleBadges = document.getElementById('toggleBadges');
const toggleTiling = document.getElementById('toggleTiling');

const runBtn = document.getElementById('runBtn');
const btnSpinner = document.getElementById('btnSpinner');
const btnText = document.getElementById('btnText');

const fishCountVal = document.getElementById('fishCountVal');
const reefCountVal = document.getElementById('reefCountVal');
const diverCountVal = document.getElementById('diverCountVal');
const totalBiomassVal = document.getElementById('totalBiomassVal');
const commercialYieldVal = document.getElementById('commercialYieldVal');
const shannonVal = document.getElementById('shannonVal');
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

  if (modelSelect) {
    modelSelect.addEventListener('change', () => runSegmentation());
  }

  if (toggleTiling) {
    toggleTiling.addEventListener('change', () => runSegmentation());
  }

  if (classifierSelect) {
    classifierSelect.addEventListener('change', () => runSegmentation());
  }

  if (presetSelect) {
    presetSelect.addEventListener('change', (e) => {
      const p = e.target.value;
      if (p === 'bruvs_rig') {
        scaleSlider.value = 22.0;
        scaleValue.textContent = '22.0';
      } else if (p === 'measuring_board') {
        scaleSlider.value = 35.0;
        scaleValue.textContent = '35.0';
      }
      runSegmentation();
    });
  }

  if (scaleSlider) {
    scaleSlider.addEventListener('input', (e) => {
      scaleValue.textContent = parseFloat(e.target.value).toFixed(1);
    });
    scaleSlider.addEventListener('change', () => {
      if (presetSelect && presetSelect.value !== 'custom') {
        presetSelect.value = 'custom';
      }
      runSegmentation();
    });
  }

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
  viewStatus.textContent = 'Running multi-model detection, taxonomic classification & biomass pipeline...';

  try {
    const formData = new FormData();
    if (state.uploadedFile) {
      formData.append('file', state.uploadedFile);
    } else {
      formData.append('sample_name', state.currentSampleName);
    }
    formData.append('conf_threshold', confSlider.value);
    if (modelSelect) {
      formData.append('model_version', modelSelect.value);
    }
    if (toggleTiling) {
      formData.append('enable_tiling', toggleTiling.checked);
    }
    if (classifierSelect) {
      formData.append('classifier_mode', classifierSelect.value);
    }
    if (presetSelect) {
      formData.append('calibration_preset', presetSelect.value);
    }
    if (scaleSlider) {
      formData.append('pixels_per_cm', scaleSlider.value);
    }

    const res = await fetch('/api/segment', {
      method: 'POST',
      body: formData
    });

    if (!res.ok) throw new Error(`Inference error: ${res.statusText}`);

    const data = await res.json();
    const elapsed = Math.round(performance.now() - startTime);

    if (data.model_title && modelStatusBadge) {
      modelStatusBadge.textContent = data.model_title;
    }
    if (subBrandModel) {
      subBrandModel.textContent = `${data.model_title} ${data.enable_tiling ? ' + Scale-Sliced Tiling' : ''}`;
    }

    state.detections = data.detections || [];
    state.counts = data.counts || { fish: 0, reefs: 0, human_divers: 0 };
    latencyVal.textContent = `${elapsed} ms`;

    fishCountVal.textContent = state.counts.fish || 0;
    reefCountVal.textContent = state.counts.reefs || 0;
    diverCountVal.textContent = state.counts.human_divers || 0;

    // Update Community Metrics (Biomass, Catch, Shannon Diversity)
    if (data.community_metrics) {
      const cm = data.community_metrics;
      if (totalBiomassVal) totalBiomassVal.textContent = `${cm.total_biomass_kg.toFixed(2)} kg`;
      if (commercialYieldVal) commercialYieldVal.textContent = `${cm.commercial_biomass_kg.toFixed(2)} kg (${Math.round(cm.commercial_ratio_pct)}%)`;
      if (shannonVal) shannonVal.textContent = cm.shannon_diversity_index.toFixed(2);
    }

    viewStatus.textContent = `Completed: ${state.detections.length} total segments identified (${data.model_title} | Mode: ${data.classifier_mode})`;

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
  btnText.textContent = isLoading ? 'Computing...' : '⚡ Run Instance Segmentation';
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

  let label = `${det.class_name.toUpperCase()} (${Math.round(det.confidence * 100)}%)`;
  let subText = `${Math.round(det.oriented_length_px)}px length | ${Math.round(det.area_px)}px²`;

  if (det.taxonomy && det.taxonomy.family) {
    const fam = det.taxonomy.family.toUpperCase();
    const cname = det.taxonomy.common_name ? ` (${det.taxonomy.common_name})` : '';
    const taxConf = Math.round((det.taxonomy.confidence || 0) * 100);
    label = `${fam}${cname} ${taxConf}%`;
    if (det.biomass && det.biomass.fork_length_cm) {
      subText = `L=${det.biomass.fork_length_cm}cm | W=${det.biomass.biomass_g}g`;
    }
  }

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

    const sourceBadge = det.source === 'consensus'
      ? `<span style="background: rgba(0, 255, 179, 0.15); color: #00ffb3; font-size: 0.65rem; padding: 2px 6px; border-radius: 4px; border: 1px solid rgba(0, 255, 179, 0.35); font-weight: 600;">Consensus ✨</span>`
      : (det.source === 'finetuned'
        ? `<span style="background: rgba(224, 86, 253, 0.15); color: #e056fd; font-size: 0.65rem; padding: 2px 6px; border-radius: 4px; border: 1px solid rgba(224, 86, 253, 0.35); font-weight: 600;">Fine-Tuned 🎯</span>`
        : `<span style="background: rgba(103, 232, 249, 0.15); color: #67e8f9; font-size: 0.65rem; padding: 2px 6px; border-radius: 4px; border: 1px solid rgba(103, 232, 249, 0.35); font-weight: 600;">Baseline 🏆</span>`);

    const scaleBadge = det.scale === 'micro'
      ? `<span style="background: rgba(255, 183, 3, 0.15); color: #ffb703; font-size: 0.65rem; padding: 2px 6px; border-radius: 4px; border: 1px solid rgba(255, 183, 3, 0.35); font-weight: 600;">Micro 📐</span>`
      : `<span style="background: rgba(255, 255, 255, 0.08); color: #94a3b8; font-size: 0.65rem; padding: 2px 6px; border-radius: 4px; font-weight: 600;">Macro 🔍</span>`;

    // Agreement and Taxonomic Consensus tags
    let agreementBadge = '';
    let votePills = '';
    let taxMetaHtml = '';

    if (det.taxonomy) {
      const tax = det.taxonomy;
      const statusColor = tax.consensus_level === 'high' ? '#34d399' : (tax.consensus_level === 'moderate' ? '#fbbf24' : '#f87171');
      agreementBadge = `<span style="background: rgba(168, 85, 247, 0.15); color: ${statusColor}; font-size: 0.65rem; padding: 2px 6px; border-radius: 4px; border: 1px solid ${statusColor}44; font-weight: 600;">${tax.agreement_status}</span>`;

      if (tax.model_votes) {
        votePills = Object.entries(tax.model_votes).map(([mName, predFam]) => {
          return `<span style="display:inline-block; font-size:0.62rem; background:rgba(255,255,255,0.06); padding:2px 5px; border-radius:3px; margin:2px; color:#cbd5e1;"><strong style="color:#a855f7;">${mName}:</strong> ${predFam}</span>`;
        }).join('');
      }

      taxMetaHtml = `
        <div style="margin-top: 6px; padding: 6px 8px; background: rgba(168, 85, 247, 0.06); border-radius: 6px; border: 1px solid rgba(168, 85, 247, 0.2);">
          <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 3px;">
            <span style="font-size: 0.72rem; color: #c084fc; font-weight: 600;">Taxon: ${tax.family} (${tax.common_name})</span>
            <span style="font-size: 0.68rem; color: #94a3b8; font-style: italic;">${tax.scientific_name}</span>
          </div>
          <div style="font-size: 0.65rem; color: #94a3b8; margin-bottom: 4px;">
            Guild: <strong style="color: #f1f5f9;">${tax.trophic_guild}</strong> | Target: <strong style="color: ${tax.is_commercial ? '#34d399' : '#94a3b8'};">${tax.is_commercial ? 'Commercial Fishery' : 'Ecological'}</strong>
          </div>
          ${votePills ? `<div style="display:flex; flex-wrap:wrap; gap:2px; margin-top:3px;">${votePills}</div>` : ''}
        </div>
      `;
    }

    // Morphometrics & Biomass table
    const bioHtml = det.biomass ? `
      <div class="instance-stats" style="margin-top: 6px; border-top: 1px solid rgba(255,255,255,0.06); padding-top: 6px;">
        <div><small>Fork Length (L)</small><strong style="color: #34d399;">${det.biomass.fork_length_cm} cm</strong></div>
        <div><small>Body Depth (D)</small><strong style="color: #38bdf8;">${det.biomass.body_depth_cm} cm</strong></div>
        <div><small>Biomass (W)</small><strong style="color: #fbbf24;">${det.biomass.biomass_g} g</strong></div>
        <div><small>Aspect (L/D)</small><strong>${det.biomass.aspect_ratio}</strong></div>
      </div>
    ` : `
      <div class="instance-stats">
        <div><small>Oriented Length</small><strong>${det.oriented_length_px} px</strong></div>
        <div><small>Body Depth</small><strong>${det.body_depth_px} px</strong></div>
        <div><small>Mask Area</small><strong>${det.area_px} px²</strong></div>
        <div><small>Polygon Points</small><strong>${det.polygon.length} pts</strong></div>
      </div>
    `;

    card.innerHTML = `
      <div class="instance-header">
        <div style="display: flex; align-items: center; gap: 6px; flex-wrap: wrap;">
          <span class="instance-class" style="color: ${palette.stroke}">#${det.id} ${det.taxonomy ? det.taxonomy.family : det.class_name}</span>
          ${sourceBadge}
          ${scaleBadge}
          ${agreementBadge}
        </div>
        <span class="instance-conf" style="background: ${palette.fill}; color: ${palette.stroke}">
          ${Math.round(det.confidence * 100)}%
        </span>
      </div>
      ${taxMetaHtml}
      ${bioHtml}
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
