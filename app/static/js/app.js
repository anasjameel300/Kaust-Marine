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

// Styling Palette per Class (Clean scientific dark theme)
const PALETTES = {
  fish: {
    stroke: '#10b981',
    fill: 'rgba(16, 185, 129, 0.18)',
    glow: 'rgba(16, 185, 129, 0.35)',
    labelBg: 'rgba(6, 14, 24, 0.92)',
    badgeColor: '#10b981'
  },
  reefs: {
    stroke: '#f59e0b',
    fill: 'rgba(245, 158, 11, 0.18)',
    glow: 'rgba(245, 158, 11, 0.35)',
    labelBg: 'rgba(24, 16, 6, 0.92)',
    badgeColor: '#f59e0b'
  },
  human_divers: {
    stroke: '#8b5cf6',
    fill: 'rgba(139, 92, 246, 0.18)',
    glow: 'rgba(139, 92, 246, 0.35)',
    labelBg: 'rgba(18, 10, 28, 0.92)',
    badgeColor: '#8b5cf6'
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
const toggleDualCrop = document.getElementById('toggleDualCrop');

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

const classifyBtn = document.getElementById('classifyBtn');
const classifySpinner = document.getElementById('classifySpinner');
const classifyBtnText = document.getElementById('classifyBtnText');
const classifyBadge = document.getElementById('classifyBadge');

// Initialize
function initApp() {
  setupEventListeners();
  loadSampleThumbnails();
}

if (document.readyState === 'loading') {
  window.addEventListener('DOMContentLoaded', initApp);
} else {
  initApp();
}

function setupEventListeners() {
  confSlider.addEventListener('input', (e) => {
    confValue.textContent = parseFloat(e.target.value).toFixed(2);
  });
  confSlider.addEventListener('change', () => runSegmentation(state.isClassified));

  if (modelSelect) {
    modelSelect.addEventListener('change', () => runSegmentation(state.isClassified));
  }

  if (toggleTiling) {
    toggleTiling.addEventListener('change', () => runSegmentation(state.isClassified));
  }

  if (classifierSelect) {
    classifierSelect.addEventListener('change', () => {
      if (state.isClassified) runSegmentation(true);
    });
  }

  if (toggleDualCrop) {
    toggleDualCrop.addEventListener('change', () => {
      if (state.isClassified) runSegmentation(true);
    });
  }

  if (classifyBtn) {
    classifyBtn.addEventListener('click', () => {
      state.isClassified = true;
      runSegmentation(true);
    });
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

  runBtn.addEventListener('click', () => runSegmentation(false));

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
  state.detections = [];
  state.selectedId = null;
  state.hoveredId = null;
  updateSidebar();
  document.querySelectorAll('.preset-thumb').forEach(t => t.classList.remove('active'));

  const reader = new FileReader();
  reader.onload = (e) => {
    loadImageAndSegment(e.target.result);
  };
  reader.readAsDataURL(file);
}

function loadImageAndSegment(url) {
  state.detections = [];
  state.selectedId = null;
  state.hoveredId = null;
  state.isClassified = false;
  if (classifyBadge) {
    classifyBadge.textContent = 'On-Demand';
    classifyBadge.style.color = 'var(--accent-cyan)';
    classifyBadge.style.background = 'rgba(56, 189, 248, 0.08)';
  }
  if (classifyBtnText) classifyBtnText.textContent = 'Run Identification (Species ID)';
  updateSidebar();

  const img = new Image();
  let done = false;
  const onReady = () => {
    if (done) return;
    done = true;
    state.imageObj = img;
    canvas.width = img.naturalWidth || 640;
    canvas.height = img.naturalHeight || 640;
    state.detections = [];
    render();
    // Auto-trigger Stage 1 instance segmentation
    runSegmentation(false);
  };

  img.onload = onReady;
  img.onerror = (err) => {
    console.error('Failed to load image:', url, err);
    runSegmentation(false);
  };

  img.src = url;
  if (img.complete && img.naturalWidth > 0) {
    onReady();
  }
}

async function runSegmentation(runClassify = false) {
  const startTime = performance.now();
  setLoading(true);

  if (runClassify && classifySpinner) {
    classifySpinner.style.display = 'inline-block';
    if (classifyBtnText) classifyBtnText.textContent = 'Classifying Species...';
    if (classifyBtn) classifyBtn.disabled = true;
  }

  viewStatus.textContent = runClassify
    ? 'Running Stage 2 taxonomic ensemble classification & allometric sizing...'
    : 'Running fast Stage 1 instance segmentation & morphometrics...';

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
    formData.append('run_classification', runClassify ? 'true' : 'false');
    if (classifierSelect) {
      formData.append('classifier_mode', classifierSelect.value);
    }
    if (toggleDualCrop) {
      formData.append('enable_dual_crop', toggleDualCrop.checked);
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
    if (latencyVal) latencyVal.textContent = `${elapsed} ms`;

    if (fishCountVal) fishCountVal.textContent = state.counts.fish || 0;
    if (reefCountVal) reefCountVal.textContent = state.counts.reefs || 0;
    if (diverCountVal) diverCountVal.textContent = state.counts.human_divers || 0;

    // Update Community Metrics (Biomass, Catch, Shannon Diversity)
    if (data.community_metrics) {
      const cm = data.community_metrics;
      if (totalBiomassVal) totalBiomassVal.textContent = `${cm.total_biomass_kg.toFixed(2)} kg`;
      if (commercialYieldVal) commercialYieldVal.textContent = `${cm.commercial_biomass_kg.toFixed(2)} kg (${Math.round(cm.commercial_ratio_pct)}%)`;
      if (shannonVal) shannonVal.textContent = cm.shannon_diversity_index.toFixed(2);
    }

    if (runClassify) {
      state.isClassified = true;
      if (classifyBadge) {
        classifyBadge.textContent = 'Classified';
        classifyBadge.style.color = '#10b981';
        classifyBadge.style.background = 'rgba(16, 185, 129, 0.15)';
      }
      if (classifyBtnText) classifyBtnText.textContent = 'Re-Identify Species';
    } else {
      state.isClassified = false;
      if (classifyBadge) {
        classifyBadge.textContent = 'On-Demand';
        classifyBadge.style.color = 'var(--accent-cyan)';
        classifyBadge.style.background = 'rgba(56, 189, 248, 0.08)';
      }
      if (classifyBtnText) classifyBtnText.textContent = 'Run Identification (Species ID)';
    }

    viewStatus.textContent = `Completed in ${elapsed}ms: ${state.detections.length} total segments (${data.model_title}${runClassify ? ' • Mode: ' + data.classifier_mode : ''})`;

    render();
    updateSidebar();

  } catch (err) {
    console.error(err);
    viewStatus.textContent = `Error: ${err.message}`;
  } finally {
    setLoading(false);
    if (classifySpinner) classifySpinner.style.display = 'none';
    if (classifyBtn) classifyBtn.disabled = false;
  }
}

function setLoading(isLoading) {
  canvasLoader.style.display = isLoading ? 'flex' : 'none';
  btnSpinner.style.display = isLoading ? 'inline-block' : 'none';
  btnText.textContent = isLoading ? 'Computing...' : 'Re-Run Segmentation';
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

  let label = `${det.class_name.toUpperCase()} ${Math.round(det.confidence * 100)}%`;
  let subText = `${Math.round(det.oriented_length_px)}px • ${Math.round(det.area_px)}px²`;

  if (det.taxonomy && det.taxonomy.family) {
    const fam = det.taxonomy.family;
    const cname = det.taxonomy.common_name ? ` (${det.taxonomy.common_name})` : '';
    const taxConf = Math.round((det.taxonomy.confidence || 0) * 100);
    const isVer = det.taxonomy.is_verified !== false;
    label = isVer ? `${fam}${cname} ${taxConf}%` : `Unverified (Guess: ${fam} ${taxConf}%)`;
    if (det.biomass && det.biomass.fork_length_cm) {
      subText = `L=${det.biomass.fork_length_cm}cm • W=${det.biomass.biomass_g}g`;
    }
  }

  ctx.save();
  ctx.font = '500 11px Outfit, sans-serif';
  const width = Math.max(ctx.measureText(label).width, ctx.measureText(subText).width) + 14;
  const height = 28;
  const x = Math.max(6, topX - width / 2);
  const y = Math.max(6, topY - height - 4);

  ctx.fillStyle = isActive ? 'rgba(7, 13, 24, 0.95)' : palette.labelBg;
  ctx.strokeStyle = isActive ? '#38bdf8' : palette.stroke;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.roundRect(x, y, width, height, 4);
  ctx.fill();
  ctx.stroke();

  ctx.fillStyle = '#f8fafc';
  ctx.fillText(label, x + 7, y + 12);
  ctx.font = '400 9px JetBrains Mono, monospace';
  ctx.fillStyle = palette.stroke;
  ctx.fillText(subText, x + 7, y + 23);
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

    const modelTag = det.source === 'finetuned' ? 'Fine-Tuned' : (det.source === 'yolov8' ? 'COCO' : 'Baseline');
    const scaleTag = det.scale === 'micro' ? 'Micro' : 'Macro';

    let taxHtml = '';
    if (det.taxonomy) {
      const tax = det.taxonomy;
      const isVer = tax.is_verified !== false;
      const titleColor = isVer ? '#f8fafc' : '#fbbf24';
      const statusNotice = isVer
        ? `<div style="font-size: 0.64rem; color: #94a3b8;">Guild: <span style="color: #cbd5e1;">${tax.trophic_guild}</span> | Fishery: <span style="color: ${tax.is_commercial ? '#10b981' : '#94a3b8'};">${tax.is_commercial ? 'Commercial' : 'Non-Target'}</span></div>`
        : `<div style="font-size: 0.64rem; color: #fbbf24; margin-top: 2px;">Specimen unverified (confidence &lt; 65% / OOD). Best guess: <strong style="color: #fde68a;">${tax.family}</strong> (${Math.round((tax.confidence || 0) * 100)}%).</div>`;

      taxHtml = `
        <div style="margin-top: 6px; padding: 6px 8px; background: ${isVer ? 'rgba(56, 189, 248, 0.04)' : 'rgba(245, 158, 11, 0.06)'}; border-radius: 4px; border: 1px solid ${isVer ? 'rgba(56, 189, 248, 0.15)' : 'rgba(245, 158, 11, 0.25)'};">
          <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 2px;">
            <strong style="font-size: 0.74rem; color: ${titleColor};">${isVer ? `${tax.family} ${tax.common_name ? '• ' + tax.common_name : ''}` : `Unverified Specimen (Guess: ${tax.family})`}</strong>
            <span style="font-size: 0.65rem; color: #94a3b8; font-style: italic;">${tax.scientific_name}</span>
          </div>
          ${statusNotice}
        </div>
      `;
    } else if (det.class_name === 'fish') {
      taxHtml = `
        <div style="margin-top: 5px; font-size: 0.65rem; color: #64748b;">
          Taxonomy unassigned • Click 'Run Identification'
        </div>
      `;
    }

    let roiHtml = '';
    if (det.taxonomy && det.taxonomy.rois && det.taxonomy.rois.head_thumb) {
      const rois = det.taxonomy.rois;
      roiHtml = `
        <div style="margin-top: 6px; padding: 6px 8px; background: rgba(15, 23, 42, 0.7); border-radius: 4px; border: 1px dashed rgba(56, 189, 248, 0.25);">
          <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px;">
            <span style="font-size: 0.62rem; color: #38bdf8; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">Dual-Crop ROIs (Fish-Vista)</span>
            <span style="font-size: 0.58rem; color: ${rois.anatomical_match ? '#34d399' : '#fbbf24'};">${rois.anatomical_match ? '✓ Anatomical Consensus' : 'Split Morphology'}</span>
          </div>
          <div style="display: flex; gap: 8px;">
            <div style="flex: 1; text-align: center; background: rgba(0,0,0,0.35); border-radius: 4px; padding: 4px;">
              <img src="${rois.head_thumb}" alt="Cephalic ROI" style="width: 100%; height: 44px; object-fit: cover; border-radius: 3px; border: 1px solid rgba(56, 189, 248, 0.35);">
              <span style="display: block; font-size: 0.58rem; color: #94a3b8; margin-top: 3px;">Cephalic (35%)</span>
              <span style="display: block; font-size: 0.60rem; color: #e2e8f0; font-weight: 500;">${rois.cephalic_family || 'Snout'}</span>
            </div>
            <div style="flex: 1; text-align: center; background: rgba(0,0,0,0.35); border-radius: 4px; padding: 4px;">
              <img src="${rois.caudal_thumb}" alt="Caudal ROI" style="width: 100%; height: 44px; object-fit: cover; border-radius: 3px; border: 1px solid rgba(56, 189, 248, 0.35);">
              <span style="display: block; font-size: 0.58rem; color: #94a3b8; margin-top: 3px;">Caudal (25%)</span>
              <span style="display: block; font-size: 0.60rem; color: #e2e8f0; font-weight: 500;">${rois.caudal_family || 'Spine'}</span>
            </div>
          </div>
        </div>
      `;
    }

    const bioHtml = det.biomass ? `
      <div class="instance-grid">
        <span>Length<strong>${det.biomass.fork_length_cm} cm</strong></span>
        <span>Depth<strong>${det.biomass.body_depth_cm} cm</strong></span>
        <span>Biomass<strong>${det.biomass.biomass_g} g</strong></span>
        <span>Aspect<strong>${det.biomass.aspect_ratio}</strong></span>
      </div>
    ` : `
      <div class="instance-grid">
        <span>Length<strong>${det.oriented_length_px} px</strong></span>
        <span>Depth<strong>${det.body_depth_px} px</strong></span>
        <span>Area<strong>${Math.round(det.area_px)} px²</strong></span>
        <span>Points<strong>${det.polygon.length}</strong></span>
      </div>
    `;

    card.innerHTML = `
      <div class="instance-header">
        <div class="instance-badges">
          <span class="instance-title">#${det.id} ${det.taxonomy ? det.taxonomy.family : det.class_name}</span>
          <span class="pill-badge pill-cyan">${modelTag}</span>
          <span class="pill-badge">${scaleTag}</span>
        </div>
        <span class="pill-badge pill-green">${Math.round(det.confidence * 100)}%</span>
      </div>
      ${taxHtml}
      ${roiHtml}
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
