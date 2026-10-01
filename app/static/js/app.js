// Application State
const state = {
  currentImageUrl: null,
  currentSampleName: 'sample_1.jpg',
  uploadedFile: null,
  imageObj: new Image(),
  detections: [],
  selectedId: null,
  hoveredId: null,
  isCalibrated: false,
  isClassified: false,
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
const calibStatusBadge = document.getElementById('calibStatusBadge');
const confirmCalibBtn = document.getElementById('confirmCalibBtn');
const confirmCalibBtnText = document.getElementById('confirmCalibBtnText');

function updateCalibUI() {
  if (!calibStatusBadge || !confirmCalibBtnText) return;
  if (state.isCalibrated) {
    const scale = parseFloat(scaleSlider ? scaleSlider.value : 25.0).toFixed(1);
    calibStatusBadge.textContent = `Calibrated (${scale} px/cm)`;
    calibStatusBadge.style.color = '#10b981';
    calibStatusBadge.style.background = 'rgba(16, 185, 129, 0.15)';
    calibStatusBadge.style.borderColor = 'rgba(16, 185, 129, 0.35)';
    confirmCalibBtnText.textContent = 'Modify / Reset Scale';
    if (confirmCalibBtn) {
      confirmCalibBtn.style.color = '#94a3b8';
      confirmCalibBtn.style.borderColor = 'rgba(148, 163, 184, 0.3)';
    }
  } else {
    calibStatusBadge.textContent = 'Uncalibrated';
    calibStatusBadge.style.color = '#fbbf24';
    calibStatusBadge.style.background = 'rgba(245, 158, 11, 0.12)';
    calibStatusBadge.style.borderColor = 'rgba(245, 158, 11, 0.25)';
    confirmCalibBtnText.textContent = 'Confirm Calibration Scale';
    if (confirmCalibBtn) {
      confirmCalibBtn.style.color = '#38bdf8';
      confirmCalibBtn.style.borderColor = 'rgba(56, 189, 248, 0.4)';
    }
  }
}

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
  setupVideoModule();
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

  // Dual-Crop Research Information Modal
  const dualCropModal = document.getElementById('dualCropModal');
  const dualCropInfoBtn = document.getElementById('dualCropInfoBtn');
  const closeDualCropModal = document.getElementById('closeDualCropModal');
  const confirmDualCropModal = document.getElementById('confirmDualCropModal');

  function openDualCropModal() {
    if (dualCropModal) dualCropModal.style.display = 'flex';
  }
  function closeDualCropModalFn() {
    if (dualCropModal) dualCropModal.style.display = 'none';
  }

  if (dualCropInfoBtn) dualCropInfoBtn.addEventListener('click', openDualCropModal);
  if (closeDualCropModal) closeDualCropModal.addEventListener('click', closeDualCropModalFn);
  if (confirmDualCropModal) confirmDualCropModal.addEventListener('click', closeDualCropModalFn);
  if (dualCropModal) {
    dualCropModal.addEventListener('click', (e) => {
      if (e.target === dualCropModal) closeDualCropModalFn();
    });
  }

  if (confirmCalibBtn) {
    confirmCalibBtn.addEventListener('click', () => {
      state.isCalibrated = !state.isCalibrated;
      updateCalibUI();
      runSegmentation(state.isClassified);
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
      if (state.isCalibrated) {
        runSegmentation(state.isClassified);
      }
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
      if (state.isCalibrated) {
        runSegmentation(state.isClassified);
      }
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
  state.isCalibrated = false;
  updateCalibUI();
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
    formData.append('is_calibrated', state.isCalibrated ? 'true' : 'false');

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
      if (cm.is_calibrated && cm.total_biomass_kg !== null) {
        if (totalBiomassVal) {
          totalBiomassVal.textContent = `${cm.total_biomass_kg.toFixed(2)} kg`;
          totalBiomassVal.style.color = 'var(--accent-cyan)';
        }
        if (commercialYieldVal) {
          commercialYieldVal.textContent = `${cm.commercial_biomass_kg.toFixed(2)} kg (${Math.round(cm.commercial_ratio_pct)}%)`;
          commercialYieldVal.style.color = 'var(--color-reefs)';
        }
      } else {
        if (totalBiomassVal) {
          totalBiomassVal.textContent = 'Uncalibrated';
          totalBiomassVal.style.color = '#fbbf24';
        }
        if (commercialYieldVal) {
          commercialYieldVal.textContent = 'Pending Scale';
          commercialYieldVal.style.color = '#94a3b8';
        }
      }
      if (shannonVal) shannonVal.textContent = (cm.shannon_diversity_index || 0).toFixed(2);
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
  let subText = `${Math.round(det.oriented_length_px)}px • Uncalibrated`;

  if (det.taxonomy && det.taxonomy.family) {
    const fam = det.taxonomy.family;
    const cname = det.taxonomy.common_name ? ` (${det.taxonomy.common_name})` : '';
    const taxConf = Math.round((det.taxonomy.confidence || 0) * 100);
    const isVer = det.taxonomy.is_verified !== false;
    label = isVer ? `${fam}${cname} ${taxConf}%` : `Unverified (Guess: ${fam} ${taxConf}%)`;
    if (det.biomass && det.biomass.is_calibrated && det.biomass.fork_length_cm) {
      subText = `L=${det.biomass.fork_length_cm}cm • W=${det.biomass.biomass_g}g`;
    } else {
      subText = `${Math.round(det.oriented_length_px)}px • Uncalibrated`;
    }
  } else if (det.biomass && det.biomass.is_calibrated && det.biomass.fork_length_cm) {
    subText = `L=${det.biomass.fork_length_cm}cm • W≈${det.biomass.biomass_g}g*`;
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
            <div style="display: flex; align-items: center; gap: 4px;">
              <span style="font-size: 0.62rem; color: #38bdf8; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">Dual-Crop ROIs</span>
              <button type="button" class="btn-roi-info" style="background: none; border: none; color: #38bdf8; cursor: pointer; padding: 0 2px; font-size: 0.70rem;" title="View Research Notes &amp; Status" onclick="document.getElementById('dualCropModal').style.display='flex'">👁</button>
            </div>
            <span style="font-size: 0.58rem; color: ${rois.consensus_color || (rois.anatomical_match ? '#34d399' : '#fbbf24')}; font-weight: 500;">✓ ${rois.consensus_badge || (rois.anatomical_match ? 'Anatomical Consensus' : 'Macro-Anchored')}</span>
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

    let bioHtml = '';
    if (det.biomass && det.biomass.is_calibrated && det.biomass.fork_length_cm) {
      const isTaxBio = det.biomass.is_taxonomic_classified;
      const weightLabel = isTaxBio ? `${det.biomass.biomass_g} g` : `~${det.biomass.biomass_g} g*`;
      const weightSub = isTaxBio
        ? `<small style="display:block; font-size:0.58rem; color:#34d399;">FishBase Allometric</small>`
        : `<small style="display:block; font-size:0.58rem; color:#fbbf24;">Heuristic (ID needed)</small>`;
      bioHtml = `
        <div class="instance-grid">
          <span>Length<strong>${det.biomass.fork_length_cm} cm</strong></span>
          <span>Depth<strong>${det.biomass.body_depth_cm} cm</strong></span>
          <span>Biomass<strong>${weightLabel}</strong>${weightSub}</span>
          <span>Aspect<strong>${det.biomass.aspect_ratio}</strong></span>
        </div>
      `;
    } else {
      const asp = (det.oriented_length_px / Math.max(1, det.body_depth_px)).toFixed(2);
      bioHtml = `
        <div class="instance-grid">
          <span>Length<strong>${Math.round(det.oriented_length_px)} px</strong></span>
          <span>Depth<strong>${Math.round(det.body_depth_px)} px</strong></span>
          <span>Area<strong>${Math.round(det.area_px)} px²</strong></span>
          <span>Aspect<strong>${asp}</strong></span>
        </div>
        <div style="font-size: 0.62rem; color: #fbbf24; margin-top: 4px; padding: 4px 6px; background: rgba(245, 158, 11, 0.08); border-radius: 4px; border: 1px dashed rgba(245, 158, 11, 0.25);">
          ⚠️ Optical scale unconfirmed — Confirm calibration scale above to unlock cm &amp; g.
        </div>
      `;
    }

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

// -------------------------------------------------------------
// BRUVs Video & MaxN Processing Module (Dr. Renato Morais ROCKNHub)
// -------------------------------------------------------------
function setupVideoModule() {
  const tabImageMode = document.getElementById('tabImageMode');
  const tabVideoMode = document.getElementById('tabVideoMode');
  const imageSidebarWrapper = document.getElementById('imageSidebarWrapper');
  const videoSidebarWrapper = document.getElementById('videoSidebarWrapper');
  const canvasWrapper = document.getElementById('canvasWrapper');
  const videoWrapper = document.getElementById('videoWrapper');
  const videoActivityBar = document.getElementById('videoActivityBar');
  const fishList = document.getElementById('fishList');
  const videoTracksList = document.getElementById('videoTracksList');
  const detailsHeaderTitle = document.getElementById('detailsHeaderTitle');

  const metricTitle1 = document.getElementById('metricTitle1');
  const metricTitle2 = document.getElementById('metricTitle2');
  const metricTitle3 = document.getElementById('metricTitle3');
  const metricTitle4 = document.getElementById('metricTitle4');
  const metricTitle5 = document.getElementById('metricTitle5');

  const videoDropZone = document.getElementById('videoDropZone');
  const videoFileInput = document.getElementById('videoFileInput');
  const videoUploadTitle = document.getElementById('videoUploadTitle');
  const videoUploadSubtitle = document.getElementById('videoUploadSubtitle');
  const presetVideoSohal = document.getElementById('presetVideoSohal');

  const videoSampleFps = document.getElementById('videoSampleFps');
  const videoMaxDuration = document.getElementById('videoMaxDuration');
  const videoClassifierSelect = document.getElementById('videoClassifierSelect');

  const runVideoBtn = document.getElementById('runVideoBtn');
  const videoBtnSpinner = document.getElementById('videoBtnSpinner');
  const videoBtnText = document.getElementById('videoBtnText');
  const videoLoader = document.getElementById('videoLoader');
  const videoLoaderText = document.getElementById('videoLoaderText');

  const bruvsVideoPlayer = document.getElementById('bruvsVideoPlayer');
  const peakFrameThumbContainer = document.getElementById('peakFrameThumbContainer');
  const peakFrameImg = document.getElementById('peakFrameImg');
  const peakFramePlaceholder = document.getElementById('peakFramePlaceholder');
  const peakTimestampBadge = document.getElementById('peakTimestampBadge');
  const maxNCountBadge = document.getElementById('maxNCountBadge');
  const timelineDurationText = document.getElementById('timelineDurationText');
  const timelineCanvas = document.getElementById('timelineCanvas');

  state.activeMode = 'image';
  state.selectedVideoSample = 'sample_redsea_sohal.mp4';
  state.uploadedVideoFile = null;
  state.videoResults = null;

  // 1. Mode Switching
  function switchToImageMode() {
    state.activeMode = 'image';
    if (tabImageMode) {
      tabImageMode.classList.add('active');
      tabImageMode.style.background = '#1e293b';
      tabImageMode.style.color = '#f8fafc';
    }
    if (tabVideoMode) {
      tabVideoMode.classList.remove('active');
      tabVideoMode.style.background = 'transparent';
      tabVideoMode.style.color = '#94a3b8';
    }

    if (imageSidebarWrapper) imageSidebarWrapper.style.display = 'block';
    if (videoSidebarWrapper) videoSidebarWrapper.style.display = 'none';
    if (canvasWrapper) canvasWrapper.style.display = 'flex';
    if (videoWrapper) videoWrapper.style.display = 'none';
    if (videoActivityBar) videoActivityBar.style.display = 'none';
    if (fishList) fishList.style.display = 'flex';
    if (videoTracksList) videoTracksList.style.display = 'none';
    if (videoLoader) videoLoader.style.display = 'none';

    if (detailsHeaderTitle) detailsHeaderTitle.textContent = 'Detected Instances';
    if (metricTitle1) metricTitle1.textContent = 'Fish Count';
    if (metricTitle2) metricTitle2.textContent = 'Total Biomass (W)';
    if (metricTitle3) metricTitle3.textContent = 'Commercial Catch';
    if (metricTitle4) metricTitle4.textContent = 'Diversity Index (H\')';
    if (metricTitle5) metricTitle5.textContent = 'Latency';

    if (totalBadge) totalBadge.textContent = state.detections.length;
    render();
  }

  function switchToVideoMode() {
    state.activeMode = 'video';
    if (tabVideoMode) {
      tabVideoMode.classList.add('active');
      tabVideoMode.style.background = '#1e293b';
      tabVideoMode.style.color = '#f8fafc';
    }
    if (tabImageMode) {
      tabImageMode.classList.remove('active');
      tabImageMode.style.background = 'transparent';
      tabImageMode.style.color = '#94a3b8';
    }

    if (imageSidebarWrapper) imageSidebarWrapper.style.display = 'none';
    if (videoSidebarWrapper) videoSidebarWrapper.style.display = 'block';
    if (canvasWrapper) canvasWrapper.style.display = 'none';
    if (videoWrapper) videoWrapper.style.display = 'flex';
    if (videoLoader) videoLoader.style.display = 'none';
    if (videoActivityBar) videoActivityBar.style.display = 'block';
    if (fishList) fishList.style.display = 'none';
    if (videoTracksList) videoTracksList.style.display = 'flex';

    if (detailsHeaderTitle) detailsHeaderTitle.textContent = 'BRUVs MaxN & Tracks';
    if (metricTitle1) metricTitle1.textContent = 'MaxN (Simultaneous)';
    if (metricTitle2) metricTitle2.textContent = 'Peak Timestamp';
    if (metricTitle3) metricTitle3.textContent = 'Unique Tracks';
    if (metricTitle4) metricTitle4.textContent = 'Duration Analyzed';
    if (metricTitle5) metricTitle5.textContent = 'Processing Time';

    // If no video is currently loaded in the player, preview the sample video
    if (bruvsVideoPlayer && !bruvsVideoPlayer.getAttribute('src')) {
      bruvsVideoPlayer.src = '/samples/sample_redsea_sohal.mp4';
      bruvsVideoPlayer.load();
    }

    if (state.videoResults) {
      updateVideoResultsUI(state.videoResults);
    } else {
      if (fishCountVal) fishCountVal.textContent = '0';
      if (totalBiomassVal) totalBiomassVal.textContent = '--:--';
      if (commercialYieldVal) commercialYieldVal.textContent = '0 Tracks';
      if (shannonVal) shannonVal.textContent = '0.0 s';
      if (latencyVal) latencyVal.textContent = '-- ms';
      if (totalBadge) totalBadge.textContent = '0';

      if (videoTracksList) {
        videoTracksList.innerHTML = `
          <div class="empty-state" style="padding: 28px 14px; text-align: center;">
            <div style="font-size: 1.8rem; margin-bottom: 8px;">🎬</div>
            <div style="font-size: 0.85rem; font-weight: 600; color: #38bdf8; margin-bottom: 4px;">Video Ready for Processing</div>
            <div style="font-size: 0.72rem; color: #94a3b8; line-height: 1.5;">
              Preview the underwater video in the center player.<br>
              When ready, click <strong style="color: #34d399;">"Process BRUVs Video &amp; MaxN"</strong> on the left to run detection, tracking, and peak MaxN calculation.
            </div>
          </div>
        `;
      }
    }
  }

  if (tabImageMode) tabImageMode.addEventListener('click', switchToImageMode);
  if (tabVideoMode) tabVideoMode.addEventListener('click', switchToVideoMode);

  // 2. Video Source Handling
  if (presetVideoSohal) {
    presetVideoSohal.addEventListener('click', () => {
      presetVideoSohal.classList.add('active');
      presetVideoSohal.style.background = 'rgba(16, 185, 129, 0.1)';
      presetVideoSohal.style.borderColor = 'rgba(16, 185, 129, 0.35)';
      state.selectedVideoSample = 'sample_redsea_sohal.mp4';
      state.uploadedVideoFile = null;
      if (videoUploadTitle) videoUploadTitle.textContent = 'Upload BRUVs Video (.mp4)';
      if (videoUploadSubtitle) videoUploadSubtitle.textContent = 'Drag & drop or click to browse';

      if (bruvsVideoPlayer) {
        bruvsVideoPlayer.src = '/samples/sample_redsea_sohal.mp4';
        bruvsVideoPlayer.load();
      }
      state.videoResults = null;
      if (videoTracksList) {
        videoTracksList.innerHTML = `
          <div class="empty-state" style="padding: 28px 14px; text-align: center;">
            <div style="font-size: 1.8rem; margin-bottom: 8px;">🎬</div>
            <div style="font-size: 0.85rem; font-weight: 600; color: #38bdf8; margin-bottom: 4px;">Sample Video Loaded</div>
            <div style="font-size: 0.72rem; color: #94a3b8; line-height: 1.5;">
              Click <strong style="color: #34d399;">"Process BRUVs Video &amp; MaxN"</strong> below to begin automated video census.
            </div>
          </div>
        `;
      }
    });
  }

  if (videoDropZone && videoFileInput) {
    videoDropZone.addEventListener('dragover', (e) => { e.preventDefault(); videoDropZone.classList.add('dragover'); });
    videoDropZone.addEventListener('dragleave', () => videoDropZone.classList.remove('dragover'));
    videoDropZone.addEventListener('drop', (e) => {
      e.preventDefault();
      videoDropZone.classList.remove('dragover');
      if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        handleVideoFile(e.dataTransfer.files[0]);
      }
    });

    videoFileInput.addEventListener('change', (e) => {
      if (e.target.files && e.target.files[0]) {
        handleVideoFile(e.target.files[0]);
      }
    });
  }

  function handleVideoFile(file) {
    state.uploadedVideoFile = file;
    state.selectedVideoSample = null;
    if (presetVideoSohal) {
      presetVideoSohal.classList.remove('active');
      presetVideoSohal.style.background = 'transparent';
      presetVideoSohal.style.borderColor = 'rgba(255, 255, 255, 0.08)';
    }
    if (videoUploadTitle) videoUploadTitle.textContent = file.name;
    if (videoUploadSubtitle) videoUploadSubtitle.textContent = `${(file.size / (1024 * 1024)).toFixed(1)} MB • Ready to analyze`;

    if (bruvsVideoPlayer) {
      bruvsVideoPlayer.src = URL.createObjectURL(file);
      bruvsVideoPlayer.load();
    }
    state.videoResults = null;
    if (videoTracksList) {
      videoTracksList.innerHTML = `
        <div class="empty-state" style="padding: 28px 14px; text-align: center;">
          <div style="font-size: 1.8rem; margin-bottom: 8px;">🎬</div>
          <div style="font-size: 0.85rem; font-weight: 600; color: #38bdf8; margin-bottom: 4px;">Custom Video Uploaded</div>
          <div style="font-size: 0.72rem; color: #94a3b8; line-height: 1.5;">
            Click <strong style="color: #34d399;">"Process BRUVs Video &amp; MaxN"</strong> on the left to analyze this video.
          </div>
        </div>
      `;
    }
  }

  // 3. Peak Frame & Timeline Seek Interaction (Pauses on seek so user can inspect frame)
  function seekToPeakAndPause() {
    if (bruvsVideoPlayer && state.videoResults && typeof state.videoResults.peak_second === 'number') {
      bruvsVideoPlayer.pause();
      bruvsVideoPlayer.currentTime = state.videoResults.peak_second;
      if (timelineCanvas && state.videoResults.timeline) {
        drawTimelineSparkline(timelineCanvas, state.videoResults.timeline, state.videoResults.peak_second, state.videoResults.peak_second);
      }
    }
  }

  if (peakFrameThumbContainer) peakFrameThumbContainer.addEventListener('click', seekToPeakAndPause);
  if (peakTimestampBadge) {
    peakTimestampBadge.style.cursor = 'pointer';
    peakTimestampBadge.title = 'Click to seek video to peak MaxN and pause';
    peakTimestampBadge.addEventListener('click', seekToPeakAndPause);
  }
  if (maxNCountBadge) {
    maxNCountBadge.style.cursor = 'pointer';
    maxNCountBadge.title = 'Click to seek video to peak MaxN and pause';
    maxNCountBadge.addEventListener('click', seekToPeakAndPause);
  }

  if (timelineCanvas && bruvsVideoPlayer) {
    timelineCanvas.style.cursor = 'crosshair';
    timelineCanvas.title = 'Click anywhere on sparkline to scrub video and pause';
    timelineCanvas.addEventListener('click', (e) => {
      if (!state.videoResults || !state.videoResults.timeline || state.videoResults.timeline.length === 0) return;
      const rect = timelineCanvas.getBoundingClientRect();
      const clickX = e.clientX - rect.left;
      const ratio = Math.max(0, Math.min(1, clickX / rect.width));
      const timeline = state.videoResults.timeline;
      const maxSec = timeline[timeline.length - 1].second || state.videoResults.duration_analyzed_sec || 1.0;
      const targetSec = ratio * maxSec;
      bruvsVideoPlayer.pause();
      bruvsVideoPlayer.currentTime = targetSec;
      drawTimelineSparkline(timelineCanvas, timeline, state.videoResults.peak_second, targetSec);
    });
  }

  if (bruvsVideoPlayer) {
    bruvsVideoPlayer.addEventListener('timeupdate', () => {
      if (state.activeMode === 'video' && state.videoResults && state.videoResults.timeline) {
        drawTimelineSparkline(timelineCanvas, state.videoResults.timeline, state.videoResults.peak_second, bruvsVideoPlayer.currentTime);
      }
    });
  }

  // 4. Run Video Analysis & MaxN
  if (runVideoBtn) {
    runVideoBtn.addEventListener('click', async () => {
      const startTime = performance.now();
      if (videoBtnSpinner) videoBtnSpinner.style.display = 'inline-block';
      if (videoBtnText) videoBtnText.textContent = 'Tracking & Computing MaxN...';
      if (runVideoBtn) runVideoBtn.disabled = true;
      if (videoLoader) videoLoader.style.display = 'flex';
      if (videoLoaderText) videoLoaderText.textContent = 'Running YOLO11s-seg + ByteTrack across video frames...';

      try {
        const formData = new FormData();
        if (state.uploadedVideoFile) {
          formData.append('file', state.uploadedVideoFile);
        } else {
          formData.append('sample_video', state.selectedVideoSample || 'sample_redsea_sohal.mp4');
        }

        formData.append('sample_fps', videoSampleFps ? videoSampleFps.value : '3.0');
        formData.append('max_duration_sec', videoMaxDuration ? videoMaxDuration.value : '30.0');
        formData.append('conf_threshold', confSlider ? confSlider.value : '0.35');
        formData.append('classifier_mode', videoClassifierSelect ? videoClassifierSelect.value : 'bioclip');
        formData.append('is_calibrated', 'true');
        formData.append('pixels_per_cm', '22.0'); // Standard BRUVs rig scale
        formData.append('run_classification', 'true');

        const res = await fetch('/api/video/process', {
          method: 'POST',
          body: formData
        });

        if (!res.ok) throw new Error(`Video processing error: ${res.statusText}`);
        const data = await res.json();
        const elapsed = Math.round(performance.now() - startTime);

        state.videoResults = data;
        updateVideoResultsUI(data, elapsed);

      } catch (err) {
        console.error('Failed to process video:', err);
        alert(`Video processing failed: ${err.message}`);
      } finally {
        if (videoBtnSpinner) videoBtnSpinner.style.display = 'none';
        if (videoBtnText) videoBtnText.textContent = 'Process BRUVs Video & MaxN';
        if (runVideoBtn) runVideoBtn.disabled = false;
        if (videoLoader) videoLoader.style.display = 'none';
      }
    });
  }

  function updateVideoResultsUI(data, elapsedMs = null) {
    if (bruvsVideoPlayer && data.video_url) {
      bruvsVideoPlayer.src = data.video_url;
      bruvsVideoPlayer.load();
    }

    if (peakFrameImg && data.peak_frame_url) {
      peakFrameImg.src = data.peak_frame_url;
      peakFrameImg.style.display = 'block';
      if (peakFramePlaceholder) peakFramePlaceholder.style.display = 'none';
    }

    if (peakTimestampBadge) {
      peakTimestampBadge.textContent = `Timestamp: ${data.peak_timestamp} (${data.peak_second}s)`;
    }

    if (maxNCountBadge) {
      maxNCountBadge.textContent = `MaxN: ${data.max_n_total} Fish (Simultaneous)`;
    }

    if (timelineDurationText) {
      timelineDurationText.textContent = `00:00 - ${data.duration_analyzed_sec}s (${data.frames_analyzed} frames)`;
    }

    // Update 5 Metrics Cards
    if (fishCountVal) {
      fishCountVal.textContent = data.max_n_total;
      fishCountVal.style.color = '#38bdf8';
    }
    if (totalBiomassVal) {
      totalBiomassVal.textContent = data.peak_timestamp;
      totalBiomassVal.style.color = '#10b981';
    }
    if (commercialYieldVal) {
      commercialYieldVal.textContent = `${data.unique_tracks_count} Tracks`;
      commercialYieldVal.style.color = '#a78bfa';
    }
    if (shannonVal) {
      shannonVal.textContent = `${data.duration_analyzed_sec} s`;
      shannonVal.style.color = '#f59e0b';
    }
    if (latencyVal && elapsedMs) {
      latencyVal.textContent = `${(elapsedMs / 1000).toFixed(1)} s`;
    }

    if (totalBadge) {
      totalBadge.textContent = data.unique_tracks_count;
    }

    // Draw Density Sparkline
    if (timelineCanvas && data.timeline) {
      drawTimelineSparkline(timelineCanvas, data.timeline, data.peak_second);
    }

    // Render Tracked Organisms
    if (videoTracksList) {
      renderVideoTracksList(data);
    }
  }

  function drawTimelineSparkline(canvasEl, timeline, peakSec, currentSec = null) {
    const ctxT = canvasEl.getContext('2d');
    const w = canvasEl.width;
    const h = canvasEl.height;
    ctxT.clearRect(0, 0, w, h);

    if (!timeline || timeline.length === 0) return;

    const maxCount = Math.max(1, ...timeline.map(p => p.count));
    const stepX = w / Math.max(1, timeline.length - 1);
    const maxSec = timeline[timeline.length - 1].second || 1.0;

    // Draw baseline
    ctxT.strokeStyle = 'rgba(255, 255, 255, 0.08)';
    ctxT.lineWidth = 1;
    ctxT.beginPath();
    ctxT.moveTo(0, h - 4);
    ctxT.lineTo(w, h - 4);
    ctxT.stroke();

    // Fill gradient area under density curve
    const grad = ctxT.createLinearGradient(0, 0, 0, h);
    grad.addColorStop(0, 'rgba(56, 189, 248, 0.35)');
    grad.addColorStop(1, 'rgba(56, 189, 248, 0.02)');

    ctxT.beginPath();
    ctxT.moveTo(0, h - 4);
    timeline.forEach((pt, i) => {
      const x = i * stepX;
      const y = (h - 6) - (pt.count / maxCount) * (h - 12);
      ctxT.lineTo(x, y);
    });
    ctxT.lineTo(w, h - 4);
    ctxT.closePath();
    ctxT.fillStyle = grad;
    ctxT.fill();

    // Stroke line
    ctxT.strokeStyle = '#38bdf8';
    ctxT.lineWidth = 1.6;
    ctxT.beginPath();
    timeline.forEach((pt, i) => {
      const x = i * stepX;
      const y = (h - 6) - (pt.count / maxCount) * (h - 12);
      if (i === 0) ctxT.moveTo(x, y);
      else ctxT.lineTo(x, y);
    });
    ctxT.stroke();

    // Highlight Peak Frame with a vertical marker
    if (typeof peakSec === 'number') {
      const peakX = Math.min(w - 2, Math.max(2, (peakSec / maxSec) * w));
      ctxT.strokeStyle = '#10b981';
      ctxT.lineWidth = 2;
      ctxT.beginPath();
      ctxT.moveTo(peakX, 2);
      ctxT.lineTo(peakX, h - 2);
      ctxT.stroke();

      // Peak circle
      ctxT.fillStyle = '#10b981';
      ctxT.beginPath();
      ctxT.arc(peakX, 4, 3, 0, Math.PI * 2);
      ctxT.fill();
    }

    // Current playhead scrubber
    if (typeof currentSec === 'number' && currentSec >= 0) {
      const curX = Math.min(w - 2, Math.max(2, (currentSec / maxSec) * w));
      ctxT.strokeStyle = '#f8fafc';
      ctxT.lineWidth = 1.5;
      ctxT.setLineDash([2, 2]);
      ctxT.beginPath();
      ctxT.moveTo(curX, 2);
      ctxT.lineTo(curX, h - 2);
      ctxT.stroke();
      ctxT.setLineDash([]);
    }
  }

  function renderVideoTracksList(data) {
    videoTracksList.innerHTML = '';

    // 1. MaxN Ecological Breakdown Card
    const famBox = document.createElement('div');
    famBox.className = 'maxn-summary-card';
    
    let famListHtml = '';
    const famBreakdown = data.max_n_by_family || {};
    for (const [fam, cnt] of Object.entries(famBreakdown)) {
      famListHtml += `
        <div class="maxn-family-row">
          <span class="maxn-family-name">${fam}</span>
          <span class="maxn-family-count">MaxN = ${cnt} fish</span>
        </div>
      `;
    }

    famBox.innerHTML = `
      <div class="maxn-summary-title">
        <span>MaxN Ecological Census</span>
        <span class="pill-badge pill-cyan">Peak @ ${data.peak_timestamp}</span>
      </div>
      <div class="maxn-summary-sub">
        Highest simultaneous count in a single frame. Guarantees zero double-counting across the survey.
      </div>
      <div style="display: flex; flex-direction: column; gap: 4px; margin-top: 4px;">
        ${famListHtml}
      </div>
    `;
    videoTracksList.appendChild(famBox);

    // 2. Individual Tracked Fish
    const tracks = data.tracks || [];
    if (tracks.length === 0) {
      videoTracksList.innerHTML += `
        <div class="empty-state" style="padding: 24px 12px;">
          <span style="color: #94a3b8; font-size: 0.74rem;">No persistent fish tracks identified in analyzed duration.</span>
        </div>
      `;
      return;
    }

    tracks.forEach(tr => {
      const card = document.createElement('div');
      card.className = `video-track-card ${tr.in_peak_frame ? 'peak-track' : ''}`;
      card.title = "Click to jump video to this fish's first appearance";

      const peakBadge = tr.in_peak_frame
        ? `<span class="pill-badge" style="background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.4);">Present at MaxN Peak</span>`
        : `<span class="pill-badge" style="background: rgba(148, 163, 184, 0.1); color: #94a3b8; border: 1px solid rgba(148, 163, 184, 0.2);">Passerby Track</span>`;

      const bio = tr.biomass || {};
      const lenText = bio.fork_length_cm ? `${bio.fork_length_cm} cm` : `${bio.length_px || '--'} px (Uncalibrated)`;
      const wtText = bio.biomass_g ? `${bio.biomass_g} g` : 'Requires Scale';

      card.innerHTML = `
        <div class="video-track-header">
          <div class="video-track-title">
            <span>Fish #${tr.track_id} ${tr.family}</span>
            ${peakBadge}
          </div>
          <span class="pill-badge pill-green">${Math.round(tr.confidence * 100)}% conf</span>
        </div>

        <div class="video-track-timing">
          <span>Active:</span> ${tr.first_seen} → ${tr.last_seen} (${tr.duration_sec}s • ${tr.frame_appearances} frames)
        </div>

        <div class="video-track-grid">
          <div>
            <span>Est. Fork Length</span>
            <strong>${lenText}</strong>
          </div>
          <div>
            <span>Biomass Weight</span>
            <strong>${wtText}</strong>
          </div>
          <div>
            <span>Commercial Species</span>
            <strong style="color: ${tr.is_commercial ? '#34d399' : '#cbd5e1'};">${tr.is_commercial ? 'Yes (Targeted)' : 'No (Non-target)'}</strong>
          </div>
          <div>
            <span>Tracking Method</span>
            <strong style="color: #38bdf8;">ByteTrack Kalman</strong>
          </div>
        </div>
      `;

      card.addEventListener('click', () => {
        // Highlight active track card
        document.querySelectorAll('.video-track-card').forEach(c => c.classList.remove('selected'));
        card.classList.add('selected');

        if (bruvsVideoPlayer && typeof tr.first_seen === 'string') {
          const parts = tr.first_seen.split(':');
          if (parts.length === 2) {
            const sec = parseFloat(parts[0]) * 60 + parseFloat(parts[1]);
            bruvsVideoPlayer.pause();
            bruvsVideoPlayer.currentTime = sec;
            if (timelineCanvas && data.timeline) {
              drawTimelineSparkline(timelineCanvas, data.timeline, data.peak_second, sec);
            }
          }
        }
      });

      videoTracksList.appendChild(card);
    });
  }
}

