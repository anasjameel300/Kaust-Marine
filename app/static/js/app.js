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
  },
  // Viewport Zoom & Pan
  zoomLevel: 1.0,
  panX: 0,
  panY: 0,
  isPanMode: false,
  isDragging: false,
  dragStartX: 0,
  dragStartY: 0
};

// Styling Palette strictly tuned for Soft Oceanic Daylight Theme
const PALETTES = {
  fish: {
    stroke: '#00e5ff',
    fill: 'rgba(0, 229, 255, 0.20)',
    glow: 'rgba(0, 229, 255, 0.55)',
    labelBg: 'rgba(10, 37, 64, 0.94)',
    badgeColor: '#00e5ff'
  },
  reefs: {
    stroke: '#f59e0b',
    fill: 'rgba(245, 158, 11, 0.18)',
    glow: 'rgba(245, 158, 11, 0.40)',
    labelBg: 'rgba(10, 37, 64, 0.94)',
    badgeColor: '#f59e0b'
  },
  human_divers: {
    stroke: '#a855f7',
    fill: 'rgba(168, 85, 247, 0.18)',
    glow: 'rgba(168, 85, 247, 0.40)',
    labelBg: 'rgba(10, 37, 64, 0.94)',
    badgeColor: '#a855f7'
  }
};

// DOM References
const canvas = document.getElementById('mainCanvas');
const ctx = canvas.getContext('2d');
const canvasWrapper = document.getElementById('canvasWrapper');
const canvasViewport = document.getElementById('canvasViewport');
const canvasLoader = document.getElementById('canvasLoader');
const canvasLoaderText = document.getElementById('canvasLoaderText');
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

// Zoom & Pan Toolbar
const btnZoomIn = document.getElementById('btnZoomIn');
const btnZoomOut = document.getElementById('btnZoomOut');
const btnResetFit = document.getElementById('btnResetFit');
const btnPanMode = document.getElementById('btnPanMode');
const btnFullscreen = document.getElementById('btnFullscreen');
const zoomLevelIndicator = document.getElementById('zoomLevelIndicator');

// Metadata & Header Chips
const canvasFilename = document.getElementById('canvasFilename');
const canvasLatencyChip = document.getElementById('canvasLatencyChip');
const canvasModelChip = document.getElementById('canvasModelChip');
const metaResolution = document.getElementById('metaResolution');
const metaFileSize = document.getElementById('metaFileSize');
const metaDateProcessed = document.getElementById('metaDateProcessed');
const metaSource = document.getElementById('metaSource');

// Biomass Card
const biomassCalibScaleText = document.getElementById('biomassCalibScaleText');
const biomassCalibSubtext = document.getElementById('biomassCalibSubtext');

const toggleOutlines = document.getElementById('toggleOutlines');
const toggleMasks = document.getElementById('toggleMasks');
const toggleBadges = document.getElementById('toggleBadges');
const toggleTiling = document.getElementById('toggleTiling');
const toggleDualCrop = document.getElementById('toggleDualCrop');

const runBtn = document.getElementById('runBtn');
const btnSpinner = document.getElementById('btnSpinner');
const btnText = document.getElementById('btnText');

const fishCountVal = document.getElementById('fishCountVal');
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

function updateCalibUI() {
  if (!calibStatusBadge || !confirmCalibBtnText) return;
  if (state.isCalibrated) {
    const scale = parseFloat(scaleSlider ? scaleSlider.value : 25.0).toFixed(1);
    calibStatusBadge.textContent = `Calibrated (${scale} px/cm)`;
    calibStatusBadge.className = 'mono-badge badge-green';
    confirmCalibBtnText.textContent = 'Modify / Reset Scale';
    if (biomassCalibScaleText) biomassCalibScaleText.textContent = `${scale} px/cm`;
    if (biomassCalibSubtext) biomassCalibSubtext.textContent = 'Scale Verified';
  } else {
    calibStatusBadge.textContent = 'Uncalibrated';
    calibStatusBadge.className = 'mono-badge badge-amber';
    confirmCalibBtnText.textContent = 'Confirm Calibration Scale';
    if (biomassCalibSubtext) biomassCalibSubtext.textContent = 'Unconfirmed Scale';
  }
}

// Initialize Application
function initApp() {
  setupEventListeners();
  loadSampleThumbnails();
  setupVideoModule();
  setupZoomPanControls();
  window.addEventListener('resize', fitCanvasToContainer);
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

  // Dual-Crop Modal
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

  downloadImgBtn.addEventListener('click', downloadCanvasImage);

  const resetViewBtn = document.getElementById('resetViewBtn');
  if (resetViewBtn) {
    resetViewBtn.addEventListener('click', () => {
      state.selectedId = null;
      document.querySelectorAll('.instance-card').forEach(c => c.classList.remove('selected'));
      resetZoomPan();
      render();
    });
  }
}

// FIXED-BOUNDS AUTO-FIT LOGIC FOR CANVAS VIEWPORT
function fitCanvasToContainer() {
  if (!state.imageObj.complete || state.imageObj.naturalWidth === 0) return;
  const wrapperRect = canvasWrapper.getBoundingClientRect();
  if (wrapperRect.width === 0 || wrapperRect.height === 0) return;

  const imgW = state.imageObj.naturalWidth;
  const imgH = state.imageObj.naturalHeight;
  const imgAspect = imgW / imgH;

  // Scale reduction (82% of container) gives comfortable breathing room
  const scaleReduction = 0.82;
  const availW = wrapperRect.width * scaleReduction;
  const availH = wrapperRect.height * scaleReduction;
  const availAspect = availW / availH;

  let baseW, baseH;
  // Preserve aspect ratio cleanly inside the non-overflowing container box
  if (imgAspect > availAspect) {
    baseW = availW;
    baseH = availW / imgAspect;
  } else {
    baseH = availH;
    baseW = availH * imgAspect;
  }

  // Apply zoom factor smoothly
  const displayW = baseW * state.zoomLevel;
  const displayH = baseH * state.zoomLevel;

  canvas.style.width = `${Math.round(displayW)}px`;
  canvas.style.height = `${Math.round(displayH)}px`;
  canvas.style.transform = `translate(${state.panX}px, ${state.panY}px)`;

  if (zoomLevelIndicator) {
    zoomLevelIndicator.textContent = `${Math.round(state.zoomLevel * 100)}%`;
  }
}

function resetZoomPan() {
  state.zoomLevel = 1.0;
  state.panX = 0;
  state.panY = 0;
  fitCanvasToContainer();
}

function setupZoomPanControls() {
  if (btnZoomIn) {
    btnZoomIn.addEventListener('click', () => {
      state.zoomLevel = Math.min(4.0, state.zoomLevel + 0.25);
      fitCanvasToContainer();
    });
  }

  if (btnZoomOut) {
    btnZoomOut.addEventListener('click', () => {
      state.zoomLevel = Math.max(0.5, state.zoomLevel - 0.25);
      fitCanvasToContainer();
    });
  }

  if (btnResetFit) {
    btnResetFit.addEventListener('click', () => {
      resetZoomPan();
    });
  }

  if (btnPanMode) {
    btnPanMode.addEventListener('click', () => {
      state.isPanMode = !state.isPanMode;
      btnPanMode.classList.toggle('active', state.isPanMode);
      canvasWrapper.classList.toggle('pan-mode', state.isPanMode);
    });
  }

  if (btnFullscreen) {
    btnFullscreen.addEventListener('click', () => {
      if (!document.fullscreenElement) {
        if (canvasViewport.requestFullscreen) {
          canvasViewport.requestFullscreen();
        }
      } else {
        if (document.exitFullscreen) {
          document.exitFullscreen();
        }
      }
      setTimeout(fitCanvasToContainer, 150);
    });
  }

  // Pan dragging
  canvasWrapper.addEventListener('mousedown', (e) => {
    if (state.isPanMode || e.button === 1) { // Left click in pan mode or middle click
      state.isDragging = true;
      state.dragStartX = e.clientX - state.panX;
      state.dragStartY = e.clientY - state.panY;
      canvasWrapper.classList.add('panning');
    }
  });

  window.addEventListener('mousemove', (e) => {
    if (state.isDragging) {
      state.panX = e.clientX - state.dragStartX;
      state.panY = e.clientY - state.dragStartY;
      canvas.style.transform = `translate(${state.panX}px, ${state.panY}px)`;
    }
  });

  window.addEventListener('mouseup', () => {
    if (state.isDragging) {
      state.isDragging = false;
      canvasWrapper.classList.remove('panning');
    }
  });
}

async function loadSampleThumbnails() {
  try {
    const res = await fetch('/api/samples');
    if (!res.ok) throw new Error('Failed to fetch samples');
    const samples = await res.json();

    presetGrid.innerHTML = '';
    samples.slice(0, 6).forEach((sample, idx) => {
      const thumb = document.createElement('div');
      thumb.className = `preset-thumb ${idx === 0 ? 'active' : ''}`;
      thumb.id = `sample-${sample.filename}`;
      thumb.title = sample.filename;

      const img = document.createElement('img');
      img.src = sample.url;
      img.alt = sample.filename;
      thumb.appendChild(img);

      thumb.addEventListener('click', () => {
        document.querySelectorAll('.preset-thumb').forEach(t => t.classList.remove('active'));
        thumb.classList.add('active');
        state.currentSampleName = sample.filename;
        state.uploadedFile = null;
        if (canvasFilename) canvasFilename.textContent = sample.filename;
        loadImageAndSegment(sample.url);
      });

      presetGrid.appendChild(thumb);
    });

    if (samples.length > 0) {
      state.currentSampleName = samples[0].filename;
      if (canvasFilename) canvasFilename.textContent = samples[0].filename;
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

  if (canvasFilename) canvasFilename.textContent = file.name;
  if (metaFileSize) metaFileSize.textContent = `${(file.size / (1024 * 1024)).toFixed(1)} MB`;

  const reader = new FileReader();
  reader.onload = (e) => {
    loadImageAndSegment(e.target.result);
  };
  reader.readAsDataURL(file);
}

function updateMetadataCard(img) {
  if (metaResolution && img) {
    metaResolution.textContent = `${img.naturalWidth} × ${img.naturalHeight}`;
  }
  if (metaDateProcessed) {
    metaDateProcessed.textContent = new Date().toLocaleDateString('en-US', {
      month: 'short', day: 'numeric', year: 'numeric'
    });
  }
}

function loadImageAndSegment(url) {
  state.detections = [];
  state.selectedId = null;
  state.hoveredId = null;
  state.isClassified = false;
  state.isCalibrated = false;
  resetZoomPan();
  updateCalibUI();

  if (classifyBadge) {
    classifyBadge.textContent = 'On-Demand';
    classifyBadge.className = 'mono-badge badge-cyan';
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
    fitCanvasToContainer();
    updateMetadataCard(img);
    state.detections = [];
    render();
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
    ? 'Running Stage 2 taxonomic classification & allometric sizing...'
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
    if (canvasModelChip && data.model_title) {
      canvasModelChip.textContent = `Model: ${data.model_title}`;
    }
    if (subBrandModel) {
      subBrandModel.textContent = `${data.model_title} ${data.enable_tiling ? ' + Tiling' : ''}`;
    }

    state.detections = data.detections || [];
    state.counts = data.counts || { fish: 0, reefs: 0, human_divers: 0 };
    
    if (latencyVal) latencyVal.textContent = `${elapsed} ms`;
    const visibleFish = getVisibleDetections();
    if (fishCountVal) fishCountVal.textContent = visibleFish.length;

    // Update Community Biomass & Ecology Metrics
    if (data.community_metrics) {
      const cm = data.community_metrics;
      if (cm.is_calibrated && cm.total_biomass_kg !== null) {
        if (totalBiomassVal) totalBiomassVal.textContent = `${cm.total_biomass_kg.toFixed(2)} kg`;
        if (commercialYieldVal) commercialYieldVal.textContent = `${cm.commercial_biomass_kg.toFixed(2)} kg (${Math.round(cm.commercial_ratio_pct)}%)`;
        if (biomassCalibScaleText) biomassCalibScaleText.textContent = `${cm.pixels_per_cm || 25.0} px/cm`;
        if (biomassCalibSubtext) biomassCalibSubtext.textContent = `Calibrated (${cm.calibration_preset || 'Preset'})`;
      } else {
        if (totalBiomassVal) totalBiomassVal.textContent = 'Uncalibrated';
        if (commercialYieldVal) commercialYieldVal.textContent = 'Scale Needed';
        if (biomassCalibScaleText) biomassCalibScaleText.textContent = `${scaleSlider ? scaleSlider.value : '25.0'} px/cm`;
        if (biomassCalibSubtext) biomassCalibSubtext.textContent = 'Unconfirmed Scale';
      }
      if (shannonVal) shannonVal.textContent = (cm.shannon_diversity_index || 0).toFixed(2);
    }

    if (runClassify) {
      state.isClassified = true;
      if (classifyBadge) {
        classifyBadge.textContent = 'Classified';
        classifyBadge.className = 'mono-badge badge-green';
      }
      if (classifyBtnText) classifyBtnText.textContent = 'Re-Identify Species';
    } else {
      state.isClassified = false;
      if (classifyBadge) {
        classifyBadge.textContent = 'On-Demand';
        classifyBadge.className = 'mono-badge badge-cyan';
      }
      if (classifyBtnText) classifyBtnText.textContent = 'Run Identification (Species ID)';
    }

    viewStatus.textContent = `Completed in ${elapsed}ms: ${visibleFish.length} fish segmented (${data.model_title})`;

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
  btnText.textContent = isLoading ? 'Computing...' : 'Re-Run Segmentation Only';
  runBtn.disabled = isLoading;
}

// STRICT FILTER: Fish only per user requirement (Sequential 1-to-1 numbering)
function getVisibleDetections() {
  const fishes = state.detections.filter(d => d.class_name === 'fish');
  fishes.forEach((det, i) => {
    det.fish_num = i + 1;
    det.tag_label = `FISH-${i + 1}`;
  });
  return fishes;
}

function render() {
  if (!state.imageObj.complete || state.imageObj.naturalWidth === 0) return;

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(state.imageObj, 0, 0);

  // Compute adaptive resolution scale factor so text and outlines are always crisp and readable
  const rect = canvas.getBoundingClientRect();
  const displayW = rect.width || canvas.clientWidth || 640;
  const drawScale = Math.max(1.0, (canvas.width / displayW) * 1.35);

  const visible = getVisibleDetections();

  visible.forEach((det) => {
    const isSelected = det.id === state.selectedId;
    const isHovered = det.id === state.hoveredId;
    const palette = PALETTES.fish;

    if (det.polygon && det.polygon.length > 2) {
      // 1. Translucent Mask Fill
      if (state.styles.masks) {
        ctx.beginPath();
        ctx.moveTo(det.polygon[0][0], det.polygon[0][1]);
        for (let i = 1; i < det.polygon.length; i++) {
          ctx.lineTo(det.polygon[i][0], det.polygon[i][1]);
        }
        ctx.closePath();
        ctx.fillStyle = isSelected || isHovered ? 'rgba(0, 242, 254, 0.45)' : palette.fill;
        ctx.fill();
      }

      // 2. Luminous Neon Contour Glow (Cyan aesthetic matching reference)
      if (state.styles.outlines) {
        ctx.save();
        ctx.beginPath();
        ctx.moveTo(det.polygon[0][0], det.polygon[0][1]);
        for (let i = 1; i < det.polygon.length; i++) {
          ctx.lineTo(det.polygon[i][0], det.polygon[i][1]);
        }
        ctx.closePath();
        ctx.lineWidth = Math.max(2.5, (isSelected || isHovered ? 4.0 : 2.8) * drawScale);
        ctx.strokeStyle = isSelected || isHovered ? '#ffffff' : palette.stroke;
        ctx.shadowColor = isSelected || isHovered ? '#00e5ff' : palette.glow;
        ctx.shadowBlur = Math.round((isSelected || isHovered ? 20 : 12) * drawScale);
        ctx.stroke();
        ctx.restore();
      }

      // 3. Floating Label Badge on Specimen
      if (state.styles.badges) {
        drawBadge(det, palette, isSelected || isHovered, drawScale);
      }
    }
  });
}

function drawBadge(det, palette, isActive, drawScale = 1.0) {
  let topY = Infinity;
  let topX = 0;
  det.polygon.forEach(pt => {
    if (pt[1] < topY) {
      topY = pt[1];
      topX = pt[0];
    }
  });

  const tagId = det.tag_label || `FISH-${det.id}`;
  let label = `${tagId} • ${Math.round(det.confidence * 100)}%`;
  let subText = `${Math.round(det.oriented_length_px)}px • Uncalibrated`;

  if (det.taxonomy && det.taxonomy.family) {
    const fam = det.taxonomy.family;
    const cname = det.taxonomy.common_name ? ` (${det.taxonomy.common_name})` : '';
    const taxConf = Math.round((det.taxonomy.confidence || 0) * 100);
    const isVer = det.taxonomy.is_verified !== false;
    label = isVer ? `${tagId} | ${taxConf}% ${fam}${cname}` : `${tagId} | Unverified (${fam})`;
    if (det.biomass && det.biomass.is_calibrated && det.biomass.fork_length_cm) {
      subText = `L=${det.biomass.fork_length_cm}cm • W=${det.biomass.biomass_g}g`;
    } else {
      subText = `${Math.round(det.oriented_length_px)}px • Uncalibrated`;
    }
  } else if (det.biomass && det.biomass.is_calibrated && det.biomass.fork_length_cm) {
    subText = `L=${det.biomass.fork_length_cm}cm • W≈${det.biomass.biomass_g}g*`;
  }

  // Adaptive font and badge metrics scaled to be comfortably readable on screen
  const titleFontSize = Math.max(12, Math.round(13 * drawScale));
  const subFontSize = Math.max(10, Math.round(10.5 * drawScale));
  const padX = Math.round(10 * drawScale);
  const cornerRadius = Math.round(6 * drawScale);
  const borderWidth = Math.max(1.5, Math.round((isActive ? 2.2 : 1.5) * drawScale));
  const height = Math.round(36 * drawScale);

  ctx.save();
  ctx.font = `700 ${titleFontSize}px Outfit, sans-serif`;
  const labelWidth = ctx.measureText(label).width;
  ctx.font = `600 ${subFontSize}px JetBrains Mono, monospace`;
  const subWidth = ctx.measureText(subText).width;

  const width = Math.max(labelWidth, subWidth) + (padX * 2);

  const x = Math.max(10 * drawScale, topX - width / 2);
  const y = Math.max(10 * drawScale, topY - height - (8 * drawScale));

  ctx.fillStyle = isActive ? 'rgba(8, 28, 50, 0.98)' : 'rgba(10, 37, 64, 0.95)';
  ctx.strokeStyle = isActive ? '#00e5ff' : palette.stroke;
  ctx.lineWidth = borderWidth;
  ctx.beginPath();
  ctx.roundRect(x, y, width, height, cornerRadius);
  ctx.fill();
  ctx.stroke();

  ctx.fillStyle = '#ffffff';
  ctx.font = `700 ${titleFontSize}px Outfit, sans-serif`;
  ctx.fillText(label, x + padX, y + Math.round(16 * drawScale));

  ctx.font = `600 ${subFontSize}px JetBrains Mono, monospace`;
  ctx.fillStyle = isActive ? '#00e5ff' : '#38bdf8';
  ctx.fillText(subText, x + padX, y + Math.round(29 * drawScale));
  ctx.restore();
}

// Generate isolated offscreen thumbnail crop of fish bounding box
function getSpecimenCropThumbnail(imageObj, box) {
  if (!imageObj || !imageObj.complete || !box) return null;
  const [x1, y1, x2, y2] = box;
  const bw = Math.max(1, x2 - x1);
  const bh = Math.max(1, y2 - y1);

  const off = document.createElement('canvas');
  off.width = 116;
  off.height = 116;
  const offCtx = off.getContext('2d');

  const padX = bw * 0.12;
  const padY = bh * 0.12;
  const sx = Math.max(0, x1 - padX);
  const sy = Math.max(0, y1 - padY);
  const sw = Math.min(imageObj.naturalWidth - sx, bw + padX * 2);
  const sh = Math.min(imageObj.naturalHeight - sy, bh + padY * 2);

  offCtx.drawImage(imageObj, sx, sy, sw, sh, 0, 0, 116, 116);
  return off.toDataURL('image/jpeg', 0.85);
}

function updateSidebar() {
  const visible = getVisibleDetections();
  if (totalBadge) totalBadge.textContent = visible.length;

  if (visible.length === 0) {
    fishList.innerHTML = `
      <div class="empty-state">
        <div class="empty-icon">🐟</div>
        <span>No fish instances detected.<br>Select a sample or upload an image on the left.</span>
      </div>
    `;
    return;
  }

  fishList.innerHTML = '';
  visible.forEach((det, idx) => {
    const card = document.createElement('div');
    card.className = `instance-card ${det.id === state.selectedId ? 'selected' : ''}`;
    card.id = `card-${det.id}`;

    const tagId = det.tag_label || `FISH-${idx + 1}`;
    const confPct = Math.round(det.confidence * 100);

    // Crop Thumbnail
    const cropUrl = getSpecimenCropThumbnail(state.imageObj, det.bbox);
    const cropThumbHtml = cropUrl
      ? `<div class="specimen-crop-box"><img src="${cropUrl}" alt="${tagId}"></div>`
      : `<div class="specimen-crop-box"><div class="specimen-crop-placeholder">🐟</div></div>`;

    let taxTitle = 'Fish Organism';
    let taxSci = 'Unclassified • Run Identification';
    if (det.taxonomy && det.taxonomy.family) {
      const isVer = det.taxonomy.is_verified !== false;
      const cname = det.taxonomy.common_name ? ` • ${det.taxonomy.common_name}` : '';
      taxTitle = isVer ? `${det.taxonomy.family}${cname}` : `Guess: ${det.taxonomy.family}`;
      taxSci = det.taxonomy.scientific_name || det.taxonomy.family;
    }

    // Morphometrics
    let morphHtml = '';
    if (det.biomass && det.biomass.is_calibrated && det.biomass.fork_length_cm) {
      morphHtml = `
        <div class="morph-grid">
          <div class="morph-item"><span>Length:</span><strong>${det.biomass.fork_length_cm} cm</strong></div>
          <div class="morph-item"><span>Depth:</span><strong>${det.biomass.body_depth_cm} cm</strong></div>
          <div class="morph-item"><span>Weight:</span><strong>${det.biomass.biomass_g} g</strong></div>
          <div class="morph-item"><span>Aspect:</span><strong>${det.biomass.aspect_ratio}</strong></div>
        </div>
      `;
    } else {
      const asp = (det.oriented_length_px / Math.max(1, det.body_depth_px)).toFixed(2);
      morphHtml = `
        <div class="morph-grid">
          <div class="morph-item"><span>Length:</span><strong>${Math.round(det.oriented_length_px)} px</strong></div>
          <div class="morph-item"><span>Depth:</span><strong>${Math.round(det.body_depth_px)} px</strong></div>
          <div class="morph-item"><span>Area:</span><strong>${Math.round(det.area_px).toLocaleString()} px²</strong></div>
          <div class="morph-item"><span>Aspect:</span><strong>${asp}</strong></div>
        </div>
      `;
    }

    // Dual-Crop ROIs
    let roiHtml = '';
    if (det.taxonomy && det.taxonomy.rois && det.taxonomy.rois.head_thumb) {
      const rois = det.taxonomy.rois;
      roiHtml = `
        <div style="margin-top: 8px; padding: 6px 8px; background: rgba(14, 165, 233, 0.05); border-radius: 6px; border: 1px dashed rgba(14, 165, 233, 0.25);">
          <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px;">
            <span style="font-size: 0.62rem; color: #0284c7; font-weight: 700; text-transform: uppercase;">Dual-Crop ROIs</span>
            <span style="font-size: 0.60rem; color: #059669; font-weight: 600;">✓ ${rois.consensus_badge || 'Anatomical Match'}</span>
          </div>
          <div style="display: flex; gap: 6px;">
            <div style="flex: 1; text-align: center; background: #ffffff; border-radius: 4px; padding: 3px; border: 1px solid rgba(14, 165, 233, 0.15);">
              <img src="${rois.head_thumb}" alt="Cephalic" style="width: 100%; height: 38px; object-fit: cover; border-radius: 2px;">
              <span style="display: block; font-size: 0.58rem; color: #64748b; margin-top: 2px;">Cephalic (35%)</span>
            </div>
            <div style="flex: 1; text-align: center; background: #ffffff; border-radius: 4px; padding: 3px; border: 1px solid rgba(14, 165, 233, 0.15);">
              <img src="${rois.caudal_thumb}" alt="Caudal" style="width: 100%; height: 38px; object-fit: cover; border-radius: 2px;">
              <span style="display: block; font-size: 0.58rem; color: #64748b; margin-top: 2px;">Caudal (25%)</span>
            </div>
          </div>
        </div>
      `;
    }

    card.innerHTML = `
      <div class="specimen-main-row">
        ${cropThumbHtml}
        <div class="specimen-info">
          <div class="specimen-title-row">
            <span class="specimen-id">${tagId}</span>
            <span class="specimen-conf">${confPct}%</span>
          </div>
          <div class="specimen-taxa">${taxTitle}</div>
          <span class="specimen-scientific">${taxSci}</span>
        </div>
      </div>
      ${morphHtml}
      ${roiHtml}
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

// Pixel-perfect normalized coordinate calculation for 8K, 4K, or 640px
function handleCanvasHover(e) {
  if (state.isDragging) return;
  const rect = canvas.getBoundingClientRect();
  if (rect.width === 0 || rect.height === 0) return;
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
  if (state.isDragging) return;
  const rect = canvas.getBoundingClientRect();
  if (rect.width === 0 || rect.height === 0) return;
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
  link.download = `kaust_annotated_${state.currentSampleName || 'image'}.png`;
  link.href = canvas.toDataURL('image/png');
  link.click();
}

// BRUVs Video Mode Module
function setupVideoModule() {
  const tabImageMode = document.getElementById('tabImageMode');
  const tabVideoMode = document.getElementById('tabVideoMode');
  const imageSidebarWrapper = document.getElementById('imageSidebarWrapper');
  const videoSidebarWrapper = document.getElementById('videoSidebarWrapper');
  const canvasViewport = document.getElementById('canvasViewport');
  const videoWrapper = document.getElementById('videoWrapper');
  const videoActivityBar = document.getElementById('videoActivityBar');
  const detailsHeaderTitle = document.getElementById('detailsHeaderTitle');
  const videoTracksList = document.getElementById('videoTracksList');

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

  let currentVideoUrl = '/samples/sample_redsea_sohal.mp4';
  let uploadedVideoFile = null;

  function switchMode(mode) {
    const taxonomySummaryCard = document.getElementById('taxonomySummaryCard');
    const metadataCard = document.querySelector('.metadata-card');

    if (mode === 'image') {
      tabImageMode.classList.add('active');
      tabVideoMode.classList.remove('active');
      imageSidebarWrapper.style.display = 'block';
      videoSidebarWrapper.style.display = 'none';
      canvasViewport.style.display = 'flex';
      videoWrapper.style.display = 'none';
      videoActivityBar.style.display = 'none';
      fishList.style.display = 'flex';
      videoTracksList.style.display = 'none';
      if (detailsHeaderTitle) detailsHeaderTitle.textContent = 'Detected Fish Instances';
      if (taxonomySummaryCard) taxonomySummaryCard.style.display = 'block';
      if (metadataCard) metadataCard.style.display = 'block';
      if (bruvsVideoPlayer) bruvsVideoPlayer.pause();
      fitCanvasToContainer();
    } else {
      tabVideoMode.classList.add('active');
      tabImageMode.classList.remove('active');
      imageSidebarWrapper.style.display = 'none';
      videoSidebarWrapper.style.display = 'block';
      canvasViewport.style.display = 'none';
      videoWrapper.style.display = 'flex';
      videoActivityBar.style.display = 'block';
      fishList.style.display = 'none';
      videoTracksList.style.display = 'flex';
      if (detailsHeaderTitle) detailsHeaderTitle.textContent = 'Tracked Fish & MaxN Census';
      if (taxonomySummaryCard) taxonomySummaryCard.style.display = 'none';
      if (metadataCard) metadataCard.style.display = 'none';

      if (bruvsVideoPlayer && !bruvsVideoPlayer.src) {
        bruvsVideoPlayer.src = currentVideoUrl;
      }
    }
  }

  tabImageMode.addEventListener('click', () => switchMode('image'));
  tabVideoMode.addEventListener('click', () => switchMode('video'));

  // Video Preset Selection
  const presetVideoSohal = document.getElementById('presetVideoSohal');
  if (presetVideoSohal) {
    presetVideoSohal.addEventListener('click', () => {
      uploadedVideoFile = null;
      currentVideoUrl = '/samples/sample_redsea_sohal.mp4';
      if (bruvsVideoPlayer) bruvsVideoPlayer.src = currentVideoUrl;
    });
  }

  // Video File Upload
  const videoDropZone = document.getElementById('videoDropZone');
  const videoFileInput = document.getElementById('videoFileInput');
  const videoUploadTitle = document.getElementById('videoUploadTitle');

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
    uploadedVideoFile = file;
    currentVideoUrl = URL.createObjectURL(file);
    if (videoUploadTitle) videoUploadTitle.textContent = file.name;
    if (bruvsVideoPlayer) {
      bruvsVideoPlayer.src = currentVideoUrl;
    }
  }

  // Process Video Inference
  if (runVideoBtn) {
    runVideoBtn.addEventListener('click', async () => {
      const videoFps = document.getElementById('videoSampleFps').value;
      const videoDuration = document.getElementById('videoMaxDuration').value;
      const videoClassifier = document.getElementById('videoClassifierSelect').value;

      runVideoBtn.disabled = true;
      if (videoBtnSpinner) videoBtnSpinner.style.display = 'inline-block';
      if (videoBtnText) videoBtnText.textContent = 'Computing MaxN & Tracking...';
      if (videoLoader) videoLoader.style.display = 'flex';

      try {
        const formData = new FormData();
        if (uploadedVideoFile) {
          formData.append('file', uploadedVideoFile);
        } else {
          formData.append('sample_video', 'sample_redsea_sohal.mp4');
        }
        formData.append('sample_fps', videoFps);
        formData.append('max_duration', videoDuration);
        formData.append('classifier_mode', videoClassifier);
        formData.append('conf_threshold', confSlider.value || '0.35');

        const res = await fetch('/api/video/process', {
          method: 'POST',
          body: formData
        });

        if (!res.ok) throw new Error(`Video processing error: ${res.statusText}`);
        const data = await res.json();

        // Highlight fishes: load and play the newly generated annotated video
        const annotUrl = data.video_url || data.annotated_video_url;
        if (annotUrl && bruvsVideoPlayer) {
          bruvsVideoPlayer.src = annotUrl;
          bruvsVideoPlayer.load();
          bruvsVideoPlayer.play().catch(() => {});
        }

        if (data.peak_frame_url && peakFrameImg) {
          peakFrameImg.src = data.peak_frame_url;
          peakFrameImg.style.display = 'block';
          if (peakFramePlaceholder) peakFramePlaceholder.style.display = 'none';
        }

        if (peakTimestampBadge) {
          peakTimestampBadge.textContent = `Timestamp: ${data.peak_timestamp} (${data.peak_second.toFixed(1)}s)`;
        }
        if (maxNCountBadge) {
          maxNCountBadge.textContent = `MaxN: ${data.max_n_total || data.max_n || 0} Fish`;
        }

        // Save timeline data for interactive scrubbing
        state.lastVideoTimeline = data.timeline;
        state.lastPeakSec = data.peak_second;

        if (timelineCanvas && data.timeline) {
          drawTimelineSparkline(timelineCanvas, data.timeline, data.peak_second, 0);
        }

        renderVideoTracksList(data);

      } catch (err) {
        console.error('Video error:', err);
        alert(`Failed to analyze video: ${err.message}`);
      } finally {
        runVideoBtn.disabled = false;
        if (videoBtnSpinner) videoBtnSpinner.style.display = 'none';
        if (videoBtnText) videoBtnText.textContent = 'Process BRUVs Video & MaxN';
        if (videoLoader) videoLoader.style.display = 'none';
      }
    });
  }

  function drawTimelineSparkline(canvasEl, timeline, peakSec, currentSec) {
    const ctxT = canvasEl.getContext('2d');
    const w = canvasEl.width;
    const h = canvasEl.height;

    ctxT.clearRect(0, 0, w, h);
    if (!timeline || timeline.length === 0) return;

    const maxCount = Math.max(1, ...timeline.map(t => t.count));
    const maxSec = timeline[timeline.length - 1].second || 1;
    const stepX = w / Math.max(1, timeline.length - 1);

    const grad = ctxT.createLinearGradient(0, 0, 0, h);
    grad.addColorStop(0, 'rgba(14, 165, 233, 0.45)');
    grad.addColorStop(1, 'rgba(14, 165, 233, 0.02)');

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

    ctxT.strokeStyle = '#0284c7';
    ctxT.lineWidth = 2;
    ctxT.beginPath();
    timeline.forEach((pt, i) => {
      const x = i * stepX;
      const y = (h - 6) - (pt.count / maxCount) * (h - 12);
      if (i === 0) ctxT.moveTo(x, y);
      else ctxT.lineTo(x, y);
    });
    ctxT.stroke();

    if (typeof peakSec === 'number') {
      const peakX = Math.min(w - 2, Math.max(2, (peakSec / maxSec) * w));
      ctxT.strokeStyle = '#059669';
      ctxT.lineWidth = 2.5;
      ctxT.beginPath();
      ctxT.moveTo(peakX, 2);
      ctxT.lineTo(peakX, h - 2);
      ctxT.stroke();

      // Peak circle
      ctxT.fillStyle = '#059669';
      ctxT.beginPath();
      ctxT.arc(peakX, 4, 3.5, 0, Math.PI * 2);
      ctxT.fill();
    }

    // Current playhead scrubber
    if (typeof currentSec === 'number' && currentSec >= 0) {
      const curX = Math.min(w - 2, Math.max(2, (currentSec / maxSec) * w));
      ctxT.strokeStyle = '#f59e0b';
      ctxT.lineWidth = 2;
      ctxT.setLineDash([2, 2]);
      ctxT.beginPath();
      ctxT.moveTo(curX, 2);
      ctxT.lineTo(curX, h - 2);
      ctxT.stroke();
      ctxT.setLineDash([]);
    }
  }

  // Interactive timeline scrubbing & playhead tracking
  if (bruvsVideoPlayer) {
    bruvsVideoPlayer.addEventListener('timeupdate', () => {
      if (state.lastVideoTimeline && timelineCanvas) {
        drawTimelineSparkline(timelineCanvas, state.lastVideoTimeline, state.lastPeakSec, bruvsVideoPlayer.currentTime);
      }
    });
  }

  if (timelineCanvas) {
    timelineCanvas.addEventListener('click', (e) => {
      if (!state.lastVideoTimeline || state.lastVideoTimeline.length === 0) return;
      const rect = timelineCanvas.getBoundingClientRect();
      const ratio = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
      const maxSec = state.lastVideoTimeline[state.lastVideoTimeline.length - 1].second || 30.0;
      if (bruvsVideoPlayer) {
        bruvsVideoPlayer.currentTime = ratio * maxSec;
      }
    });
  }

  if (peakFrameThumbContainer) {
    peakFrameThumbContainer.addEventListener('click', () => {
      if (typeof state.lastPeakSec === 'number' && bruvsVideoPlayer) {
        bruvsVideoPlayer.currentTime = state.lastPeakSec;
      }
    });
  }

  function renderVideoTracksList(data) {
    videoTracksList.innerHTML = '';
    const tracks = data.tracks || [];

    if (tracks.length === 0) {
      videoTracksList.innerHTML = `
        <div class="empty-state">
          <span>No persistent fish tracks identified in analyzed sequence.</span>
        </div>
      `;
      return;
    }

    tracks.forEach(tr => {
      const card = document.createElement('div');
      card.className = `instance-card ${tr.in_peak_frame ? 'selected' : ''}`;

      const bio = tr.biomass || {};
      const lenText = bio.fork_length_cm ? `${bio.fork_length_cm} cm` : `${bio.length_px || '--'} px`;
      const wtText = bio.biomass_g ? `${bio.biomass_g} g` : 'Requires Scale';

      card.innerHTML = `
        <div class="specimen-main-row">
          <div class="specimen-crop-box"><div class="specimen-crop-placeholder">🐟</div></div>
          <div class="specimen-info">
            <div class="specimen-title-row">
              <span class="specimen-id">Track #${tr.track_id}</span>
              <span class="specimen-conf">${Math.round(tr.confidence * 100)}%</span>
            </div>
            <div class="specimen-taxa">${tr.family}</div>
            <span class="specimen-scientific">${tr.in_peak_frame ? '★ Present at MaxN Peak' : 'Passerby Track'}</span>
          </div>
        </div>
        <div class="morph-grid">
          <div class="morph-item"><span>Active:</span><strong>${tr.duration_sec}s</strong></div>
          <div class="morph-item"><span>Est. Length:</span><strong>${lenText}</strong></div>
          <div class="morph-item"><span>Biomass:</span><strong>${wtText}</strong></div>
          <div class="morph-item"><span>Frames:</span><strong>${tr.frame_appearances}</strong></div>
        </div>
      `;

      card.addEventListener('click', () => {
        document.querySelectorAll('.instance-card').forEach(c => c.classList.remove('selected'));
        card.classList.add('selected');
        if (bruvsVideoPlayer && typeof tr.first_seen === 'string') {
          const parts = tr.first_seen.split(':');
          if (parts.length === 2) {
            const sec = parseFloat(parts[0]) * 60 + parseFloat(parts[1]);
            bruvsVideoPlayer.pause();
            bruvsVideoPlayer.currentTime = sec;
          }
        }
      });

      videoTracksList.appendChild(card);
    });
  }
}
