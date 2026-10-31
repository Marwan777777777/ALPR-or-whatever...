/**
 * Mission 012: Independent Vehicle Ground Truth Annotation Tool
 * Complete client-side annotation logic with canvas interaction, zoom/pan,
 * box drawing/resizing, multi-vehicle labeling, and persistence.
 */

// Vehicle Classes with distinct colors
const CLASS_COLORS = {
  heavy_truck: '#ef4444',
  light_commercial: '#f97316',
  passenger_car: '#38bdf8',
  bus: '#a855f7',
  microbus_van: '#ec4899',
  motorcycle: '#10b981',
  tricycle: '#eab308',
  other_vehicle: '#94a3b8'
};

const CLASS_HOTKEYS = {
  '1': 'passenger_car',
  '2': 'light_commercial',
  '3': 'heavy_truck',
  '4': 'bus',
  '5': 'microbus_van',
  '6': 'motorcycle',
  '7': 'tricycle',
  '8': 'other_vehicle'
};

// Application State
const state = {
  samples: [],
  currentIndex: 0,
  currentImage: null,
  annotations: {}, // Keyed by sample_id
  activeBoxIndex: -1,
  isDrawing: false,
  isDragging: false,
  isResizing: false,
  resizeHandle: null,
  isPanning: false,
  panMode: false,
  dragStart: { x: 0, y: 0 },
  zoom: 1.0,
  pan: { x: 0, y: 0 },
  brightness: 100,
  contrast: 100,
  HANDLE_SIZE: 8,
};

// DOM Elements
const canvas = document.getElementById('annotationCanvas');
const ctx = canvas.getContext('2d');
const container = document.getElementById('canvasContainer');

// UI Controls
const sampleIndexText = document.getElementById('sampleIndexText');
const metaSampleId = document.getElementById('metaSampleId');
const metaTimestamp = document.getElementById('metaTimestamp');
const metaResolution = document.getElementById('metaResolution');
const metaAnnotStatus = document.getElementById('metaAnnotStatus');
const lightingBadge = document.getElementById('lightingBadge');
const boxList = document.getElementById('boxList');
const boxCount = document.getElementById('boxCount');
const emptyBanner = document.getElementById('emptyBanner');
const boxEditorSection = document.getElementById('boxEditorSection');
const coordsReadout = document.getElementById('coordsReadout');
const sizeReadout = document.getElementById('sizeReadout');
const chkAdjudication = document.getElementById('chkAdjudication');
const boxNotes = document.getElementById('boxNotes');
const zoomLevelText = document.getElementById('zoomLevelText');
const brightnessSlider = document.getElementById('brightnessSlider');
const contrastSlider = document.getElementById('contrastSlider');
const statAnnotated = document.getElementById('statAnnotated');
const statBoxes = document.getElementById('statBoxes');
const statFlagged = document.getElementById('statFlagged');
const statEmpty = document.getElementById('statEmpty');
const progressBar = document.getElementById('progressBar');
const progressPercent = document.getElementById('progressPercent');

// Initialize
async function init() {
  loadStoredAnnotations();
  await fetchSampleManifest();
  setupEventListeners();
  loadSample(0);
}

// Fetch Sample Manifest from server or generate fallback for offline mode
async function fetchSampleManifest() {
  try {
    const res = await fetch('/api/images');
    if (res.ok) {
      const data = await res.json();
      state.samples = data.samples;
      console.log(`Loaded ${state.samples.length} samples from server manifest.`);
      return;
    }
  } catch (e) {
    console.warn('Backend API not reachable. Initializing standard 340-sample fallback index.');
  }

  // Fallback generation for 340 samples
  state.samples = [];
  for (let i = 1; i <= 340; i++) {
    const sampleId = `sample_${String(i).padStart(3, '0')}`;
    state.samples.push({
      index: i - 1,
      sample_id: sampleId,
      anonymous_id: `SAMPLE-${String(i).padStart(3, '0')}`,
      image_url: `/api/image?index=${i - 1}`,
      timestamp: 'Unknown',
      lighting_condition: i <= 10 ? 'NIGHT_IR' : (i <= 209 ? 'DAYLIGHT' : 'EVENING_IR'),
      width: 3840,
      height: 2160
    });
  }
}

// Load Stored Annotations from LocalStorage
function loadStoredAnnotations() {
  try {
    const stored = localStorage.getItem('mission_012_vehicle_annotations');
    if (stored) {
      state.annotations = JSON.parse(stored);
      console.log('Restored annotations from LocalStorage.');
    }
  } catch (e) {
    console.error('Failed to parse stored annotations:', e);
  }
}

// Save Annotations to LocalStorage and Backend API
async function saveAnnotations(notify = true) {
  try {
    localStorage.setItem('mission_012_vehicle_annotations', JSON.stringify(state.annotations));
    
    // Attempt backend save if available
    try {
      await fetch('/api/annotations/save', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(state.annotations)
      });
    } catch (e) {
      // Offline fallback
    }

    updateStats();
    if (notify) {
      const btn = document.getElementById('btnSave');
      const originalText = btn.innerHTML;
      btn.innerHTML = '✓ Saved!';
      btn.style.backgroundColor = 'var(--success)';
      setTimeout(() => {
        btn.innerHTML = originalText;
        btn.style.backgroundColor = '';
      }, 1200);
    }
  } catch (e) {
    console.error('Error saving annotations:', e);
  }
}

// Load Specific Sample
function loadSample(index) {
  if (index < 0 || index >= state.samples.length) return;
  state.currentIndex = index;
  state.activeBoxIndex = -1;
  const sample = state.samples[index];

  // Initialize annotation entry if missing
  if (!state.annotations[sample.sample_id]) {
    state.annotations[sample.sample_id] = {
      sample_id: sample.sample_id,
      anonymous_id: sample.anonymous_id,
      timestamp: sample.timestamp,
      lighting_condition: sample.lighting_condition,
      is_empty_frame: false,
      boxes: []
    };
  }

  // Update UI Meta
  sampleIndexText.textContent = `Sample ${index + 1} of ${state.samples.length}`;
  metaSampleId.textContent = sample.anonymous_id;
  metaTimestamp.textContent = sample.timestamp || '--:--:--';
  metaResolution.textContent = `${sample.width || 3840} × ${sample.height || 2160}`;
  
  // Lighting Badge
  lightingBadge.textContent = sample.lighting_condition;
  lightingBadge.className = 'badge';
  if (sample.lighting_condition === 'DAYLIGHT') lightingBadge.classList.add('badge-daylight');
  else if (sample.lighting_condition === 'EVENING_IR') lightingBadge.classList.add('badge-evening');
  else lightingBadge.classList.add('badge-night');

  // Load Image
  const img = new Image();
  img.crossOrigin = 'anonymous';
  img.onload = () => {
    state.currentImage = img;
    sample.width = img.naturalWidth;
    sample.height = img.naturalHeight;
    metaResolution.textContent = `${img.naturalWidth} × ${img.naturalHeight}`;
    resetZoomFit();
    render();
    updateBoxList();
  };
  img.src = sample.image_url;

  updateStats();
}

// Reset Zoom to Fit Container
function resetZoomFit() {
  if (!state.currentImage) return;
  const containerW = container.clientWidth;
  const containerH = container.clientHeight;
  const imgW = state.currentImage.naturalWidth;
  const imgH = state.currentImage.naturalHeight;

  const scale = Math.min((containerW - 40) / imgW, (containerH - 40) / imgH);
  state.zoom = scale;
  state.pan = {
    x: (containerW - imgW * scale) / 2,
    y: (containerH - imgH * scale) / 2
  };
  zoomLevelText.textContent = `${Math.round(scale * 100)}%`;
}

// Convert Screen (Canvas Container) Coordinates to Image Pixel Coordinates
function screenToImageCoords(screenX, screenY) {
  const rect = container.getBoundingClientRect();
  const relX = screenX - rect.left - state.pan.x;
  const relY = screenY - rect.top - state.pan.y;
  return {
    x: Math.max(0, Math.min(state.currentImage.naturalWidth, relX / state.zoom)),
    y: Math.max(0, Math.min(state.currentImage.naturalHeight, relY / state.zoom))
  };
}

// Convert Image Pixel Coordinates to Screen Coordinates
function imageToScreenCoords(imgX, imgY) {
  return {
    x: imgX * state.zoom + state.pan.x,
    y: imgY * state.zoom + state.pan.y
  };
}

// Render Canvas
function render() {
  if (!state.currentImage) return;

  const w = container.clientWidth;
  const h = container.clientHeight;
  canvas.width = w;
  canvas.height = h;

  ctx.clearRect(0, 0, w, h);

  // Apply Brightness & Contrast Filter
  ctx.save();
  ctx.filter = `brightness(${state.brightness}%) contrast(${state.contrast}%)`;
  
  // Draw Scaled Image
  const imgW = state.currentImage.naturalWidth * state.zoom;
  const imgH = state.currentImage.naturalHeight * state.zoom;
  ctx.drawImage(state.currentImage, state.pan.x, state.pan.y, imgW, imgH);
  ctx.restore();

  // Draw Bounding Boxes
  const currentAnnot = state.annotations[state.samples[state.currentIndex].sample_id];
  if (currentAnnot && !currentAnnot.is_empty_frame) {
    currentAnnot.boxes.forEach((box, idx) => {
      drawBox(box, idx === state.activeBoxIndex);
    });
  }
}

// Draw Single Bounding Box on Canvas
function drawBox(box, isActive) {
  const p1 = imageToScreenCoords(box.x1, box.y1);
  const p2 = imageToScreenCoords(box.x2, box.y2);
  const bw = p2.x - p1.x;
  const bh = p2.y - p1.y;

  const color = CLASS_COLORS[box.class] || '#38bdf8';

  // Box Fill & Stroke
  ctx.save();
  ctx.lineWidth = isActive ? 3 : 2;
  ctx.strokeStyle = color;
  ctx.fillStyle = isActive ? `${color}33` : `${color}15`;
  ctx.fillRect(p1.x, p1.y, bw, bh);
  ctx.strokeRect(p1.x, p1.y, bw, bh);

  // Class Label Tag
  const label = `${box.class.replace('_', ' ')} [${box.visibility.substring(0, 4)}]`;
  ctx.font = 'bold 12px sans-serif';
  const textWidth = ctx.measureText(label).width;
  ctx.fillStyle = color;
  ctx.fillRect(p1.x - 1, p1.y - 18, textWidth + 10, 18);
  ctx.fillStyle = '#ffffff';
  ctx.fillText(label, p1.x + 4, p1.y - 4);

  // Draw Resize Handles if Active
  if (isActive) {
    drawHandle(p1.x, p1.y); // NW
    drawHandle(p2.x, p1.y); // NE
    drawHandle(p2.x, p2.y); // SE
    drawHandle(p1.x, p2.y); // SW
  }
  ctx.restore();
}

function drawHandle(x, y) {
  const s = state.HANDLE_SIZE;
  ctx.fillStyle = '#ffffff';
  ctx.strokeStyle = '#0b0f19';
  ctx.lineWidth = 2;
  ctx.fillRect(x - s / 2, y - s / 2, s, s);
  ctx.strokeRect(x - s / 2, y - s / 2, s, s);
}

// Check which handle was clicked
function getHandleAt(screenX, screenY, box) {
  const s = state.HANDLE_SIZE * 1.5;
  const p1 = imageToScreenCoords(box.x1, box.y1);
  const p2 = imageToScreenCoords(box.x2, box.y2);
  const handles = [
    { name: 'nw', x: p1.x, y: p1.y },
    { name: 'ne', x: p2.x, y: p1.y },
    { name: 'se', x: p2.x, y: p2.y },
    { name: 'sw', x: p1.x, y: p2.y }
  ];
  for (const h of handles) {
    if (Math.abs(screenX - h.x) <= s && Math.abs(screenY - h.y) <= s) {
      return h.name;
    }
  }
  return null;
}

// Event Listeners Setup
function setupEventListeners() {
  window.addEventListener('resize', () => {
    render();
  });

  // Navigation
  document.getElementById('btnPrev').addEventListener('click', () => {
    if (state.currentIndex > 0) loadSample(state.currentIndex - 1);
  });
  document.getElementById('btnNext').addEventListener('click', () => {
    if (state.currentIndex < state.samples.length - 1) loadSample(state.currentIndex + 1);
  });
  document.getElementById('btnJump').addEventListener('click', () => {
    const val = parseInt(document.getElementById('jumpIndex').value);
    if (val >= 1 && val <= state.samples.length) loadSample(val - 1);
  });
  document.getElementById('btnNextUnannotated').addEventListener('click', jumpToNextUnannotated);

  // Persistence
  document.getElementById('btnSave').addEventListener('click', () => saveAnnotations(true));
  document.getElementById('btnExportJSON').addEventListener('click', exportJSON);
  document.getElementById('btnExportCSV').addEventListener('click', exportCSV);
  document.getElementById('btnImportJSON').addEventListener('click', () => {
    document.getElementById('importFileInput').click();
  });
  document.getElementById('importFileInput').addEventListener('change', importJSON);

  // Enhancements
  brightnessSlider.addEventListener('input', (e) => {
    state.brightness = e.target.value;
    render();
  });
  contrastSlider.addEventListener('input', (e) => {
    state.contrast = e.target.value;
    render();
  });
  document.getElementById('btnResetEnhance').addEventListener('click', () => {
    state.brightness = 100;
    state.contrast = 100;
    brightnessSlider.value = 100;
    contrastSlider.value = 100;
    render();
  });

  // Zoom Controls
  document.getElementById('btnZoomIn').addEventListener('click', () => {
    state.zoom *= 1.25;
    zoomLevelText.textContent = `${Math.round(state.zoom * 100)}%`;
    render();
  });
  document.getElementById('btnZoomOut').addEventListener('click', () => {
    state.zoom /= 1.25;
    zoomLevelText.textContent = `${Math.round(state.zoom * 100)}%`;
    render();
  });
  document.getElementById('btnZoomFit').addEventListener('click', resetZoomFit);
  document.getElementById('btnTogglePan').addEventListener('click', () => {
    state.panMode = !state.panMode;
    document.getElementById('btnTogglePan').style.backgroundColor = state.panMode ? 'var(--accent)' : '';
  });

  // Canvas Mouse Events
  canvas.addEventListener('mousedown', handleMouseDown);
  window.addEventListener('mousemove', handleMouseMove);
  window.addEventListener('mouseup', handleMouseUp);
  canvas.addEventListener('wheel', handleWheel, { passive: false });

  // Class Selection
  document.querySelectorAll('.class-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const cls = btn.getAttribute('data-class');
      setClassForActiveBox(cls);
    });
  });

  // Visibility Selection
  document.querySelectorAll('.vis-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const vis = btn.getAttribute('data-vis');
      setVisibilityForActiveBox(vis);
    });
  });

  // Adjudication & Notes
  chkAdjudication.addEventListener('change', (e) => {
    const box = getActiveBox();
    if (box) {
      box.is_adjudication = e.target.checked;
      saveAnnotations(false);
      updateBoxList();
    }
  });

  boxNotes.addEventListener('input', (e) => {
    const box = getActiveBox();
    if (box) {
      box.notes = e.target.value;
      saveAnnotations(false);
    }
  });

  // Delete Box
  document.getElementById('btnDeleteBox').addEventListener('click', deleteActiveBox);

  // Mark Empty Frame
  document.getElementById('btnMarkEmpty').addEventListener('click', toggleEmptyFrame);

  // Keyboard Hotkeys
  window.addEventListener('keydown', handleKeyDown);
}

// Mouse Event Handlers
function handleMouseDown(e) {
  if (!state.currentImage) return;

  const rect = container.getBoundingClientRect();
  const screenX = e.clientX;
  const screenY = e.clientY;

  // Pan Mode Check
  if (state.panMode || e.spaceKey || e.button === 1) {
    state.isPanning = true;
    state.dragStart = { x: screenX - state.pan.x, y: screenY - state.pan.y };
    return;
  }

  const coords = screenToImageCoords(screenX, screenY);
  const currentAnnot = state.annotations[state.samples[state.currentIndex].sample_id];

  // 1. Check if clicking on an active box handle
  const activeBox = getActiveBox();
  if (activeBox) {
    const handle = getHandleAt(screenX, screenY, activeBox);
    if (handle) {
      state.isResizing = true;
      state.resizeHandle = handle;
      state.dragStart = coords;
      return;
    }
  }

  // 2. Check if clicking inside an existing box to select / move
  let clickedIndex = -1;
  if (currentAnnot && !currentAnnot.is_empty_frame) {
    for (let i = currentAnnot.boxes.length - 1; i >= 0; i--) {
      const b = currentAnnot.boxes[i];
      if (coords.x >= b.x1 && coords.x <= b.x2 && coords.y >= b.y1 && coords.y <= b.y2) {
        clickedIndex = i;
        break;
      }
    }
  }

  if (clickedIndex !== -1) {
    state.activeBoxIndex = clickedIndex;
    state.isDragging = true;
    state.dragStart = coords;
    updateBoxEditor();
    updateBoxList();
    render();
    return;
  }

  // 3. Otherwise, start drawing a new box
  if (currentAnnot.is_empty_frame) {
    currentAnnot.is_empty_frame = false;
    emptyBanner.style.display = 'none';
  }

  state.isDrawing = true;
  const newBox = {
    box_id: `box_${Date.now()}`,
    x1: coords.x,
    y1: coords.y,
    x2: coords.x,
    y2: coords.y,
    class: 'heavy_truck',
    visibility: 'fully_visible',
    is_adjudication: false,
    notes: ''
  };
  currentAnnot.boxes.push(newBox);
  state.activeBoxIndex = currentAnnot.boxes.length - 1;
  state.dragStart = coords;
  updateBoxEditor();
  render();
}

function handleMouseMove(e) {
  if (state.isPanning) {
    state.pan.x = e.clientX - state.dragStart.x;
    state.pan.y = e.clientY - state.dragStart.y;
    render();
    return;
  }

  const coords = screenToImageCoords(e.clientX, e.clientY);
  const box = getActiveBox();

  if (state.isDrawing && box) {
    box.x2 = coords.x;
    box.y2 = coords.y;
    normalizeBoxCoords(box);
    updateBoxEditor();
    render();
  } else if (state.isDragging && box) {
    const dx = coords.x - state.dragStart.x;
    const dy = coords.y - state.dragStart.y;
    const w = box.x2 - box.x1;
    const h = box.y2 - box.y1;

    box.x1 = Math.max(0, Math.min(state.currentImage.naturalWidth - w, box.x1 + dx));
    box.y1 = Math.max(0, Math.min(state.currentImage.naturalHeight - h, box.y1 + dy));
    box.x2 = box.x1 + w;
    box.y2 = box.y1 + h;

    state.dragStart = coords;
    updateBoxEditor();
    render();
  } else if (state.isResizing && box) {
    if (state.resizeHandle === 'nw') {
      box.x1 = Math.min(coords.x, box.x2 - 10);
      box.y1 = Math.min(coords.y, box.y2 - 10);
    } else if (state.resizeHandle === 'ne') {
      box.x2 = Math.max(coords.x, box.x1 + 10);
      box.y1 = Math.min(coords.y, box.y2 - 10);
    } else if (state.resizeHandle === 'se') {
      box.x2 = Math.max(coords.x, box.x1 + 10);
      box.y2 = Math.max(coords.y, box.y1 + 10);
    } else if (state.resizeHandle === 'sw') {
      box.x1 = Math.min(coords.x, box.x2 - 10);
      box.y2 = Math.max(coords.y, box.y1 + 10);
    }
    updateBoxEditor();
    render();
  }
}

function handleMouseUp() {
  if (state.isDrawing) {
    const box = getActiveBox();
    if (box) {
      normalizeBoxCoords(box);
      // Remove degenerate zero-area boxes
      if ((box.x2 - box.x1) < 15 || (box.y2 - box.y1) < 15) {
        deleteActiveBox();
      }
    }
    state.isDrawing = false;
    saveAnnotations(false);
    updateBoxList();
  }
  state.isDragging = false;
  state.isResizing = false;
  state.isPanning = false;
  render();
}

function handleWheel(e) {
  e.preventDefault();
  const delta = e.deltaY < 0 ? 1.15 : 0.85;
  const rect = container.getBoundingClientRect();
  const mouseX = e.clientX - rect.left;
  const mouseY = e.clientY - rect.top;

  const newZoom = state.zoom * delta;
  if (newZoom < 0.1 || newZoom > 10.0) return;

  state.pan.x = mouseX - (mouseX - state.pan.x) * delta;
  state.pan.y = mouseY - (mouseY - state.pan.y) * delta;
  state.zoom = newZoom;
  zoomLevelText.textContent = `${Math.round(state.zoom * 100)}%`;
  render();
}

// Coordinate Normalization Helper
function normalizeBoxCoords(box) {
  const xmin = Math.min(box.x1, box.x2);
  const xmax = Math.max(box.x1, box.x2);
  const ymin = Math.min(box.y1, box.y2);
  const ymax = Math.max(box.y1, box.y2);
  box.x1 = Math.round(xmin);
  box.y1 = Math.round(ymin);
  box.x2 = Math.round(xmax);
  box.y2 = Math.round(ymax);
}

// Active Box Helpers
function getActiveBox() {
  const currentAnnot = state.annotations[state.samples[state.currentIndex]?.sample_id];
  if (!currentAnnot || state.activeBoxIndex < 0 || state.activeBoxIndex >= currentAnnot.boxes.length) {
    return null;
  }
  return currentAnnot.boxes[state.activeBoxIndex];
}

function setClassForActiveBox(cls) {
  const box = getActiveBox();
  if (box) {
    box.class = cls;
    updateBoxEditor();
    updateBoxList();
    render();
    saveAnnotations(false);
  }
}

function setVisibilityForActiveBox(vis) {
  const box = getActiveBox();
  if (box) {
    box.visibility = vis;
    updateBoxEditor();
    updateBoxList();
    render();
    saveAnnotations(false);
  }
}

function deleteActiveBox() {
  const currentAnnot = state.annotations[state.samples[state.currentIndex]?.sample_id];
  if (!currentAnnot || state.activeBoxIndex < 0) return;
  currentAnnot.boxes.splice(state.activeBoxIndex, 1);
  state.activeBoxIndex = currentAnnot.boxes.length - 1;
  updateBoxEditor();
  updateBoxList();
  render();
  saveAnnotations(false);
}

function toggleEmptyFrame() {
  const currentAnnot = state.annotations[state.samples[state.currentIndex]?.sample_id];
  if (!currentAnnot) return;
  currentAnnot.is_empty_frame = !currentAnnot.is_empty_frame;
  if (currentAnnot.is_empty_frame) {
    currentAnnot.boxes = [];
    state.activeBoxIndex = -1;
    emptyBanner.style.display = 'block';
  } else {
    emptyBanner.style.display = 'none';
  }
  updateBoxEditor();
  updateBoxList();
  render();
  saveAnnotations(false);
}

// Update UI Views
function updateBoxEditor() {
  const box = getActiveBox();
  if (!box) {
    coordsReadout.textContent = '[--]';
    sizeReadout.textContent = '0 × 0 px';
    chkAdjudication.checked = false;
    boxNotes.value = '';
    return;
  }

  coordsReadout.textContent = `[${box.x1}, ${box.y1}, ${box.x2}, ${box.y2}]`;
  sizeReadout.textContent = `${box.x2 - box.x1} × ${box.y2 - box.y1} px`;
  chkAdjudication.checked = box.is_adjudication || false;
  boxNotes.value = box.notes || '';

  // Class Buttons
  document.querySelectorAll('.class-btn').forEach(btn => {
    btn.classList.toggle('selected', btn.getAttribute('data-class') === box.class);
  });

  // Visibility Buttons
  document.querySelectorAll('.vis-btn').forEach(btn => {
    btn.classList.toggle('selected', btn.getAttribute('data-vis') === box.visibility);
  });
}

function updateBoxList() {
  const currentAnnot = state.annotations[state.samples[state.currentIndex]?.sample_id];
  boxList.innerHTML = '';

  if (!currentAnnot || currentAnnot.is_empty_frame || currentAnnot.boxes.length === 0) {
    boxCount.textContent = '0';
    if (currentAnnot?.is_empty_frame) {
      emptyBanner.style.display = 'block';
    } else {
      emptyBanner.style.display = 'none';
    }
    return;
  }

  emptyBanner.style.display = 'none';
  boxCount.textContent = currentAnnot.boxes.length;

  currentAnnot.boxes.forEach((box, idx) => {
    const item = document.createElement('div');
    item.className = `box-item ${idx === state.activeBoxIndex ? 'active' : ''}`;
    item.innerHTML = `
      <div class="box-item-indicator" style="background-color: ${CLASS_COLORS[box.class]};"></div>
      <div class="box-item-info">
        <div class="box-item-name">${box.class.replace('_', ' ')} ${box.is_adjudication ? '⚠️' : ''}</div>
        <div class="box-item-meta">${box.visibility} | ${box.x2 - box.x1}×${box.y2 - box.y1}px</div>
      </div>
    `;
    item.addEventListener('click', () => {
      state.activeBoxIndex = idx;
      updateBoxEditor();
      updateBoxList();
      render();
    });
    boxList.appendChild(item);
  });
}

// Update Overall Benchmark Stats
function updateStats() {
  let annotatedCount = 0;
  let totalBoxes = 0;
  let flaggedCount = 0;
  let emptyCount = 0;

  state.samples.forEach(s => {
    const annot = state.annotations[s.sample_id];
    if (annot) {
      if (annot.is_empty_frame) {
        annotatedCount++;
        emptyCount++;
      } else if (annot.boxes && annot.boxes.length > 0) {
        annotatedCount++;
        totalBoxes += annot.boxes.length;
        if (annot.boxes.some(b => b.is_adjudication)) flaggedCount++;
      }
    }
  });

  statAnnotated.textContent = `${annotatedCount} / ${state.samples.length}`;
  statBoxes.textContent = totalBoxes;
  statFlagged.textContent = flaggedCount;
  statEmpty.textContent = emptyCount;

  const pct = Math.round((annotatedCount / state.samples.length) * 100);
  progressPercent.textContent = `${pct}%`;
  progressBar.style.width = `${pct}%`;

  // Status for active sample
  const current = state.annotations[state.samples[state.currentIndex]?.sample_id];
  if (current && (current.is_empty_frame || (current.boxes && current.boxes.length > 0))) {
    metaAnnotStatus.textContent = 'Completed';
    metaAnnotStatus.style.color = 'var(--success)';
  } else {
    metaAnnotStatus.textContent = 'Pending';
    metaAnnotStatus.style.color = 'var(--warning)';
  }
}

function jumpToNextUnannotated() {
  for (let i = 0; i < state.samples.length; i++) {
    const idx = (state.currentIndex + 1 + i) % state.samples.length;
    const sample = state.samples[idx];
    const annot = state.annotations[sample.sample_id];
    if (!annot || (!annot.is_empty_frame && (!annot.boxes || annot.boxes.length === 0))) {
      loadSample(idx);
      return;
    }
  }
  alert('All 340 gate captures have been annotated!');
}

// Keyboard Navigation and Hotkeys
function handleKeyDown(e) {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

  if (e.key === 'p' || e.key === 'P' || e.key === '[') {
    if (state.currentIndex > 0) loadSample(state.currentIndex - 1);
  } else if (e.key === 'n' || e.key === 'N' || e.key === ']') {
    if (state.currentIndex < state.samples.length - 1) loadSample(state.currentIndex + 1);
  } else if (e.key === 'Delete' || e.key === 'Backspace') {
    deleteActiveBox();
  } else if (e.key === 's' || e.key === 'S') {
    e.preventDefault();
    saveAnnotations(true);
  } else if (e.key === 'e' || e.key === 'E') {
    toggleEmptyFrame();
  } else if (CLASS_HOTKEYS[e.key]) {
    setClassForActiveBox(CLASS_HOTKEYS[e.key]);
  }
}

// Export to JSON
function exportJSON() {
  const dataStr = 'data:text/json;charset=utf-8,' + encodeURIComponent(JSON.stringify(state.annotations, null, 2));
  const dlAnchor = document.createElement('a');
  dlAnchor.setAttribute('href', dataStr);
  dlAnchor.setAttribute('download', 'vehicle_ground_truth.json');
  dlAnchor.click();
}

// Export to CSV
function exportCSV() {
  const headers = ['sample_id', 'anonymous_id', 'lighting_condition', 'is_empty_frame', 'box_id', 'class', 'x1', 'y1', 'x2', 'y2', 'visibility', 'is_adjudication', 'notes'];
  const rows = [headers.join(',')];

  state.samples.forEach(s => {
    const annot = state.annotations[s.sample_id];
    if (annot) {
      if (annot.is_empty_frame) {
        rows.push([s.sample_id, s.anonymous_id, s.lighting_condition, 'True', '', '', '', '', '', '', '', '', ''].join(','));
      } else if (annot.boxes) {
        annot.boxes.forEach(b => {
          rows.push([
            s.sample_id,
            s.anonymous_id,
            s.lighting_condition,
            'False',
            b.box_id,
            b.class,
            b.x1, b.y1, b.x2, b.y2,
            b.visibility,
            b.is_adjudication ? 'True' : 'False',
            `"${(b.notes || '').replace(/"/g, '""')}"`
          ].join(','));
        });
      }
    }
  });

  const blob = new Blob([rows.join('\n')], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const dlAnchor = document.createElement('a');
  dlAnchor.setAttribute('href', url);
  dlAnchor.setAttribute('download', 'vehicle_ground_truth.csv');
  dlAnchor.click();
}

// Import JSON
function importJSON(e) {
  const file = e.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = (event) => {
    try {
      const data = JSON.parse(event.target.result);
      state.annotations = data;
      saveAnnotations(true);
      loadSample(state.currentIndex);
      alert('Annotations imported successfully.');
    } catch (err) {
      alert('Failed to parse JSON file.');
    }
  };
  reader.readAsText(file);
}

// Run Initialization on Load
window.addEventListener('DOMContentLoaded', init);
