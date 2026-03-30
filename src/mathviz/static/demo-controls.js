/**
 * UI controls wiring for the static demo page.
 */
import * as THREE from 'three';
import {
  addCrystalLedLight, removeCrystalLedLight, updateColorMap,
} from './demo-crystal.js';
import {exportTurntable} from './demo-export.js';

const TURNTABLE_BASE_SPEED = 2;

/** Wire all UI controls to scene actions. */
export function wireControls(ctx) {
  const {state, scene, renderer, camera, controls, container, actions} = ctx;
  const {
    applyViewMode, updateDensitySliderVisibility, applyDensityFilter,
    updatePointSize, applyStretch, syncStretchUI, fitCamera,
    removeAxes, createAxesWithLabels, applyLightIntensities, loadVisualization,
  } = actions;
  const DARK_COLOR = ctx.DARK_COLOR;
  const LIGHT_COLOR = ctx.LIGHT_COLOR;

  /* Slider value labels */
  document.querySelectorAll('input[type="range"][data-label]').forEach(slider => {
    slider.addEventListener('input', () => {
      document.getElementById(slider.dataset.label).textContent = slider.value;
    });
  });

  /* View mode */
  document.getElementById('view-mode').addEventListener('change', (e) => {
    state.viewMode = e.target.value;
    applyViewMode();
    updateDensitySliderVisibility();
  });

  /* Point size */
  document.getElementById('point-size').addEventListener('input', (e) => {
    updatePointSize(parseFloat(e.target.value));
  });

  /* Density */
  let densityDirty = false;
  document.getElementById('density-slider').addEventListener('input', (e) => {
    state.density = parseFloat(e.target.value);
    if (!densityDirty) {
      densityDirty = true;
      requestAnimationFrame(() => { densityDirty = false; applyDensityFilter(); });
    }
  });

  /* Bounding box */
  document.getElementById('show-bbox').addEventListener('change', (e) => {
    if (state.bboxHelper) state.bboxHelper.visible = e.target.checked;
  });

  /* Axes */
  document.getElementById('show-axes').addEventListener('change', (e) => {
    if (e.target.checked) { removeAxes(); createAxesWithLabels(); } else { removeAxes(); }
  });

  /* Background toggle */
  document.getElementById('toggle-bg').addEventListener('change', (e) => {
    const isLight = e.target.checked;
    state.darkBg = !isLight;
    if (state.crystalActive) {
      scene.background = new THREE.Color(0x000000);
    } else {
      scene.background = new THREE.Color(isLight ? LIGHT_COLOR : DARK_COLOR);
    }
    applyLightIntensities(!isLight);
    document.getElementById('controls').classList.toggle('light-bg', isLight);
    document.getElementById('info-panel').classList.toggle('light-bg', isLight);
    document.getElementById('gallery-panel').classList.toggle('light-bg', isLight);
  });

  /* Camera lock */
  document.getElementById('lock-camera').addEventListener('click', () => {
    const cycle = {render: 'full', full: 'off', off: 'render'};
    const labels = {render: '\u{1f512} Render Lock', full: '\u{1f512} Full Lock', off: '\u{1f513} Free'};
    const next = cycle[state.cameraLocked] ?? 'render';
    state.cameraLocked = next;
    const btn = document.getElementById('lock-camera');
    btn.dataset.mode = next;
    btn.textContent = labels[next];
    btn.title = next === 'render' ? 'Camera: Locked (movable)'
      : next === 'full' ? 'Camera: Frozen' : 'Camera: Free';
    controls.enabled = next !== 'full';
    renderer.domElement.style.cursor = next === 'full' ? 'not-allowed' : '';
  });

  _wireStretchControls(state, applyStretch, syncStretchUI);
  _wireResetView(state, controls, fitCamera);
  _wireScreenshot(renderer, scene, camera);
  _wireCrystalControls(state, scene);
  _wireColormapControls(state, scene);
  _wireTurntable(state, controls, renderer, camera, scene, container);

  /* Visualization selector */
  document.getElementById('viz-selector').addEventListener('change', (e) => {
    const selected = e.target.selectedOptions[0];
    if (!selected) return;
    loadVisualization(
      selected.textContent,
      selected.dataset.mesh,
      selected.dataset.cloud,
    );
  });
}

function _wireStretchControls(state, applyStretch, syncStretchUI) {
  ['stretch-x', 'stretch-y', 'stretch-z'].forEach((id) => {
    const axis = id.split('-')[1];
    const slider = document.getElementById(id);
    const numInput = document.getElementById(id + '-num');
    slider.addEventListener('input', (e) => {
      const val = parseFloat(e.target.value);
      numInput.value = val.toFixed(1);
      state.stretch[axis] = val;
      applyStretch();
    });
    numInput.addEventListener('input', (e) => {
      const parsed = parseFloat(e.target.value);
      if (isNaN(parsed)) return;
      const val = Math.max(0.1, Math.min(3, parsed));
      slider.value = val;
      state.stretch[axis] = val;
      applyStretch();
    });
  });

  document.getElementById('reset-scale-btn').addEventListener('click', () => {
    state.stretch = {x: 1, y: 1, z: 1};
    syncStretchUI(state.stretch);
    applyStretch();
  });

  const stretchToggle = document.getElementById('stretch-toggle');
  const stretchPanel = document.getElementById('stretch-panel');
  stretchToggle.addEventListener('click', () => {
    stretchPanel.classList.toggle('collapsed');
    const chevron = stretchToggle.querySelector('.chevron');
    chevron.style.transform = stretchPanel.classList.contains('collapsed') ? '' : 'rotate(90deg)';
  });
}

function _wireResetView(state, controls, fitCamera) {
  const resetViewBtn = document.getElementById('reset-view-btn');
  resetViewBtn.addEventListener('click', () => {
    const active = state.meshGroup || state.cloudPoints;
    if (!active) return;
    const wasFullLock = state.cameraLocked === 'full';
    if (wasFullLock) controls.enabled = true;
    try { fitCamera(active); } finally { if (wasFullLock) controls.enabled = false; }
  });

  document.addEventListener('keydown', (e) => {
    if (e.key !== 'Home') return;
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    if (resetViewBtn.disabled) return;
    const tag = e.target.tagName;
    if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA') return;
    e.preventDefault();
    resetViewBtn.click();
  });
}

function _wireScreenshot(renderer, scene, camera) {
  document.getElementById('screenshot-btn').addEventListener('click', () => {
    renderer.render(scene, camera);
    const link = document.createElement('a');
    link.download = 'mathviz-screenshot.png';
    link.href = renderer.domElement.toDataURL('image/png');
    link.click();
  });
}

function _wireCrystalControls(state, scene) {
  document.getElementById('crystal-glass-tint').addEventListener('input', (e) => {
    if (state.crystalGlassBlock) state.crystalGlassBlock.material.color.set(e.target.value);
  });
  document.getElementById('crystal-bloom').addEventListener('input', (e) => {
    if (state.crystalComposer) {
      const bloomPass = state.crystalComposer.passes.find(p => p.name === 'crystal-bloom');
      if (bloomPass) bloomPass.strength = parseFloat(e.target.value);
    }
  });
  document.getElementById('crystal-brightness').addEventListener('input', (e) => {
    const brightness = parseFloat(e.target.value);
    scene.traverse((child) => {
      if (child.isPoints && child.material && child.material.blending === THREE.AdditiveBlending) {
        child.material.opacity = brightness;
      }
    });
  });
  document.getElementById('crystal-led-base').addEventListener('change', (e) => {
    if (!state.crystalActive) return;
    if (e.target.checked) addCrystalLedLight(state, scene);
    else removeCrystalLedLight(state, scene);
  });
  document.getElementById('crystal-led-color').addEventListener('input', (e) => {
    if (state.crystalLedLight) state.crystalLedLight.color.set(e.target.value);
  });
}

function _wireColormapControls(state, scene) {
  document.getElementById('colormap-metric').addEventListener('change', (e) => {
    state.colormapMetric = e.target.value;
    updateColorMap(state, scene);
  });
  document.getElementById('colormap-gradient').addEventListener('change', (e) => {
    state.colormapGradient = e.target.value;
    document.getElementById('colormap-custom-colors').style.display =
      e.target.value === 'custom' ? 'block' : 'none';
    updateColorMap(state, scene);
  });
  document.getElementById('colormap-start-color').addEventListener('input', () => {
    if (state.colormapGradient === 'custom') updateColorMap(state, scene);
  });
  document.getElementById('colormap-end-color').addEventListener('input', () => {
    if (state.colormapGradient === 'custom') updateColorMap(state, scene);
  });
}

function _wireTurntable(state, controls, renderer, camera, scene, container) {
  function setTurntable(enabled) {
    state.turntable = enabled;
    controls.autoRotate = enabled;
    controls.autoRotateSpeed = enabled ? state.turntableSpeed * TURNTABLE_BASE_SPEED : 0;
    document.getElementById('turntable-speed').disabled = !enabled;
    const section = document.getElementById('turntable-export-section');
    if (enabled) section.classList.add('visible');
    else section.classList.remove('visible');
  }

  document.getElementById('turntable-toggle').addEventListener('change', (e) => {
    setTurntable(e.target.checked);
  });

  document.getElementById('turntable-speed').addEventListener('input', (e) => {
    const speed = parseFloat(e.target.value);
    state.turntableSpeed = speed;
    controls.autoRotateSpeed = speed * TURNTABLE_BASE_SPEED;
    document.getElementById('turntable-speed-label').textContent = speed + 'x';
  });

  document.getElementById('export-btn').addEventListener('click', () => {
    exportTurntable({state, renderer, camera, controls, scene, container});
  });
}
