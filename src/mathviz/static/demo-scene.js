/**
 * Main entry point for the MathViz static demo page.
 * Sets up the Three.js scene, loads manifest.json, and wires UI controls.
 */
import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import {createDisplayManager} from './demo-display.js';
import {wireControls} from './demo-controls.js';
import {buildGallery, getQueryParamName, resolveItemPaths} from './demo-gallery.js';

/* ── Constants ── */
const DARK_COLOR = 0x1a1a2e;
const LIGHT_COLOR = 0xf0f0f0;
const HEMI_SKY_COLOR = 0x8899cc;
const HEMI_GROUND_COLOR = 0x443322;
const KEY_LIGHT_POSITION = [3, 5, 4];
const FILL_LIGHT_POSITION = [-3, 2, -2];
const RIM_LIGHT_POSITION = [0, 3, -5];
const LIGHT_INTENSITIES = {
  dark: {key: 1.2, fill: 0.4, rim: 0.3, hemi: 0.3, ambient: 0.15},
  light: {key: 0.8, fill: 0.3, rim: 0.2, hemi: 0.2, ambient: 0.1},
};

/* ── State ── */
const state = {
  meshGroup: null, cloudPoints: null, bboxHelper: null,
  axesHelper: null, axisLabels: [],
  viewMode: 'vertex', pointSize: 0.05, sceneExtent: 1, darkBg: true,
  cameraLocked: 'render', stretch: {x: 1, y: 1, z: 1},
  crystalGlassBlock: null, crystalComposer: null, crystalLedLight: null,
  crystalActive: false, crystalEnvRT: null, crystalTemplateMat: null,
  density: 1.0, fullCloudPositions: null, fullCloudCount: 0,
  turntable: false, turntableSpeed: 1.0,
  exporting: false, exportCancelled: false,
  colormapActive: false, colormapMetric: 'height', colormapGradient: 'viridis',
};
/* Expose for export cancel button */
window._demoState = state;

/* ── Scene setup ── */
const container = document.getElementById('canvas-container');
const renderer = new THREE.WebGLRenderer({antialias: true, preserveDrawingBuffer: true});
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(container.clientWidth, container.clientHeight);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.0;
container.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(DARK_COLOR);

const camera = new THREE.PerspectiveCamera(
  50, container.clientWidth / container.clientHeight, 0.01, 100,
);
camera.position.set(2, 1.5, 3);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;

/* ── Lighting ── */
const mainLights = _createLightingRig();
scene.add(mainLights.hemi, mainLights.ambient, mainLights.key, mainLights.fill, mainLights.rim);

function _createLightingRig() {
  const hemi = new THREE.HemisphereLight(HEMI_SKY_COLOR, HEMI_GROUND_COLOR, LIGHT_INTENSITIES.dark.hemi);
  const ambient = new THREE.AmbientLight(0xffffff, LIGHT_INTENSITIES.dark.ambient);
  const key = new THREE.DirectionalLight(0xffffff, LIGHT_INTENSITIES.dark.key);
  key.position.set(...KEY_LIGHT_POSITION);
  key.castShadow = true;
  key.shadow.mapSize.set(2048, 2048);
  key.shadow.camera.near = 0.1; key.shadow.camera.far = 50;
  key.shadow.camera.left = -5; key.shadow.camera.right = 5;
  key.shadow.camera.top = 5; key.shadow.camera.bottom = -5;
  const fill = new THREE.DirectionalLight(0xffffff, LIGHT_INTENSITIES.dark.fill);
  fill.position.set(...FILL_LIGHT_POSITION);
  const rim = new THREE.DirectionalLight(0xffffff, LIGHT_INTENSITIES.dark.rim);
  rim.position.set(...RIM_LIGHT_POSITION);
  return {hemi, ambient, key, fill, rim};
}

function applyLightIntensities(isDark) {
  const vals = isDark ? LIGHT_INTENSITIES.dark : LIGHT_INTENSITIES.light;
  mainLights.key.intensity = vals.key;
  mainLights.fill.intensity = vals.fill;
  mainLights.rim.intensity = vals.rim;
  mainLights.hemi.intensity = vals.hemi;
  mainLights.ambient.intensity = vals.ambient;
}

/* ── WASD Camera Controls ── */
const _wasdKeys = {w: false, a: false, s: false, d: false, ' ': false, shift: false};

function _isFormElement(el) {
  const tag = el.tagName;
  return tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA' || el.isContentEditable;
}

document.addEventListener('keydown', (e) => {
  if (_isFormElement(e.target)) return;
  const k = e.key === 'Shift' ? 'shift' : e.key.toLowerCase();
  if (k in _wasdKeys) { e.preventDefault(); _wasdKeys[k] = true; }
});
document.addEventListener('keyup', (e) => {
  const k = e.key === 'Shift' ? 'shift' : e.key.toLowerCase();
  if (k in _wasdKeys) _wasdKeys[k] = false;
});
window.addEventListener('blur', () => { for (const k in _wasdKeys) _wasdKeys[k] = false; });

function _applyWASD() {
  if (state.cameraLocked === 'full') return;
  const dist = camera.position.distanceTo(controls.target);
  const speed = dist * 0.015;
  const any = _wasdKeys.w || _wasdKeys.s || _wasdKeys.a || _wasdKeys.d
    || _wasdKeys[' '] || _wasdKeys.shift;
  if (!any) return;
  const prev = camera.position.clone();
  if (_wasdKeys.w) camera.translateZ(-speed);
  if (_wasdKeys.s) camera.translateZ(speed);
  if (_wasdKeys.a) camera.translateX(-speed);
  if (_wasdKeys.d) camera.translateX(speed);
  if (_wasdKeys[' ']) camera.translateY(speed);
  if (_wasdKeys.shift) camera.translateY(-speed);
  const delta = camera.position.clone().sub(prev);
  controls.target.add(delta);
}

/* ── Resize ── */
window.addEventListener('resize', () => {
  camera.aspect = container.clientWidth / container.clientHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(container.clientWidth, container.clientHeight);
});

/* ── FPS counter ── */
let frameCount = 0, lastFpsTime = performance.now();
function updateFPS() {
  frameCount++;
  const now = performance.now();
  if (now - lastFpsTime >= 1000) {
    document.getElementById('info-fps').textContent =
      'FPS: ' + Math.round(frameCount * 1000 / (now - lastFpsTime));
    frameCount = 0;
    lastFpsTime = now;
  }
}

/* ── Create display manager and wire controls ── */
const sceneCtx = {state, scene, renderer, camera, controls, container, DARK_COLOR, LIGHT_COLOR};
const display = createDisplayManager(sceneCtx);

wireControls({
  state, scene, renderer, camera, controls, container,
  DARK_COLOR, LIGHT_COLOR,
  actions: {
    ...display,
    applyLightIntensities,
  },
});

/* ── Render loop ── */
function animate() {
  requestAnimationFrame(animate);
  _applyWASD();
  controls.update();
  if (state.crystalActive && state.crystalComposer) {
    state.crystalComposer.render();
  } else {
    renderer.render(scene, camera);
  }
  updateFPS();
}
animate();

/* ── Load manifest and init ── */
async function init() {
  const selector = document.getElementById('viz-selector');
  const galleryPanel = document.getElementById('gallery-panel');
  const galleryToggle = document.getElementById('gallery-toggle');

  /* Wire gallery toggle/close buttons */
  galleryToggle.addEventListener('click', () => {
    galleryPanel.classList.remove('collapsed');
    galleryToggle.classList.add('hidden');
  });
  document.getElementById('gallery-close').addEventListener('click', () => {
    galleryPanel.classList.add('collapsed');
    galleryToggle.classList.remove('hidden');
  });

  try {
    const resp = await fetch('./manifest.json');
    if (!resp.ok) throw new Error('manifest.json not found');
    const manifest = await resp.json();
    const items = (manifest.visualizations || manifest)
      .filter((item) => {
        if (!item.name) {
          console.warn('Skipping manifest entry missing "name" field:', item);
          return false;
        }
        return true;
      });

    if (items.length === 0) {
      _addDisabledOption(selector, 'No visualizations available');
      return;
    }

    /* Populate dropdown selector (kept for keyboard/programmatic use) */
    for (const item of items) {
      const paths = resolveItemPaths(item);
      const opt = document.createElement('option');
      opt.value = item.name;
      opt.textContent = item.display_name || item.name;
      opt.dataset.mesh = paths.mesh;
      opt.dataset.cloud = paths.cloud;
      selector.appendChild(opt);
    }

    /* Build gallery UI */
    const gallery = buildGallery(galleryPanel, items, (item) => {
      const paths = resolveItemPaths(item);
      selector.value = item.name;
      display.loadVisualization(item.display_name || item.name, paths.mesh, paths.cloud);
    });

    /* Deep-link via ?name= query param, or select first item */
    const queryName = getQueryParamName();
    if (!queryName || !gallery.selectByName(queryName)) {
      gallery.selectByName(items[0].name);
    }
  } catch (err) {
    console.warn('Could not load manifest.json:', err.message);
    _addDisabledOption(selector, 'No manifest.json found');
  }
}

function _addDisabledOption(selector, text) {
  const opt = document.createElement('option');
  opt.textContent = text;
  opt.disabled = true;
  selector.appendChild(opt);
}

init();
