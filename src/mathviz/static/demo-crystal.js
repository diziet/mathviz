/**
 * Crystal preview mode and color map mode for the static demo page.
 */
import * as THREE from 'three';
import {EffectComposer} from 'three/addons/postprocessing/EffectComposer.js';
import {RenderPass} from 'three/addons/postprocessing/RenderPass.js';
import {UnrealBloomPass} from 'three/addons/postprocessing/UnrealBloomPass.js';
import {computeAdaptivePointSize} from './demo-render.js';

/* ── Baked container dimensions (100mm cube, 5mm margins) ── */
const CONTAINER = {width_mm: 100, height_mm: 100, depth_mm: 100};

/* ── Crystal mode ── */

/** Create the glass block mesh. */
function _createGlassBlock(renderer) {
  const geom = new THREE.BoxGeometry(CONTAINER.width_mm, CONTAINER.height_mm, CONTAINER.depth_mm);
  const pmremScene = new THREE.Scene();
  pmremScene.background = new THREE.Color(0x111122);
  pmremScene.add(new THREE.AmbientLight(0x444466, 1));
  const pmrem = new THREE.PMREMGenerator(renderer);
  pmrem.compileCubemapShader();
  const envRT = pmrem.fromScene(pmremScene, 0.04);
  pmrem.dispose();

  const mat = new THREE.MeshPhysicalMaterial({
    color: new THREE.Color(document.getElementById('crystal-glass-tint').value),
    roughness: 0.02, clearcoat: 1.0, clearcoatRoughness: 0.05,
    opacity: 0.15, transparent: true, envMap: envRT.texture, envMapIntensity: 0.3,
    side: THREE.BackSide, depthWrite: false,
  });
  return {mesh: new THREE.Mesh(geom, mat), envRT};
}

/** Create the crystal-style point material. */
function _createCrystalPointsMaterial(state) {
  const size = 64;
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d');
  const gradient = ctx.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
  gradient.addColorStop(0, 'rgba(255,255,255,1)');
  gradient.addColorStop(0.3, 'rgba(232,240,255,0.6)');
  gradient.addColorStop(1, 'rgba(232,240,255,0)');
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, size, size);
  const texture = new THREE.CanvasTexture(canvas);
  const brightness = parseFloat(document.getElementById('crystal-brightness').value);
  return new THREE.PointsMaterial({
    color: 0xe8f0ff, size: computeAdaptivePointSize(state.pointSize, state.sceneExtent) * 1.2,
    map: texture, transparent: true, opacity: brightness,
    blending: THREE.AdditiveBlending, depthWrite: false, sizeAttenuation: true,
  });
}

/** Set up bloom post-processing composer. */
function _setupCrystalComposer(renderer, scene, camera, container) {
  const composer = new EffectComposer(renderer);
  composer.addPass(new RenderPass(scene, camera));
  const bloomStrength = parseFloat(document.getElementById('crystal-bloom').value);
  const bloomPass = new UnrealBloomPass(
    new THREE.Vector2(container.clientWidth, container.clientHeight),
    bloomStrength, 0.3, 0.6,
  );
  bloomPass.name = 'crystal-bloom';
  composer.addPass(bloomPass);
  return composer;
}

/** Enter crystal preview mode. */
export function enterCrystalMode(state, scene, renderer, camera, container) {
  state.crystalActive = true;
  scene.background = new THREE.Color(0x000000);

  const {mesh, envRT} = _createGlassBlock(renderer);
  mesh.name = 'crystal-glass-block';
  mesh.renderOrder = 0;
  scene.add(mesh);
  state.crystalGlassBlock = mesh;
  state.crystalEnvRT = envRT;

  const crystalMat = _createCrystalPointsMaterial(state);
  state.crystalTemplateMat = crystalMat;
  scene.traverse((child) => {
    if (child.isPoints && child.material) {
      child.userData.originalMaterial = child.material;
      const cloned = crystalMat.clone();
      child.material = cloned;
      child.renderOrder = 1;
      child.material.depthTest = false;
      child.visible = true;
    }
  });

  if (state.meshGroup) {
    state.meshGroup.traverse((child) => {
      if (child.name === 'shaded' || child.name === 'wireframe') child.visible = false;
      if (child.name === 'points') child.visible = true;
    });
  }
  if (state.cloudPoints) state.cloudPoints.visible = true;

  state.crystalComposer = _setupCrystalComposer(renderer, scene, camera, container);
  document.getElementById('crystal-controls').style.display = 'block';

  if (document.getElementById('crystal-led-base').checked) {
    addCrystalLedLight(state, scene);
  }
}

/** Exit crystal preview mode. */
export function exitCrystalMode(state, scene, DARK_COLOR, LIGHT_COLOR) {
  state.crystalActive = false;
  scene.background = new THREE.Color(state.darkBg ? DARK_COLOR : LIGHT_COLOR);

  if (state.crystalEnvRT) { state.crystalEnvRT.dispose(); state.crystalEnvRT = null; }
  if (state.crystalGlassBlock) {
    scene.remove(state.crystalGlassBlock);
    state.crystalGlassBlock.geometry.dispose();
    state.crystalGlassBlock.material.dispose();
    state.crystalGlassBlock = null;
  }

  scene.traverse((child) => {
    if (child.isPoints && child.userData.originalMaterial) {
      child.material.dispose();
      child.material = child.userData.originalMaterial;
      child.renderOrder = 0;
      delete child.userData.originalMaterial;
    }
  });

  if (state.crystalTemplateMat) {
    if (state.crystalTemplateMat.map) state.crystalTemplateMat.map.dispose();
    state.crystalTemplateMat.dispose();
    state.crystalTemplateMat = null;
  }

  removeCrystalLedLight(state, scene);
  if (state.crystalComposer) { state.crystalComposer.dispose(); state.crystalComposer = null; }
  document.getElementById('crystal-controls').style.display = 'none';
}

/** Add LED base light. */
export function addCrystalLedLight(state, scene) {
  if (state.crystalLedLight) return;
  const color = new THREE.Color(document.getElementById('crystal-led-color').value);
  const light = new THREE.PointLight(color, 2, CONTAINER.height_mm * 3);
  light.position.set(0, -CONTAINER.height_mm / 2 - 5, 0);
  light.name = 'crystal-led-base';
  scene.add(light);
  state.crystalLedLight = light;
}

/** Remove LED base light. */
export function removeCrystalLedLight(state, scene) {
  if (!state.crystalLedLight) return;
  scene.remove(state.crystalLedLight);
  state.crystalLedLight = null;
}

/* ── Color map mode ── */

const COLORMAP_GRADIENTS = {
  viridis: [
    [0.267,0.004,0.329],[0.283,0.141,0.458],[0.254,0.265,0.530],
    [0.207,0.372,0.553],[0.164,0.471,0.558],[0.128,0.567,0.551],
    [0.135,0.659,0.518],[0.267,0.749,0.441],[0.478,0.821,0.318],
    [0.741,0.873,0.150],[0.993,0.906,0.144],
  ],
  inferno: [
    [0.001,0.000,0.014],[0.106,0.042,0.233],[0.258,0.039,0.406],
    [0.417,0.056,0.424],[0.578,0.105,0.350],[0.735,0.206,0.224],
    [0.863,0.344,0.114],[0.949,0.517,0.030],[0.982,0.706,0.124],
    [0.961,0.895,0.371],[0.988,1.000,0.644],
  ],
  coolwarm: [
    [0.230,0.299,0.754],[0.400,0.450,0.850],[0.580,0.600,0.920],
    [0.740,0.740,0.960],[0.870,0.870,0.970],[0.970,0.970,0.970],
    [0.960,0.830,0.830],[0.940,0.680,0.680],[0.890,0.490,0.490],
    [0.800,0.300,0.300],[0.706,0.016,0.150],
  ],
  rainbow: [
    [1,0,0],[1,0.5,0],[1,1,0],[0,1,0],[0,1,1],[0,0,1],[0.5,0,1],
  ],
};

/** Sample a color from a gradient at position t ∈ [0, 1]. */
function _sampleGradient(stops, t) {
  const clamped = Math.max(0, Math.min(1, t));
  const idx = clamped * (stops.length - 1);
  const lo = Math.floor(idx);
  const hi = Math.min(lo + 1, stops.length - 1);
  const frac = idx - lo;
  return [
    stops[lo][0] + (stops[hi][0] - stops[lo][0]) * frac,
    stops[lo][1] + (stops[hi][1] - stops[lo][1]) * frac,
    stops[lo][2] + (stops[hi][2] - stops[lo][2]) * frac,
  ];
}

/** Get gradient stops for current settings. */
function _getGradientStops(state) {
  if (state.colormapGradient === 'custom') {
    const startHex = document.getElementById('colormap-start-color').value;
    const endHex = document.getElementById('colormap-end-color').value;
    const s = new THREE.Color(startHex);
    const e = new THREE.Color(endHex);
    return [[s.r, s.g, s.b], [e.r, e.g, e.b]];
  }
  return COLORMAP_GRADIENTS[state.colormapGradient] || COLORMAP_GRADIENTS.viridis;
}

/** Compute metric values for vertices. */
function _computeMetricValues(posArray, count, state) {
  const values = new Float32Array(count);
  const metric = state.colormapMetric;
  if (metric === 'height') {
    for (let i = 0; i < count; i++) values[i] = posArray[i * 3 + 2];
  } else if (metric === 'distance') {
    for (let i = 0; i < count; i++) {
      const x = posArray[i * 3], y = posArray[i * 3 + 1], z = posArray[i * 3 + 2];
      values[i] = Math.sqrt(x * x + y * y + z * z);
    }
  } else if (metric === 'curvature') {
    _computeCurvatureMetric(posArray, count, values);
  } else if (metric === 'velocity') {
    _computeVelocityMetric(posArray, count, values);
  }
  return _normalizeValues(values);
}

function _computeCurvatureMetric(posArray, count, values) {
  for (let i = 0; i < count; i++) {
    const prev = Math.max(0, i - 1);
    const next = Math.min(count - 1, i + 1);
    const dx = posArray[next * 3] - 2 * posArray[i * 3] + posArray[prev * 3];
    const dy = posArray[next * 3 + 1] - 2 * posArray[i * 3 + 1] + posArray[prev * 3 + 1];
    const dz = posArray[next * 3 + 2] - 2 * posArray[i * 3 + 2] + posArray[prev * 3 + 2];
    values[i] = Math.sqrt(dx * dx + dy * dy + dz * dz);
  }
}

function _computeVelocityMetric(posArray, count, values) {
  for (let i = 0; i < count; i++) {
    const next = Math.min(count - 1, i + 1);
    const dx = posArray[next * 3] - posArray[i * 3];
    const dy = posArray[next * 3 + 1] - posArray[i * 3 + 1];
    const dz = posArray[next * 3 + 2] - posArray[i * 3 + 2];
    values[i] = Math.sqrt(dx * dx + dy * dy + dz * dz);
  }
}

function _normalizeValues(values) {
  let min = Infinity, max = -Infinity;
  for (let i = 0; i < values.length; i++) {
    if (values[i] < min) min = values[i];
    if (values[i] > max) max = values[i];
  }
  const range = max - min || 1;
  for (let i = 0; i < values.length; i++) values[i] = (values[i] - min) / range;
  return values;
}

/** Build vertex colors array from position data. */
export function buildVertexColors(posArray, count, state) {
  const values = _computeMetricValues(posArray, count, state);
  const stops = _getGradientStops(state);
  const colors = new Float32Array(count * 3);
  for (let i = 0; i < count; i++) {
    const [r, g, b] = _sampleGradient(stops, values[i]);
    colors[i * 3] = r;
    colors[i * 3 + 1] = g;
    colors[i * 3 + 2] = b;
  }
  return colors;
}

/** Apply color map to a Three.js object. */
export function applyColorMapToObject(obj, state) {
  const geom = obj.geometry;
  const pos = geom.getAttribute('position');
  const colors = buildVertexColors(pos.array, pos.count, state);
  geom.setAttribute('color', new THREE.BufferAttribute(colors, 3));

  if (obj.isPoints) {
    const newMat = new THREE.PointsMaterial({
      size: obj.material.size, vertexColors: true, sizeAttenuation: true,
    });
    obj.material.dispose();
    obj.material = newMat;
  } else if (obj.isMesh) {
    const newMat = new THREE.MeshPhysicalMaterial({
      vertexColors: true, metalness: 0.1, roughness: 0.6, side: THREE.DoubleSide,
    });
    obj.material.dispose();
    obj.material = newMat;
  }
}

/** Enter color map mode. */
export function enterColorMapMode(state, scene) {
  state.colormapActive = true;
  document.getElementById('colormap-controls').style.display = 'block';
  _applyColorMapToScene(state, scene);
}

/** Exit color map mode. */
export function exitColorMapMode(state) {
  state.colormapActive = false;
  document.getElementById('colormap-controls').style.display = 'none';
}

/** Update color map on all visible objects. */
export function updateColorMap(state, scene) {
  if (!state.colormapActive) return;
  _applyColorMapToScene(state, scene);
}

function _applyColorMapToScene(state, scene) {
  if (state.meshGroup) {
    state.meshGroup.traverse((child) => {
      if (child.isMesh && child.visible) applyColorMapToObject(child, state);
      if (child.isPoints && child.visible) applyColorMapToObject(child, state);
    });
  }
  if (state.cloudPoints && state.cloudPoints.visible) {
    applyColorMapToObject(state.cloudPoints, state);
  }
}
