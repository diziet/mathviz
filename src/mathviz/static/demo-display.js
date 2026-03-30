/**
 * Scene display helpers: bounding box, axes, view modes, density, geometry loading.
 */
import * as THREE from 'three';
import {loadMeshFromGLB, loadCloudFromPLY, computeAdaptivePointSize} from './demo-render.js';
import {
  enterCrystalMode, exitCrystalMode,
  enterColorMapMode, exitColorMapMode,
} from './demo-crystal.js';

/** Create display manager bound to a specific scene context. */
export function createDisplayManager(ctx) {
  const {state, scene, renderer, camera, controls, container, DARK_COLOR, LIGHT_COLOR} = ctx;

  function clearScene() {
    if (state.meshGroup) { scene.remove(state.meshGroup); state.meshGroup = null; }
    if (state.cloudPoints) { scene.remove(state.cloudPoints); state.cloudPoints = null; }
    if (state.bboxHelper) { scene.remove(state.bboxHelper); state.bboxHelper = null; }
    removeAxes();
    state.fullCloudPositions = null;
    state.fullCloudCount = 0;
  }

  function fitCamera(object3d) {
    const box = new THREE.Box3().setFromObject(object3d);
    const size = new THREE.Vector3();
    box.getSize(size);
    const center = new THREE.Vector3();
    box.getCenter(center);
    const maxDim = Math.max(size.x, size.y, size.z);
    const dist = maxDim * 1.8;
    controls.target.copy(center);
    camera.position.set(center.x + dist * 0.6, center.y + dist * 0.4, center.z + dist * 0.8);
    camera.lookAt(center);
    controls.update();
  }

  function setupCameraForObject(object3d) {
    addBoundingBox(object3d);
    fitCamera(object3d);
    const box = new THREE.Box3().setFromObject(object3d);
    camera.near = 0.001;
    const size = new THREE.Vector3();
    box.getSize(size);
    camera.far = Math.max(size.x, size.y, size.z) * 20 + 100;
    camera.updateProjectionMatrix();
  }

  /* ── Bounding box ── */
  function addBoundingBox(object3d) {
    if (state.bboxHelper) scene.remove(state.bboxHelper);
    const box = new THREE.Box3().setFromObject(object3d);
    state.bboxHelper = new THREE.Box3Helper(box, 0x888888);
    state.bboxHelper.material.transparent = true;
    state.bboxHelper.material.opacity = 0.4;
    state.bboxHelper.visible = document.getElementById('show-bbox').checked;
    scene.add(state.bboxHelper);
  }

  /* ── Axes ── */
  function createAxesWithLabels() {
    const helper = new THREE.AxesHelper(1.5);
    scene.add(helper);
    state.axesHelper = helper;
    const labels = [
      {text: 'X', color: '#ff4444', pos: [1.7, 0, 0]},
      {text: 'Y', color: '#44ff44', pos: [0, 1.7, 0]},
      {text: 'Z', color: '#4444ff', pos: [0, 0, 1.7]},
    ];
    for (const l of labels) {
      const sprite = _createAxisLabelSprite(l.text, l.color);
      sprite.position.set(...l.pos);
      scene.add(sprite);
      state.axisLabels.push(sprite);
    }
  }

  function removeAxes() {
    if (state.axesHelper) { scene.remove(state.axesHelper); state.axesHelper = null; }
    for (const s of state.axisLabels) scene.remove(s);
    state.axisLabels = [];
  }

  function refreshAxesIfVisible() {
    if (document.getElementById('show-axes').checked) {
      removeAxes();
      createAxesWithLabels();
    }
  }

  /* ── View mode ── */
  function isPointLikeMode(mode) {
    return mode === 'vertex' || mode === 'dense' || mode === 'surface' || mode === 'edge_cloud';
  }

  function viewModeNeedsMesh() {
    return state.viewMode === 'shaded' || state.viewMode === 'wireframe';
  }

  function applyViewMode() {
    if (state.viewMode === 'crystal' && !state.crystalActive) {
      enterCrystalMode(state, scene, renderer, camera, container);
    } else if (state.viewMode !== 'crystal' && state.crystalActive) {
      exitCrystalMode(state, scene, DARK_COLOR, LIGHT_COLOR);
    }
    if (state.viewMode === 'colormap' && !state.colormapActive) {
      enterColorMapMode(state, scene);
    } else if (state.viewMode !== 'colormap' && state.colormapActive) {
      exitColorMapMode(state);
    }
    if (state.viewMode !== 'colormap') {
      _applyViewModeTo(state.meshGroup, state.cloudPoints);
    }
  }

  function _applyViewModeTo(meshGroup, cloudPoints) {
    const mode = state.viewMode;
    if (mode === 'crystal' || mode === 'colormap') return;
    const pointLike = isPointLikeMode(mode);
    if (meshGroup) {
      meshGroup.traverse((child) => {
        if (child.name === 'shaded') child.visible = (mode === 'shaded');
        if (child.name === 'wireframe') child.visible = (mode === 'wireframe');
        if (child.name === 'points') child.visible = pointLike;
      });
    }
    if (cloudPoints) cloudPoints.visible = pointLike || !meshGroup;
  }

  /* ── Density filtering ── */
  function applyDensityFilter() {
    if (!state.cloudPoints || !state.fullCloudPositions) return;
    const full = state.fullCloudPositions;
    const totalPoints = state.fullCloudCount;
    const density = state.density;
    const posAttr = state.cloudPoints.geometry.getAttribute('position');
    const displayCount = density >= 1.0
      ? totalPoints : Math.max(1, Math.round(totalPoints * density));
    if (density >= 1.0) {
      posAttr.array.set(full);
    } else {
      const step = totalPoints / displayCount;
      for (let i = 0; i < displayCount; i++) {
        const srcIdx = Math.floor(i * step) * 3;
        posAttr.array[i * 3] = full[srcIdx];
        posAttr.array[i * 3 + 1] = full[srcIdx + 1];
        posAttr.array[i * 3 + 2] = full[srcIdx + 2];
      }
    }
    posAttr.needsUpdate = true;
    state.cloudPoints.geometry.setDrawRange(0, displayCount);
    _updateDensityInfo(displayCount, totalPoints);
  }

  function _updateDensityInfo(displayed, total) {
    const pct = Math.round((displayed / total) * 100);
    const label = displayed === total
      ? total.toLocaleString()
      : displayed.toLocaleString() + ' / ' + total.toLocaleString() + ' (' + pct + '%)';
    document.getElementById('info-points').textContent = 'Points: ' + label;
  }

  function updateDensitySliderVisibility() {
    const show = isPointLikeMode(state.viewMode);
    document.getElementById('density-control').style.display = show ? '' : 'none';
    document.getElementById('density-slider').style.display = show ? '' : 'none';
    document.getElementById('density-value').style.display = show ? '' : 'none';
  }

  /* ── Point size ── */
  function updatePointSize(size) {
    state.pointSize = size;
    const matSize = computeAdaptivePointSize(size, state.sceneExtent);
    scene.traverse((child) => {
      if (child.isPoints && child.material) child.material.size = matSize;
    });
  }

  /* ── Stretch ── */
  function applyStretch() {
    const s = state.stretch;
    if (state.meshGroup) state.meshGroup.scale.set(s.x, s.y, s.z);
    if (state.cloudPoints) state.cloudPoints.scale.set(s.x, s.y, s.z);
  }

  function syncStretchUI(values) {
    ['x', 'y', 'z'].forEach((a) => {
      document.getElementById('stretch-' + a).value = values[a];
      document.getElementById('stretch-' + a + '-num').value = Number(values[a]).toFixed(1);
    });
  }

  /* ── Mesh density extraction ── */
  function _setupMeshDensity(meshGroup) {
    const posArrays = [];
    const pointsChildren = [];
    meshGroup.traverse((child) => {
      if (child.isPoints && child.name === 'points') {
        pointsChildren.push(child);
        posArrays.push(new Float32Array(child.geometry.getAttribute('position').array));
      }
    });
    if (posArrays.length === 0) return;
    const totalFloats = posArrays.reduce((sum, a) => sum + a.length, 0);
    const fullPositions = new Float32Array(totalFloats);
    let offset = 0;
    for (const arr of posArrays) { fullPositions.set(arr, offset); offset += arr.length; }
    const geom = new THREE.BufferGeometry();
    const dynBuf = new THREE.BufferAttribute(new Float32Array(totalFloats), 3);
    dynBuf.usage = THREE.DynamicDrawUsage;
    geom.setAttribute('position', dynBuf);
    const mat = new THREE.PointsMaterial({
      color: 0x44ccff, size: computeAdaptivePointSize(state.pointSize, state.sceneExtent),
    });
    const cloudPoints = new THREE.Points(geom, mat);
    for (const child of pointsChildren) child.parent.remove(child);
    scene.add(cloudPoints);
    state.cloudPoints = cloudPoints;
    state.fullCloudPositions = fullPositions;
    state.fullCloudCount = totalFloats / 3;
    applyDensityFilter();
    updateDensitySliderVisibility();
  }

  /* ── Display helpers ── */
  async function displayMesh(url, label) {
    const {group, extent, totalVertices, totalFaces} = await loadMeshFromGLB(url, state);
    state.meshGroup = group;
    scene.add(group);
    setupCameraForObject(group);
    document.getElementById('reset-view-btn').disabled = false;
    updateInfo({generator: label, vertices: totalVertices, faces: totalFaces});
    return extent;
  }

  function displayCloud(points, vertexCount, label) {
    state.cloudPoints = points;
    const posAttr = points.geometry.getAttribute('position');
    state.fullCloudPositions = new Float32Array(posAttr.array);
    state.fullCloudCount = vertexCount;
    const dynBuf = new THREE.BufferAttribute(new Float32Array(state.fullCloudPositions.length), 3);
    dynBuf.usage = THREE.DynamicDrawUsage;
    points.geometry.setAttribute('position', dynBuf);
    scene.add(points);
    if (!state.meshGroup) {
      setupCameraForObject(points);
      document.getElementById('view-mode').value = 'vertex';
      state.viewMode = 'vertex';
    }
    document.getElementById('reset-view-btn').disabled = false;
    updateInfo({generator: label});
    applyDensityFilter();
    updateDensitySliderVisibility();
  }

  /* ── Info panel ── */
  const infoState = {generator: '—', vertices: '—', faces: '—', points: '—'};
  function updateInfo(updates) {
    Object.assign(infoState, updates);
    document.getElementById('info-generator').textContent = 'Generator: ' + (infoState.generator || '—');
    document.getElementById('info-vertices').textContent = 'Vertices: ' + (infoState.vertices || '—');
    document.getElementById('info-faces').textContent = 'Faces: ' + (infoState.faces || '—');
    document.getElementById('info-points').textContent = 'Points: ' + (infoState.points || '—');
  }

  /* ── Load a visualization by name ── */
  async function loadVisualization(name, basePath) {
    const loadingEl = document.getElementById('loading');
    loadingEl.style.display = 'block';
    document.getElementById('loading-text').textContent = 'Loading ' + name + '...';

    try {
      const savedCamera = _saveCameraIfLocked();
      clearScene();

      let hasMesh = false;
      let meshExtent = 0;
      try {
        meshExtent = await displayMesh(basePath + '/mesh.glb', name);
        hasMesh = true;
      } catch (err) {
        if (!_isNotFoundError(err)) console.warn('Failed to load mesh:', err);
      }

      let hasCloud = false;
      let cloudExtent = 0;
      try {
        const {points, extent, vertexCount} = await loadCloudFromPLY(basePath + '/cloud.ply', state);
        displayCloud(points, vertexCount, name);
        cloudExtent = extent;
        hasCloud = true;
      } catch (err) {
        if (!_isNotFoundError(err)) console.warn('Failed to load cloud:', err);
      }

      state.sceneExtent = Math.max(meshExtent, cloudExtent, 1);
      if (hasMesh || hasCloud) _reapplyPointSizing();
      if (hasMesh && !hasCloud) _setupMeshDensity(state.meshGroup);
      if (!hasMesh && !hasCloud) updateInfo({generator: name + ' (no geometry found)'});
      if (viewModeNeedsMesh() && !hasMesh) {
        state.viewMode = 'vertex';
        document.getElementById('view-mode').value = 'vertex';
      }

      _restoreCameraIfSaved(savedCamera);
      applyViewMode();
      applyStretch();
      refreshAxesIfVisible();
    } finally {
      loadingEl.style.display = 'none';
    }
  }

  function _saveCameraIfLocked() {
    if (state.cameraLocked === 'off') return null;
    return {pos: camera.position.clone(), target: controls.target.clone()};
  }

  function _restoreCameraIfSaved(saved) {
    if (!saved) return;
    camera.position.copy(saved.pos);
    controls.target.copy(saved.target);
    controls.update();
  }

  function _isNotFoundError(err) {
    if (err && err.message && err.message.includes('404')) return true;
    if (err && err.status === 404) return true;
    if (err instanceof Response && err.status === 404) return true;
    return false;
  }

  function _reapplyPointSizing() {
    const matSize = computeAdaptivePointSize(state.pointSize, state.sceneExtent);
    scene.traverse((child) => {
      if (child.isPoints && child.material) child.material.size = matSize;
    });
  }

  return {
    fitCamera, applyViewMode, updateDensitySliderVisibility, applyDensityFilter,
    updatePointSize, applyStretch, syncStretchUI,
    removeAxes, createAxesWithLabels, loadVisualization,
  };
}

/* ── Private helpers ── */

function _createAxisLabelSprite(text, color) {
  const canvas = document.createElement('canvas');
  canvas.width = 64; canvas.height = 64;
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = color;
  ctx.font = 'bold 48px sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(text, 32, 32);
  const texture = new THREE.CanvasTexture(canvas);
  const mat = new THREE.SpriteMaterial({map: texture, depthTest: false});
  const sprite = new THREE.Sprite(mat);
  sprite.scale.set(0.3, 0.3, 0.3);
  return sprite;
}
