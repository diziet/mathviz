/**
 * Geometry loading and display helpers for the static demo page.
 */
import * as THREE from 'three';
import {GLTFLoader} from 'three/addons/loaders/GLTFLoader.js';

/* ── PLY type sizes ── */
const PLY_TYPE_SIZES = {
  'char': 1, 'uchar': 1, 'int8': 1, 'uint8': 1,
  'short': 2, 'ushort': 2, 'int16': 2, 'uint16': 2,
  'int': 4, 'uint': 4, 'int32': 4, 'uint32': 4,
  'float': 4, 'float32': 4, 'double': 8, 'float64': 8,
};

const PLY_TYPE_READERS = {
  'char': (dv, o) => dv.getInt8(o),
  'uchar': (dv, o) => dv.getUint8(o),
  'int8': (dv, o) => dv.getInt8(o),
  'uint8': (dv, o) => dv.getUint8(o),
  'short': (dv, o) => dv.getInt16(o, true),
  'ushort': (dv, o) => dv.getUint16(o, true),
  'int16': (dv, o) => dv.getInt16(o, true),
  'uint16': (dv, o) => dv.getUint16(o, true),
  'int': (dv, o) => dv.getInt32(o, true),
  'uint': (dv, o) => dv.getUint32(o, true),
  'int32': (dv, o) => dv.getInt32(o, true),
  'uint32': (dv, o) => dv.getUint32(o, true),
  'float': (dv, o) => dv.getFloat32(o, true),
  'float32': (dv, o) => dv.getFloat32(o, true),
  'double': (dv, o) => dv.getFloat64(o, true),
  'float64': (dv, o) => dv.getFloat64(o, true),
};

/** Parse binary little-endian PLY buffer. */
export function parseBinaryPLY(buffer) {
  const headerEnd = _findHeaderEnd(buffer);
  if (headerEnd < 0) return null;

  const headerText = new TextDecoder().decode(new Uint8Array(buffer, 0, headerEnd));
  const lines = headerText.split('\n').map(l => l.trim());

  let vertexCount = 0;
  const properties = [];

  let inVertexElement = false;
  for (const line of lines) {
    if (line.startsWith('element vertex')) {
      vertexCount = parseInt(line.split(/\s+/)[2], 10);
      inVertexElement = true;
    } else if (line.startsWith('element ')) {
      inVertexElement = false;
    } else if (inVertexElement && line.startsWith('property') && !line.includes('list')) {
      const parts = line.split(/\s+/);
      properties.push({type: parts[1], name: parts[2]});
    }
  }

  if (vertexCount === 0) return null;

  const dataOffset = headerEnd;
  const dv = new DataView(buffer, dataOffset);
  let stride = 0;
  const propOffsets = {};

  for (const prop of properties) {
    propOffsets[prop.name] = {offset: stride, type: prop.type};
    stride += PLY_TYPE_SIZES[prop.type] || 4;
  }

  const positions = new Float32Array(vertexCount * 3);
  const xProp = propOffsets['x'], yProp = propOffsets['y'], zProp = propOffsets['z'];
  if (!xProp || !yProp || !zProp) return null;

  const xReader = PLY_TYPE_READERS[xProp.type];
  const yReader = PLY_TYPE_READERS[yProp.type];
  const zReader = PLY_TYPE_READERS[zProp.type];

  for (let i = 0; i < vertexCount; i++) {
    const base = i * stride;
    positions[i * 3] = xReader(dv, base + xProp.offset);
    positions[i * 3 + 1] = yReader(dv, base + yProp.offset);
    positions[i * 3 + 2] = zReader(dv, base + zProp.offset);
  }

  return {positions, vertexCount};
}

/** Compute adaptive point size based on scene extent. */
export function computeAdaptivePointSize(sliderValue, extent) {
  const e = extent !== undefined ? extent : 1;
  return e * sliderValue * 0.03;
}

/** Compute max extent from a geometry's bounding box. */
export function getGeometryExtent(geometry) {
  geometry.computeBoundingBox();
  const box = geometry.boundingBox;
  if (!box) return 1;
  const size = new THREE.Vector3();
  box.getSize(size);
  return Math.max(size.x, size.y, size.z, 1);
}

/** Load a GLB file and return {group, totalVertices, totalFaces}. */
export function loadMeshFromGLB(url, state) {
  return new Promise((resolve, reject) => {
    new GLTFLoader().load(url, (gltf) => {
      const group = new THREE.Group();
      let totalVertices = 0, totalFaces = 0;

      const unionBox = new THREE.Box3();
      gltf.scene.traverse((child) => {
        if (!child.isMesh) return;
        child.geometry.computeBoundingBox();
        if (child.geometry.boundingBox) unionBox.union(child.geometry.boundingBox);
      });
      const unionSize = new THREE.Vector3();
      unionBox.getSize(unionSize);
      const meshExtent = Math.max(unionSize.x, unionSize.y, unionSize.z, 1);

      gltf.scene.traverse((child) => {
        if (!child.isMesh) return;
        const geom = child.geometry;
        geom.computeVertexNormals();
        totalVertices += geom.attributes.position.count;
        totalFaces += geom.index ? geom.index.count / 3 : geom.attributes.position.count / 3;

        const shadedMat = new THREE.MeshPhysicalMaterial({
          color: 0x4488cc, metalness: 0.1, roughness: 0.6, side: THREE.DoubleSide,
        });
        const wireMat = new THREE.MeshBasicMaterial({color: 0x44aaff, wireframe: true});
        const shadedMesh = new THREE.Mesh(geom, shadedMat);
        shadedMesh.name = 'shaded';
        shadedMesh.visible = (state.viewMode === 'shaded');
        shadedMesh.castShadow = true;
        shadedMesh.receiveShadow = true;
        const wireMesh = new THREE.Mesh(geom, wireMat);
        wireMesh.name = 'wireframe';
        wireMesh.visible = (state.viewMode === 'wireframe');

        const ptSize = computeAdaptivePointSize(state.pointSize, meshExtent);
        const pointsMat = new THREE.PointsMaterial({color: 0x44ccff, size: ptSize});
        const pts = new THREE.Points(geom, pointsMat);
        pts.name = 'points';
        pts.visible = (state.viewMode === 'vertex');

        group.add(shadedMesh, wireMesh, pts);
      });

      resolve({group, extent: meshExtent, totalVertices: Math.round(totalVertices), totalFaces: Math.round(totalFaces)});
    }, undefined, reject);
  });
}

/** Load a PLY file and return {points, vertexCount}. */
export function loadCloudFromPLY(url, state) {
  return fetch(url).then(r => {
    if (!r.ok) throw Object.assign(new Error('HTTP ' + r.status), {status: r.status});
    return r.arrayBuffer();
  }).then(buf => {
    const parsed = parseBinaryPLY(buf);
    if (!parsed) throw new Error('Failed to parse PLY');
    const geom = new THREE.BufferGeometry();
    geom.setAttribute('position', new THREE.BufferAttribute(parsed.positions, 3));
    const cloudExtent = getGeometryExtent(geom);
    const ptSize = computeAdaptivePointSize(state.pointSize, cloudExtent);
    const mat = new THREE.PointsMaterial({color: 0x44ccff, size: ptSize});
    const points = new THREE.Points(geom, mat);
    return {points, extent: cloudExtent, vertexCount: parsed.vertexCount};
  });
}

/* ── Private helpers ── */

/** Find the end of the PLY header in a buffer. */
function _findHeaderEnd(buffer) {
  const bytes = new Uint8Array(buffer);
  const marker = [101, 110, 100, 95, 104, 101, 97, 100, 101, 114, 10]; // "end_header\n"
  for (let i = 0; i < Math.min(bytes.length, 4096); i++) {
    let found = true;
    for (let j = 0; j < marker.length; j++) {
      if (bytes[i + j] !== marker[j]) { found = false; break; }
    }
    if (found) return i + marker.length;
  }
  return -1;
}
