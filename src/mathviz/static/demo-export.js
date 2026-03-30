/**
 * Turntable GIF/WebM export for the static demo page.
 */

/** Render a single frame at a given orbital angle. */
function _renderFrameAtAngle(ctx, angle, pivotTarget, radius, startAngle, camY) {
  const {camera, controls, renderer, scene, state} = ctx;
  const theta = startAngle + angle;
  camera.position.set(
    pivotTarget.x + radius * Math.sin(theta),
    camY,
    pivotTarget.z + radius * Math.cos(theta),
  );
  camera.lookAt(pivotTarget);

  if (state.crystalActive && state.crystalComposer) {
    state.crystalComposer.render();
  } else {
    renderer.render(scene, camera);
  }
}

/** Download a Blob as a file. */
function _downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.download = filename;
  link.href = url;
  link.click();
  URL.revokeObjectURL(url);
}

/** Update the capture progress UI. */
function _updateProgress(i, totalFrames, progressText, progressFill, scale) {
  progressText.textContent = 'Capturing frame ' + (i + 1) + '/' + totalFrames + '...';
  progressFill.style.width = Math.round(((i + 1) / totalFrames) * scale) + '%';
}

/** Export turntable as WebM video. */
async function _exportWebM(ctx, cfg) {
  const {totalFrames, fps, anglePerFrame, pivotTarget, radius, startAngle, camY, progressText, progressFill} = cfg;
  const stream = ctx.renderer.domElement.captureStream(0);
  const recorder = new MediaRecorder(stream, {mimeType: 'video/webm;codecs=vp9'});
  const chunks = [];

  recorder.ondataavailable = (e) => { if (e.data.size > 0) chunks.push(e.data); };
  const donePromise = new Promise((resolve) => { recorder.onstop = () => resolve(); });
  const track = stream.getVideoTracks()[0];
  const canRequestFrame = typeof track.requestFrame === 'function';

  recorder.start();

  let cancelled = false;
  for (let i = 0; i < totalFrames; i++) {
    if (ctx.state.exportCancelled) { cancelled = true; break; }
    _renderFrameAtAngle(ctx, i * anglePerFrame, pivotTarget, radius, startAngle, camY);
    if (canRequestFrame) track.requestFrame();
    _updateProgress(i, totalFrames, progressText, progressFill, 100);
    await new Promise((r) => setTimeout(r, 1000 / fps));
  }

  recorder.stop();
  await donePromise;
  if (!cancelled) {
    _downloadBlob(new Blob(chunks, {type: 'video/webm'}), cfg.filename + '_turntable.webm');
  }
}

/** Export turntable as animated GIF. */
async function _exportGIF(ctx, cfg) {
  const {totalFrames, fps, anglePerFrame, pivotTarget, radius, startAngle, camY,
    progressText, progressFill, exportWidth, exportHeight} = cfg;
  const canvas = document.createElement('canvas');
  canvas.width = exportWidth;
  canvas.height = exportHeight;
  const drawCtx = canvas.getContext('2d');

  const delay = Math.round(1000 / fps);
  const gifEncoder = _createGIFEncoder(exportWidth, exportHeight, delay);

  let cancelled = false;
  for (let i = 0; i < totalFrames; i++) {
    if (ctx.state.exportCancelled) { cancelled = true; break; }
    _renderFrameAtAngle(ctx, i * anglePerFrame, pivotTarget, radius, startAngle, camY);
    drawCtx.drawImage(ctx.renderer.domElement, 0, 0, exportWidth, exportHeight);
    const imageData = drawCtx.getImageData(0, 0, exportWidth, exportHeight);
    gifEncoder.addFrame(imageData.data);
    _updateProgress(i, totalFrames, progressText, progressFill, 100);
    await new Promise((r) => setTimeout(r, 0));
  }

  if (!cancelled) {
    progressText.textContent = 'Finalizing GIF...';
    const gif = gifEncoder.finish();
    _downloadBlob(new Blob([gif], {type: 'image/gif'}), cfg.filename + '_turntable.gif');
  }
}

/** Create a minimal GIF89a encoder. */
function _createGIFEncoder(width, height, delay) {
  const chunks = [];
  let chunkBuf = new Uint8Array(65536);
  let chunkPos = 0;

  function flush() {
    if (chunkPos > 0) { chunks.push(chunkBuf.slice(0, chunkPos)); chunkPos = 0; }
  }
  function ensureCapacity(needed) {
    if (chunkPos + needed > chunkBuf.length) {
      flush();
      if (needed > chunkBuf.length) chunkBuf = new Uint8Array(needed);
    }
  }
  function writeByte(b) { ensureCapacity(1); chunkBuf[chunkPos++] = b & 0xff; }
  function writeShort(s) { writeByte(s & 0xff); writeByte((s >> 8) & 0xff); }
  function writeString(s) { for (let i = 0; i < s.length; i++) writeByte(s.charCodeAt(i)); }

  writeString('GIF89a');
  writeShort(width);
  writeShort(height);
  writeByte(0xf7); writeByte(0); writeByte(0);

  for (let r = 0; r < 8; r++)
    for (let g = 0; g < 8; g++)
      for (let b = 0; b < 4; b++) {
        writeByte(Math.round(r * 255 / 7));
        writeByte(Math.round(g * 255 / 7));
        writeByte(Math.round(b * 255 / 3));
      }

  writeByte(0x21); writeByte(0xff); writeByte(11);
  writeString('NETSCAPE2.0');
  writeByte(3); writeByte(1); writeShort(0); writeByte(0);

  function addFrame(data) {
    writeByte(0x21); writeByte(0xf9); writeByte(4);
    writeByte(0x00); writeShort(Math.round(delay / 10)); writeByte(0); writeByte(0);
    writeByte(0x2c); writeShort(0); writeShort(0); writeShort(width); writeShort(height);
    writeByte(0);

    const minCodeSize = 8;
    writeByte(minCodeSize);
    const pixelCount = width * height;
    const pixels = new Uint8Array(pixelCount);
    for (let i = 0; i < pixelCount; i++) {
      const off = i * 4;
      pixels[i] = ((data[off] >> 5) & 0x07) << 5
        | ((data[off + 1] >> 5) & 0x07) << 2
        | ((data[off + 2] >> 6) & 0x03);
    }
    const encoded = _lzwEncode(pixels, minCodeSize);
    let pos = 0;
    while (pos < encoded.length) {
      const chunk = Math.min(255, encoded.length - pos);
      writeByte(chunk);
      ensureCapacity(chunk);
      chunkBuf.set(encoded.subarray(pos, pos + chunk), chunkPos);
      chunkPos += chunk;
      pos += chunk;
    }
    writeByte(0);
  }

  function finish() {
    writeByte(0x3b); flush();
    let totalLen = 0;
    for (const c of chunks) totalLen += c.length;
    const result = new Uint8Array(totalLen);
    let offset = 0;
    for (const c of chunks) { result.set(c, offset); offset += c.length; }
    return result;
  }

  return {addFrame, finish};
}

/** LZW encoder for GIF. */
function _lzwEncode(pixels, minCodeSize) {
  const clearCode = 1 << minCodeSize;
  const eoiCode = clearCode + 1;
  let codeSize = minCodeSize + 1;
  let nextCode = eoiCode + 1;
  const maxCode = 4096;

  let trie = new Array(clearCode);
  let outBuf = new Uint8Array(pixels.length + 1024);
  let outPos = 0;
  let bitBuf = 0;
  let bitCount = 0;

  function emit(code) {
    bitBuf |= code << bitCount;
    bitCount += codeSize;
    while (bitCount >= 8) {
      if (outPos >= outBuf.length) {
        const newBuf = new Uint8Array(outBuf.length * 2);
        newBuf.set(outBuf);
        outBuf = newBuf;
      }
      outBuf[outPos++] = bitBuf & 0xff;
      bitBuf >>= 8;
      bitCount -= 8;
    }
  }

  function resetTrie() {
    for (let i = 0; i < clearCode; i++) trie[i] = null;
    nextCode = eoiCode + 1;
    codeSize = minCodeSize + 1;
  }

  emit(clearCode);
  resetTrie();

  let currentCode = pixels[0];
  let currentNode = trie[currentCode];

  for (let i = 1; i < pixels.length; i++) {
    const px = pixels[i];
    const child = currentNode ? currentNode.get(px) : undefined;
    if (child !== undefined) {
      currentCode = child;
      currentNode = trie[currentCode] || (currentCode < trie.length ? trie[currentCode] : null);
    } else {
      emit(currentCode);
      if (nextCode < maxCode) {
        if (!currentNode && currentCode < trie.length) {
          trie[currentCode] = new Map();
          currentNode = trie[currentCode];
        } else if (!currentNode) {
          if (trie.length <= currentCode) trie.length = currentCode + 1;
          trie[currentCode] = new Map();
          currentNode = trie[currentCode];
        }
        currentNode.set(px, nextCode);
        if (nextCode >= trie.length) trie.length = nextCode + 1;
        trie[nextCode] = null;
        nextCode++;
        if (nextCode >= (1 << codeSize) && codeSize < 12) codeSize++;
      } else {
        emit(clearCode);
        resetTrie();
      }
      currentCode = px;
      currentNode = trie[px];
    }
  }

  emit(currentCode);
  emit(eoiCode);
  if (bitCount > 0) {
    if (outPos >= outBuf.length) {
      const newBuf = new Uint8Array(outBuf.length + 1);
      newBuf.set(outBuf);
      outBuf = newBuf;
    }
    outBuf[outPos++] = bitBuf & 0xff;
  }
  return outBuf.subarray(0, outPos);
}

/** Run the turntable export (main entry point). */
export async function exportTurntable(ctx) {
  const {state, renderer, camera, controls, container} = ctx;
  if (state.exporting) return;
  state.exporting = true;
  state.exportCancelled = false;

  const format = document.getElementById('export-format').value;
  const resMult = parseInt(document.getElementById('export-resolution').value, 10);
  const progressEl = document.getElementById('export-progress');
  const progressText = document.getElementById('export-progress-text');
  const progressFill = document.getElementById('export-progress-fill');

  progressEl.classList.add('visible');
  progressFill.style.width = '0%';

  const fps = 30;
  const duration = 360 / (state.turntableSpeed * 10);
  const totalFrames = Math.round(fps * duration);

  const origWidth = renderer.domElement.width;
  const origHeight = renderer.domElement.height;
  const exportWidth = origWidth * resMult;
  const exportHeight = origHeight * resMult;
  renderer.setSize(renderer.domElement.clientWidth * resMult, renderer.domElement.clientHeight * resMult, false);

  const wasAutoRotate = controls.autoRotate;
  controls.autoRotate = false;

  const anglePerFrame = (2 * Math.PI) / totalFrames;
  const pivotTarget = controls.target.clone();
  const startPos = camera.position.clone();
  const offset = startPos.clone().sub(pivotTarget);
  const radius = Math.sqrt(offset.x ** 2 + offset.z ** 2);
  const startAngle = Math.atan2(offset.x, offset.z);
  const camY = startPos.y;

  const vizName = document.getElementById('viz-selector').selectedOptions[0]?.text || 'mathviz';
  const filename = vizName.replace(/[^a-zA-Z0-9_-]/g, '_');

  try {
    const cfg = {
      totalFrames, fps, anglePerFrame, pivotTarget, radius, startAngle,
      camY, progressText, progressFill, exportWidth, exportHeight, filename,
    };
    if (format === 'webm') {
      await _exportWebM(ctx, cfg);
    } else {
      await _exportGIF(ctx, cfg);
    }
  } finally {
    camera.position.copy(startPos);
    camera.lookAt(pivotTarget);
    controls.update();
    controls.autoRotate = wasAutoRotate;
    renderer.setSize(origWidth, origHeight, false);
    progressEl.classList.remove('visible');
    state.exporting = false;
  }
}
