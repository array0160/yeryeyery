import "./style.css";
import * as ort from "onnxruntime-web";
import {
  DetectionService,
  TextImageUnwarpingService,
  getTextImageUnwarpingPresetOptions,
} from "paddleocr";

const UVDOC_URL =
  "https://huggingface.co/PaddlePaddle/UVDoc_onnx/resolve/main/inference.onnx";
const DET_URL =
  "https://huggingface.co/PaddlePaddle/PP-OCRv5_mobile_det_onnx/resolve/main/inference.onnx";

const MODEL_CACHE = "bookocr-models-v2";
const MAX_PAGE_SIDE = 2400;

ort.env.wasm.numThreads = 1;
ort.env.wasm.simd = true;
ort.env.wasm.wasmPaths =
  "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.27.0/dist/";

const $ = (id) => document.getElementById(id);

const state = {
  file: null,
  bitmap: null,
  rightInput: null,
  leftInput: null,
  rightFlat: null,
  leftFlat: null,
  uvdoc: null,
  uvSession: null,
  detector: null,
  detSession: null,
};

function setStatus(text, detail = "", progress = null) {
  $("status").textContent = text;
  $("detail").textContent = detail;
  if (progress != null) {
    $("progressBar").style.width = `${Math.max(0, Math.min(100, progress))}%`;
  }
}

function showCanvas(canvasId, emptyId) {
  $(canvasId).style.display = "block";
  $(emptyId).style.display = "none";
}

function clearCanvas(canvasId, emptyId, text) {
  const c = $(canvasId);
  c.width = 1;
  c.height = 1;
  c.style.display = "none";
  $(emptyId).style.display = "block";
  $(emptyId).textContent = text;
}

function drawBitmapCropToCanvas(bitmap, sx, sy, sw, sh, canvas) {
  const scale = Math.min(1, MAX_PAGE_SIDE / Math.max(sw, sh));
  canvas.width = Math.max(1, Math.round(sw * scale));
  canvas.height = Math.max(1, Math.round(sh * scale));
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(bitmap, sx, sy, sw, sh, 0, 0, canvas.width, canvas.height);
}

function canvasToPixels(canvas) {
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  return {
    width: canvas.width,
    height: canvas.height,
    data: new Uint8Array(
      imageData.data.buffer.slice(
        imageData.data.byteOffset,
        imageData.data.byteOffset + imageData.data.byteLength,
      ),
    ),
  };
}

function pixelsToCanvas(image, canvas) {
  const { width, height, data } = image;
  canvas.width = width;
  canvas.height = height;

  let rgba;
  if (data.length === width * height * 4) {
    rgba = new Uint8ClampedArray(
      data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength),
    );
  } else if (data.length === width * height * 3) {
    rgba = new Uint8ClampedArray(width * height * 4);
    for (let si = 0, di = 0; si < data.length; ) {
      rgba[di++] = data[si++];
      rgba[di++] = data[si++];
      rgba[di++] = data[si++];
      rgba[di++] = 255;
    }
  } else if (data.length === width * height) {
    rgba = new Uint8ClampedArray(width * height * 4);
    for (let i = 0, j = 0; i < data.length; i++) {
      const v = data[i];
      rgba[j++] = v;
      rgba[j++] = v;
      rgba[j++] = v;
      rgba[j++] = 255;
    }
  } else {
    throw new Error(
      `未知像素格式：${data.length} bytes for ${width}x${height}`,
    );
  }

  canvas.getContext("2d").putImageData(new ImageData(rgba, width, height), 0, 0);
}

async function fetchArrayBufferCached(url, label, progressBase, progressSpan) {
  const cache = "caches" in window ? await caches.open(MODEL_CACHE) : null;

  if (cache) {
    const cached = await cache.match(url);
    if (cached) {
      setStatus(`載入 ${label}…`, `${label} 已從瀏覽器快取取得。`, progressBase + progressSpan);
      return cached.arrayBuffer();
    }
  }

  const response = await fetch(url, { mode: "cors" });
  if (!response.ok) {
    throw new Error(`${label} 下載失敗：HTTP ${response.status}`);
  }

  const total = Number(response.headers.get("content-length")) || 0;
  if (!response.body) {
    const buffer = await response.arrayBuffer();
    if (cache) {
      await cache.put(url, new Response(buffer.slice(0)));
    }
    return buffer;
  }

  const reader = response.body.getReader();
  const chunks = [];
  let received = 0;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.byteLength;

    const ratio = total ? received / total : 0.2;
    setStatus(
      `下載 ${label}…`,
      total
        ? `${(received/1024/1024).toFixed(1)} / ${(total/1024/1024).toFixed(1)} MB`
        : `${(received/1024/1024).toFixed(1)} MB`,
      progressBase + progressSpan * ratio,
    );
  }

  const merged = new Uint8Array(received);
  let offset = 0;
  for (const chunk of chunks) {
    merged.set(chunk, offset);
    offset += chunk.byteLength;
  }

  if (cache) {
    await cache.put(url, new Response(merged.slice().buffer));
  }
  return merged.buffer;
}

async function ensureUvDoc() {
  if (state.uvdoc) return state.uvdoc;

  const model = await fetchArrayBufferCached(UVDOC_URL, "UVDoc", 2, 45);
  setStatus("建立 UVDoc session…", "ONNX Runtime Web / WASM", 50);

  state.uvSession = await ort.InferenceSession.create(model, {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "all",
  });

  state.uvdoc = new TextImageUnwarpingService(
    ort,
    state.uvSession,
    getTextImageUnwarpingPresetOptions("UVDoc"),
  );
  return state.uvdoc;
}

function detectorRuntimeOptions() {
  return {
    // These map to the Colab/PaddleOCR values we settled on.
    textPixelThreshold: Number($("pixelThreshold").value),
    boxScoreThreshold: Number($("boxThreshold").value),
    unclipRatio: Number($("unclipRatio").value),
    maxSideLength: Number($("maxSideLength").value),
    limitType: "max",
    maxSideLimit: 4000,

    // Paddle's official PP-OCRv5 inference config decodes BGR.
    channelOrder: "bgr",

    // Keep raw DB boxes as geometry hints. Do not add service-level padding.
    paddingBoxVertical: 0,
    paddingBoxHorizontal: 0,
    minimumAreaThreshold: 20,
    maxCandidates: 1000,
    boxType: "quad",
  };
}

async function ensureDetector() {
  if (state.detector) return state.detector;

  const model = await fetchArrayBufferCached(DET_URL, "PP-OCRv5 mobile detector", 50, 30);
  setStatus("建立 Detector session…", "PP-OCRv5_mobile_det ONNX", 82);

  state.detSession = await ort.InferenceSession.create(model, {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "all",
  });

  state.detector = new DetectionService(
    ort,
    state.detSession,
    detectorRuntimeOptions(),
  );

  return state.detector;
}

function splitPages() {
  if (!state.bitmap) return;

  const w = state.bitmap.width;
  const h = state.bitmap.height;
  const splitPct = Number($("splitRange").value) / 100;
  const gutterPct = Number($("gutterRange").value) / 100;
  const splitX = w * splitPct;
  const halfGutter = (w * gutterPct) / 2;
  const leftEnd = Math.max(1, splitX - halfGutter);
  const rightStart = Math.min(w - 1, splitX + halfGutter);

  const rc = document.createElement("canvas");
  const lc = document.createElement("canvas");

  drawBitmapCropToCanvas(state.bitmap, rightStart, 0, w - rightStart, h, rc);
  drawBitmapCropToCanvas(state.bitmap, 0, 0, leftEnd, h, lc);

  state.rightInput = canvasToPixels(rc);
  state.leftInput = canvasToPixels(lc);
  state.rightFlat = null;
  state.leftFlat = null;

  clearCanvas("rightFlat", "rightFlatEmpty", "等待 UVDoc");
  clearCanvas("leftFlat", "leftFlatEmpty", "等待 UVDoc");
  clearCanvas("rightOverlay", "rightOverlayEmpty", "等待 Detector");
  clearCanvas("leftOverlay", "leftOverlayEmpty", "等待 Detector");
  $("rightStats").textContent = "尚未偵測";
  $("leftStats").textContent = "尚未偵測";

  setStatus("左右頁切割完成。", "下一步可跑 UVDoc。", 0);
}

async function runUvDoc() {
  if (!state.rightInput || !state.leftInput) splitPages();

  const uv = await ensureUvDoc();

  setStatus("UVDoc：正在展平右頁…", "", 55);
  state.rightFlat = (await uv.run(state.rightInput)).doctrImage;
  pixelsToCanvas(state.rightFlat, $("rightFlat"));
  showCanvas("rightFlat", "rightFlatEmpty");

  setStatus("UVDoc：正在展平左頁…", "", 72);
  state.leftFlat = (await uv.run(state.leftInput)).doctrImage;
  pixelsToCanvas(state.leftFlat, $("leftFlat"));
  showCanvas("leftFlat", "leftFlatEmpty");

  setStatus("UVDoc 完成。", "現在可以跑 Detector + PCA 中心線。", 100);
}

function boxPoints(box) {
  if (Array.isArray(box.points) && box.points.length >= 4) {
    return box.points.map((p) => ({ x: Number(p.x), y: Number(p.y) }));
  }
  if (Array.isArray(box.polygon) && box.polygon.length >= 4) {
    return box.polygon.map((p) => ({ x: Number(p.x), y: Number(p.y) }));
  }
  return [
    { x: box.x, y: box.y },
    { x: box.x + box.width, y: box.y },
    { x: box.x + box.width, y: box.y + box.height },
    { x: box.x, y: box.y + box.height },
  ];
}

function pcaGeometry(points) {
  const n = points.length;
  if (n < 2) return null;

  let cx = 0, cy = 0;
  for (const p of points) { cx += p.x; cy += p.y; }
  cx /= n; cy /= n;

  let a = 0, b = 0, d = 0;
  for (const p of points) {
    const x = p.x - cx;
    const y = p.y - cy;
    a += x * x;
    b += x * y;
    d += y * y;
  }
  a /= n;
  b /= n;
  d /= n;

  // Principal eigenvector for symmetric 2x2 covariance matrix.
  const trace = a + d;
  const disc = Math.sqrt(Math.max(0, (a - d) * (a - d) + 4 * b * b));
  const lambda1 = (trace + disc) / 2;

  let vx, vy;
  if (Math.abs(b) > 1e-8) {
    vx = lambda1 - d;
    vy = b;
  } else if (a >= d) {
    vx = 1; vy = 0;
  } else {
    vx = 0; vy = 1;
  }

  const norm = Math.hypot(vx, vy) || 1;
  vx /= norm; vy /= norm;
  if (vy < 0) { vx = -vx; vy = -vy; }

  const mx = -vy, my = vx;
  let majorMin = Infinity, majorMax = -Infinity;
  let minorMin = Infinity, minorMax = -Infinity;

  for (const p of points) {
    const dx = p.x - cx;
    const dy = p.y - cy;
    const q1 = dx * vx + dy * vy;
    const q2 = dx * mx + dy * my;
    majorMin = Math.min(majorMin, q1);
    majorMax = Math.max(majorMax, q1);
    minorMin = Math.min(minorMin, q2);
    minorMax = Math.max(minorMax, q2);
  }

  const top = { x: cx + vx * majorMin, y: cy + vy * majorMin };
  const bottom = { x: cx + vx * majorMax, y: cy + vy * majorMax };
  const length = majorMax - majorMin;
  const width = minorMax - minorMin;
  const dy = bottom.y - top.y;
  if (Math.abs(dy) < 3) return null;

  const slope = (bottom.x - top.x) / dy;
  const intercept = top.x - slope * top.y;

  return {
    center: { x: cx, y: cy },
    top,
    bottom,
    length,
    width,
    slope,
    intercept,
    y0: Math.min(top.y, bottom.y),
    y1: Math.max(top.y, bottom.y),
  };
}

function median(values) {
  if (!values.length) return 0;
  const x = [...values].sort((a, b) => a - b);
  const m = Math.floor(x.length / 2);
  return x.length % 2 ? x[m] : (x[m - 1] + x[m]) / 2;
}

function lineX(c, y) {
  return c.slope * y + c.intercept;
}

function makeColumns(boxes, imageHeight) {
  const cols = [];

  for (const box of boxes) {
    const points = boxPoints(box);
    const g = pcaGeometry(points);
    if (!g) continue;

    // Same intent as Colab: keep long vertical-ish text regions,
    // including shorter headings, but reject tiny/non-column shapes.
    if (g.length < g.width * 2) continue;
    if (g.length < imageHeight * 0.025) continue;

    cols.push({ ...g, points, rawBox: box });
  }

  if (!cols.length) return [];

  const stableSlopes = cols
    .map((c) => c.slope)
    .filter((s) => Math.abs(s) < 0.5);

  const pageMedianSlope = stableSlopes.length ? median(stableSlopes) : 0;

  for (const c of cols) {
    if (Math.abs(c.slope - pageMedianSlope) > 0.12) {
      c.slope = pageMedianSlope;
      c.intercept = c.center.x - pageMedianSlope * c.center.y;
    }
    c.xRef = lineX(c, imageHeight * 0.5);
  }

  // Traditional vertical reading order: right to left.
  cols.sort((a, b) => b.xRef - a.xRef);
  return cols;
}

function drawOverlay(image, boxes, cols, canvas) {
  pixelsToCanvas(image, canvas);
  const ctx = canvas.getContext("2d");

  ctx.lineJoin = "round";
  ctx.lineCap = "round";

  // Detector polygons: red.
  ctx.strokeStyle = "rgba(220, 55, 65, .82)";
  ctx.lineWidth = Math.max(1.5, canvas.width / 650);

  for (const box of boxes) {
    const pts = boxPoints(box);
    if (!pts.length) continue;
    ctx.beginPath();
    ctx.moveTo(pts[0].x, pts[0].y);
    for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i].x, pts[i].y);
    ctx.closePath();
    ctx.stroke();
  }

  // PCA centerlines: blue.
  ctx.strokeStyle = "rgba(25, 105, 215, .95)";
  ctx.fillStyle = "rgba(25, 105, 215, .95)";
  ctx.lineWidth = Math.max(2.2, canvas.width / 430);
  ctx.font = `bold ${Math.max(15, canvas.width / 42)}px system-ui`;

  cols.forEach((c, i) => {
    const y0 = Math.max(0, c.y0);
    const y1 = Math.min(canvas.height - 1, c.y1);
    const x0 = lineX(c, y0);
    const x1 = lineX(c, y1);

    ctx.beginPath();
    ctx.moveTo(x0, y0);
    ctx.lineTo(x1, y1);
    ctx.stroke();

    const labelX = Math.max(4, Math.min(canvas.width - 50, x0 + 5));
    const labelY = Math.max(20, y0 + 18);
    ctx.fillText(String(i + 1), labelX, labelY);
  });
}

async function detectPage(image, side) {
  const detector = await ensureDetector();
  const options = detectorRuntimeOptions();

  const boxes = await detector.run(image, {
    ...options,
    onProgress(event) {
      const stageName = {
        preprocess: "前處理",
        infer: "模型推理",
        postprocess: "DB 後處理",
      }[event.stage] || event.stage;

      setStatus(
        `Detector：${side === "right" ? "右頁" : "左頁"} ${stageName}…`,
        event.detectedCount != null ? `偵測到 ${event.detectedCount} 個區域` : "",
        side === "right" ? 45 : 78,
      );
    },
  });

  const cols = makeColumns(boxes, image.height);
  const canvas = $(side === "right" ? "rightOverlay" : "leftOverlay");
  drawOverlay(image, boxes, cols, canvas);

  showCanvas(
    side === "right" ? "rightOverlay" : "leftOverlay",
    side === "right" ? "rightOverlayEmpty" : "leftOverlayEmpty",
  );

  $(side === "right" ? "rightStats" : "leftStats").textContent =
    `紅框 ${boxes.length} · 藍線 ${cols.length}`;

  return { boxes, cols };
}

async function runDetector() {
  if (!state.rightFlat || !state.leftFlat) {
    await runUvDoc();
  }

  setStatus("Detector：右頁…", "使用 PP-OCRv5_mobile_det", 35);
  await detectPage(state.rightFlat, "right");

  setStatus("Detector：左頁…", "使用 PP-OCRv5_mobile_det", 68);
  await detectPage(state.leftFlat, "left");

  setStatus(
    "中心線完成。",
    "請看藍線是否大致一欄一條；數字已依右 → 左排序。",
    100,
  );
}

async function withBusy(fn) {
  const buttons = ["splitBtn", "uvBtn", "detBtn", "allBtn"];
  buttons.forEach((id) => $(id).disabled = true);
  try {
    await fn();
  } catch (error) {
    console.error(error);
    setStatus(
      "處理失敗。",
      error instanceof Error ? error.message : String(error),
      0,
    );
  } finally {
    if (state.bitmap) {
      buttons.forEach((id) => $(id).disabled = false);
    }
  }
}

async function loadFile(file) {
  state.file = file;
  state.bitmap?.close?.();
  state.bitmap = await createImageBitmap(file, { imageOrientation: "from-image" });

  $("photoPreview").src = URL.createObjectURL(file);
  $("photoPreview").style.display = "block";
  $("dropHint").style.display = "none";

  ["splitBtn", "uvBtn", "detBtn", "allBtn"].forEach((id) => $(id).disabled = false);
  splitPages();
}

$("fileInput").addEventListener("change", (event) => {
  const file = event.target.files?.[0];
  if (file) loadFile(file);
});

const dropZone = $("dropZone");
["dragenter", "dragover"].forEach((name) => {
  dropZone.addEventListener(name, (event) => {
    event.preventDefault();
    dropZone.classList.add("drag");
  });
});
["dragleave", "drop"].forEach((name) => {
  dropZone.addEventListener(name, (event) => {
    event.preventDefault();
    dropZone.classList.remove("drag");
  });
});
dropZone.addEventListener("drop", (event) => {
  const file = event.dataTransfer?.files?.[0];
  if (file?.type.startsWith("image/")) loadFile(file);
});

$("splitRange").addEventListener("input", () => {
  $("splitValue").textContent = `${Number($("splitRange").value).toFixed(1)}%`;
  if (state.bitmap) splitPages();
});
$("gutterRange").addEventListener("input", () => {
  $("gutterValue").textContent = `${Number($("gutterRange").value).toFixed(1)}%`;
  if (state.bitmap) splitPages();
});

$("splitBtn").addEventListener("click", () => splitPages());
$("uvBtn").addEventListener("click", () => withBusy(runUvDoc));
$("detBtn").addEventListener("click", () => withBusy(runDetector));
$("allBtn").addEventListener("click", () => withBusy(async () => {
  splitPages();
  await runUvDoc();
  await runDetector();
}));

$("clearCacheBtn").addEventListener("click", async () => {
  if ("caches" in window) await caches.delete(MODEL_CACHE);
  state.uvdoc = null;
  state.uvSession = null;
  state.detector = null;
  state.detSession = null;
  setStatus("模型快取已清除。", "下一次執行會重新下載 UVDoc 與 Detector。", 0);
});

$("splitValue").textContent = `${Number($("splitRange").value).toFixed(1)}%`;
$("gutterValue").textContent = `${Number($("gutterRange").value).toFixed(1)}%`;
