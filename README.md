# BookOCR Web v2 — UVDoc + Detector + PCA Centerlines

這版完全不需要 Python / BAT / 本機 server。

目前流程：

```text
跨頁照片
→ 右頁 / 左頁切割
→ Browser UVDoc
→ PP-OCRv5_mobile_det ONNX
→ detector quadrilateral
→ PCA 長軸中心線
→ 短欄 slope regularization
→ 每頁右 → 左排序
```

## 最重要：以後不需要再手動建立 YML

這包已經包含：

```text
.github/
└── workflows/
    └── deploy-pages.yml
```

所以更新同一個 GitHub repo 時，只要把新版檔案整批上傳 / 覆蓋後 Commit。
GitHub Actions 會自己觸發，不需要再到 Actions 建 workflow。

## Repo 根目錄應該長這樣

```text
.github/
public/
src/
index.html
package.json
vite.config.js
README.md
```

## GitHub Pages

第一次 repo 只需要做一次：

Settings → Pages → Source = GitHub Actions

之後不必再設定。

## 本版 Detector 預設

對應我們 Colab 最後使用的參數：

```text
thresh / textPixelThreshold = 0.20
box_thresh / boxScoreThreshold = 0.35
unclip_ratio = 1.15
limit_side_len / maxSideLength = 2000
limit_type = max
```

另將 service-level box padding 設為 0，因為 detector box 在我們的流程裡只是幾何定位線索，
不直接當作 OCR 裁切邊界。

PP-OCRv5 官方 inference config 使用 BGR，因此 detector runtime 設為 BGR channel order。

## 本版驗收

只看藍色中心線：

- 是否大致一欄一條
- 是否能跟著斜欄
- 短標題 / 章名不要全部消失
- 順序是否由右至左

如果這一關過，下一版才搬 V3：

```text
centerline x(y)
→ neighboring midpoint corridor
→ slope regularization
→ scanline remap
→ Column 01 / 02 / 03...
```

## 模型

瀏覽器第一次會下載：

- UVDoc ONNX：大約 30 MB
- PP-OCRv5_mobile_det ONNX：大約 4.8 MB

下載後使用 Cache Storage 保存。
