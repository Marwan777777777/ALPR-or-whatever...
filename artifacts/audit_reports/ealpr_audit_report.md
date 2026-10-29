# EALPR Dataset Audit Report

**Dataset Root:** `C:\Users\MARWAN\plate_project\images_raw\EALPR-master`  
**Provenance:** Youssef et al., ACIRS 2022 (doi: 10.1109/ACIRS55390.2022.9845514)  
**License Status:** `UNSPECIFIED_NO_LICENSE_FILE (Academic citation requested in README.md, no commercial license grant included)`  

---

## 1. Executive Summary & Inventory Counts

| Subset | Images / Crops | Annotations | Total Boxes | Key Classes |
|---|---|---|---|---|
| **Vehicles (Plate Detection)** | 2087 images | 2087 files | 2143 boxes | `license_plate` (100%) |
| **Plates (Cropped ROIs)** | 2069 images | N/A (Image crops) | N/A | Cropped plate crops |
| **Characters (OCR & Detection)** | 10505 crops | 1983 files | 10797 boxes | 27 classes (17 Arabic letters + 10 digits) |

> [!IMPORTANT]
> **Dataset Label Meaning Finding:**  
> The annotations in `EALPR Vechicles dataset/Vehicles Labeling` contain **License Plate bounding boxes** (Class 0: `license_plate`), with mean aspect ratio ~2.01 and normalized width ~0.18.  
> They do **NOT** contain vehicle bounding boxes, make/model, or color labels. This dataset is a **Plate Detection & OCR dataset**, not a Vehicle Classification dataset.

---

## 2. Image Dimensions & Integrity

### Vehicles
- Min Resolution: 240x342
- Max Resolution: 2048x1536
- Mean Resolution: 825.1x872.0
- Corrupt Images: 0
- Exact MD5 Duplicates: 4 groups

### Plates
- Min Resolution: 43x29
- Max Resolution: 777x531
- Mean Resolution: 160.1x88.3
- Corrupt Images: 0
- Exact MD5 Duplicates: 55 groups

---

## 3. Character Class Distribution (27 Classes)

Total character boxes: **10797** across **1983** plate files.

| Class ID | Character | Count | Percentage |
|---|---|---|---|
| `١` | ١ | 1408 | 13.04% |
| `٦` | ٦ | 1058 | 9.80% |
| `٥` | ٥ | 644 | 5.96% |
| `م` | م | 616 | 5.71% |
| `٤` | ٤ | 615 | 5.70% |
| `٧` | ٧ | 550 | 5.09% |
| `أ` | أ | 465 | 4.31% |
| `٢` | ٢ | 435 | 4.03% |
| `س` | س | 434 | 4.02% |
| `٩` | ٩ | 403 | 3.73% |
| `٣` | ٣ | 400 | 3.70% |
| `ر` | ر | 383 | 3.55% |
| `د` | د | 358 | 3.32% |
| `ع` | ع | 323 | 2.99% |
| `٨` | ٨ | 291 | 2.70% |
| `ھ` | ھ | 280 | 2.59% |
| `و` | و | 273 | 2.53% |
| `ن` | ن | 272 | 2.52% |
| `ج` | ج | 224 | 2.07% |
| `ل` | ل | 221 | 2.05% |
| `ى` | ى | 217 | 2.01% |
| `ب` | ب | 211 | 1.95% |
| `ق` | ق | 211 | 1.95% |
| `ف` | ف | 191 | 1.77% |
| `ط` | ط | 163 | 1.51% |
| `ص` | ص | 150 | 1.39% |
| `٠` | ٠ | 1 | 0.01% |

---

## 4. Hierarchical Linkage Between Subsets

- Total Vehicle Images: **2087**
- Vehicles with Linked Plate Crops: **2027** (97.1%)
- Unlinked Vehicles (No Plate Crop): **50**
- Plates with Character Annotations: **1983** / 2069
- Vehicles with Full Chain (Vehicle -> Plate -> Characters): **1955**

---

## 5. Discovered Errors and Exclusions

1. **Missing Label Files**: 0 vehicle images lack labels. 1 label file `classes.txt` exists in both label directories.
2. **Empty Character Annotations**: 5 plate files have 0 characters labeled: `['0075_license_plate_2.txt', '0286_license_plate_1.txt', '0742_license_plate_1.txt', '1080_license_plate_1.txt', '1080_license_plate_3.txt']`.
3. **Malformed Lines**: 0 syntax or out-of-bounds coordinate errors found across all files.
4. **Duplicate Images**: 4 vehicle duplicate groups and 55 plate duplicate groups must be isolated during train/val/test splitting to prevent data leakage.
