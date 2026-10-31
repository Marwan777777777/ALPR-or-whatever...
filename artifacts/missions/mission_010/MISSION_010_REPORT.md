# MISSION 010: Vehicle-Level Detection, Tracking, and Plate Association

## Executive Summary

Mission 010 transitions the Egyptian Vehicle Intelligence / ALPR pipeline from **plate-level tracking** to **vehicle-level tracking**. 

Prior to this milestone, the pipeline tracked license plate bounding boxes as an ad-hoc proxy for vehicles. Vehicle identity did not exist independently of license plate identity: if a plate was missed or occluded, the vehicle ceased to exist. When multiple vehicles or adjacent lanes were present, plates were vulnerable to track swapping and misattribution.

This engineering milestone establishes:
1. **Architectural separation of vehicle identity and plate identity** via independent vehicle track IDs.
2. **A configurable, replaceable vehicle detection interface** with comprehensive dependency licensing analysis.
3. **A robust multi-object vehicle tracker** (`CentroidIoUTracker`) handling linear motion, multi-vehicle scenes, temporary detection loss, and trajectory history.
4. **A confidence-aware plate-to-vehicle association layer** (`PlateVehicleAssociator`) utilizing spatial containment ($\text{IoP}$), relative geometry, scale plausibility, and ambiguity detection without ever forcing uncertain associations.
5. **An upgraded vehicle transit event model** (`VehicleTransitEvent`) capturing multi-frame evidence, alternative readings, competing vehicle IDs, and review flags.
6. **Eight deterministic synthetic integration tests** covering all edge cases without GPU or external media dependencies.
7. **Empirical gate validation on 340 chronological captures**, honestly documenting why discrete trigger snapshots cannot be treated as continuous video trajectories.

---

## 1. Architectural Audit of Existing Tracking Limitations

An architectural review of `pipeline/alpr_pipeline.py` and `scripts/alpr_cli.py` prior to Mission 010 revealed six fundamental tracking limitations:

| Architectural Dimension | Historical ALPR Pipeline Behavior | Engineering Deficiency & Failure Mode | Mission 010 Resolution |
| :--- | :--- | :--- | :--- |
| **Tracking Target** | Operated solely on **plate bounding boxes** wrapped in dummy `VehicleDetection(box=plate_box)`. | Vehicle body bounding boxes were never detected or tracked. | Tracking now operates directly on vehicle bodies with independent vehicle track IDs. |
| **Vehicle Identity Independence** | Did not exist. Vehicle ID was strictly derived from plate presence. | If a plate was occluded, dirty, or missed by the plate detector for several frames, the vehicle ceased to exist. When the plate reappeared, a new track ID was spawned. | Vehicle identity persists across frames even when plate detection is intermittently lost. |
| **Plate Association Mechanism** | Trivial self-match: `t_veh.box.iou(r.bounding_box) >= 0.50`. | Because the "vehicle" box was the plate box, association was simply matching a plate to itself. | Independent `PlateVehicleAssociator` calculates multi-factor spatial containment ($\text{IoP}$), vertical/horizontal alignment, and scale ratios. |
| **Multi-Vehicle Handling** | Multiple vehicles in one frame were treated as disjoint plate detections with no parent vehicle envelope. | If two vehicles appeared simultaneously, the system could not determine which vehicle belonged to which lane or track. | Vehicles receive distinct track IDs; plates are associated to enclosing vehicle candidates via bipartite geometric scoring. |
| **Wrong Assignment Risk** | High during tailgating, passing, or queueing. | When two vehicles were close, a slight plate detection shift could cause the plate to snap to the wrong vehicle track. | Multi-vehicle ambiguity detection: if candidate score delta $\Delta S < 0.15$ or multiple vehicles enclose a plate, the system flags `AMBIGUOUS`, preserves competing IDs, and marks for review. |
| **Replay & Timestamps** | Replay relied on simulated timestamps (`frame_idx / fps` or `idx * 0.1`). | Filename-derived plate strings (legacy ANPR) were sometimes conflated with ground truth. | Full decoupling: timestamps are parsed solely for chronological ordering and elapsed-time calculation ($\Delta t$). Filename metadata is never used as ground truth. |

---

## 2. Vehicle Detection & Model Licensing Analysis

### 2.1 Replaceable Detector Interface
Vehicle detection is abstracted behind the `BaseVehicleDetector` interface in `vehicle_intelligence/core/interfaces.py`:
```python
class BaseVehicleDetector(ABC):
    @abstractmethod
    def detect(self, image: np.ndarray) -> List[VehicleDetection]:
        pass
```
The implementation `PyTorchYoloVehicleDetector` in `vehicle_intelligence/detectors/pytorch_yolo.py`:
- Detects cars, trucks, buses, motorcycles, vans, and SUVs.
- Configurable weights path, confidence threshold, IoU threshold, input size, and execution device (`cuda` or `cpu`).
- Maps COCO class IDs (2: car, 3: motorcycle, 5: bus, 7: truck) to domain `VehicleClass` enums.
- Strictly validates weight existence: never fabricates detections if weights are missing.

### 2.2 Dependency & Model Licensing Audit
A commercial product must avoid unintended copyleft obligations:

1. **Ultralytics YOLOv8 (AGPL-3.0 License):**
   - The YOLOv8 model architecture and code repository are licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
   - Under AGPL-3.0, if this software is run on a server and interacted with over a network (e.g., CCTV API or SaaS), the entire host application source code may be subject to copyleft disclosure requirements.
   - For internal evaluation, research, and non-commercial benchmarking, YOLOv8 is acceptable. However, for a commercial proprietary gate product, commercial licensing from Ultralytics or an open-source permissive alternative is required.

2. **Permissively Licensed Alternatives (Apache 2.0 / MIT):**
   - **RT-DETR (Real-Time DEtection TRansformer):** Licensed under **Apache 2.0**. Provides transformer-based real-time object detection with zero copyleft obligations.
   - **YOLOv10 / ONNX Runtime:** Available under **Apache 2.0**.
   - **Architecture Protection:** Because `BaseVehicleDetector` decouples the inference pipeline from the specific neural network, `PyTorchYoloVehicleDetector` can be replaced with an `ONNXRTDETRVehicleDetector` without modifying a single line of tracking, association, or event aggregation logic.

### 2.3 Real-World Gate Domain Shift Findings
In evaluating `yolov8n.pt` across the 340 gate frames in `D:/TruckEntry`:
- **Night / Infrared Illumination Domain Shift:** 
  In night-time frames (01:39 to 06:29), the gate camera operates under active infrared (monochrome/grayscale). Off-the-shelf COCO models (trained on daylight color imagery) detected 0 vehicles in almost all night frames. In contrast, the domain-adapted Egyptian plate detector detected plates with high confidence ($0.88 - 0.90$).
- **Extreme Close-Up Perspective:**
  The barrier gate camera is mounted high, pointing down at the front grille/windshield of large cab-over trucks occupying up to 90% of the 4K frame. The lack of visible wheels or full vehicle silhouettes causes natural-scene vehicle detectors to exhibit lower confidence or partial misclassification (e.g. classifying a massive truck cab as "car" or "bus").
- **Engineering Conclusion:** Rather than retraining or fine-tuning models prematurely, the pipeline incorporates fallback awareness: plates detected without vehicle bodies are preserved as `UNASSIGNED` rather than discarded.

---

## 3. Vehicle-Level Tracking Architecture

Vehicle-level tracking is implemented in `CentroidIoUTracker` (`vehicle_intelligence/tracking/tracker.py`).

### 3.1 Tracking Algorithm & Distance Metric
To prevent track hopping when vehicles pass close together or enter adjacent gate lanes, the cost matrix combines bounding box $\text{IoU}$ with a **normalized centroid distance** relative to vehicle dimensions:

$$\Delta x_{\text{norm}} = \frac{|c_{x1} - c_{x2}|}{w_{\text{avg}}}, \quad \Delta y_{\text{norm}} = \frac{|c_{y1} - c_{y2}|}{h_{\text{avg}}}$$

$$d_{\text{norm}} = \sqrt{\Delta x_{\text{norm}}^2 + \Delta y_{\text{norm}}^2}$$

$$\text{Score}(T_i, D_j) = 2.0 \cdot \text{IoU}(T_i, D_j) + \max\left(0, 1.0 - \frac{d_{\text{norm}}}{d_{\text{max\_norm}}}\right) + \text{Bonus}_{\text{class}}$$

Where:
- $w_{\text{avg}} = (w_T + w_D) / 2$, $h_{\text{avg}} = (h_T + h_D) / 2$.
- $\text{Bonus}_{\text{class}} = 0.2$ if vehicle classes match and are not `UNKNOWN`.
- A candidate match is accepted only if $\text{IoU} \ge 0.20$ or ($d_{\text{norm}} \le 1.0$ and absolute Euclidean distance $\le d_{\text{max}}$).

### 3.2 State Management & Track Recovery
- **Independent Track IDs:** Initialized via monotonically increasing integer IDs.
- **Trajectory & History Retention:** Each track maintains `history_boxes`, `confidence_history`, `timestamps`, `frame_ids`, and total `hits`.
- **Temporary Detection Loss:** If a vehicle is missed due to motion blur or barrier gate occlusion, the track increments `lost`. It is maintained up to `max_lost_frames` (default: 5 to 15 frames) and re-associated if the vehicle reappears within spatial bounds.

---

## 4. Confidence-Aware Plate-to-Vehicle Association

Plate-to-vehicle association is decoupled from detection and tracking, residing in `PlateVehicleAssociator` (`vehicle_intelligence/association/plate_vehicle_associator.py`).

### 4.1 Association Criteria & Math Formulation
For any plate detection $P$ and vehicle candidate $V$:

1. **Spatial Containment ($\text{IoP}$ - Intersection over Plate Area):**
   $$\text{IoP}(P, V) = \frac{\text{Area}(P \cap V)}{\text{Area}(P)}$$
   A genuine license plate must be physically mounted on the vehicle ($\text{IoP} \ge 0.50$, typically $\ge 0.90$). If $\text{IoP} < 0.50$, the candidate is rejected.

2. **Relative Horizontal Center Offset:**
   Plates are roughly horizontally centered on the front or rear bumper:
   $$dx_{\text{rel}} = \frac{|c_{x}^P - c_{x}^V|}{w_V}, \quad S_{\text{horiz}} = \max\left(0.0, 1.0 - \frac{dx_{\text{rel}}}{0.50}\right)$$

3. **Relative Vertical Placement:**
   Plates are mounted on the lower front grille or bumper ($y_{\text{rel}} \in [0.25, 0.98]$):
   $$y_{\text{rel}} = \frac{c_{y}^P - y_{1}^V}{h_V}$$
   If $y_{\text{rel}} < 0.25$ (e.g. cab roof/windshield header), a penalty is applied (`HIGH_PLATE_PLACEMENT`).

4. **Physical Scale Consistency:**
   $$\text{Ratio}_{\text{area}} = \frac{\text{Area}(P)}{\text{Area}(V)} \in [0.0005, 0.35]$$

5. **Composite Association Score:**
   $$S_{\text{assoc}} = 0.50 \cdot \text{IoP} + 0.20 \cdot S_{\text{horiz}} + 0.15 \cdot (S_{\text{vert}} \cdot S_{\text{scale}}) + 0.15 \cdot \text{Conf}_V$$

### 4.2 Ambiguity Resolution & Unassigned Handling
The associator enforces strict, conservative assignment rules:
- **Case 1: Confident Association:** Single candidate with $S_{\text{assoc}} \ge 0.65$ and $\text{IoP} \ge 0.75 \implies \text{CONFIDENT}$.
- **Case 2: Ambiguous Multiple Candidates:** If multiple vehicles compete for the same plate and $|S_1 - S_2| < 0.15$ or both enclose the plate ($\text{IoP}_1 \ge 0.70$ and $\text{IoP}_2 \ge 0.50$):
  - Association status is set to `AMBIGUOUS`.
  - All competing candidate IDs are recorded in `competing_vehicle_ids`.
  - `needs_review` is set to `True`.
  - The association is **never arbitrarily forced**.
- **Case 3: Unassigned Plates:** If no vehicle candidate achieves $S_{\text{assoc}} \ge 0.35$ (e.g. night IR vehicle detection miss or roadside sign):
  - Preserved with `status = UNASSIGNED`, `vehicle_track_id = None`.
  - `needs_review = True`, reason: `"UNASSIGNED_PLATE: No enclosing vehicle found"`.
  - The plate observation is never discarded.

---

## 5. Upgraded Vehicle Transit Event Model

The consolidated vehicle transit event is formalized in `VehicleTransitEvent` (`vehicle_intelligence/core/domain.py`):

```python
class VehicleTransitEvent(BaseModel):
    event_type: str = "VEHICLE_TRANSIT"
    vehicle_track_id: int
    vehicle_class: VehicleClass
    vehicle_confidence: float
    first_seen_timestamp: float
    last_seen_timestamp: float
    total_frames_tracked: int
    plate_reading: Optional[PlateTextResult] = None
    plate_text: str = ""
    letters: str = ""
    digits: str = ""
    plate_confidence: float = 0.0
    plate_association_confidence: float = 0.0
    association_status: AssociationStatus  # CONFIDENT, AMBIGUOUS, UNASSIGNED
    best_vehicle_box: BoundingBox
    best_plate_box: Optional[BoundingBox] = None
    best_evidence_frame_id: Optional[int] = None
    best_crop_path: Optional[str] = None
    best_evidence_source_id: Optional[str] = None
    observation_timestamps: List[float] = Field(default_factory=list)
    alternative_readings: List[PlateTextResult] = Field(default_factory=list)
    has_conflicting_reads: bool = False
    has_conflicting_association: bool = False
    competing_vehicle_ids: List[int] = Field(default_factory=list)
    attributes: Optional[VehicleAttributes] = None
    needs_review: bool = False
    review_reasons: List[str] = Field(default_factory=list)
```

### Constraint Compliance on Vehicle Attributes
- **No Fabricated Colors or Models:** Vehicle make, model, and color are only populated if an attribute classifier provides positive evidence.
- **Monochrome IR Limitation:** Active infrared night footage is inherently single-channel intensity data; it physically cannot determine surface paint color. The pipeline explicitly rejects claiming vehicle color from infrared captures.

---

## 6. Deterministic Synthetic Integration Tests

To guarantee reproducibility and regression protection across CI environments without requiring GPU hardware or external video files, eight deterministic integration fixtures were implemented in `tests/test_mission_010_vehicle_tracking.py`:

| Scenario ID | Test Name | Injected Test Conditions | Expected Behavior | Outcome |
| :---: | :--- | :--- | :--- | :---: |
| **1** | `test_scenario_1_single_vehicle_single_plate_multiple_frames` | 1 vehicle moving linearly across 5 frames with plate inside lower bumper ($\text{IoP} \ge 0.99$). | Track ID persists across all 5 frames; plate associated confidently; coherent transit event generated with 5 timestamps. | **PASSED** |
| **2** | `test_scenario_2_two_vehicles_in_same_frame` | Lane 1 (Car at $x=50..250$) and Lane 2 (Truck at $x=500..750$) simultaneously in frame. | Two independent track IDs assigned; disjoint spatial matching; plates mapped to correct parent vehicles with zero cross-talk. | **PASSED** |
| **3** | `test_scenario_3_two_vehicles_passing_close_together` | Adjacent vehicles passing in opposite directions within 10px boundary. | Normalized centroid distance metric prevents track ID swaps; each vehicle retains its own track ID across frames. | **PASSED** |
| **4** | `test_scenario_4_intermittent_plate_detection` | Vehicle tracked across 6 frames; plate visible only in frames 1, 2, and 5 (missed in 3, 4, 6). | Vehicle track remains continuous without fragmentation; consolidated event created with plate evidence from visible frames. | **PASSED** |
| **5** | `test_scenario_5_plate_detected_without_vehicle_detection` | Plate detected at image center; 0 vehicles detected (simulating night IR domain shift). | Handled gracefully without crash; plate preserved as `UNASSIGNED` with `vehicle_track_id=None` and flagged for review. | **PASSED** |
| **6** | `test_scenario_6_ambiguous_plate_vehicle_association` | Two overlapping vehicles (e.g. towing/tight queue) both enclose plate with score delta $< 0.15$. | Flags `AMBIGUOUS`, preserves competing candidate vehicle IDs in `competing_vehicle_ids`, marks `needs_review=True`. | **PASSED** |
| **7** | `test_scenario_7_conflicting_ocr_observations_for_one_vehicle` | Same vehicle observed across frames: frame 1 reads "س ع د 1234" (0.72), frame 2 reads "س ع ر 1234" (0.91). | Highest confidence reading selected as primary; `has_conflicting_reads=True`; alternatives preserved in `alternative_readings`. | **PASSED** |
| **8** | `test_scenario_8_temporary_vehicle_detection_loss_and_recovery` | Vehicle detected in frames 1–2; missed in frames 3–4; reappears in frame 5 within `max_lost_frames`. | Existing track ID recovered on reappearance; does not spawn duplicate track; track continuity preserved. | **PASSED** |

**Test Execution Result:** `8 passed in 0.04s` (captured in `artifacts/missions/mission_010/integration_test_results.txt`).
**Full Suite Regression Result:** `101 passed, 1 warning in 8.89s` (zero regressions across all past milestones).

---

## 7. Gate Footage Chronological Validation Analysis

### 7.1 Chronological Gap & Trajectory Feasibility Audit
All 340 gate frame captures from `D:/TruckEntry` were parsed chronologically using only the timestamp prefix `YYYYMMDD_HHMMSS_mmm`:

```text
Total parsed gate frames: 340
Start time: 2026-10-15 01:39:42.111000
End time: 2026-10-15 22:35:17.659000
Total elapsed time span: 20.93 hours
Minimum inter-frame delta: 3.221s
Maximum inter-frame delta: 5134.314s (1.43 hours)
Median inter-frame delta: 116.1s (~2 minutes)
Frames with delta < 5.0s: 4 (1.2%)
Frames with delta < 30.0s: 71 (20.9%)
Frames with delta > 60.0s: 228 (67.1%)
Frames with delta > 300.0s: 68 (20.0%)
```

### 7.2 Critical Empirical Finding Regarding Gate Footage
> [!IMPORTANT]
> **The available gate images in `D:/TruckEntry` are discrete, event-triggered vehicle snapshot captures spanning nearly 21 hours—NOT continuous 25/30 FPS video.**
> 
> With a median interval of 116.1 seconds between captures and over 67% of frames separated by more than a minute, consecutive images represent completely different vehicles arriving hours apart. 
> 
> Attempting to evaluate multi-frame trajectory persistence on these still images as if they were consecutive video frames would be fundamentally invalid engineering and would manufacture a false video benchmark.
> 
> The system handles these real-world captures with complete integrity:
> 1. Because $\Delta t > 15.0\text{s}$ between most captures, the vehicle tracker correctly expires the preceding vehicle track and instantiates a new independent track for the newly arriving vehicle.
> 2. Where rare double-triggers or close queues occur ($\Delta t < 15.0\text{s}$), the tracker clusters them into continuous transit events.
> 3. All 340 gate images were successfully processed, producing 322 plate associations and consolidating 201 distinct vehicle transit events.

### 7.3 Gate Validation Outputs Summary
The validation script `scripts/run_mission_010_validation.py` generated the four required CSV artifacts in `artifacts/missions/mission_010/`:
- [vehicle_detection_results.csv](file:///c:/Users/MARWAN/computer%20vision/artifacts/missions/mission_010/vehicle_detection_results.csv): 340 rows documenting per-frame vehicle detections, classes, confidence scores, and bounding boxes.
- [plate_vehicle_association_results.csv](file:///c:/Users/MARWAN/computer%20vision/artifacts/missions/mission_010/plate_vehicle_association_results.csv): 322 rows documenting plate detections, $\text{IoP}$ containment ratios, association status, association confidence, and competing candidate IDs.
- [vehicle_track_analysis.csv](file:///c:/Users/MARWAN/computer%20vision/artifacts/missions/mission_010/vehicle_track_analysis.csv): 201 rows documenting vehicle transit tracks, time spans, continuity classifications, and gap notes.
- [event_examples.csv](file:///c:/Users/MARWAN/computer%20vision/artifacts/missions/mission_010/event_examples.csv): 50 representative consolidated vehicle transit events demonstrating the updated data model.

---

## 8. Acceptance Criteria Compliance Matrix

| Acceptance Criterion | Verification Status | Verification Evidence / Implementation Reference |
| :--- | :---: | :--- |
| **1. Vehicle detection is integrated through a replaceable interface.** | **MET** | Defined `BaseVehicleDetector` interface in `vehicle_intelligence/core/interfaces.py`. Implemented `PyTorchYoloVehicleDetector` in `vehicle_intelligence/detectors/pytorch_yolo.py` with configurable weights, thresholds, and COCO mapping. |
| **2. Vehicle tracking has independent IDs.** | **MET** | `CentroidIoUTracker` maintains independent integer track IDs (`track_id`) based on vehicle body dimensions and normalized centroid distances, completely decoupled from plate detection. |
| **3. Plate-to-vehicle association is implemented and confidence-aware.** | **MET** | Implemented `PlateVehicleAssociator` in `vehicle_intelligence/association/plate_vehicle_associator.py` computing containment ($\text{IoP}$), horizontal alignment, vertical placement, and scale ratio. |
| **4. Ambiguous associations are preserved for review.** | **MET** | When candidate score delta $\Delta S < 0.15$ or multiple vehicles enclose a plate, `PlateVehicleAssociator` flags `AssociationStatus.AMBIGUOUS`, records `competing_vehicle_ids`, and sets `needs_review=True` without forcing arbitrary assignment. |
| **5. Vehicle transit events contain required evidence and metadata.** | **MET** | Upgraded `VehicleTransitEvent` in `vehicle_intelligence/core/domain.py` and `MultiFrameEvidenceAggregator` supporting vehicle track ID, class, confidence, plate reading, association status, best evidence frames, alternative readings, and review triggers. |
| **6. All eight synthetic scenarios pass.** | **MET** | All 8 scenarios in `tests/test_mission_010_vehicle_tracking.py` pass cleanly in 0.04s. Outputs recorded in `artifacts/missions/mission_010/integration_test_results.txt`. |
| **7. Existing ALPR regression tests continue to pass.** | **MET** | Full repository test suite executed: `101 passed, 1 warning in 8.89s` with zero regressions across any existing test suite. |
| **8. Validation limitations are documented honestly.** | **MET** | Documented exact empirical timing distribution of `D:/TruckEntry` captures (median $\Delta t = 116.1$s). Explicitly acknowledged that discrete snapshot triggers cannot establish continuous video trajectories. |
