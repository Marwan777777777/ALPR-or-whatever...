# Egyptian License Plate Annotation Guide (Gate CCTV Ground Truth)

This guide establishes the standardized ground truth annotation protocol for Egyptian license plates captured by the gate CCTV camera at `TruckEntry`.

---

## 1. Annotation Status Classes

Every sampled image must be assigned exactly one of the following statuses:

| Status | Definition | Action Required |
|---|---|---|
| `readable` | All plate characters can be confidently established by the human annotator. | Enter all letters in `true_letters` and all numerals in `true_digits`. |
| `partially_readable` | The plate is visible, but one or more characters are obscured, motion-blurred, or uncertain. | Enter established characters and use `?` for uncertain positions (e.g. `س ? ج`). |
| `unreadable` | A plate is physically present on the vehicle, but severe blur, occlusion, or glare prevents confident character identification. | Leave `true_letters` and `true_digits` empty. Mark `plate_visible_to_eye = yes` or `partial`. |
| `no_plate` | No license plate is visible in the frame (e.g. vehicle rear missing plate, camera angle obstructed, or empty gate). | Leave `true_letters` and `true_digits` empty. Mark `plate_visible_to_eye = no`. |
| `multiple_plates` | More than one plate is visible in the frame (e.g. two vehicles in frame, or truck tractor + trailer plates). | Transcribe the primary foreground vehicle plate and record details in `notes`. |

> [!IMPORTANT]
> **Zero Guessing Rule:** Never guess blurred or occluded characters. If a letter or digit cannot be independently verified, mark `?` or set status to `unreadable`. Uncertain characters must **never** be turned into confident ground truth.

---

## 2. Character Set & Unicode Standards (The 27 Classes)

The Egyptian license plate specification recognizes exactly **17 Arabic letters** and **10 Eastern Arabic numerals**:

### A. Arabic Letters (17 Classes)
*Entered in standard Egyptian plate reading order (Right to Left: e.g. first letter on right, second in middle, third on left).*

| Class ID | Glyph | Unicode Codepoint | Official Name | Keypad & Normalization Rule |
|:---:|:---:|:---:|---|---|
| **0** | `أ` | `U+0623` | Arabic Letter Alef with Hamza Above | Standard Alef with Hamza |
| **1** | `ب` | `U+0628` | Arabic Letter Beh | Beh |
| **2** | `ج` | `U+062C` | Arabic Letter Jeem | Jeem |
| **3** | `د` | `U+062F` | Arabic Letter Dal | Dal |
| **4** | `ر` | `U+0631` | Arabic Letter Reh | Reh |
| **5** | `س` | `U+0633` | Arabic Letter Seen | Seen |
| **6** | `ص` | `U+0635` | Arabic Letter Sad | Sad |
| **7** | `ط` | `U+0637` | Arabic Letter Tah | Tah |
| **8** | `ع` | `U+0639` | Arabic Letter Ain | Ain |
| **9** | `ف` | `U+0641` | Arabic Letter Feh | Feh |
| **10** | `ق` | `U+0642` | Arabic Letter Qaf | Qaf |
| **11** | `ل` | `U+0644` | Arabic Letter Lam | Lam |
| **12** | `م` | `U+0645` | Arabic Letter Meem | Meem |
| **13** | `ن` | `U+0646` | Arabic Letter Noon | Noon |
| **14** | `ھ` | `U+06BE` | Arabic Letter Heh Doachashmee | **Critical:** EALPR uses `U+06BE` (`ھ`). Standard keyboard `ه` (`U+0647`) is automatically normalized to `U+06BE`. |
| **15** | `و` | `U+0648` | Arabic Letter Waw | Waw |
| **16** | `ى` | `U+0649` | Arabic Letter Alef Maksura | **Critical:** EALPR uses `U+0649` (`ى` without dots). Standard keyboard `ي` (`U+064A`) is automatically normalized to `U+0649`. |

### B. Eastern Arabic Numerals (10 Classes)
*Entered in standard numeral reading order (Left to Right: e.g. highest place value first).*

| Class ID | Glyph | Unicode Codepoint | ASCII Digit | Normalization Rule |
|:---:|:---:|:---:|:---:|---|
| **17** | `٠` | `U+0660` | 0 | `0` auto-converts to `٠` |
| **18** | `١` | `U+0661` | 1 | `1` auto-converts to `١` |
| **19** | `٢` | `U+0662` | 2 | `2` auto-converts to `٢` |
| **20** | `٣` | `U+0663` | 3 | `3` auto-converts to `٣` |
| **21** | `٤` | `U+0664` | 4 | `4` auto-converts to `٤` |
| **22** | `٥` | `U+0665` | 5 | `5` auto-converts to `٥` |
| **23** | `٦` | `U+0666` | 6 | `6` auto-converts to `٦` |
| **24** | `٧` | `U+0667` | 7 | `7` auto-converts to `٧` |
| **25** | `٨` | `U+0668` | 8 | `8` auto-converts to `٨` |
| **26** | `٩` | `U+0669` | 9 | `9` auto-converts to `٩` |

---

## 3. Reading Order & Field Separation

Egyptian license plates have two distinct sections separated by visual layout:
1. **Arabic Letters (`true_letters`):**
   - Transcribed in **reading order from right to left**.
   - Example: If the rightmost letter is `س`, middle is `ق`, leftmost is `ج`, enter: `سقج`.
2. **Arabic Numerals (`true_digits`):**
   - Transcribed in **reading order from left to right**.
   - Example: If the digits are four-digit `8568`, enter: `٨٥٦٨`.

The labeling tool provides dedicated inputs for both fields. The scoring script merges them canonically as:
$$\text{Full Plate Text} = \text{true\_letters} + \text{" "} + \text{true\_digits}$$

---

## 4. Uncertainty & Anomaly Notation

- **Uncertain Characters:** Use the ASCII question mark `?` (`U+003F`) in place of any obscured character (e.g. `true_letters = س ? ج`, `true_digits = ٨ ٥ ? ٨`).
- **No Plate:** If no plate exists on the vehicle, set `status = no_plate` and `plate_visible_to_eye = no`. Leave letters and digits blank.
- **Occluded / Blurry Plate:** Set `status = unreadable`. Leave letters and digits blank.
- **Multiple Plates:** Set `status = multiple_plates`. Transcribe the primary front plate and write note in `notes` (e.g. `trailer plate also visible: ط ص د 1234`).

---

## 5. Exporting and Saving Annotations

1. In the HTML tool, complete the annotations.
2. Click **Export Completed Annotations**.
3. Save the downloaded file to `artifacts/triage/completed_annotations.csv`.
4. Run the validation and scoring suite:
   ```bash
   python scripts/score_labeled_triage.py artifacts/triage/completed_annotations.csv
   ```
