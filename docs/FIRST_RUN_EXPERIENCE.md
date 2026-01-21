docs/FIRST_RUN_EXPERIENCE.md
# First Run Experience (FRE) Strategy

## Philosophy: "Zero Barriers to Work"
VideoForge is a professional tool. Our users are editors, archivists, and engineers. They do not read "Welcome" carousels. They learn by poking the interface.

**Goal:** The user should successfully process their first second of video within 30 seconds of launch without reading a modal.

## 1. The Empty State
The initial state of the application must not be a blank grey screen. It must be an invitation.

**Center Viewport (Preview):**
- **Visual:** Subtle, large SVG icon of a "Plus" or "Import".
- **Text:** "Drag and Drop Video or Image to Begin" (Sans-serif, Grey 500).
- **Subtext:** "Supports MP4, MKV, MOV, PNG, JPG" (Grey 700).
- **Behavior:** Clicking anywhere in this zone opens the system file picker.

**Right Panel (Queue):**
- **Visual:** "No Active Jobs".
- **Text:** "Exported files will appear here."

## 2. Default Configuration (Safe Mode)
The app must launch with safe, high-probability-of-success defaults.
- **Model:** `RealESRGAN_x4plus` (Reliable, general purpose).
- **Scale:** `4x` (Standard).
- **Crop:** Disabled.
- **FPS:** Native (No interpolation).
- **Format:** MP4 (h.264) / PNG.

## 3. The "Golden Path" Hints
Instead of popups, use subtle UI states to guide the eye.
1. **Input:** When no file is loaded, the "Start Processing" button is Disabled (Grey).
2. **Loaded:** Once a file is dropped, the "Start Processing" button turns **Brand Green**.
3. **Action:** The bright green button becomes the visual anchor, screaming "Click me next."

## 4. Error Recovery (The "Panic" Button)
If the Python engine fails to start on first run (common issue with AV/Firewalls):
- Do NOT show a stack trace immediately.
- Show a toast: "Engine Initialization Failed."
- Change the "Start" button to "Retry Engine" (Yellow).
- Only show the log panel if the user clicks "Retry" and it fails again.

## 5. What We Do NOT Do
- **No Account Creation:** Local-first means local-first.
- **No "Tour":** Do not dim the screen and highlight buttons.
- **No "Tip of the Day":** It's annoying.
- **No Watermarks:** Even on free/trial versions (if applicable). Trust is currency.

