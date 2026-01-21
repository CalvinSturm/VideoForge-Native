docs/UX_LAWS.md
# VideoForge UX Laws

## 1. The Law of Telemetry vs. Intent
**Rule:** Never conflate what a file *is* with what the user *wants* it to be.
**Rationale:** Professional workflows require absolute clarity between source data (immutable facts) and export targets (mutable goals). Ambiguity causes errors.
**Correct:** Display "Source: 1080p (Read-only)" separate from "Target: 4K (Editable)".
**Violation:** A single dropdown showing "1080p" that acts as both a status indicator and a setting.

## 2. The Law of Explicit State
**Rule:** Every control must have a binary, unambiguous state; "maybe" is not a state.
**Rationale:** In a dark UI, subtle greys look like disabled states. Active states must use high-contrast brand colors to be scannable from a distance.
**Correct:** Active toggle is solid Brand Green with black text. Inactive is transparent with grey text.
**Violation:** Active state is Grey 700; Inactive state is Grey 800.

## 3. The Law of Destructive Friction
**Rule:** Actions that destroy work or close views must require physical intent or offer immediate recovery.
**Rationale:** Misclicks happen. Closing a panel without a way to get it back breaks trust. Deleting a job without undo is hostile.
**Correct:** Closing a panel moves it to a "View" menu. Delete requires a distinct, red-hover interaction or confirmation.
**Violation:** Clicking "X" on a panel permanently removes it from the DOM with no menu option to restore it.

## 4. The Law of Deterministic Layouts
**Rule:** The interface layout must persist exactly as left between sessions.
**Rationale:** Professionals build muscle memory. If panels shift or reset on restart, the tool feels fragile and amateur.
**Correct:** App opens with the exact window size, panel split ratios, and scroll positions from the last session.
**Violation:** App resets to "Default View" every time it launches.

## 5. The Law of Fitts’s Precision
**Rule:** Interactive elements must have a hit area larger than their visible footprint, especially in timelines.
**Rationale:** A 1px timeline playhead line is visible but unclickable. Frustration accumulates with every missed click.
**Correct:** A 1px visual line has a 12px transparent padding for mouse interaction.
**Violation:** Requiring pixel-perfect mouse positioning to grab a trim handle.

## 6. The Law of Passive Onboarding
**Rule:** Teach through affordance and empty states, never through modals or wizards.
**Rationale:** Pros install software to use it, not to read about it. Modals block intent. Empty states invite intent.
**Correct:** An empty job queue says "Drag media here to begin."
**Violation:** A 5-step "Welcome to VideoForge" carousel overlaying the interface on first launch.

## 7. The Law of Synchronized Reality
**Rule:** In split or comparison views, temporal and spatial manipulation must occur exactly in lockstep.
**Rationale:** If the "After" view lags the "Before" view by even one frame, the user cannot trust the enhancement.
**Correct:** Scrubbing the timeline updates source and result frames in the same render cycle.
**Violation:** The source video plays while the result video buffers or loads asynchronously.

## 8. The Law of No "Apply"
**Rule:** Tools should be modeless or auto-saving; explicit "Apply" buttons should be reserved for expensive batch commits.
**Rationale:** "Apply" buttons separate the adjustment from the result, slowing down the feedback loop.
**Correct:** Changing a crop region immediately queues the new parameters. Toggling "Crop" enables the edit mode.
**Violation:** Adjusting crop handles, then needing to click a detached "Apply" button to see the change take effect.