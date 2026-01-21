docs/V1_RELEASE_CHECKLIST.md
# v1.0.0 Release Checklist

## 🔴 Blocking UX Issues (Must Fix)
- [ ] **Timeline Hit-Testing:** Can I grab the trim handle on the first try with a trackpad?
- [ ] **Split View Visibility:** Is the split line visible on a pure white AND pure black video?
- [ ] **File Path Truncation:** Does a long filename break the layout in the settings panel?
- [ ] **Job Queue Scroll:** Does the queue scroll correctly when 50 items are added?
- [ ] **Process Cancellation:** Does clicking "Stop" actually kill the Python process immediately?
- [ ] **App Close:** Does closing the window orphan any zombie `python.exe` processes?

## 🟡 Quality of Life (Must Pass 80%)
- [ ] **Tooltips:** Do all icon-only buttons have hover tooltips?
- [ ] **Tab Order:** Can I tab through the settings panel inputs logically?
- [ ] **Default Save Path:** Does it default to the source folder (good) or System32 (bad)?
- [ ] **Error Toast:** Do errors show a human-readable toast ("Out of Memory") vs a raw code ("Exit 1")?
- [ ] **Reset Layout:** Does the "View -> Reset Layout" button work?

## 🟢 Visual Polish (Deferrable to v1.1)
- [ ] **Custom Scrollbars:** Are scrollbars styled to match the theme? (Nice to have).
- [ ] **Button Active States:** Do buttons have a distinct "Down" state (click feedback)?
- [ ] **Loading Skeleton:** Do we show a skeleton while the model loads?
- [ ] **Icon Alignment:** Are all icons perfectly optically centered?

## 🏁 Go/No-Go Criteria
1. **Zero Data Loss:** The app never deletes the source file.
2. **Deterministic Output:** The same input + same settings = exact same binary output.
3. **No Zombies:** App exit cleans up 100% of child processes.
4. **Offline Capable:** App runs without internet (after initial asset download).

**Signed off by:** Product Lead & Engineering Lead
**Target Date:** [Insert Date]