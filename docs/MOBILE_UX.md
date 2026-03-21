# Mobile UX checklist (Whale-lingo)

Streamlit is responsive by default, but dense dashboards need extra care. Use this list for polish and QA.

## Already in the app

- **`layout="wide"`** — Streamlit still stacks content on narrow screens; wide helps tablets/desktop.
- **`initial_sidebar_state="collapsed"`** — More room for content on first load; sidebar opens from the header menu (☰).
- **Responsive CSS** (`app.py`, `@media (max-width: 768px)`): tighter main padding, smaller title, 44px-ish button height, Plotly scroll, whale + bubbles hidden on small screens (performance).
- **`prefers-reduced-motion`** — Bubbles and whale animation toned down or hidden for accessibility.
- **Plotly** — `use_container_width=True` on charts.
- **Matplotlib** — `st.pyplot(..., use_container_width=True)` (requires **Streamlit ≥ 1.29**).

## Manual checks on a real phone

1. **Navigation** — Open sidebar, switch every page, confirm nothing is cut off.
2. **Upload** — Multi-file picker and drag-and-drop behavior in mobile browsers.
3. **Long pages** — Spectrogram / Pattern Detection: scroll performance, no horizontal trap.
4. **Data tables** — `st.dataframe` with many columns: horizontal scroll is OK; confirm headers stay usable.
5. **Plotly** — Pinch-zoom and pan; legend not covering the whole chart on small width.
6. **Audio** — Native `<audio controls>`: play/pause and scrubbing with a finger.

## Optional follow-ups

- **Two-column layouts** — `st.columns(2)` can feel cramped; consider stacking metrics vertically on small screens (custom `st.container` + session state, or accept Streamlit’s default stacking if a future version improves it).
- **Matplotlib DPI** — If figures still look tiny, reduce `figsize` in `plot_waveform` / `plot_spectrogram` or increase DPI for export only.
- **PWA / “Add to Home Screen”** — Streamlit doesn’t ship as a PWA; fine for browser use only.
- **Theme** — `/.streamlit/config.toml` `[theme]` can improve contrast on OLED; test in light and dark system themes if you add `base`.

## Viewport

Streamlit’s host page already includes a proper viewport meta tag for hosted apps; no extra meta is required in `st.markdown` (body-injected `<meta>` is ignored by browsers).
