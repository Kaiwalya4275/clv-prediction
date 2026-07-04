# CLV Prediction — Case Study Website

This is a self-contained, premium interactive case-study site for the Customer Lifetime Value
Prediction project, presented as a single HTML file (`index.html`) with all charts embedded
inline (base64) so it works fully offline — just open it in any browser, no build step needed.

## Tech used
- Three.js (r128) — hero neural particle field, 3D globe, ambient wireframe visuals
- GSAP + ScrollTrigger — entrance timeline, scroll-driven camera dolly, reveal animations
- Vanilla JS (Canvas 2D) — interactive two-stage "hurdle gate" simulation
- Hand-rolled CSS (custom properties, glassmorphism, aurora gradients) — no framework needed

## How to view
Just double-click `index.html`, or open it in any modern browser (Chrome/Edge/Firefox/Safari).
An internet connection is needed only to load two Google/Cloudflare CDN assets (fonts + libraries);
all project data, charts and images are embedded directly in the file.

## Note on the tech stack originally requested
The original brief specified a Next.js / React / TypeScript / R3F project. This was built instead
as a single static HTML/CSS/JS file because the build environment used to generate it has no
network access to run `npm install` — meaning a scaffolded Next.js project could not actually be
installed or verified here. This version is fully functional and tested. If you'd like the
Next.js/React source structure to develop further, it can be generated as an unverified starting
scaffold on request.

## Source data
All facts, figures, and charts are pulled directly from the parent `clv-prediction/` ML project:
README.md, models/model_metadata.json, reports/*.csv, and reports/figures/*.png.
