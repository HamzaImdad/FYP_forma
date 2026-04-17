# Root Directory Cleanup Log — 2026-04-17

All files moved, nothing deleted. Original location: project root (`/`).

## Moved to `docs/screenshots/` (61 files, ~30 MB)

Dev iteration screenshots — none referenced by application code.

| # | Filename | Type |
|---|----------|------|
| 1 | a-about.jpeg | UI screenshot |
| 2 | a-dev.jpeg | UI screenshot |
| 3 | a-home-fixed.jpeg | UI screenshot |
| 4 | a-quiet-honest.jpeg | UI screenshot |
| 5 | about-hero.jpg | Hero draft |
| 6 | about-with-new-footer.jpeg | UI screenshot |
| 7 | cinematic-grid.jpg | Design exploration |
| 8 | cinematic-preview.jpg | Design exploration |
| 9 | cinematic-preview2.jpg | Design exploration |
| 10 | features-block-03.jpg | Features page draft |
| 11 | features-desktop-hero.jpg | Features page draft |
| 12 | features-full-v2.jpg | Features page draft |
| 13 | features-full.png | Features page draft |
| 14 | features-v2.jpg | Features page draft |
| 15 | final-features-hero.jpg | Features page draft |
| 16 | footer-marquee.jpeg | Footer draft |
| 17 | footer-top.jpeg | Footer draft |
| 18 | footer-watermark.jpeg | Footer draft |
| 19 | footer-with-marquee.jpeg | Footer draft |
| 20 | forma-dashboard-after-login.png | Dashboard screenshot |
| 21 | forma-final.jpeg | UI screenshot |
| 22 | forma-hero-fixed.jpeg | Hero draft |
| 23 | forma-hero-settled.jpeg | Hero draft |
| 24 | forma-hero.jpeg | Hero draft |
| 25 | forma-login-page.png | Login screenshot |
| 26 | home-check.jpeg | Home page draft |
| 27 | home-dev.jpeg | Home page draft |
| 28 | home-full-flask.jpeg | Home page draft |
| 29 | home-full.jpeg | Home page draft |
| 30 | home-hero-v2.jpg | Hero draft |
| 31 | home-hero.jpg | Hero draft |
| 32 | home-section-stats.jpg | Home page draft |
| 33 | howitworks-hero.jpg | How It Works draft |
| 34 | login-v2.jpg | Login draft |
| 35 | login-v3-seamless.jpg | Login draft |
| 36 | nav-inline.png | Nav screenshot |
| 37 | page-chatbot.jpeg | Chatbot page screenshot |
| 38 | page-dev.jpeg | Dev page screenshot |
| 39 | page-milestones.jpeg | Milestones page screenshot |
| 40 | page-plans.jpeg | Plans page screenshot |
| 41 | page-voice.jpeg | Voice page screenshot |
| 42 | phase1-hello-vite-dev.png | Phase 1 dev screenshot |
| 43 | phase1-hello.png | Phase 1 screenshot |
| 44 | phase1-hero-fixed.png | Phase 1 screenshot |
| 45 | phase1-hero.png | Phase 1 screenshot |
| 46 | phase2-exercises-grid.jpeg | Phase 2 screenshot |
| 47 | phase2-hero-viewport.jpeg | Phase 2 screenshot |
| 48 | phase2-home-final.jpeg | Phase 2 screenshot |
| 49 | phase2-home-fixed.jpeg | Phase 2 screenshot |
| 50 | phase2-home-full.jpeg | Phase 2 screenshot |
| 51 | phase2-home-full2.jpeg | Phase 2 screenshot |
| 52 | phase2-home-hero.jpeg | Phase 2 screenshot |
| 53 | phase2-login.png | Phase 2 screenshot |
| 54 | phase2-mid-sections.jpeg | Phase 2 screenshot |
| 55 | phase2-signup.png | Phase 2 screenshot |
| 56 | phase3-howitworks.png | Phase 3 screenshot |
| 57 | phase4-features-full.png | Phase 4 screenshot |
| 58 | phase4-features-hero.png | Phase 4 screenshot |
| 59 | phase5-widget-closed.png | Phase 5 screenshot |
| 60 | phase5-widget-open.png | Phase 5 screenshot |
| 61 | test1-preview.png | Test screenshot |

## Moved to `docs/` (4 files)

| File | Original Location | Reason |
|------|-------------------|--------|
| `FEATURES.md` | `/FEATURES.md` (git tracked) | Spec document, not referenced by app |
| `forma.db` | `/forma.db` (gitignored) | 0-byte empty DB, never referenced |
| `exervision.db.bak-20260416-220547` | `/` (gitignored) | DB backup, not referenced by app code |

## .gitignore fixes

- Added `*.jpg` to root screenshot exclusion (was missing — 15 `.jpg` files slipped through)
- Added `!app/static/images/cinematic/*.jpg` exception for cinematic hero images
- Added `docs/screenshots/` exclusion

## NOT moved (confirmed dependencies)

| Item | Why it stays at root |
|------|---------------------|
| `app/` | Core application — server.py computes PROJECT_ROOT from here |
| `src/` | Core library — imported by app via sys.path |
| `models/` | MediaPipe + DTW templates — hardcoded paths in server.py + config.py |
| `exervision.db` | Hardcoded in server.py line 111 |
| `.env` | Loaded by python-dotenv from root |
| `reports/` | Referenced by server.py, train_bilstm.py, pipeline.py, evaluate_models.py |
| `data/` | Training data (34 GB, gitignored) |
| `_archive/` | Already organized archive (1.3 GB) |
| `scripts/` | Training/pipeline scripts |
| `tests/` | Test files |
| `Dockerfile`, `nixpacks.toml` | Deploy config |
| `requirements.txt`, `requirements-deploy.txt` | Python dependencies |
| `runtime.txt` | Python version spec |
| `README.md`, `pytest.ini` | Root-level essentials |
