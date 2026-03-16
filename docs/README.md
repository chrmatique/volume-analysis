# Codebase documentation

Visual codebase map and file reference for **mkt-noise-analysis**.

## Viewing

Open `index.html` in a web browser:

```bash
# From project root
start docs/index.html      # Windows
open docs/index.html       # macOS
xdg-open docs/index.html   # Linux
```

Or serve locally (required for some browsers due to CORS on local files):

```bash
cd docs && python -m http.server 8000
# Then open http://localhost:8000
```

## Contents

- **index.html** — Visual codebase map: module dependency flow, file tree, module overview cards, Cargo dependency table, data flow diagram, AppState field reference
- **file-reference.html** — File-by-file descriptions for every Rust source file, grouped by module

## Requirements

- Modern browser (Chrome, Firefox, Edge, Safari)
- Internet connection for Mermaid.js CDN (diagram rendering in index.html)
