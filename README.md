# BNNT Profile Tagger

A Panel-based GUI for tagging boron nitride nanotube (BNNT) height profiles extracted with [Gwyddion](http://gwyddion.net/).

## Features

- Load `.gwy` or `.txt` (Gwyddion profile export) files
- Interactive AFM image viewer — click a line to select and tag it
- **Cycle mode**: repeating tag pattern per BNNT (e.g. longitudinal → cross_1 → cross_2 → cross_3)
- **Block mode**: all longitudinals first, then cross sections grouped by tube
- Editable profile table — change tags/BNNT numbers inline, delete rows
- Plot grouped profiles and height distributions with configurable bin count
- Export tagged profiles to CSV
- Save all plots as PNG files

## Installation

### Option 1 — pip from GitHub (recommended)

```bash
pip install git+https://github.com/crisavp/bnnt-tagger.git
```

Then launch from anywhere:

```bash
bnnt-tagger
```

### Option 2 — conda environment

```bash
git clone https://github.com/crisavp/bnnt-tagger.git
cd bnnt-tagger
conda env create -f environment.yml
conda activate bnnt-tagger
pip install -e .
```

### Option 3 — pip from source

```bash
git clone https://github.com/YOUR_USERNAME/bnnt-tagger.git
cd bnnt-tagger
pip install .
```

## Usage

```bash
bnnt-tagger                        # open with file browser
bnnt-tagger path/to/file.gwy       # load a .gwy file directly
bnnt-tagger path/to/profiles.txt   # load a Gwyddion .txt export directly
```

The app opens in your browser at `http://localhost:5006/bnnt_tagger`.

> **SSH / remote server**: forward port 5006 in VS Code (Ports tab) or with  
> `ssh -L 5006:localhost:5006 user@host`, then open `http://localhost:5006/bnnt_tagger`.

## Workflow

1. **Load** a `.gwy` file or Gwyddion profile export (`.txt`)
2. **Set tagging mode** — Cycle or Block — and click **Tag**
3. For `.gwy` files: click lines on the AFM image to inspect/edit individual tags; hover for details
4. Fine-tune tags directly in the profile table or use the Tag Editor panel
5. **Plot profiles** and **Distributions** to review
6. **Export CSV** or **Save plots (PNG)**

## Tagging modes

| Mode      | Pattern                                                                                      |
| --------- | -------------------------------------------------------------------------------------------- |
| **Cycle** | Repeating sequence: `longitudinal, cross_1, cross_2, cross_3, longitudinal, …`               |
| **Block** | All longitudinals first (one per tube), then cross groups: `L1, L2, L3, L4, S1₁S2₁S3₁, S1₂…` |

## Requirements

- Python ≥ 3.9
- panel ≥ 1.3, bokeh ≥ 3.3, numpy, pandas, scipy, matplotlib, gwyfile