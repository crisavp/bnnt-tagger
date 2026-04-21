#!/usr/bin/env python3
"""
BNNT Profile Tagger — Panel web interface v3
=============================================
Supports .gwy and .txt (Gwyddion export) input files.

Run with:
    panel serve --show bnnt_tagger.py
    panel serve --show bnnt_tagger.py --args path/to/file.gwy
    panel serve --show bnnt_tagger.py --args path/to/file.txt
"""

import sys, os, csv, re, io
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy.ndimage import map_coordinates
from scipy.signal import find_peaks
import gwyfile

import panel as pn
from bokeh.plotting import figure
from bokeh.models import (
    ColumnDataSource, ColorBar, LinearColorMapper,
    Range1d, Label, HoverTool, TapTool, Line as BokehLine,
)
from bokeh.events import Tap
from bokeh.palettes import Inferno256
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_CSS = """
.bk-root .bk-btn, button.bk-btn {
    padding: 2px 6px !important;
    font-size: 12px !important;
    line-height: 1.3 !important;
    min-height: 0 !important;
}
.bk-root .bk-input, input.bk-input, select.bk-input {
    font-size: 12px !important;
    padding: 1px 5px !important;
    height: 28px !important;
    min-height: 0 !important;
}
.bk-root label, .bk-root .bk-widget-form-label {
    font-size: 11px !important;
    margin-bottom: 1px !important;
}
.card-header { font-size: 13px !important; }
.tabulator .tabulator-cell { font-size: 11px !important; padding: 2px 4px !important; }
.tabulator .tabulator-col-title { font-size: 11px !important; }
"""

pn.extension('tabulator', notifications=True, sizing_mode='stretch_width', raw_css=[_CSS])

# ── Configuration ──────────────────────────────────────────────────────────────

DEFAULT_CYCLE = ['longitudinal', 'cross_1', 'cross_2', 'cross_3']
VALID_TAGS    = ['longitudinal', 'cross_1', 'cross_2', 'cross_3', 'cross_4', 'cross_5']

TAG_COLORS = {
    'longitudinal': '#FF4444',
    'cross_1':      '#44AAFF',
    'cross_2':      '#44FF88',
    'cross_3':      '#FFAA44',
    'cross_4':      '#FF88FF',
    'cross_5':      '#88FFFF',
    'unassigned':   '#AAAAAA',
}

PLOT_COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
    '#9467bd', '#8c564b', '#e377c2', '#17becf',
]

PROFILE_POINTS = 256
HIST_COLOR     = '#F4A460'

# ── Pure helper functions ──────────────────────────────────────────────────────

def load_gwy(fp):
    obj          = gwyfile.load(fp)
    d            = obj['/0/data']
    xres, yres   = d['xres'], d['yres']
    xreal, yreal = d['xreal'], d['yreal']
    image        = d['data'].reshape(yres, xres).copy()
    if d['si_unit_z']['unitstr'] == 'm':
        image *= 1e9
    raw  = obj['/0/select/line']['data']
    n    = len(raw) // 4
    if n == 0:
        raise ValueError("No line profiles found. Draw profiles in Gwyddion and save.")
    lines = [dict(zip(['x1','y1','x2','y2'], raw[i*4:i*4+4])) for i in range(n)]
    return image, lines, xreal, yreal, xres, yres


def parse_txt(fp):
    with open(fp) as f:
        content = f.readlines()
    header             = content[0]
    all_profile_tokens = re.findall(r'Profile\s+\d+', header)
    if not all_profile_tokens:
        raise ValueError("Cannot detect profiles in text file. Expected Gwyddion export format.")

    if ';' in header:
        n_profiles = len(all_profile_tokens) // 2
        raw = [{'x': [], 'y': []} for _ in range(n_profiles)]
        for line in content[3:]:
            tokens = line.rstrip('\r\n').split(';')
            for i in range(n_profiles):
                xi, yi = i * 2, i * 2 + 1
                if xi < len(tokens) and yi < len(tokens):
                    xv, yv = tokens[xi].strip(), tokens[yi].strip()
                    if xv and yv and xv != '-' and yv != '-':
                        try:
                            raw[i]['x'].append(float(xv))
                            raw[i]['y'].append(float(yv))
                        except ValueError:
                            pass
    else:
        n_profiles = len(all_profile_tokens)
        raw = [{'x': [], 'y': []} for _ in range(n_profiles)]
        for line in content[3:]:
            tokens = line.split()
            for i in range(n_profiles):
                xi, yi = i * 2, i * 2 + 1
                if xi < len(tokens) and yi < len(tokens):
                    xv, yv = tokens[xi], tokens[yi]
                    if xv != '-' and yv != '-':
                        try:
                            raw[i]['x'].append(float(xv))
                            raw[i]['y'].append(float(yv))
                        except ValueError:
                            pass

    profiles = []
    for p in raw:
        if p['x']:
            x_um = np.array(p['x']) * 1e6
            y_nm = np.array(p['y']) * 1e9
            profiles.append((x_um.tolist(), y_nm.tolist()))
    if not profiles:
        raise ValueError("No valid profile data found in text file.")
    return profiles


def extract_profile(image, line, xreal, yreal, xres, yres):
    x1p = line['x1'] / xreal * xres
    y1p = line['y1'] / yreal * yres
    x2p = line['x2'] / xreal * xres
    y2p = line['y2'] / yreal * yres
    xs  = np.linspace(x1p, x2p, PROFILE_POINTS)
    ys  = np.linspace(y1p, y2p, PROFILE_POINTS)
    h   = map_coordinates(image, [ys, xs], order=1, mode='nearest')
    L   = np.hypot((line['x2'] - line['x1']) * 1e6,
                   (line['y2'] - line['y1']) * 1e6)
    d   = np.linspace(0, L, PROFILE_POINTS)
    return d.tolist(), h.tolist()


def compute_metric(tag, distances, heights):
    d, h = np.array(distances), np.array(heights)
    return float(d[-1]) * 1000 if tag == 'longitudinal' else float(np.max(h))


def _analyze_longitudinal_data(d, h, baseline_pct, prominence_pct, min_dist):
    """Core peak/valley detection — operates on raw (d_µm, h_nm) arrays."""
    d_nm    = np.array(d) * 1000
    h_arr   = np.array(h)
    baseline = (float(np.percentile(h_arr, baseline_pct))
                if baseline_pct > 0 else float(h_arr.min()))
    h_corr  = h_arr - baseline
    h_range = float(h_corr.max()) - float(h_corr.min())
    prom     = max(0.0, prominence_pct / 100.0 * h_range)
    dist_arg = max(1, int(min_dist))
    pk_idx, pk_props = find_peaks(h_corr, prominence=prom, distance=dist_arg)
    vl_idx, vl_props = find_peaks(-h_corr, prominence=max(0.0, prom * 0.5),
                                  distance=dist_arg)
    peaks = dict(
        pos        = d_nm[pk_idx].tolist(),
        height     = h_corr[pk_idx].tolist(),
        prominence = pk_props['prominences'].tolist(),
    )
    valleys = dict(
        pos        = d_nm[vl_idx].tolist(),
        height     = h_corr[vl_idx].tolist(),
        prominence = vl_props['prominences'].tolist(),
    )
    return d_nm.tolist(), h_corr.tolist(), baseline, peaks, valleys


def analyze_longitudinal(prof_idx, baseline_pct, prominence_pct, min_dist):
    d, h = profiles[prof_idx]
    return _analyze_longitudinal_data(d, h, baseline_pct, prominence_pct, min_dist)


def arrange_rows(figs, n_cols=2):
    rows = []
    for i in range(0, len(figs), n_cols):
        rows.append(pn.Row(*figs[i:i + n_cols]))
    return pn.Column(*rows) if rows else pn.Column()


def _setup_line_tap(pf, srcs):
    """Register figure-level Tap handler; returns a state dict {key} updated on each tap.
    srcs: list of (ColumnDataSource, key). Nearest line by normalised distance wins."""
    state = {'key': None}
    def _on_tap(event):
        tap_x, tap_y = event.x, event.y
        best_key, best_d = None, float('inf')
        for src, key in srcs:
            xs = np.array(src.data['x'])
            ys = np.array(src.data['y'])
            if not len(xs):
                continue
            xr = float(xs.max() - xs.min()) or 1.0
            yr = float(ys.max() - ys.min()) or 1.0
            d = float(np.min(np.hypot((xs - tap_x) / xr, (ys - tap_y) / yr)))
            if d < best_d:
                best_d, best_key = d, key
        state['key'] = best_key
    pf.on_event(Tap, _on_tap)
    return state


def get_session_summary_df():
    rows = []
    for entry in _session:
        for i, (d, p) in enumerate(entry['profiles']):
            tag    = entry['tags'][i]
            metric = (float(np.array(d)[-1]) * 1000 if tag == 'longitudinal'
                      else float(np.max(np.array(p))))
            rows.append({
                'filename':     entry['file_stem'],
                'line_index':   i + 1,
                'bnnt':         entry['bnnts'][i],
                'tag':          tag,
                'metric_nm':    round(metric, 4),
                'metric_label': 'length_nm' if tag == 'longitudinal' else 'peak_height_nm',
            })
    return pd.DataFrame(rows)


def get_session_full_df():
    rows = []
    for entry in _session:
        for i, (d, p) in enumerate(entry['profiles']):
            for dist, ht in zip(d, p):
                rows.append({
                    'filename':    entry['file_stem'],
                    'line_index':  i + 1,
                    'bnnt':        entry['bnnts'][i],
                    'tag':         entry['tags'][i],
                    'distance_um': round(dist, 6),
                    'height_nm':   round(ht, 4),
                })
    return pd.DataFrame(rows)


def _csv_download_btn(get_df_fn, filename):
    def _cb():
        buf = io.StringIO()
        get_df_fn().to_csv(buf, index=False)
        return io.BytesIO(buf.getvalue().encode())
    return pn.widgets.FileDownload(
        callback=_cb, filename=filename,
        label='⬇ CSV', button_type='light',
        height=26, width=90, embed=False,
    )


def _plot_with_dl(fig, dl_btn, stretch=False, remove_btn=None):
    kw = {'sizing_mode': 'stretch_width'} if stretch else {}
    top_items = []
    if remove_btn is not None:
        top_items.append(remove_btn)
    top_items += [pn.Spacer(), dl_btn]
    return pn.Column(
        pn.Row(*top_items),
        pn.pane.Bokeh(fig, **kw),
    )


def auto_tag(n, cyc):
    return [cyc[i % len(cyc)] for i in range(n)]

def auto_bnnt(n, cyc_len):
    return [i // cyc_len + 1 for i in range(n)]

def block_tag(n, n_tubes, cross_cycle):
    out = []
    for t in range(1, n_tubes + 1):
        out.append('longitudinal')
    for t in range(1, n_tubes + 1):
        for c in cross_cycle:
            out.append(c)
    return out[:n]

def block_bnnt(n, n_tubes, cross_cycle):
    out = []
    for t in range(1, n_tubes + 1):
        out.append(t)
    for t in range(1, n_tubes + 1):
        for _ in cross_cycle:
            out.append(t)
    return out[:n]

def short_label(tag, bnnt):
    s = 'L' if tag == 'longitudinal' else tag.replace('cross_', 'S')
    return f'T{bnnt}-{s}'

def parse_cycle_text(text):
    parts   = [t.strip() for t in text.split(',') if t.strip()]
    if not parts:
        return None, "Cycle cannot be empty."
    invalid = [p for p in parts if p not in VALID_TAGS]
    if invalid:
        return None, f"Unknown tag(s): {', '.join(invalid)}"
    return parts, None

def _get_subdirs(folder):
    try:
        return sorted(
            [e.name for e in os.scandir(folder)
             if e.is_dir() and not e.name.startswith('.')],
            key=str.lower,
        )
    except (PermissionError, OSError):
        return []

# ── Mutable application state ──────────────────────────────────────────────────

filepath     = None
is_gwy       = False
profiles     = []
n_lines      = 0
image        = None
lines        = None
xreal = yreal = xres = yres = None
cycle        = list(DEFAULT_CYCLE)
tags         = []
bnnts        = []
selected_idx = [None]
_file_dir    = os.getcwd()
_file_stem   = ''
_tag_mode    = 'cycle'
_block_tubes = 4
_session     = []   # list of dicts: {file_stem, filepath, profiles, tags, bnnts}

# GWY data sources — reassigned in load_file for each GWY file
line_source  = ColumnDataSource(data=dict(
    xs=[], ys=[], colors=[], labels=[],
    line_num=[], tag_val=[], bnnt_val=[],
))
label_source = ColumnDataSource(data=dict(x=[], y=[], text=[]))

# ── CLI args ───────────────────────────────────────────────────────────────────

_cli_args = [a for a in sys.argv[1:] if not a.startswith('--')]
if _cli_args:
    _cli_abs     = os.path.abspath(_cli_args[0])
    _initial_dir = os.path.dirname(_cli_abs)
else:
    _cli_abs     = None
    _initial_dir = os.getcwd()

# ── Input file browser ─────────────────────────────────────────────────────────

_in_nav_dir = [_initial_dir]

def _get_browser_items(folder):
    items = []
    try:
        entries = sorted(os.scandir(folder),
                         key=lambda e: (not e.is_dir(), e.name.lower()))
        for e in entries:
            if e.is_dir() and not e.name.startswith('.'):
                items.append('📁 ' + e.name)
            elif e.is_file() and e.name.lower().endswith(('.gwy', '.txt')):
                items.append('📄 ' + e.name)
    except (PermissionError, OSError):
        pass
    return items

in_nav_path_html = pn.pane.HTML('', sizing_mode='stretch_width')
in_nav_list      = pn.widgets.MultiSelect(
    name='', options=[], height=130, sizing_mode='stretch_width',
)
in_nav_up_btn    = pn.widgets.Button(name='↑ Up',    width=70, height=26)
in_nav_enter_btn = pn.widgets.Button(name='↓ Enter', width=80, height=26)

def _refresh_in_nav(folder=None):
    d = folder or _in_nav_dir[0]
    _in_nav_dir[0] = d
    label = d if len(d) <= 46 else '…' + d[-44:]
    in_nav_path_html.object = (
        f'<span style="font-size:11px;color:#444;word-break:break-all">'
        f'📁 <b>{label}</b></span>'
    )
    items = _get_browser_items(d)
    parent = os.path.dirname(d)
    if parent and parent != d:
        items = ['..'] + items
    in_nav_list.options = items if items else ['(empty)']
    in_nav_list.value   = []

def _on_in_up(_event):
    parent = os.path.dirname(_in_nav_dir[0])
    if parent and parent != _in_nav_dir[0]:
        _refresh_in_nav(parent)

def _on_in_enter(_event):
    sel = in_nav_list.value
    if not sel or sel[0] == '(empty)':
        return
    item = sel[0]
    if item == '..':
        _on_in_up(None)
    elif item.startswith('📁 '):
        new = os.path.join(_in_nav_dir[0], item[2:])
        if os.path.isdir(new):
            _refresh_in_nav(new)

def _on_in_select(event):
    sel = event.new
    if not sel:
        return
    item = sel[0]
    if item == '..':
        _on_in_up(None)
    elif item.startswith('📁 '):
        new = os.path.join(_in_nav_dir[0], item[2:])
        if os.path.isdir(new):
            _refresh_in_nav(new)

in_nav_up_btn.on_click(_on_in_up)
in_nav_enter_btn.on_click(_on_in_enter)
in_nav_list.param.watch(_on_in_select, 'value')
_refresh_in_nav()

load_btn = pn.widgets.Button(
    name='▶  Load selected', button_type='success',
    sizing_mode='stretch_width', height=30,
)
load_status = pn.pane.HTML(
    '<span style="color:#aaa;font-size:12px">No file loaded.</span>',
    sizing_mode='stretch_width',
)

# Pre-navigate to CLI file's directory if given
if _cli_abs and os.path.isfile(_cli_abs):
    _refresh_in_nav(os.path.dirname(_cli_abs))

# ── Output directory browser ───────────────────────────────────────────────────

_out_dir = [_initial_dir]  # mutable container

out_dir_path_html = pn.pane.HTML('', sizing_mode='stretch_width')
out_dir_sublist   = pn.widgets.MultiSelect(
    name='Subdirectories:',
    options=[],
    height=80,
    sizing_mode='stretch_width',
)
out_nav_up_btn    = pn.widgets.Button(name='↑ Up',    width=70,  height=26)
out_nav_enter_btn = pn.widgets.Button(name='↓ Enter', width=80,  height=26)

def _refresh_out_nav(folder=None):
    d = folder or _out_dir[0]
    _out_dir[0] = d
    label = d if len(d) <= 48 else '…' + d[-46:]
    out_dir_path_html.object = (
        f'<span style="font-size:11px;color:#444;word-break:break-all">'
        f'📁 <b>{label}</b></span>'
    )
    subdirs = _get_subdirs(d)
    parent = os.path.dirname(d)
    if parent and parent != d:
        subdirs = ['..'] + subdirs
    out_dir_sublist.options = subdirs if subdirs else ['(no subdirectories)']
    out_dir_sublist.value   = []

def _on_out_up(_event):
    parent = os.path.dirname(_out_dir[0])
    if parent and parent != _out_dir[0]:
        _refresh_out_nav(parent)

def _on_out_enter(_event):
    sel = out_dir_sublist.value
    if not sel or sel[0] == '(no subdirectories)':
        return
    item = sel[0]
    if item == '..':
        _on_out_up(None)
    else:
        new = os.path.join(_out_dir[0], item)
        if os.path.isdir(new):
            _refresh_out_nav(new)

def _on_out_select(event):
    sel = event.new
    if not sel:
        return
    item = sel[0]
    if item == '..':
        _on_out_up(None)
    elif item != '(no subdirectories)':
        new = os.path.join(_out_dir[0], item)
        if os.path.isdir(new):
            _refresh_out_nav(new)

out_nav_up_btn.on_click(_on_out_up)
out_nav_enter_btn.on_click(_on_out_enter)
out_dir_sublist.param.watch(_on_out_select, 'value')
_refresh_out_nav()

# ── GWY helper functions ───────────────────────────────────────────────────────

def build_line_data():
    xs, ys, colors, lbls = [], [], [], []
    ln_nums, tag_vals, bnnt_vals = [], [], []
    for i, ln in enumerate(lines):
        xs.append([ln['x1'] * 1e6, ln['x2'] * 1e6])
        ys.append([(yreal - ln['y1']) * 1e6, (yreal - ln['y2']) * 1e6])
        colors.append('#FFFFFF')
        lbls.append(str(bnnts[i]))
        ln_nums.append(i + 1)
        tag_vals.append(tags[i])
        bnnt_vals.append(bnnts[i])
    return dict(xs=xs, ys=ys, colors=colors, labels=lbls,
                line_num=ln_nums, tag_val=tag_vals, bnnt_val=bnnt_vals)

def build_label_data():
    lx, ly, lt = [], [], []
    for i, ln in enumerate(lines):
        x1 = ln['x1'] * 1e6;  x2 = ln['x2'] * 1e6
        y1 = (yreal - ln['y1']) * 1e6;  y2 = (yreal - ln['y2']) * 1e6
        dx = x2 - x1;  dy = y2 - y1
        length = (dx**2 + dy**2) ** 0.5 or 1.0
        # unit perpendicular (rotated 90° CCW)
        px = -dy / length;  py = dx / length
        offset = length * 0.09
        lx.append(x1 + offset * px)
        ly.append(y1 + offset * py)
        lt.append(str(bnnts[i]))
    return dict(x=lx, y=ly, text=lt)

# ── GWY tag-editor widgets ─────────────────────────────────────────────────────

tag_status  = pn.pane.HTML('', width=245)
sel_info    = pn.pane.HTML('', width=245)
tag_select  = pn.widgets.Select(name='Tag:', value='', options=[], width=215, disabled=True)
bnnt_select = pn.widgets.Select(name='BNNT number:', value='1', options=['1'],
                                width=215, disabled=True)
apply_btn   = pn.widgets.Button(name='Apply changes',  button_type='primary', width=215, disabled=True)
reset_btn   = pn.widgets.Button(name='Reset all tags', button_type='warning',  width=215)

def on_line_selected(attr, old, new):
    if not new:
        selected_idx[0] = None
        sel_info.object = '<span style="color:#aaa;font-size:12px">No line selected.</span>'
        tag_select.disabled = bnnt_select.disabled = apply_btn.disabled = True
        return
    i = new[0]
    selected_idx[0]   = i
    tag_select.value  = tags[i]
    bnnt_select.value = str(bnnts[i])
    tag_select.disabled = bnnt_select.disabled = apply_btn.disabled = False
    c = TAG_COLORS.get(tags[i], '#aaa')
    sel_info.object = (
        f'<b>Line {i+1}</b> selected<br>'
        f'Tag: <b style="color:{c}">{tags[i]}</b> | BNNT: <b>{bnnts[i]}</b>'
    )

def on_apply(_event):
    i = selected_idx[0]
    if i is None:
        return
    tags[i]  = tag_select.value
    bnnts[i] = int(bnnt_select.value)
    line_source.data  = build_line_data()
    label_source.data = build_label_data()
    c = TAG_COLORS.get(tags[i], '#aaa')
    sel_info.object = (
        f'<b>Line {i+1}</b> updated<br>'
        f'Tag: <b style="color:{c}">{tags[i]}</b> | BNNT: <b>{bnnts[i]}</b>'
    )
    tag_status.object = f'<span style="color:green">✔ Line {i+1} saved.</span>'
    update_profile_table()

def on_reset(_event):
    global tags, bnnts
    tags  = auto_tag(n_lines, cycle)
    bnnts = auto_bnnt(n_lines, len(cycle))
    line_source.data    = build_line_data()
    label_source.data   = build_label_data()
    selected_idx[0]     = None
    tag_select.disabled = bnnt_select.disabled = apply_btn.disabled = True
    tag_select.options  = list(cycle) + ['unassigned']
    tag_select.value    = cycle[0]
    sel_info.object   = '<span style="color:#aaa;font-size:12px">No line selected.</span>'
    tag_status.object = '<span style="color:orange">↺ Tags reset.</span>'
    update_profile_table()

apply_btn.on_click(on_apply)
reset_btn.on_click(on_reset)

# ── Cycle editor ───────────────────────────────────────────────────────────────

tag_mode_select = pn.widgets.Select(
    name='Mode:', value='Cycle',
    options=['Cycle', 'Block (longitudinals first)'],
    width=230,
)
n_tubes_input = pn.widgets.IntInput(
    name='N tubes:', value=4, start=1, step=1,
    width=100, visible=False,
)
tag_mode_select.param.watch(
    lambda e: setattr(n_tubes_input, 'visible', e.new.startswith('Block')),
    'value',
)
cycle_input = pn.widgets.TextInput(
    name='Cross pattern (comma-separated):',
    value=', '.join(DEFAULT_CYCLE),
    sizing_mode='stretch_width',
)
cycle_apply_btn = pn.widgets.Button(name='Tag', button_type='primary', width=90, height=30)
cycle_status    = pn.pane.HTML(
    f'<span style="color:#555;font-size:12px">'
    f'Current: {" → ".join(DEFAULT_CYCLE)}</span>',
    sizing_mode='stretch_width',
)
valid_tags_hint = pn.pane.HTML(
    f'<span style="color:#999;font-size:11px">'
    f'Valid tags: {", ".join(VALID_TAGS)}</span>',
    sizing_mode='stretch_width',
)

def on_cycle_apply(_event):
    global cycle, tags, bnnts, _tag_mode, _block_tubes
    new_cycle, err = parse_cycle_text(cycle_input.value)
    if err:
        cycle_status.object = f'<span style="color:red">✗ {err}</span>'
        return
    cycle     = new_cycle
    _tag_mode = 'block' if tag_mode_select.value.startswith('Block') else 'cycle'
    _block_tubes = n_tubes_input.value

    if _tag_mode == 'block':
        tags  = block_tag(n_lines, _block_tubes, cycle)
        bnnts = block_bnnt(n_lines, _block_tubes, cycle)
        cycle_status.object = (
            f'<span style="color:green">✔ Block: {_block_tubes} tubes, '
            f'cross: {" → ".join(cycle)}</span>'
        )
    else:
        tags  = auto_tag(n_lines, cycle)
        bnnts = auto_bnnt(n_lines, len(cycle))
        cycle_status.object = (
            f'<span style="color:green">✔ Cycle: {" → ".join(cycle)}</span>'
        )
    update_profile_table()
    if is_gwy and lines:
        line_source.data   = build_line_data()
        label_source.data  = build_label_data()
        tag_select.options = list(VALID_TAGS) + ['unassigned']
        tag_select.value   = cycle[0]

cycle_apply_btn.on_click(on_cycle_apply)

# ── Action buttons + output containers ────────────────────────────────────────

_BTN = dict(height=30, sizing_mode='stretch_width')
plot_btn        = pn.widgets.Button(name='▶  Plot profiles',   button_type='success', disabled=True, **_BTN)
dist_btn        = pn.widgets.Button(name='📊  Distributions',   button_type='primary', disabled=True, **_BTN)
export_btn      = pn.widgets.Button(name='⬇  Export CSV',       button_type='default', disabled=True, **_BTN)
save_plots_btn  = pn.widgets.Button(name='💾  Save plots (PNG)', button_type='default', disabled=True, **_BTN)
clear_plots_btn = pn.widgets.Button(name='✕  Clear plots',      button_type='danger',  disabled=True, **_BTN)
legend_toggle   = pn.widgets.Toggle(name='Legends: ON', value=True, button_type='default', height=30, width=150)

save_status = pn.pane.HTML('', width=900)

profiles_header = pn.pane.HTML('', width=1000)
profiles_col    = pn.Column()
dist_header     = pn.pane.HTML('', width=1000)
dist_col        = pn.Column()

_profile_figs         = []
_dist_figs            = []
_profile_data         = []
_session_profile_figs = []
_sf_tap_selected   = {}   # id(src) -> global_profile_idx
_sess_tap_selected = {}   # id(src) -> (entry_idx, prof_idx)
_dist_data    = []

# ── Session widgets ────────────────────────────────────────────────────────────

session_status_html = pn.pane.HTML(
    '<span style="color:#aaa;font-size:12px">No files in session.</span>',
    sizing_mode='stretch_width',
)
session_table = pn.widgets.Tabulator(
    pd.DataFrame({'File': pd.Series(dtype=str), 'Profiles': pd.Series(dtype=int)}),
    width=318, height=110, show_index=False, disabled=True,
    selectable='checkbox',
)
session_add_btn    = pn.widgets.Button(
    name='➕  Add current file', button_type='success',
    sizing_mode='stretch_width', height=28, disabled=True,
)
session_remove_btn = pn.widgets.Button(
    name='⌫  Remove selected', button_type='warning',
    sizing_mode='stretch_width', height=28,
)
session_clear_btn  = pn.widgets.Button(
    name='🗑  Clear session', button_type='danger',
    sizing_mode='stretch_width', height=28,
)
session_plot_btn   = pn.widgets.Button(
    name='▶  Plot all', button_type='success',
    width=148, height=28, disabled=True,
)
session_dist_btn   = pn.widgets.Button(
    name='📊  Dist all', button_type='primary',
    width=148, height=28, disabled=True,
)
session_export_summary_btn = pn.widgets.FileDownload(
    callback=lambda: io.BytesIO(get_session_summary_df().to_csv(index=False).encode()),
    filename='session_summary.csv',
    label='⬇  Summary CSV', button_type='light',
    height=26, sizing_mode='stretch_width', embed=False, disabled=True,
)
session_export_full_btn = pn.widgets.FileDownload(
    callback=lambda: io.BytesIO(get_session_full_df().to_csv(index=False).encode()),
    filename='session_full.csv',
    label='⬇  Full CSV', button_type='light',
    height=26, sizing_mode='stretch_width', embed=False, disabled=True,
)
session_long_btn = pn.widgets.Button(
    name='🔍  Analyze longitudinals', button_type='default',
    sizing_mode='stretch_width', height=28, disabled=True,
)

# Session output placeholders (main area)
session_output_header = pn.pane.HTML('', sizing_mode='stretch_width')
session_profiles_col  = pn.Column()
session_dist_header   = pn.pane.HTML('', sizing_mode='stretch_width')
session_dist_col      = pn.Column()
session_long_header   = pn.pane.HTML('', sizing_mode='stretch_width')
session_long_col      = pn.Column()

# ── Single profile plot ────────────────────────────────────────────────────────

single_select   = pn.widgets.Select(
    name='Single profile:',
    options=[],
    sizing_mode='stretch_width',
    disabled=True,
)
plot_single_btn = pn.widgets.Button(
    name='📈  Plot selected',
    button_type='primary',
    disabled=True,
    height=30,
    width=150,
)
single_plot_section = pn.Column(visible=False)

def _make_single_options():
    return [f'Profile {i+1}  —  {tags[i]}  —  BNNT {bnnts[i]}'
            for i in range(n_lines)]

def on_plot_single(_event):
    sel = single_select.value
    if not sel:
        return
    # parse "Profile N  —  tag  —  BNNT M"
    try:
        idx = int(sel.split('—')[0].replace('Profile', '').strip()) - 1
    except (ValueError, IndexError):
        return
    if idx < 0 or idx >= n_lines:
        return
    d, p   = profiles[idx]
    d_nm   = [x * 1000 for x in d]
    tag_str = tags[idx]
    bnnt_v  = bnnts[idx]
    color   = TAG_COLORS.get(tag_str, '#1f77b4')

    pf = figure(
        width=700, height=350,
        title=f'Profile {idx+1}  ·  {tag_str}  ·  BNNT {bnnt_v}',
        x_axis_label='Distance (nm)',
        y_axis_label='Height (nm)',
        tools='pan,wheel_zoom,reset,save',
        toolbar_location='above',
    )
    pf.line(d_nm, p, line_color=color, line_width=2)
    pf.title.text_font_size = '12px'
    pf.grid.grid_line_alpha = 0.25

    _df = pd.DataFrame({'distance_nm': d_nm, 'height_nm': p})
    dl  = _csv_download_btn(lambda df=_df: df,
                            f'profile_{idx+1}_bnnt{bnnt_v}.csv')
    single_plot_section.objects = [
        pn.pane.HTML('<b style="font-size:13px">Single profile:</b>'),
        _plot_with_dl(pf, dl, stretch=True),
    ]
    single_plot_section.visible = True

plot_single_btn.on_click(on_plot_single)

# ── Longitudinal peak / valley analysis ───────────────────────────────────────

long_select = pn.widgets.Select(
    name='Longitudinal profile:',
    options=[], sizing_mode='stretch_width', disabled=True,
)
baseline_pct_input = pn.widgets.FloatInput(
    name='Baseline percentile (0 = min):', value=0.0,
    start=0.0, end=50.0, step=1.0, width=195,
)
prominence_pct_input = pn.widgets.FloatInput(
    name='Min prominence (% of range):', value=10.0,
    start=0.0, end=100.0, step=1.0, width=195,
)
min_dist_input = pn.widgets.IntInput(
    name='Min distance (samples):', value=5, start=1, step=1, width=140,
)
analyze_btn = pn.widgets.Button(
    name='🔍  Detect peaks & valleys',
    button_type='primary', disabled=True,
    sizing_mode='stretch_width', height=30,
)
analysis_status  = pn.pane.HTML('', sizing_mode='stretch_width')
analysis_section = pn.Column(visible=False, sizing_mode='stretch_width')


def _make_dist_panel(vals, color, title, x_label, n_bins_init,
                     dl_btn=None, width=370, height=270, fill_alpha=0.75):
    arr = np.array(vals)
    if arr.size == 0:
        return None
    n_b  = max(2, n_bins_init)
    hist, edges = np.histogram(arr, bins=n_b)
    xpad = (edges[-1] - edges[0]) * 0.08 if edges[-1] != edges[0] else 0.5

    source = ColumnDataSource(data=dict(
        top=hist.tolist(), bottom=[0] * len(hist),
        left=edges[:-1].tolist(), right=edges[1:].tolist(),
    ))
    pf = figure(
        width=width, height=height, title=title,
        x_axis_label=x_label, y_axis_label='Count',
        x_range=Range1d(edges[0] - xpad, edges[-1] + xpad),
        y_range=Range1d(0, float(hist.max()) * 1.20),
        tools='pan,wheel_zoom,reset,save', toolbar_location='above',
    )
    pf.title.text_font_size  = '11px'
    pf.grid.grid_line_alpha  = 0.25
    pf.xgrid.grid_line_color = None
    pf.quad(
        top='top', bottom='bottom', left='left', right='right', source=source,
        fill_color=color, fill_alpha=fill_alpha, line_color='white', line_width=1,
    )
    n_lbl = Label(
        x=float(edges[-1]) + xpad * 0.3, y=float(hist.max()) * 1.06,
        text=f'n = {len(arr)}',
        text_font_size='9pt', text_color='#555', text_align='right',
    )
    pf.add_layout(n_lbl)

    bins_input = pn.widgets.IntInput(value=n_b, start=2, step=1, name='', width=60, height=26)

    def _on_bins(event, src=source, fig=pf, a=arr, lbl=n_lbl):
        nb = max(2, event.new)
        h, e = np.histogram(a, bins=nb)
        src.data = dict(
            top=h.tolist(), bottom=[0] * len(h),
            left=e[:-1].tolist(), right=e[1:].tolist(),
        )
        fig.y_range.end = float(h.max()) * 1.20
        lbl.y           = float(h.max()) * 1.06

    bins_input.param.watch(_on_bins, 'value')

    header = [
        pn.pane.HTML('<span style="font-size:11px;color:#555">Bins:</span>'),
        bins_input,
        pn.Spacer(),
    ]
    if dl_btn:
        header.append(dl_btn)
    return pn.Column(pn.Row(*header, align='center'), pn.pane.Bokeh(pf))


def on_analyze(_event):
    sel = long_select.value
    if not sel:
        return
    try:
        idx = int(sel.split('—')[0].replace('Profile', '').strip()) - 1
    except (ValueError, IndexError):
        return
    if idx < 0 or idx >= n_lines:
        return

    bpct = baseline_pct_input.value
    ppct = prominence_pct_input.value
    mdist = min_dist_input.value

    d_nm, h_corr, baseline, peaks, valleys = analyze_longitudinal(
        idx, baseline_pct=bpct, prominence_pct=ppct, min_dist=mdist,
    )

    # ── Individual profile plot with markers ──────────────────────────────
    pf = figure(
        width=700, height=320,
        title=(f'Profile {idx+1}  ·  BNNT {bnnts[idx]}'
               f'  ·  baseline = {baseline:.2f} nm'),
        x_axis_label='Distance (nm)',
        y_axis_label='Height above baseline (nm)',
        tools='pan,wheel_zoom,reset,save',
        toolbar_location='above',
    )
    pf.title.text_font_size = '11px'
    pf.grid.grid_line_alpha = 0.25
    pf.line(d_nm, h_corr, line_color='#1f77b4', line_width=2,
            legend_label='Profile')
    pf.line([d_nm[0], d_nm[-1]], [0, 0],
            line_color='#888', line_width=1, line_dash='dashed',
            legend_label='Baseline')
    if peaks['pos']:
        pf.scatter(peaks['pos'], peaks['height'], marker='triangle',
                   size=11, color='#d62728', legend_label='Peaks')
    if valleys['pos']:
        pf.scatter(valleys['pos'], valleys['height'], marker='inverted_triangle',
                   size=11, color='#2ca02c', legend_label='Valleys')
    pf.legend.label_text_font_size = '8pt'
    pf.legend.location             = 'top_right'

    # ── Individual tables ─────────────────────────────────────────────────
    pk_df = pd.DataFrame({
        '#':               range(1, len(peaks['pos']) + 1),
        'Position (nm)':   [f'{v:.1f}' for v in peaks['pos']],
        'Height (nm)':     [f'{v:.2f}' for v in peaks['height']],
        'Prominence (nm)': [f'{v:.2f}' for v in peaks['prominence']],
    })
    vl_df = pd.DataFrame({
        '#':               range(1, len(valleys['pos']) + 1),
        'Position (nm)':   [f'{v:.1f}' for v in valleys['pos']],
        'Height (nm)':     [f'{v:.2f}' for v in valleys['height']],
        'Prominence (nm)': [f'{v:.2f}' for v in valleys['prominence']],
    })
    pk_table = pn.widgets.Tabulator(
        pk_df, width=390, height=150, show_index=False, disabled=True,
    )
    vl_table = pn.widgets.Tabulator(
        vl_df, width=390, height=150, show_index=False, disabled=True,
    )

    # ── Individual single-profile distributions ───────────────────────────
    single_dist_panels = []
    for vals, color, title, col_name, fname_suffix in [
        (peaks['height'],   '#d62728', 'Peak heights — this profile',
         'peak_height_nm',   f'profile_{idx+1}_peak_heights.csv'),
        (valleys['height'], '#2ca02c', 'Valley heights — this profile',
         'valley_height_nm', f'profile_{idx+1}_valley_heights.csv'),
    ]:
        if not vals:
            continue
        n_b  = max(2, int(np.ceil(np.log2(len(vals)) + 1)))
        dl   = _csv_download_btn(
            lambda v=vals, c=col_name: pd.DataFrame({c: v}), fname_suffix
        )
        panel = _make_dist_panel(vals, color, title, 'Height above baseline (nm)', n_b, dl_btn=dl)
        if panel is not None:
            single_dist_panels.append(panel)

    # ── Aggregate: all longitudinal profiles ─────────────────────────────
    long_indices = [i for i in range(n_lines) if tags[i] == 'longitudinal']
    all_pk_heights, all_pk_proms = [], []
    all_vl_heights, all_vl_proms = [], []
    for li in long_indices:
        _, _, _, pk_i, vl_i = analyze_longitudinal(
            li, baseline_pct=bpct, prominence_pct=ppct, min_dist=mdist,
        )
        all_pk_heights.extend(pk_i['height'])
        all_pk_proms.extend(pk_i['prominence'])
        all_vl_heights.extend(vl_i['height'])
        all_vl_proms.extend(vl_i['prominence'])

    agg_panels = []
    for vals, color, title, xlabel, col_name, fname_suffix in [
        (all_pk_heights, '#d62728', 'Peak heights — all longitudinals',
         'Height above baseline (nm)', 'peak_height_nm',    'all_peak_heights.csv'),
        (all_vl_heights, '#2ca02c', 'Valley heights — all longitudinals',
         'Height above baseline (nm)', 'valley_height_nm',  'all_valley_heights.csv'),
        (all_pk_proms,   '#e07070', 'Peak prominence — all longitudinals',
         'Prominence (nm)',            'peak_prominence_nm', 'all_peak_prominence.csv'),
        (all_vl_proms,   '#70c070', 'Valley prominence — all longitudinals',
         'Prominence (nm)',            'valley_prominence_nm','all_valley_prominence.csv'),
    ]:
        if not vals:
            continue
        n_b  = max(2, int(np.ceil(np.log2(len(vals)) + 1)))
        dl   = _csv_download_btn(
            lambda v=vals, c=col_name: pd.DataFrame({c: v}),
            f'{_file_stem}_{fname_suffix}',
        )
        panel = _make_dist_panel(vals, color, title, xlabel, n_b, dl_btn=dl)
        if panel is not None:
            agg_panels.append(panel)

    n_long = len(long_indices)
    analysis_status.object = (
        f'<span style="color:green">✔ '
        f'Selected: <b>{len(peaks["pos"])} peaks</b>, <b>{len(valleys["pos"])} valleys</b>  |  '
        f'All {n_long} longitudinals: '
        f'<b>{len(all_pk_heights)} peaks</b>, <b>{len(all_vl_heights)} valleys</b></span>'
    )

    # ── Profile plot download ─────────────────────────────────────────────
    _prof_df = pd.DataFrame({'distance_nm': d_nm, 'height_corr_nm': h_corr})
    prof_dl  = _csv_download_btn(lambda df=_prof_df: df,
                                 f'profile_{idx+1}_longitudinal_corrected.csv')

    # ── Peak/valley table download ────────────────────────────────────────
    pk_dl = _csv_download_btn(lambda df=pk_df: df,
                              f'profile_{idx+1}_peaks.csv')
    vl_dl = _csv_download_btn(lambda df=vl_df: df,
                              f'profile_{idx+1}_valleys.csv')

    section_objects = [
        pn.pane.HTML('<hr style="border-color:#e0e0e0;margin:8px 0">'),
        pn.pane.HTML('<b style="font-size:13px">Longitudinal peak/valley analysis</b>'),
        pn.Spacer(height=6),
        _plot_with_dl(pf, prof_dl, stretch=True),
        pn.Spacer(height=10),
        pn.Row(
            pn.Column(
                pn.Row(pn.pane.HTML('<b style="font-size:12px;color:#d62728">▲ Peaks</b>'),
                       pn.Spacer(), pk_dl),
                pk_table,
            ),
            pn.Spacer(width=20),
            pn.Column(
                pn.Row(pn.pane.HTML('<b style="font-size:12px;color:#2ca02c">▼ Valleys</b>'),
                       pn.Spacer(), vl_dl),
                vl_table,
            ),
        ),
    ]
    if single_dist_panels:
        section_objects += [
            pn.Spacer(height=10),
            pn.pane.HTML('<b style="font-size:12px">Distributions — selected profile</b>'),
            pn.Row(*single_dist_panels),
        ]
    if agg_panels:
        section_objects += [
            pn.Spacer(height=14),
            pn.pane.HTML(
                f'<b style="font-size:12px">Distributions — all {n_long} longitudinal profiles</b>'
            ),
            pn.Row(*agg_panels[:2]),
            pn.Row(*agg_panels[2:]),
        ]

    analysis_section.objects = section_objects
    analysis_section.visible = True


analyze_btn.on_click(on_analyze)

# ── Profile summary table ──────────────────────────────────────────────────────

profile_table = pn.widgets.Tabulator(
    pd.DataFrame({'#': pd.Series(dtype=int),
                  'BNNT': pd.Series(dtype=int),
                  'Tag': pd.Series(dtype=str)}),
    width=310, height=220, show_index=False, disabled=False,
    sizing_mode='fixed',
    editors={
        '#':    None,
        'BNNT': {'type': 'number', 'min': 1, 'step': 1},
        'Tag':  {'type': 'list', 'values': list(VALID_TAGS) + ['unassigned']},
    },
    selectable='checkbox',
)
delete_btn = pn.widgets.Button(
    name='⌫  Delete selected', button_type='danger',
    width=310, disabled=True,
)
table_cycle_header = pn.pane.HTML('', sizing_mode='stretch_width')
table_section = pn.Column(
    table_cycle_header,
    profile_table,
    delete_btn,
    visible=False,
)

def _cycle_header_html():
    if _tag_mode == 'block':
        return (
            f'<b style="font-size:12px">Loaded profiles</b> '
            f'<span style="font-size:11px;color:#666">({n_lines} total)</span>'
            f'<br><span style="font-size:11px;color:#888">'
            f'Block: {_block_tubes} tubes | cross: {" → ".join(cycle)}</span>'
        )
    return (
        f'<b style="font-size:12px">Loaded profiles</b> '
        f'<span style="font-size:11px;color:#666">({n_lines} total)</span>'
        f'<br><span style="font-size:11px;color:#888">Cycle: {" → ".join(cycle)}</span>'
    )

def update_profile_table():
    profile_table.value = pd.DataFrame({
        '#':    [i + 1    for i in range(n_lines)],
        'BNNT': [bnnts[i] for i in range(n_lines)],
        'Tag':  [tags[i]  for i in range(n_lines)],
    })
    table_cycle_header.object = _cycle_header_html()
    table_section.visible = (n_lines > 0)
    # update single profile select
    opts = _make_single_options()
    single_select.options    = opts
    single_select.value      = opts[0] if opts else None
    single_select.disabled   = not opts
    plot_single_btn.disabled = not opts
    # update longitudinal analysis select
    long_opts = [
        f'Profile {i+1}  —  BNNT {bnnts[i]}'
        for i in range(n_lines) if tags[i] == 'longitudinal'
    ]
    long_select.options    = long_opts
    long_select.value      = long_opts[0] if long_opts else None
    long_select.disabled   = not long_opts
    analyze_btn.disabled   = not long_opts

def _do_remove_profiles(sorted_desc_indices):
    """Remove profiles at the given indices (sorted descending). Updates all dependent state."""
    global n_lines
    for i in sorted_desc_indices:
        if i < len(profiles):
            profiles.pop(i)
            tags.pop(i)
            bnnts.pop(i)
            if lines is not None and i < len(lines):
                lines.pop(i)
    n_lines = len(profiles)
    selected_idx[0] = None
    if is_gwy:
        if n_lines > 0:
            line_source.data  = build_line_data()
            label_source.data = build_label_data()
        else:
            line_source.data  = dict(xs=[], ys=[], colors=[], labels=[])
            label_source.data = dict(x=[], y=[], text=[])
    update_profile_table()
    profile_table.selection = []


def on_table_edit(event):
    row = event.row
    col = event.column
    val = event.value
    if col == 'Tag' and row < n_lines:
        tags[row] = val
    elif col == 'BNNT' and row < n_lines:
        bnnts[row] = int(val) if val else 1
    if is_gwy and n_lines > 0:
        line_source.data  = build_line_data()
        label_source.data = build_label_data()
    update_profile_table()

profile_table.on_edit(on_table_edit)

def on_delete(_event):
    sel = sorted(profile_table.selection, reverse=True)
    if not sel:
        return
    _do_remove_profiles(sel)

delete_btn.on_click(on_delete)

# Placeholder — filled with image + tag-editor for GWY files
gwy_section = pn.Column()

# ── Legend toggle ──────────────────────────────────────────────────────────────

def on_legend_toggle(event):
    legend_toggle.name = 'Legends: ON' if event.new else 'Legends: OFF'
    for pf in _profile_figs + _session_profile_figs:
        for lg in pf.legend:
            lg.visible = event.new

legend_toggle.param.watch(on_legend_toggle, 'value')

# ── load_file ──────────────────────────────────────────────────────────────────

def load_file(fp):
    global filepath, is_gwy, profiles, n_lines
    global image, lines, xreal, yreal, xres, yres
    global cycle, tags, bnnts, selected_idx
    global _file_dir, _file_stem
    global line_source, label_source
    global _tag_mode, _block_tubes

    ext    = os.path.splitext(fp)[1].lower()
    is_gwy = (ext == '.gwy')

    if is_gwy:
        image, lines, xreal, yreal, xres, yres = load_gwy(fp)
        n_lines  = len(lines)
        profiles = [extract_profile(image, ln, xreal, yreal, xres, yres)
                    for ln in lines]
    else:
        profiles = parse_txt(fp)
        n_lines  = len(profiles)
        image = lines = xreal = yreal = xres = yres = None

    filepath     = fp
    cycle        = list(DEFAULT_CYCLE)
    tags         = auto_tag(n_lines, cycle)
    bnnts        = auto_bnnt(n_lines, len(cycle))
    selected_idx = [None]
    _file_dir    = os.path.dirname(os.path.abspath(fp))
    _file_stem   = os.path.splitext(os.path.basename(fp))[0]

    # sync browsers to file's directory
    _refresh_in_nav(_file_dir)
    _refresh_out_nav(_file_dir)

    _tag_mode    = 'cycle'
    _block_tubes = 4
    cycle_input.value        = ', '.join(DEFAULT_CYCLE)
    tag_mode_select.value    = 'Cycle'
    n_tubes_input.visible    = False
    cycle_status.object = (
        f'<span style="color:#555;font-size:12px">'
        f'Current: {" → ".join(DEFAULT_CYCLE)}</span>'
    )

    for btn in (plot_btn, dist_btn, export_btn, save_plots_btn, clear_plots_btn, delete_btn):
        btn.disabled = False
    session_add_btn.disabled = False

    single_plot_section.visible = False
    single_plot_section.objects = []
    analysis_section.visible    = False
    analysis_section.objects    = []
    analysis_status.object      = ''
    on_clear_plots(None)

    # ── Build GWY section ─────────────────────────────────────────────────────
    if is_gwy:
        line_source  = ColumnDataSource(build_line_data())
        label_source = ColumnDataSource(build_label_data())
        line_source.selected.on_change('indices', on_line_selected)

        mapper = LinearColorMapper(
            palette=Inferno256,
            low=float(image.min()), high=float(image.max()),
        )
        hover = HoverTool(
            tooltips="""
            <div style="padding:4px 8px;font-size:11px;background:rgba(0,0,0,0.75);color:#fff;border-radius:4px">
                <b>Line @line_num</b><br/>
                BNNT: <b>@bnnt_val</b><br/>
                Tag: @tag_val
            </div>
            """,
            line_policy='nearest',
        )
        p_img = figure(
            width=610, height=570,
            x_range=Range1d(0, xreal * 1e6),
            y_range=Range1d(0, yreal * 1e6),
            title=os.path.basename(fp),
            tools='pan,wheel_zoom,reset,tap',
            toolbar_location='above',
            x_axis_label='x (µm)', y_axis_label='y (µm)',
        )
        p_img.add_tools(hover)
        p_img.image(
            image=[np.flipud(image)], x=0, y=0,
            dw=xreal * 1e6, dh=yreal * 1e6,
            color_mapper=mapper,
        )
        p_img.add_layout(
            ColorBar(color_mapper=mapper, label_standoff=8, width=12, location=(0, 0)),
            'right',
        )
        lines_glyph = p_img.multi_line(
            xs='xs', ys='ys', line_color='white', line_width=1.5,
            source=line_source,
            selection_line_width=3, selection_line_color='#FFFF00',
            nonselection_line_alpha=0.45, nonselection_line_color='white',
        )
        hover.renderers = [lines_glyph]
        p_img.text(
            x='x', y='y', text='text', source=label_source,
            text_font_size='8pt', text_color='white',
            text_align='center', text_baseline='middle',
        )

        tag_status.object = '<span style="color:#555;font-size:12px">Click a line to select it.</span>'
        sel_info.object   = '<span style="color:#aaa;font-size:12px">No line selected.</span>'
        tag_select.options  = list(VALID_TAGS) + ['unassigned']
        tag_select.value    = cycle[0]
        tag_select.disabled = bnnt_select.disabled = apply_btn.disabled = True
        bnnt_select.options = [str(x) for x in range(1, n_lines + 1)]
        bnnt_select.value   = '1'

        tag_controls = pn.Column(
            pn.pane.HTML('<h3 style="margin:4px 0;color:#333">Tag Editor</h3>'),
            tag_status,
            pn.Spacer(height=6),
            sel_info,
            pn.Spacer(height=10),
            tag_select,
            bnnt_select,
            pn.Spacer(height=4),
            apply_btn,
            pn.Spacer(height=14),
            reset_btn,
            width=255,
        )
        gwy_section.objects = [
            pn.Row(p_img, pn.Spacer(width=20), tag_controls),
            pn.pane.HTML('<hr style="border-color:#e0e0e0;margin:4px 0">'),
        ]
    else:
        gwy_section.objects = []

    update_profile_table()
    load_status.object = (
        f'<span style="color:green;font-size:12px">'
        f'✔ Loaded <b>{os.path.basename(fp)}</b> — {n_lines} profiles</span>'
    )


def on_load_file(_event):
    sel = in_nav_list.value
    if not sel or sel[0] == '(empty)':
        load_status.object = '<span style="color:orange">⚠ Select a file from the list.</span>'
        return
    item = sel[0]
    if not item.startswith('📄 '):
        load_status.object = '<span style="color:orange">⚠ Select a file (📄), not a folder.</span>'
        return
    fp = os.path.join(_in_nav_dir[0], item[2:])
    try:
        load_file(fp)
    except Exception as exc:
        load_status.object = f'<span style="color:red">✗ {exc}</span>'

load_btn.on_click(on_load_file)

# ── Plot profiles ──────────────────────────────────────────────────────────────

def on_plot(_event):
    global _profile_figs, _profile_data, _sf_tap_selected
    _sf_tap_selected = {}
    plot_tags = ['longitudinal'] + [t for t in cycle if t != 'longitudinal']
    groups = {t: [] for t in plot_tags}
    for i, (d, p) in enumerate(profiles):
        if tags[i] in groups:
            groups[tags[i]].append((bnnts[i], i, d, p))

    active = [(t, e) for t, e in groups.items() if e]
    if not active:
        profiles_header.object = '<span style="color:red">No tagged profiles to plot.</span>'
        return

    figs = []
    profile_data_entries = []
    rm_btns = []
    for tag, entries in active:
        is_cross = (tag != 'longitudinal')
        x_label  = 'Distance from peak (nm)' if is_cross else 'Distance (nm)'
        pf = figure(
            width=500, height=360,
            title=tag.replace('_', ' ').capitalize(),
            x_axis_label=x_label,
            y_axis_label='Height (nm)',
            tools='pan,wheel_zoom,tap,reset,save',
            toolbar_location='above',
        )
        pf.title.text_color     = '#000000'
        pf.title.text_font_size = '12px'
        pf.grid.grid_line_alpha = 0.25
        tag_entry_data = []
        fig_src_ids = []
        fig_srcs = []
        for j, (bnnt, global_idx, d, p) in enumerate(sorted(entries)):
            d_plot = [x * 1000 for x in d]
            if is_cross:
                peak_pos = d_plot[int(np.argmax(np.array(p)))]
                d_plot   = [x - peak_pos for x in d_plot]
            color = PLOT_COLORS[j % len(PLOT_COLORS)]
            label = f'BNNT {bnnt}'
            src = ColumnDataSource({'x': d_plot, 'y': list(p),
                                    'label': [label] * len(d_plot)})
            renderer = pf.line('x', 'y', source=src,
                               line_color=color, line_width=1.6,
                               legend_label=label)
            renderer.selection_glyph    = BokehLine(line_color=color, line_width=3.0, line_alpha=1.0)
            renderer.nonselection_glyph = BokehLine(line_color=color, line_width=1.6, line_alpha=0.2)
            fig_srcs.append((src, global_idx))
            tag_entry_data.append((bnnt, d_plot, list(p)))
        tap_state = _setup_line_tap(pf, fig_srcs)
        pf.add_tools(HoverTool(
            tooltips=[('Profile', '@label'),
                      ('Distance', '@x{0.0} nm'),
                      ('Height',   '@y{0.000} nm')],
            mode='mouse',
        ))
        pf.legend.label_text_font_size = '7pt'
        pf.legend.glyph_height         = 10
        pf.legend.glyph_width          = 12
        pf.legend.spacing              = 1
        pf.legend.location             = 'top_right'
        pf.legend.visible              = legend_toggle.value
        figs.append(pf)
        profile_data_entries.append((tag, tag_entry_data))

        rm_btn = pn.widgets.Button(name='✕ Remove', button_type='danger',
                                   height=22, width=80)
        def _make_rm_cb(st):
            def _cb(_event):
                idx = st['key']
                if idx is None:
                    return
                _do_remove_profiles([idx])
                on_plot(None)
            return _cb
        rm_btn.on_click(_make_rm_cb(tap_state))
        rm_btns.append(rm_btn)

    _profile_figs = figs
    _profile_data = profile_data_entries
    panels = []
    for pf, (tag, entries_data), rm_btn in zip(figs, _profile_data, rm_btns):
        rows_list = []
        for bnnt, d_nm_l, heights in entries_data:
            for d, h in zip(d_nm_l, heights):
                rows_list.append({'bnnt': bnnt, 'distance_nm': round(d, 4), 'height_nm': round(h, 4)})
        _df  = pd.DataFrame(rows_list)
        dl   = _csv_download_btn(lambda df=_df: df, f'{_file_stem}_{tag}_profiles.csv')
        panels.append(_plot_with_dl(pf, dl, remove_btn=rm_btn))
    profiles_col.objects   = arrange_rows(panels, n_cols=2).objects
    profiles_header.object = '<b style="font-size:13px">Grouped profiles:</b>'

plot_btn.on_click(on_plot)

# ── Distributions ──────────────────────────────────────────────────────────────

def on_dist(_event):
    global _dist_figs, _dist_data
    # Always include longitudinal regardless of what is in the cross cycle
    plot_tags = ['longitudinal'] + [t for t in cycle if t != 'longitudinal']
    groups = {t: [] for t in plot_tags}
    for i, (d, p) in enumerate(profiles):
        if tags[i] in groups:
            metric = compute_metric(tags[i], d, p)
            groups[tags[i]].append((bnnts[i], metric))

    active = [(t, e) for t, e in groups.items() if e]
    if not active:
        dist_header.object = '<span style="color:red">No tagged profiles for distribution.</span>'
        return

    _dist_figs = []
    _dist_data = []
    panels = []
    for tag, entries in active:
        values  = np.array([v for _, v in entries])
        x_label = 'Length (nm)' if tag == 'longitudinal' else 'Peak height (nm)'
        n_bins  = max(5, int(np.ceil(np.log2(len(values))) + 1))
        col_name = x_label.lower().replace(' ', '_').replace('(', '').replace(')', '')
        _df = pd.DataFrame({col_name: values})
        dl  = _csv_download_btn(lambda df=_df: df, f'{_file_stem}_{tag}_distribution.csv')
        panel = _make_dist_panel(
            values, HIST_COLOR,
            tag.replace('_', ' ').capitalize(),
            x_label, n_bins,
            dl_btn=dl, width=500, height=340, fill_alpha=0.85,
        )
        if panel is not None:
            panels.append(panel)
            _dist_data.append((tag, values, x_label, n_bins))

    dist_col.objects   = arrange_rows(panels, n_cols=2).objects
    dist_header.object = '<b style="font-size:13px">Distributions:</b>'

dist_btn.on_click(on_dist)

# ── Export CSV ─────────────────────────────────────────────────────────────────

def on_export(_event):
    out = os.path.join(_out_dir[0], _file_stem + '_tagged_profiles.csv')
    with open(out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['line_index', 'bnnt', 'tag', 'distance_um', 'height_nm'])
        for i, (d, p) in enumerate(profiles):
            for dist, ht in zip(d, p):
                w.writerow([i + 1, bnnts[i], tags[i], f'{dist:.6f}', f'{ht:.4f}'])
    cycle_status.object = (
        f'<span style="color:green">✔ Exported → {os.path.basename(out)}</span>'
    )

export_btn.on_click(on_export)

# ── Clear plots ────────────────────────────────────────────────────────────────

def on_clear_plots(_event):
    global _profile_figs, _dist_figs, _profile_data, _dist_data
    _profile_figs = []
    _dist_figs    = []
    _profile_data = []
    _dist_data    = []
    profiles_col.objects = []
    dist_col.objects     = []
    profiles_header.object = ''
    dist_header.object     = ''
    save_status.object     = ''

clear_plots_btn.on_click(on_clear_plots)

# ── Session callbacks ──────────────────────────────────────────────────────────

def update_session_table():
    has = len(_session) > 0
    session_table.value = pd.DataFrame({
        'File':     [e['file_stem']        for e in _session],
        'Profiles': [len(e['profiles'])    for e in _session],
    })
    if has:
        session_status_html.object = (
            f'<span style="font-size:12px;color:#333">'
            f'<b>{len(_session)}</b> file(s) in session</span>'
        )
    else:
        session_status_html.object = (
            '<span style="color:#aaa;font-size:12px">No files in session.</span>'
        )
    for w in (session_plot_btn, session_dist_btn, session_long_btn,
              session_export_summary_btn, session_export_full_btn):
        w.disabled = not has


def on_session_add(_event):
    if not profiles:
        return
    # Replace entry if the same file_stem is already present
    for k, e in enumerate(_session):
        if e['file_stem'] == _file_stem:
            _session[k] = dict(
                file_stem=_file_stem, filepath=filepath,
                profiles=[tuple(p) for p in profiles],
                tags=list(tags), bnnts=list(bnnts),
            )
            update_session_table()
            session_status_html.object = (
                f'<span style="color:#e07800;font-size:12px">'
                f'↺ Updated <b>{_file_stem}</b> in session.</span>'
            )
            return
    _session.append(dict(
        file_stem=_file_stem, filepath=filepath,
        profiles=[tuple(p) for p in profiles],
        tags=list(tags), bnnts=list(bnnts),
    ))
    update_session_table()
    session_status_html.object = (
        f'<span style="color:green;font-size:12px">'
        f'✔ Added <b>{_file_stem}</b> ({len(profiles)} profiles).</span>'
    )


def on_session_remove(_event):
    sel = sorted(session_table.selection, reverse=True)
    for i in sel:
        if i < len(_session):
            _session.pop(i)
    session_table.selection = []
    update_session_table()


def on_session_clear(_event):
    _session.clear()
    session_table.selection = []
    update_session_table()
    session_output_header.object = ''
    session_profiles_col.objects = []
    session_dist_header.object   = ''
    session_dist_col.objects     = []
    session_long_header.object   = ''
    session_long_col.objects     = []


def on_session_plot(_event):
    global _session_profile_figs, _sess_tap_selected
    if not _session:
        return
    _session_profile_figs = []
    _sess_tap_selected = {}
    # Collect all (file_stem, bnnt) pairs for color assignment
    all_pairs = []
    for entry in _session:
        for b in sorted(set(entry['bnnts'])):
            pair = (entry['file_stem'], b)
            if pair not in all_pairs:
                all_pairs.append(pair)
    color_map = {pair: PLOT_COLORS[i % len(PLOT_COLORS)]
                 for i, pair in enumerate(all_pairs)}

    # Build per-tag groups across all session entries
    tag_set = ['longitudinal']
    for entry in _session:
        for t in entry['tags']:
            if t not in tag_set:
                tag_set.append(t)
    groups = {t: [] for t in tag_set}
    for entry_idx, entry in enumerate(_session):
        for prof_idx, (d, p) in enumerate(entry['profiles']):
            t = entry['tags'][prof_idx]
            if t in groups:
                groups[t].append((entry['file_stem'], entry['bnnts'][prof_idx],
                                  d, p, entry_idx, prof_idx))

    panels = []
    for tag, entries in groups.items():
        if not entries:
            continue
        is_cross = (tag != 'longitudinal')
        x_label  = 'Distance from peak (nm)' if is_cross else 'Distance (nm)'
        pf = figure(
            width=500, height=360,
            title=f'{tag.replace("_", " ").capitalize()}  —  {len(_session)} file(s)',
            x_axis_label=x_label, y_axis_label='Height (nm)',
            tools='pan,wheel_zoom,tap,reset,save', toolbar_location='above',
        )
        pf.title.text_font_size = '12px'
        pf.grid.grid_line_alpha = 0.25
        rows_list = []
        fig_srcs = []
        for file_stem, bnnt, d, p, entry_idx, prof_idx in entries:
            d_plot = [x * 1000 for x in d]
            if is_cross:
                peak_pos = d_plot[int(np.argmax(np.array(p)))]
                d_plot   = [x - peak_pos for x in d_plot]
            color = color_map.get((file_stem, bnnt), PLOT_COLORS[0])
            label = f'{file_stem} · BNNT {bnnt}'
            src = ColumnDataSource({'x': d_plot, 'y': list(p),
                                    'label': [label] * len(d_plot)})
            renderer = pf.line('x', 'y', source=src, line_color=color, line_width=1.4,
                               legend_label=label)
            renderer.selection_glyph    = BokehLine(line_color=color, line_width=3.0, line_alpha=1.0)
            renderer.nonselection_glyph = BokehLine(line_color=color, line_width=1.4, line_alpha=0.2)
            fig_srcs.append((src, (entry_idx, prof_idx)))
            for dist, ht in zip(d_plot, p):
                rows_list.append({'file': file_stem, 'bnnt': bnnt,
                                  'distance_nm': round(dist, 4),
                                  'height_nm': round(ht, 4)})
        tap_state = _setup_line_tap(pf, fig_srcs)
        pf.add_tools(HoverTool(
            tooltips=[('Profile', '@label'),
                      ('Distance', '@x{0.0} nm'),
                      ('Height',   '@y{0.000} nm')],
            mode='mouse',
        ))
        pf.legend.label_text_font_size = '7pt'
        pf.legend.glyph_height         = 10
        pf.legend.glyph_width          = 12
        pf.legend.spacing              = 1
        pf.legend.location             = 'top_right'
        pf.legend.visible              = legend_toggle.value
        _session_profile_figs.append(pf)

        rm_btn = pn.widgets.Button(name='✕ Remove', button_type='danger',
                                   height=22, width=80)
        def _make_session_rm_cb(st):
            def _cb(_event):
                key = st['key']
                if key is None:
                    return
                ei, pi = key
                _session[ei]['profiles'].pop(pi)
                _session[ei]['tags'].pop(pi)
                _session[ei]['bnnts'].pop(pi)
                _session[:] = [e for e in _session if e['profiles']]
                update_session_table()
                on_session_plot(None)
            return _cb
        rm_btn.on_click(_make_session_rm_cb(tap_state))

        _df = pd.DataFrame(rows_list)
        dl  = _csv_download_btn(lambda df=_df: df,
                                f'session_{tag}_profiles.csv')
        panels.append(_plot_with_dl(pf, dl, remove_btn=rm_btn))

    session_profiles_col.objects  = arrange_rows(panels, n_cols=2).objects
    session_output_header.object  = (
        f'<b style="font-size:13px">Session grouped profiles '
        f'({len(_session)} file(s)):</b>'
    )


def on_session_dist(_event):
    if not _session:
        return
    tag_set = ['longitudinal']
    for entry in _session:
        for t in entry['tags']:
            if t not in tag_set:
                tag_set.append(t)

    panels = []
    for tag in tag_set:
        metrics = []
        for entry in _session:
            for i, (d, p) in enumerate(entry['profiles']):
                if entry['tags'][i] == tag:
                    metrics.append(compute_metric(tag, d, p))
        if not metrics:
            continue
        x_label = 'Length (nm)' if tag == 'longitudinal' else 'Peak height (nm)'
        n_bins  = max(5, int(np.ceil(np.log2(len(metrics))) + 1))
        title   = (f'{tag.replace("_", " ").capitalize()}  —  '
                   f'{len(_session)} file(s), n={len(metrics)}')
        col_name = x_label.lower().replace(' ', '_').replace('(', '').replace(')', '')
        _df = pd.DataFrame({col_name: metrics})
        dl  = _csv_download_btn(lambda df=_df: df,
                                f'session_{tag}_distribution.csv')
        panel = _make_dist_panel(
            metrics, HIST_COLOR, title, x_label, n_bins,
            dl_btn=dl, width=500, height=340, fill_alpha=0.85,
        )
        if panel is not None:
            panels.append(panel)

    session_dist_col.objects   = arrange_rows(panels, n_cols=2).objects
    session_dist_header.object = (
        f'<b style="font-size:13px">Session distributions '
        f'({len(_session)} file(s)):</b>'
    )


def on_session_long_analyze(_event):
    if not _session:
        return
    bpct  = baseline_pct_input.value
    ppct  = prominence_pct_input.value
    mdist = min_dist_input.value

    all_pk_heights, all_pk_proms = [], []
    all_vl_heights, all_vl_proms = [], []

    summary_rows = []
    for entry in _session:
        n_long = n_pk = n_vl = 0
        for i, (d, p) in enumerate(entry['profiles']):
            if entry['tags'][i] != 'longitudinal':
                continue
            n_long += 1
            _, _, _, pk, vl = _analyze_longitudinal_data(
                d, p, bpct, ppct, mdist,
            )
            all_pk_heights.extend(pk['height'])
            all_pk_proms.extend(pk['prominence'])
            all_vl_heights.extend(vl['height'])
            all_vl_proms.extend(vl['prominence'])
            n_pk += len(pk['height'])
            n_vl += len(vl['height'])
        if n_long:
            summary_rows.append({
                'File':         entry['file_stem'],
                'Longitudinals': n_long,
                'Peaks':         n_pk,
                'Valleys':       n_vl,
            })

    if not summary_rows:
        session_long_header.object = (
            '<span style="color:orange">⚠ No longitudinal profiles found in session.</span>'
        )
        session_long_col.objects = []
        return

    # Summary table
    summary_df    = pd.DataFrame(summary_rows)
    summary_table = pn.widgets.Tabulator(
        summary_df, width=560, height=min(30 * len(summary_rows) + 40, 220),
        show_index=False, disabled=True,
    )
    summary_dl = _csv_download_btn(lambda df=summary_df: df,
                                   'session_longitudinal_summary.csv')

    # Four aggregate histograms
    panels = []
    for vals, color, title, xlabel, col_name, fname in [
        (all_pk_heights, '#d62728', 'Peak heights — all session longitudinals',
         'Height above baseline (nm)', 'peak_height_nm',    'session_all_peak_heights.csv'),
        (all_vl_heights, '#2ca02c', 'Valley heights — all session longitudinals',
         'Height above baseline (nm)', 'valley_height_nm',  'session_all_valley_heights.csv'),
        (all_pk_proms,   '#e07070', 'Peak prominence — all session longitudinals',
         'Prominence (nm)',            'peak_prominence_nm', 'session_all_peak_prominence.csv'),
        (all_vl_proms,   '#70c070', 'Valley prominence — all session longitudinals',
         'Prominence (nm)',            'valley_prominence_nm','session_all_valley_prominence.csv'),
    ]:
        if not vals:
            continue
        n_b   = max(2, int(np.ceil(np.log2(len(vals)) + 1)))
        _df   = pd.DataFrame({col_name: vals})
        dl    = _csv_download_btn(lambda df=_df: df, fname)
        panel = _make_dist_panel(vals, color, title, xlabel, n_b, dl_btn=dl)
        if panel is not None:
            panels.append(panel)

    total_long = sum(r['Longitudinals'] for r in summary_rows)
    session_long_header.object = (
        f'<b style="font-size:13px">Session longitudinal analysis — '
        f'{len(_session)} file(s), {total_long} longitudinal profile(s) '
        f'(baseline={bpct}%, prominence={ppct}%, min dist={mdist})</b>'
    )
    session_long_col.objects = [
        pn.Row(
            pn.pane.HTML('<b style="font-size:12px">Per-file summary:</b>'),
            pn.Spacer(), summary_dl,
        ),
        summary_table,
        pn.Spacer(height=10),
        *arrange_rows(panels, n_cols=2).objects,
    ]


session_add_btn.on_click(on_session_add)
session_remove_btn.on_click(on_session_remove)
session_clear_btn.on_click(on_session_clear)
session_plot_btn.on_click(on_session_plot)
session_dist_btn.on_click(on_session_dist)
session_long_btn.on_click(on_session_long_analyze)

# ── Save plots (PNG via matplotlib) ───────────────────────────────────────────

def on_save_plots(_event):
    if not _profile_data and not _dist_data:
        save_status.object = (
            '<span style="color:orange">⚠ No plots to save — '
            'generate profiles or distributions first.</span>'
        )
        return
    saved = []
    _out = _out_dir[0]

    for tag, entries in _profile_data:
        slug    = tag.replace(' ', '_')
        out     = os.path.join(_out, f'{_file_stem}_profile_{slug}.png')
        x_label = 'Distance from peak (nm)' if tag != 'longitudinal' else 'Distance (nm)'
        fig, ax = plt.subplots(figsize=(6, 4))
        for j, (bnnt, d, p) in enumerate(entries):
            ax.plot(d, p, color=PLOT_COLORS[j % len(PLOT_COLORS)],
                    linewidth=1.4, label=f'BNNT {bnnt}')
        ax.set_title(tag.replace('_', ' ').capitalize(), fontsize=12)
        ax.set_xlabel(x_label)
        ax.set_ylabel('Height (nm)')
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        plt.close(fig)
        saved.append(os.path.basename(out))

    for tag, values, x_label, n_bins in _dist_data:
        slug = tag.replace(' ', '_')
        out  = os.path.join(_out, f'{_file_stem}_dist_{slug}.png')
        hist, edges = np.histogram(values, bins=n_bins)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(edges[:-1], hist, width=np.diff(edges), align='edge',
               color=HIST_COLOR, edgecolor='white', linewidth=0.8, alpha=0.85)
        ax.set_title(tag.replace('_', ' ').capitalize(), fontsize=12)
        ax.set_xlabel(x_label)
        ax.set_ylabel('Count')
        ax.yaxis.grid(alpha=0.25, color='#cccccc')
        ax.set_axisbelow(True)
        ax.text(0.98, 0.97, f'n = {len(values)}',
                transform=ax.transAxes, ha='right', va='top',
                fontsize=8, color='#555555')
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        plt.close(fig)
        saved.append(os.path.basename(out))

    save_status.object = (
        f'<span style="color:green">✔ Saved {len(saved)} file(s): '
        f'{", ".join(saved)}</span>'
    )

save_plots_btn.on_click(on_save_plots)

# ── Layout assembly ────────────────────────────────────────────────────────────

file_browser_section = pn.Card(
    in_nav_path_html,
    pn.Spacer(height=4),
    in_nav_list,
    pn.Spacer(height=4),
    pn.Row(in_nav_up_btn, pn.Spacer(width=6), in_nav_enter_btn,
           sizing_mode='stretch_width'),
    pn.Spacer(height=6),
    load_btn,
    pn.Spacer(height=4),
    load_status,
    title='📂  File',
    collapsible=False,
    header_background='#f5f5f5',
    styles={'border': '1px solid #e0e0e0'},
    sizing_mode='stretch_width',
)

out_dir_section = pn.Card(
    out_dir_path_html,
    pn.Spacer(height=4),
    out_dir_sublist,
    pn.Spacer(height=4),
    pn.Row(out_nav_up_btn, pn.Spacer(width=6), out_nav_enter_btn,
           sizing_mode='stretch_width'),
    title='📁  Output folder',
    collapsible=True,
    header_background='#f5f5f5',
    styles={'border': '1px solid #e0e0e0'},
    sizing_mode='stretch_width',
)

cycle_card = pn.Card(
    valid_tags_hint,
    pn.Spacer(height=4),
    pn.Row(tag_mode_select, pn.Spacer(width=8), n_tubes_input),
    pn.Spacer(height=4),
    pn.Row(cycle_input, pn.Spacer(width=8), cycle_apply_btn, align='end', sizing_mode='stretch_width'),
    cycle_status,
    title='🔁  Cycle / Block Editor',
    collapsible=True,
    header_background='#f5f5f5',
    styles={'border': '1px solid #e0e0e0'},
    sizing_mode='stretch_width',
)

action_row = pn.Column(
    pn.Row(
        plot_btn, pn.Spacer(width=6),
        dist_btn, pn.Spacer(width=6),
        legend_toggle,
        align='center',
        sizing_mode='stretch_width', max_width=760,
    ),
    pn.Spacer(height=6),
    pn.Row(
        export_btn,     pn.Spacer(width=6),
        save_plots_btn, pn.Spacer(width=6),
        clear_plots_btn,
        sizing_mode='stretch_width', max_width=760,
    ),
    pn.Spacer(height=8),
    pn.Row(single_select, pn.Spacer(width=6), plot_single_btn,
           sizing_mode='stretch_width', max_width=760, align='end'),
)

output_section = pn.Column(
    pn.Spacer(height=10),
    action_row,
    save_status,
    pn.Spacer(height=16),
    single_plot_section,
    analysis_section,
    pn.Spacer(height=8),
    profiles_header,
    profiles_col,
    pn.Spacer(height=16),
    dist_header,
    dist_col,
    pn.Spacer(height=24),
    pn.pane.HTML('<hr style="border-color:#ccc;margin:4px 0">'),
    session_output_header,
    session_profiles_col,
    pn.Spacer(height=16),
    session_dist_header,
    session_dist_col,
    pn.Spacer(height=16),
    session_long_header,
    session_long_col,
    sizing_mode='stretch_width',
)

# Auto-load CLI-supplied file if one was given
if _cli_abs:
    try:
        load_file(_cli_abs)
    except Exception as exc:
        load_status.object = f'<span style="color:red">✗ {exc}</span>'

analysis_card = pn.Card(
    long_select,
    pn.Spacer(height=4),
    baseline_pct_input,
    prominence_pct_input,
    min_dist_input,
    pn.Spacer(height=4),
    analyze_btn,
    analysis_status,
    title='📈  Longitudinal Analysis',
    collapsible=True,
    header_background='#f5f5f5',
    styles={'border': '1px solid #e0e0e0'},
    sizing_mode='stretch_width',
)

session_card = pn.Card(
    session_status_html,
    pn.Spacer(height=4),
    session_table,
    pn.Spacer(height=4),
    session_add_btn,
    pn.Row(session_remove_btn, session_clear_btn),
    pn.Spacer(height=6),
    pn.Row(session_plot_btn, session_dist_btn, align='center'),
    pn.Spacer(height=4),
    session_long_btn,
    pn.Spacer(height=4),
    session_export_summary_btn,
    session_export_full_btn,
    title='📋  Session',
    collapsible=True,
    header_background='#f5f5f5',
    styles={'border': '1px solid #e0e0e0'},
    sizing_mode='stretch_width',
)

controls_col = pn.Column(
    file_browser_section,
    pn.Spacer(height=8),
    out_dir_section,
    pn.Spacer(height=8),
    session_card,
    pn.Spacer(height=8),
    cycle_card,
    pn.Spacer(height=8),
    analysis_card,
    pn.Spacer(height=8),
    table_section,
    width=360,
)

root = pn.Column(
    pn.pane.HTML('<h2 style="margin:6px 0 10px;color:#333;font-size:18px">BNNT Profile Tagger</h2>'),
    pn.Row(
        controls_col,
        pn.Spacer(width=16),
        pn.Column(gwy_section, output_section, sizing_mode='stretch_width'),
        sizing_mode='stretch_width',
    ),
    sizing_mode='stretch_width',
)

root.servable(title='BNNT Profile Tagger')


def _cli_main():
    import subprocess
    cmd = [sys.executable, '-m', 'panel', 'serve', '--show', __file__]
    if len(sys.argv) > 1:
        cmd += ['--args'] + sys.argv[1:]
    subprocess.run(cmd)

if __name__ == '__main__':
    _cli_main()
