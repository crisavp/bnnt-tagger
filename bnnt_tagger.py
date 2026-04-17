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

import sys, os, csv, re
import numpy as np
import pandas as pd
from scipy.ndimage import map_coordinates
import gwyfile

import panel as pn
from bokeh.plotting import figure
from bokeh.models import (
    ColumnDataSource, ColorBar, LinearColorMapper,
    Range1d, Label, HoverTool,
)
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

pn.extension('bokeh', 'tabulator', notifications=True, sizing_mode='stretch_width', raw_css=[_CSS])

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


def arrange_rows(figs, n_cols=2):
    rows = []
    for i in range(0, len(figs), n_cols):
        rows.append(pn.Row(*figs[i:i + n_cols]))
    return pn.Column(*rows) if rows else pn.Column()


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

def list_data_files(folder):
    try:
        return sorted(f for f in os.listdir(folder)
                      if f.lower().endswith(('.txt', '.gwy')))
    except Exception:
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

# GWY data sources — reassigned in load_file for each GWY file
line_source  = ColumnDataSource(data=dict(
    xs=[], ys=[], colors=[], labels=[],
    line_num=[], tag_val=[], bnnt_val=[],
))
label_source = ColumnDataSource(data=dict(x=[], y=[], text=[]))

# ── File browser ───────────────────────────────────────────────────────────────

_cli_args = [a for a in sys.argv[1:] if not a.startswith('--')]
if _cli_args:
    _cli_abs     = os.path.abspath(_cli_args[0])
    _initial_dir = os.path.dirname(_cli_abs)
else:
    _cli_abs     = None
    _initial_dir = os.getcwd()

folder_input = pn.widgets.TextInput(name='Folder:', value=_initial_dir, sizing_mode='stretch_width', height=30)
file_select  = pn.widgets.Select(name='File:', value='', options=[''], sizing_mode='stretch_width')
refresh_btn  = pn.widgets.Button(name='↺  Refresh', button_type='default', width=100, height=30)
load_btn     = pn.widgets.Button(name='▶  Load',    button_type='success',  width=100, height=30)
load_status  = pn.pane.HTML(
    '<span style="color:#aaa;font-size:12px">No file loaded.</span>',
    sizing_mode='stretch_width',
)

def refresh_file_list(folder=None):
    folder = (folder or folder_input.value).strip()
    files  = list_data_files(folder)
    if files:
        file_select.options = files
        if file_select.value not in files:
            file_select.value = files[0]
    else:
        file_select.options = ['(no .txt / .gwy files found)']
        file_select.value   = file_select.options[0]

folder_input.param.watch(lambda e: refresh_file_list(e.new), 'value')
refresh_btn.on_click(lambda _e: refresh_file_list())

refresh_file_list(_initial_dir)
if _cli_abs:
    fname = os.path.basename(_cli_abs)
    if fname in file_select.options:
        file_select.value = fname

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
dist_bins_input = pn.widgets.TextInput(name='', placeholder='bins (auto)', value='', width=100, height=30)

save_status = pn.pane.HTML('', width=900)

profiles_header = pn.pane.HTML('', width=1000)
profiles_col    = pn.Column()
dist_header     = pn.pane.HTML('', width=1000)
dist_col        = pn.Column()

_profile_figs = []
_dist_figs    = []
_profile_data = []
_dist_data    = []

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
    global profiles, tags, bnnts, n_lines, selected_idx, lines
    sel = sorted(profile_table.selection, reverse=True)
    if not sel:
        return
    for i in sel:
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

delete_btn.on_click(on_delete)

# Placeholder — filled with image + tag-editor for GWY files
gwy_section = pn.Column()

# ── Legend toggle ──────────────────────────────────────────────────────────────

def on_legend_toggle(event):
    legend_toggle.name = 'Legends: ON' if event.new else 'Legends: OFF'
    for pf in _profile_figs:
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
    sel = file_select.value
    if not sel or sel.startswith('('):
        load_status.object = '<span style="color:orange">⚠ Select a valid file first.</span>'
        return
    fp = os.path.join(folder_input.value.strip(), sel)
    try:
        load_file(fp)
    except Exception as exc:
        load_status.object = f'<span style="color:red">✗ {exc}</span>'

load_btn.on_click(on_load_file)

# ── Plot profiles ──────────────────────────────────────────────────────────────

def on_plot(_event):
    global _profile_figs, _profile_data
    groups = {t: [] for t in cycle}
    for i, (d, p) in enumerate(profiles):
        if tags[i] in groups:
            groups[tags[i]].append((bnnts[i], i + 1, d, p))

    active = [(t, e) for t, e in groups.items() if e]
    if not active:
        profiles_header.object = '<span style="color:red">No tagged profiles to plot.</span>'
        return

    figs = []
    for tag, entries in active:
        pf = figure(
            width=500, height=360,
            title=tag.replace('_', ' ').capitalize(),
            x_axis_label='Distance (nm)',
            y_axis_label='Height (nm)',
            tools='pan,wheel_zoom,reset,save',
            toolbar_location='above',
        )
        pf.title.text_color     = '#000000'
        pf.title.text_font_size = '12px'
        pf.grid.grid_line_alpha = 0.25
        for j, (bnnt, _lidx, d, p) in enumerate(sorted(entries)):
            d_plot = [x * 1000 for x in d]
            pf.line(d_plot, p,
                    line_color=PLOT_COLORS[j % len(PLOT_COLORS)],
                    line_width=1.6,
                    legend_label=f'BNNT {bnnt}')
        pf.legend.label_text_font_size = '7pt'
        pf.legend.glyph_height         = 10
        pf.legend.glyph_width          = 12
        pf.legend.spacing              = 1
        pf.legend.location             = 'top_right'
        pf.legend.visible              = legend_toggle.value
        figs.append(pf)

    _profile_figs         = figs
    _profile_data         = [(tag, [(b, [x * 1000 for x in d], p)
                                    for b, _, d, p in sorted(entries)])
                              for tag, entries in active]
    profiles_col.objects  = arrange_rows(figs, n_cols=2).objects
    profiles_header.object = '<b style="font-size:13px">Grouped profiles:</b>'

plot_btn.on_click(on_plot)

# ── Distributions ──────────────────────────────────────────────────────────────

def on_dist(_event):
    global _dist_figs, _dist_data
    groups = {t: [] for t in cycle}
    for i, (d, p) in enumerate(profiles):
        if tags[i] in groups:
            metric = compute_metric(tags[i], d, p)
            groups[tags[i]].append((bnnts[i], metric))

    active = [(t, e) for t, e in groups.items() if e]
    if not active:
        dist_header.object = '<span style="color:red">No tagged profiles for distribution.</span>'
        return

    try:
        user_bins = int(dist_bins_input.value.strip())
        if user_bins < 1:
            raise ValueError
    except (ValueError, AttributeError):
        user_bins = None

    figs = []
    for tag, entries in active:
        values  = np.array([v for _, v in entries])
        x_label = 'Length (nm)' if tag == 'longitudinal' else 'Peak height (nm)'
        n_bins  = (user_bins if user_bins is not None
                   else max(5, int(np.ceil(np.log2(len(values))) + 1)))
        hist, edges = np.histogram(values, bins=n_bins)

        xpad = (edges[-1] - edges[0]) * 0.08
        pf = figure(
            width=500, height=340,
            title=tag.replace('_', ' ').capitalize(),
            x_axis_label=x_label,
            y_axis_label='Count',
            x_range=Range1d(edges[0] - xpad, edges[-1] + xpad),
            y_range=Range1d(0, float(hist.max()) * 1.18),
            tools='pan,wheel_zoom,reset,save',
            toolbar_location='above',
        )
        pf.title.text_color      = '#000000'
        pf.title.text_font_size  = '12px'
        pf.grid.grid_line_alpha  = 0.25
        pf.grid.grid_line_color  = '#cccccc'
        pf.xgrid.grid_line_color = None

        pf.quad(
            top=hist.tolist(), bottom=[0] * len(hist),
            left=edges[:-1].tolist(), right=edges[1:].tolist(),
            fill_color=HIST_COLOR, fill_alpha=0.85,
            line_color='white', line_width=1.2,
        )
        pf.add_layout(Label(
            x=float(edges[-1]) + xpad * 0.3,
            y=float(hist.max()) * 1.05,
            text=f'n = {len(values)}',
            text_font_size='9pt', text_color='#555',
            text_align='right',
        ))
        figs.append(pf)

    _dist_figs         = figs
    _dist_data         = [(tag, np.array([v for _, v in entries]),
                           'Length (nm)' if tag == 'longitudinal' else 'Peak height (nm)',
                           (user_bins if user_bins is not None
                            else max(5, int(np.ceil(np.log2(len(entries))) + 1))))
                          for tag, entries in active]
    dist_col.objects   = arrange_rows(figs, n_cols=2).objects
    dist_header.object = '<b style="font-size:13px">Distributions:</b>'

dist_btn.on_click(on_dist)

# ── Export CSV ─────────────────────────────────────────────────────────────────

def on_export(_event):
    out = os.path.join(_file_dir, _file_stem + '_tagged_profiles.csv')
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

# ── Save plots (PNG via matplotlib) ───────────────────────────────────────────

def on_save_plots(_event):
    if not _profile_data and not _dist_data:
        save_status.object = (
            '<span style="color:orange">⚠ No plots to save — '
            'generate profiles or distributions first.</span>'
        )
        return
    saved = []

    for tag, entries in _profile_data:
        slug = tag.replace(' ', '_')
        out  = os.path.join(_file_dir, f'{_file_stem}_profile_{slug}.png')
        fig, ax = plt.subplots(figsize=(6, 4))
        for j, (bnnt, d, p) in enumerate(entries):
            ax.plot(d, p, color=PLOT_COLORS[j % len(PLOT_COLORS)],
                    linewidth=1.4, label=f'BNNT {bnnt}')
        ax.set_title(tag.replace('_', ' ').capitalize(), fontsize=12)
        ax.set_xlabel('Distance (nm)')
        ax.set_ylabel('Height (nm)')
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        plt.close(fig)
        saved.append(os.path.basename(out))

    for tag, values, x_label, n_bins in _dist_data:
        slug = tag.replace(' ', '_')
        out  = os.path.join(_file_dir, f'{_file_stem}_dist_{slug}.png')
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
    pn.Row(folder_input, pn.Column(pn.Spacer(height=20), refresh_btn), sizing_mode='stretch_width'),
    pn.Spacer(height=4),
    pn.Row(file_select,  pn.Column(pn.Spacer(height=20), load_btn), sizing_mode='stretch_width'),
    pn.Spacer(height=4),
    load_status,
    title='📂  File',
    collapsible=False,
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
        legend_toggle, pn.Spacer(width=6),
        dist_bins_input,
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
)

output_section = pn.Column(
    pn.Spacer(height=10),
    action_row,
    save_status,
    pn.Spacer(height=16),
    profiles_header,
    profiles_col,
    pn.Spacer(height=16),
    dist_header,
    dist_col,
    sizing_mode='stretch_width',
)

# Auto-load CLI-supplied file if one was given
if _cli_abs:
    try:
        load_file(_cli_abs)
    except Exception as exc:
        load_status.object = f'<span style="color:red">✗ {exc}</span>'

controls_col = pn.Column(
    file_browser_section,
    pn.Spacer(height=8),
    cycle_card,
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