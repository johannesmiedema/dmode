import os
import glob
import base64
import mimetypes
import html
import pandas as pd
from datetime import datetime

# Known RNA modification folder names mapped to display labels
MODIFICATION_SECTIONS = {
    'm6a': 'm6A',
    'am': 'Am',
    'ino': 'Ino',
    'pseu': 'pseU',
    'um': 'Um',
    'gm': 'Gm',
    'cm': 'Cm',
    'm5c': 'm5C',
    'all_modifications': 'All Modifications'
}

# Analyses that should not be split by modification folders
NON_MODIFICATION_ANALYSES = {'diff_gene_expression', 'diff_transcript_expression'}

def _sanitize_id(s: str) -> str:
    """Sanitize string for use as HTML ID."""
    return ''.join(c if c.isalnum() or c in ('_', '-') else '_' for c in s)

def _get_plot_title(filename: str) -> str:
    """Convert plot filename to descriptive title."""
    # Remove extension
    name = os.path.splitext(filename)[0]
    
    # Define mapping for common plot types
    title_mapping = {
        'UMAP': 'UMAP Plot',
        'PCA': 'PCA Plot',
        'Modsite_Barplot': 'Modification Site Count',
        'Mod_Distribution': 'Modification Distribution',
        'Mod_Violin': 'Modification Violin Plot',
        'Modsite_RMBase_Intersection_Proportion': 'RMBase Intersection Proportion',
        'Modsite_RMBase_Intersection': 'RMBase Intersection Count',
        'volcano_plot': 'Volcano Plot',
        'ma_plot': 'MA Plot',
        'scatter': 'Scatter Plot',
        'heatmap': 'Heatmap',
        'barplot': 'Bar Plot',
        'correlation': 'Correlation Plot'
    }
    
    # Try to match known patterns
    for pattern, title in title_mapping.items():
        if pattern in name:
            return title
    
    # Fallback: convert underscores to spaces and title case
    return name.replace('_', ' ').title()

def _detect_modification_section(path_parts):
    """Return display label if any folder matches a known modification."""
    for part in reversed(path_parts):
        key = part.lower()
        if key in MODIFICATION_SECTIONS:
            return MODIFICATION_SECTIONS[key]
    return None

def _image_to_data_uri(image_path: str) -> str | None:
    """Read an image file and return a data URI string."""
    mime_type, _ = mimetypes.guess_type(image_path)
    if not mime_type:
        ext = os.path.splitext(image_path)[1].lower()
        if ext == '.svg':
            mime_type = 'image/svg+xml'
        else:
            mime_type = 'application/octet-stream'

    try:
        if mime_type == 'image/svg+xml':
            with open(image_path, 'r', encoding='utf-8') as fh:
                raw = fh.read().encode('utf-8')
        else:
            with open(image_path, 'rb') as fh:
                raw = fh.read()
    except OSError:
        return None

    b64 = base64.b64encode(raw).decode('ascii')
    return f'data:{mime_type};base64,{b64}'


def generate_html_report(output_dir: str, report_filename: str = 'dmode_report.html') -> str:
    """
    Scan ALL analysis subfolders under `output_dir` and produce a single comprehensive
    HTML report organized by analysis types. Each analysis gets a dedicated page
    with sections for different modification types or comparison folders.
    
    This function discovers all analysis folders and organizes outputs by analysis
    type first, then by modification type or comparison within each analysis.
    
    Parameters
    ----------
    output_dir : str
        Root output directory containing analysis subfolders
    report_filename : str
        Name of the HTML file to generate (default: 'dmode_report.html')
    
    Returns
    -------
    str
        Path to the written HTML file
    """
    if not os.path.isdir(output_dir):
        raise FileNotFoundError(f"Output directory not found: {output_dir}")
    
    # Scan for all PNG and SVG files recursively
    all_images = []
    all_images.extend(glob.glob(os.path.join(output_dir, '**', '*.png'), recursive=True))
    all_images.extend(glob.glob(os.path.join(output_dir, '**', '*.svg'), recursive=True))
    
    # Organize by analysis type, then by modification type or comparison
    # Expected structures:
    # - analysis_type/statistics_gene/modification_type/plot.png (basic stats)
    # - analysis_type/modification_type/plot.png (other analyses)
    # - analysis_type/comparison/plot.png (diff analyses without mods)
    # - analysis_type/plot.png (direct files)
    analyses = {}
    
    # Track seen images to avoid duplicates
    seen_images = set()
    seen_image_bases = set()
    
    # Known intermediate folders and skip folders
    intermediate_folders = {'statistics_gene', 'statistics_transcript'}
    skip_folders = {'qvalues'}
    exclude_suffixes = {'qvalues', 'pvalues', 'scatterplot', 'heatmap', 'barplot', 
                       'plot', 'data', 'table', 'results', 'summary'}
    
    for img_path in all_images:
        # Skip if we've already processed this image
        if img_path in seen_images:
            continue
        seen_images.add(img_path)

        # Skip alternate formats of the same plot (e.g., PNG + SVG)
        image_base = os.path.splitext(img_path)[0]
        if image_base in seen_image_bases:
            continue
        seen_image_bases.add(image_base)
        
        rel_path = os.path.relpath(img_path, output_dir)
        parts = rel_path.split(os.sep)
        
        # Skip files in skip_folders
        if any(part in skip_folders for part in parts[:-1]):
            continue
        
        # Extract analysis type and candidate subsection (comparison folder or inferred)
        analysis_type = None
        subsection = None
        context_section = None
        context_from_intermediate = False
        
        if len(parts) >= 4:
            analysis_type = parts[0]
            if parts[1] in intermediate_folders:
                context_section = parts[2]
                context_from_intermediate = True
            else:
                context_section = parts[1]
        elif len(parts) >= 3:
            analysis_type = parts[0]
            context_section = parts[1]
        elif len(parts) == 2:
            analysis_type = parts[0]
            filename = os.path.basename(img_path)
            for ext in ['.png', '.svg']:
                if filename.endswith(ext):
                    base = filename[:-len(ext)]
                    if '_' in base:
                        potential_sub = base.split('_')[-1]
                        if (len(potential_sub) < 20 and 
                            potential_sub.lower() not in exclude_suffixes):
                            context_section = potential_sub
                            break
        
        folder_hierarchy = parts[1:-1] if len(parts) > 2 else []
        mod_section = None
        if analysis_type and analysis_type not in NON_MODIFICATION_ANALYSES and folder_hierarchy:
            mod_section = _detect_modification_section(folder_hierarchy)

        if mod_section:
            if (context_section and not context_from_intermediate and
                    context_section.lower() != mod_section.lower()):
                subsection = f"{context_section} / {mod_section}"
            else:
                subsection = mod_section
        else:
            subsection = context_section

        # Skip if subsection is in skip_folders
        if subsection and subsection in skip_folders:
            continue
        
        # Use "General" if no subsection found
        if not subsection:
            subsection = "General"
        
        if analysis_type:
            if analysis_type not in analyses:
                analyses[analysis_type] = {}
            
            if subsection not in analyses[analysis_type]:
                analyses[analysis_type][subsection] = []
            
            data_uri = _image_to_data_uri(img_path)
            if not data_uri:
                continue

            analyses[analysis_type][subsection].append({
                'path': rel_path,
                'title': _get_plot_title(os.path.basename(img_path)),
                'filename': os.path.basename(img_path),
                'data_uri': data_uri
            })
    
    # Sort analyses and subsections
    sorted_analyses = sorted(analyses.keys())
    
    # If no analyses found, return early with a message
    if not sorted_analyses:
        html_parts = ['<!doctype html><html><head><title>dmode Report</title></head><body>']
        html_parts.append('<div style="padding: 2rem; text-align: center;">')
        html_parts.append('<h1>No Analysis Results Found</h1>')
        html_parts.append(f'<p>No plot files were found in {output_dir}</p>')
        html_parts.append('<p>Make sure your analysis has completed successfully.</p>')
        html_parts.append('</div></body></html>')
        outpath = os.path.join(output_dir, report_filename)
        with open(outpath, 'w', encoding='utf-8') as fh:
            fh.write('\n'.join(html_parts))
        return outpath
    
    # Build HTML
    html_parts = []
    html_parts.append('<!doctype html>')
    html_parts.append('<html lang="en">')
    html_parts.append('<head>')
    html_parts.append('<meta charset="utf-8">')
    html_parts.append('<meta name="viewport" content="width=device-width, initial-scale=1">')
    html_parts.append('<title>dmode Analysis Report</title>')
    html_parts.append('<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">')
    html_parts.append('<style>')
    html_parts.append('body { background-color: #f8f9fa; }')
    html_parts.append('.mod-card { transition: transform 0.2s, box-shadow 0.2s; cursor: pointer; }')
    html_parts.append('.mod-card:hover { transform: translateY(-5px); box-shadow: 0 4px 12px rgba(0,0,0,0.15); }')
    html_parts.append('.analysis-section { margin-top: 2rem; padding-top: 2rem; border-top: 2px solid #dee2e6; }')
    html_parts.append('.analysis-section:first-child { border-top: none; padding-top: 0; }')
    html_parts.append('.plot-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(400px, 1fr)); gap: 1.5rem; margin-bottom: 2rem; }')
    html_parts.append('.plot-card { background: white; border-radius: 8px; padding: 1.5rem; box-shadow: 0 2px 4px rgba(0,0,0,0.1); cursor: zoom-in; }')
    html_parts.append('.plot-card img { max-width: 100%; height: auto; border-radius: 4px; cursor: pointer; transition: transform 0.2s; }')
    html_parts.append('.plot-card img:hover { transform: scale(1.02); }')
    html_parts.append('.back-btn { position: fixed; bottom: 2rem; right: 2rem; z-index: 1000; }')
    html_parts.append('.analysis-title { color: #0d6efd; font-weight: 600; margin-bottom: 1.5rem; padding-bottom: 0.5rem; border-bottom: 2px solid #0d6efd; }')
    html_parts.append('</style>')
    html_parts.append('</head>')
    html_parts.append('<body>')
    
    # Navigation bar
    html_parts.append('<nav class="navbar navbar-dark bg-dark sticky-top">')
    html_parts.append('<div class="container-fluid">')
    html_parts.append('<span class="navbar-brand mb-0 h1">dmode Analysis Report</span>')
    html_parts.append(f'<span class="navbar-text text-white-50">Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</span>')
    html_parts.append('</div>')
    html_parts.append('</nav>')
    
    # Main container
    html_parts.append('<div class="container my-5">')
    
    # Front page - Analysis type selector
    html_parts.append('<div id="home-page">')
    html_parts.append('<h1 class="mb-4">Select an Analysis Type</h1>')
    html_parts.append(f'<p class="lead text-muted">Found {len(sorted_analyses)} analysis types with results</p>')
    html_parts.append('<div class="row row-cols-1 row-cols-md-3 g-4 mt-3">')
    
    for analysis in sorted_analyses:
        aid = _sanitize_id(analysis)
        # Count total plots and subsections
        total_plots = sum(len(plots) for plots in analyses[analysis].values())
        subsection_count = len(analyses[analysis])
        analysis_display = analysis.replace('_', ' ').title()
        
        html_parts.append('<div class="col">')
        html_parts.append(f'<div class="card mod-card h-100" onclick="showAnalysis(\'{aid}\')">')
        html_parts.append('<div class="card-body text-center">')
        html_parts.append(f'<h3 class="card-title">{analysis_display}</h3>')
        html_parts.append(f'<p class="card-text text-muted">{subsection_count} section{"s" if subsection_count != 1 else ""}, {total_plots} plot{"s" if total_plots != 1 else ""}</p>')
        html_parts.append('<i class="bi bi-arrow-right-circle" style="font-size: 2rem;"></i>')
        html_parts.append('</div>')
        html_parts.append('</div>')
        html_parts.append('</div>')
    
    html_parts.append('</div>')
    html_parts.append('</div>')
    
    # Analysis detail pages with modification/comparison sections
    for analysis in sorted_analyses:
        aid = _sanitize_id(analysis)
        analysis_display = analysis.replace('_', ' ').title()
        
        html_parts.append(f'<div id="analysis-{aid}" class="modification-page" style="display: none;">')
        html_parts.append(f'<div class="d-flex justify-content-between align-items-center mb-4">')
        html_parts.append(f'<h1>{analysis_display}</h1>')
        html_parts.append('<button class="btn btn-outline-primary" onclick="showHome()">← Back to Overview</button>')
        html_parts.append('</div>')
        
        # Group by subsection (modification type or comparison)
        sorted_subsections = sorted(analyses[analysis].keys())
        
        for idx, subsection in enumerate(sorted_subsections):
            subsection_display = subsection.replace('_', ' ')
            html_parts.append(f'<div class="analysis-section">')
            html_parts.append(f'<h3 class="analysis-title">{subsection_display}</h3>')
            html_parts.append('<div class="plot-grid">')
            
            for plot in analyses[analysis][subsection]:
                escaped_title = html.escape(plot["title"], quote=True)
                escaped_filename = html.escape(plot["filename"], quote=True)
                html_parts.append(
                    f'<div class="plot-card" role="button" tabindex="0" '
                    f'data-plot-title="{escaped_title}" '
                    f'data-plot-src="{plot["data_uri"]}" '
                    'onclick="openPlotModalFromElement(this)" '
                    'onkeypress="if(event.key===\'Enter\') openPlotModalFromElement(this)">'
                )
                html_parts.append(f'<h5 class="mb-3">{plot["title"]}</h5>')
                html_parts.append(f'<img src="{plot["data_uri"]}" alt="{escaped_title}" loading="lazy">')
                html_parts.append(f'<p class="text-muted small mt-2 mb-0">{escaped_filename}</p>')
                html_parts.append('</div>')
            
            html_parts.append('</div>')
            html_parts.append('</div>')
        
        html_parts.append('</div>')
    
    html_parts.append('</div>')
    
    # Modal for zoomed plots
    html_parts.append('<div class="modal fade" id="plotModal" tabindex="-1" aria-labelledby="plotModalLabel" aria-hidden="true">')
    html_parts.append('<div class="modal-dialog modal-xl modal-dialog-centered">')
    html_parts.append('<div class="modal-content">')
    html_parts.append('<div class="modal-header">')
    html_parts.append('<h5 class="modal-title" id="plotModalLabel">Plot Preview</h5>')
    html_parts.append('<button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>')
    html_parts.append('</div>')
    html_parts.append('<div class="modal-body">')
    html_parts.append('<img id="modalPlotImage" src="" alt="Plot" style="width: 100%; height: auto;">')
    html_parts.append('</div>')
    html_parts.append('<div class="modal-footer">')
    html_parts.append('<button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>')
    html_parts.append('<button type="button" class="btn btn-primary" onclick="openPlotInNewTab()">Open in New Tab</button>')
    html_parts.append('</div>')
    html_parts.append('</div>')
    html_parts.append('</div>')
    html_parts.append('</div>')

    # Back to top button
    html_parts.append('<button class="btn btn-primary back-btn" onclick="window.scrollTo({top: 0, behavior: \'smooth\'})">↑ Top</button>')
    
    # Footer
    html_parts.append('<footer class="bg-light text-center py-3 mt-5">')
    html_parts.append('<p class="text-muted mb-0">Generated by dmode analysis pipeline</p>')
    html_parts.append('</footer>')
    
    # Scripts
    html_parts.append('<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>')
    html_parts.append('<script>')
    html_parts.append('let currentPlotDataUri = null;')
    html_parts.append('const plotModal = new bootstrap.Modal(document.getElementById("plotModal"));')
    html_parts.append('function showAnalysis(analysisId) {')
    html_parts.append('  document.getElementById("home-page").style.display = "none";')
    html_parts.append('  document.querySelectorAll(".modification-page").forEach(p => p.style.display = "none");')
    html_parts.append('  document.getElementById("analysis-" + analysisId).style.display = "block";')
    html_parts.append('  window.scrollTo({top: 0, behavior: "smooth"});')
    html_parts.append('}')
    html_parts.append('function showHome() {')
    html_parts.append('  document.querySelectorAll(".modification-page").forEach(p => p.style.display = "none");')
    html_parts.append('  document.getElementById("home-page").style.display = "block";')
    html_parts.append('  window.scrollTo({top: 0, behavior: "smooth"});')
    html_parts.append('}')
    html_parts.append('function openPlotModalFromElement(card) {')
    html_parts.append('  const title = card.getAttribute("data-plot-title") || "Plot Preview";')
    html_parts.append('  const src = card.getAttribute("data-plot-src");')
    html_parts.append('  if (!src) { return; }')
    html_parts.append('  currentPlotDataUri = src;')
    html_parts.append('  document.getElementById("plotModalLabel").textContent = title;')
    html_parts.append('  const img = document.getElementById("modalPlotImage");')
    html_parts.append('  img.src = src;')
    html_parts.append('  img.alt = title;')
    html_parts.append('  plotModal.show();')
    html_parts.append('}')
    html_parts.append('function openPlotInNewTab() {')
    html_parts.append('  if (currentPlotDataUri) { window.open(currentPlotDataUri, "_blank"); }')
    html_parts.append('}')
    html_parts.append('</script>')
    html_parts.append('</body></html>')
    
    outpath = os.path.join(output_dir, report_filename)
    with open(outpath, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(html_parts))
    
    return outpath
