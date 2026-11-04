import os

import matplotlib as mpl
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------- #
# Pest Management Science artwork settings
#   Initial submission: JPEG (EPS and PDF are not accepted)
#   Revision: TIF or JPEG, one file per figure
#   Resolution: >= 800 dpi for graphs and line drawings at final size
# --------------------------------------------------------------------------- #
MM = 1.0 / 25.4
FULL_MM = 170.0
HALF_MM = 82.0
MAX_H_MM = 240.0
PNG_DPI = 600
SUBMIT_DPI = 800
TIFF_DPI = SUBMIT_DPI          # kept for backward compatibility
JPEG_QUALITY = 95
PANEL_LETTERS = "ABCDEFGH"
DEFAULT_FORMATS = tuple(f.strip() for f in os.environ.get("FIG_FORMATS", "png,jpg,tif").split(",")
                        if f.strip())
COLOR_MODE = os.environ.get("FIG_COLOR", "rgb").strip().lower()
CMYK_ICC_CANDIDATES = [
    os.environ.get("FIG_CMYK_ICC", ""),
    os.path.join(os.environ.get("SystemRoot", r"C:\Windows"),
                 "System32", "spool", "drivers", "color", "RSWOP.icm"),
]


def apply_style(base_pt=8.5):
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "font.size": base_pt,
        "axes.titlesize": base_pt + 1.0,
        "axes.titleweight": "bold",
        "axes.labelsize": base_pt,
        "axes.labelweight": "regular",
        "xtick.labelsize": base_pt - 0.5,
        "ytick.labelsize": base_pt - 0.5,
        "legend.fontsize": base_pt - 0.5,
        "figure.titlesize": base_pt + 1.0,
        "axes.linewidth": 0.7,
        "grid.linewidth": 0.5,
        "lines.linewidth": 1.1,
        "patch.linewidth": 0.6,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.minor.width": 0.5,
        "ytick.minor.width": 0.5,
        "axes.edgecolor": "#000000",
        "text.color": "#000000",
        "axes.labelcolor": "#000000",
        "xtick.color": "#000000",
        "ytick.color": "#000000",
        "legend.frameon": False,
        "figure.dpi": 300,
        "savefig.dpi": PNG_DPI,
        "savefig.bbox": "standard",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


# --------------------------------------------------------------------------- #
# Figure size and layout helpers
# --------------------------------------------------------------------------- #
def size(width="full", height_mm=None, aspect=None):
    w_mm = float({"full": FULL_MM, "half": HALF_MM}.get(width, width))
    if height_mm is None:
        height_mm = w_mm * (aspect if aspect else 0.62)
    return (w_mm * MM, height_mm * MM)


def rows_height_mm(nrow, per_row_mm, extra_mm=0.0):
    return min(nrow * per_row_mm + extra_mm, MAX_H_MM)


def add_row_labels(fig, pairs, fontsize=9.0, gap_mm=1.6):
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    w_in = fig.get_size_inches()[0]
    need = (fontsize * 1.6 / 72.0 + gap_mm * MM) / w_in
    x0 = min(ax.get_tightbbox(rend).transformed(inv).x0 for ax, _ in pairs)
    if x0 < need:
        fig.subplots_adjust(left=fig.subplotpars.left + (need - x0))
        fig.canvas.draw()
    gap = gap_mm * MM / w_in
    for ax, lab in pairs:
        bb = ax.get_tightbbox(rend).transformed(inv)
        pos = ax.get_position()
        fig.text(bb.x0 - gap, (pos.y0 + pos.y1) / 2, lab, rotation=90,
                 va="center", ha="right", fontsize=fontsize, fontweight="bold", color="black")


# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #
def _has_transparency(fig):
    for art in fig.findobj():
        a = getattr(art, "get_alpha", lambda: None)()
        if a is not None and a < 1.0:
            return True
    return False


def _to_cmyk(img):
    from PIL import ImageCms
    for prof in CMYK_ICC_CANDIDATES:
        if prof and os.path.isfile(prof):
            out = ImageCms.profileToProfile(img, ImageCms.createProfile("sRGB"),
                                            ImageCms.getOpenProfile(prof), outputMode="CMYK",
                                            renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC)
            with open(prof, "rb") as fh:
                return out, fh.read()
    return img.convert("CMYK"), None


def _flatten_rgb(im):
    from PIL import Image
    im.load()
    if im.mode in ("RGBA", "LA", "PA") or (im.mode == "P" and "transparency" in im.info):
        im = im.convert("RGBA")
        flat = Image.new("RGB", im.size, "white")
        flat.paste(im, mask=im.split()[-1])
        return flat
    return im.convert("RGB")


def _finalize_raster(path, dpi):
    from PIL import Image
    with Image.open(path) as im:
        flat = _flatten_rgb(im)
    icc = None
    if COLOR_MODE == "cmyk":
        flat, icc = _to_cmyk(flat)
    if os.path.splitext(path)[1].lower() in (".jpg", ".jpeg"):
        kw = {"quality": JPEG_QUALITY, "subsampling": 0, "dpi": (dpi, dpi)}
    else:
        kw = {"compression": "tiff_lzw", "dpi": (dpi, dpi)}
    if icc:
        kw["icc_profile"] = icc
    flat.save(path, **kw)


def png_to_submission(src, target_mm=FULL_MM, formats=("jpg", "tif")):
    """Convert an externally exported PNG (e.g. Fig. 1) to JPEG and TIFF.

    The dpi written to the file header is derived from the pixel width and the
    final printed width, so the image is not resampled.
    """
    from PIL import Image
    stem = os.path.splitext(src)[0]
    with Image.open(src) as im:
        base = _flatten_rgb(im)
    dpi = int(round(base.size[0] * 25.4 / float(target_mm)))
    written = []
    for ext in formats:
        dst = f"{stem}.{ext}"
        if ext == "jpg":
            base.save(dst, quality=JPEG_QUALITY, subsampling=0)
        else:
            base.save(dst, compression="tiff_lzw")
        _finalize_raster(dst, dpi)
        written.append(dst)
    if dpi < SUBMIT_DPI:
        print(f"Warning: {os.path.basename(src)} is {dpi} dpi at {target_mm:.0f} mm "
              f"(< {SUBMIT_DPI} dpi).")
    return written


def save_journal(stem, slug, outdir, fig=None, formats=None, tight=True):
    fig = fig or plt.gcf()
    formats = DEFAULT_FORMATS if formats is None else formats
    os.makedirs(outdir, exist_ok=True)
    if tight:
        try:
            fig.tight_layout()
        except Exception:
            pass
    if "eps" in formats and _has_transparency(fig):
        formats = tuple(f for f in formats if f != "eps")
    for ext in formats:
        if ext == "png":
            fig.savefig(os.path.join(outdir, f"{stem}_{slug}.png"), dpi=PNG_DPI, bbox_inches=None)
        elif ext in ("jpg", "jpeg"):
            path = os.path.join(outdir, f"{stem}.jpg")
            fig.savefig(path, dpi=SUBMIT_DPI, bbox_inches=None,
                        pil_kwargs={"quality": JPEG_QUALITY, "subsampling": 0})
            _finalize_raster(path, SUBMIT_DPI)
        elif ext in ("tif", "tiff"):
            path = os.path.join(outdir, f"{stem}.tif")
            fig.savefig(path, dpi=SUBMIT_DPI, bbox_inches=None,
                        pil_kwargs={"compression": "tiff_lzw"})
            _finalize_raster(path, SUBMIT_DPI)
        elif ext == "eps":
            fig.savefig(os.path.join(outdir, f"{stem}.eps"), format="eps", bbox_inches=None)
        elif ext == "pdf":
            fig.savefig(os.path.join(outdir, f"{stem}.pdf"), format="pdf", bbox_inches=None)
        elif ext == "svg":
            fig.savefig(os.path.join(outdir, f"{stem}.svg"), format="svg", bbox_inches=None,
                        metadata={"Date": None})
    plt.close(fig)


if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    width = FULL_MM
    if len(args) >= 2 and args[0] == "--width-mm":
        width, args = float(args[1]), args[2:]
    for f in args:
        png_to_submission(f, target_mm=width)