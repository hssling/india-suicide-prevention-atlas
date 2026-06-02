"""Build the IJMR submission bundle for meta_suicide_reconciliation_india."""
from __future__ import annotations
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANUSCRIPT = ROOT / "manuscript"
OUTPUTS = ROOT / "outputs"
OUT_FIGS = OUTPUTS / "figures"
OUT_TABLES = OUTPUTS / "tables"
SUBMISSION = ROOT / "submission"
BUNDLE = SUBMISSION / "IJMR_submission_ready"

MANUSCRIPT_FILES = [
    "00_title_page",
    "01_blinded_manuscript_IJMR",
    "02_cover_letter_IJMR",
    "03_declarations",
    "04_supplementary",
    "strobe_checklist",
    "gather_checklist",
    "robust_mc_checklist",
]


def _pandoc_available() -> bool:
    try:
        return subprocess.run(["pandoc", "--version"], capture_output=True, text=True, timeout=10).returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _render(src: Path, dest_dir: Path) -> dict[str, str]:
    status = {}
    for fmt in ("docx", "pdf"):
        out = dest_dir / (src.stem + f".{fmt}")
        r = subprocess.run(["pandoc", str(src), "-o", str(out)], capture_output=True, text=True, timeout=120)
        status[fmt] = "OK" if r.returncode == 0 else f"SKIPPED ({r.stderr[:80]})"
    return status


def main() -> None:
    BUNDLE.mkdir(parents=True, exist_ok=True)
    pandoc_ok = _pandoc_available()
    render_log = []

    for stem in MANUSCRIPT_FILES:
        src = MANUSCRIPT / (stem + ".md")
        if not src.exists():
            print(f"WARNING: {src} not found")
            continue
        shutil.copy2(src, BUNDLE / src.name)
        print(f"copied {src.name}")
        if pandoc_ok:
            for fmt, st in _render(src, BUNDLE).items():
                render_log.append(f"  {stem}.{fmt}: {st}")

    # figures.docx
    figures = MANUSCRIPT / "figures.docx"
    if figures.exists():
        shutil.copy2(figures, BUNDLE / figures.name)
        print(f"copied {figures.name}")

    # tables (xlsx + csv)
    for f in sorted(list(OUT_TABLES.glob("*.xlsx")) + list(OUT_TABLES.glob("*.csv"))):
        shutil.copy2(f, BUNDLE / f.name)
    print(f"copied {len(list(OUT_TABLES.glob('*')))} table files")

    # figure files (all formats)
    for f in sorted(OUT_FIGS.iterdir()):
        if f.is_file():
            shutil.copy2(f, BUNDLE / f.name)
    print(f"copied {len(list(OUT_FIGS.glob('*')))} figure files")

    readme = f"""# IJMR Submission Bundle -- meta_suicide_reconciliation_india

**Title:** State-level reconciliation and equity decomposition of suicide mortality in India: triangulating NCRB and GBD with a Bayesian hierarchical model, 2023

**Author:** Dr Siddalingaiah H S (sole author)
**Bundle created:** 2026-05-30
**Pandoc available:** {pandoc_ok}

## Headline numbers (verified, in manuscript)

- National NCRB count 2023: 170,380
- National GBD modelled 2023: 199,474
- National NCRB/GBD ratio: 0.854
- Avertable Scenario A (Indian top-quartile): 69,496 deaths/yr
- Avertable Scenario B (national average): 24,895 deaths/yr
- Avertable Scenario C (WHO global): 81,753 deaths/yr
- SII state-level: 15.79 / 100,000
- RII state-level: 1.09
- Concentration Index state-level: +0.18
- Theil index: 0.063
- Sex SII: -22.62 / 100,000 (males higher)
- Bayesian max R-hat: 1.0100
- Bayesian min ESS bulk: 309 (borderline; longer chain run recommended pre-publication)

## File manifest

### Manuscript
- `00_title_page.{{md,docx,pdf}}` -- with author identification
- `01_blinded_manuscript_IJMR.{{md,docx,pdf}}` -- main manuscript with tables and figure legends appended
- `02_cover_letter_IJMR.{{md,docx,pdf}}`
- `03_declarations.{{md,docx,pdf}}`
- `04_supplementary.{{md,docx,pdf}}` -- S1-S8

### Checklists
- `strobe_checklist.{{md,docx,pdf}}`
- `gather_checklist.{{md,docx,pdf}}`
- `robust_mc_checklist.{{md,docx,pdf}}`

### Tables (4)
- `table1_burden_2023.{{xlsx,csv}}`
- `table2_avertable_burden.{{xlsx,csv}}`
- `table3_equity_indices.{{xlsx,csv}}`
- `table4_bayesian_rates.{{xlsx,csv}}`

### Figures (6, in 3 formats each)
- `fig1_ncrb_gbd_ratio_{{colour,bw}}.{{png,pdf,tif}}`
- `fig2_avertable_scenario_C_{{colour,bw}}.{{png,pdf,tif}}`
- `fig3_forest_three_scenarios.{{png,pdf,tif}}`
- `fig4_concentration_curve.{{png,pdf,tif}}`
- `fig5_bayesian_state_rates.{{png,pdf,tif}}`
- `fig6_under_registration_drivers.{{png,pdf,tif}}`
- `figures.docx` (all 6 figures embedded with legends)

## Pre-submission user actions

- [ ] Bayesian re-fit with 4 chains x 4,000 draws to push ESS over 1,000 (currently 309; borderline)
- [ ] Verify ORCID, ICMJE COI forms ready for upload
- [ ] Optional: NCRB 2018-2022 historical fetch for trend layer (current paper is 2023 cross-sectional)

## Pandoc render log

{chr(10).join(render_log)}
"""
    (BUNDLE / "SUBMISSION_README.md").write_text(readme, encoding="utf-8")
    print("wrote SUBMISSION_README.md")

    zip_path = SUBMISSION / "IJMR_Submission_Final"
    shutil.make_archive(str(zip_path), "zip", str(BUNDLE))
    print(f"created {zip_path}.zip")


if __name__ == "__main__":
    sys.exit(main() or 0)
