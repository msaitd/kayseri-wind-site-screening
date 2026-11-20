# Kayseri Wind Farm Site Screening with GIS and SVM

Analysis code and research materials for *Wind farm site screening in Kayseri Province, Türkiye, using geographic information systems and support vector machines*, by Abdurrahman Eymen, Burak Dundar and Mehmet Sait Dundar. The study uses a support vector machine to identify cells resembling known wind farm footprints, then applies nine geographic criteria to screen and rank those cells. The map represents a regional planning screen. Site feasibility requires project-level investigation.

**Repository** [https://github.com/msaitd/kayseri-wind-site-screening](https://github.com/msaitd/kayseri-wind-site-screening)

## Reproduce the screening areas

From the repository root, run the example with Python 3.12. It uses the standard library and the bundled study data. It works offline.

```shell
python examples/area_by_grade.py
```

On Windows, `py -3.12 examples/area_by_grade.py` is another way to select Python.

| Screening group | Cells | Area (km²) |
| --- | ---: | ---: |
| First-stage candidates inside Kayseri | 863,207 | 647.46 |
| Excluded by at least one geographic criterion | 569,573 | 427.19 |
| High suitability | 4,083 | 3.07 |
| Suitable | 228,949 | 171.85 |
| Moderate suitability | 60,602 | 45.35 |
| Total retained | 293,634 | 220.27 |

The three suitability classes partition the retained area. The public sample groups provincial candidate cells by their weighted GIS score and preserves their ellipsoidal area. It has no point coordinates, project identifiers or infrastructure geometries. It supports the area calculation in Table 7. Rebuilding the map and refitting the classifier require the spatial inputs described below.

## Repository contents

| Path | Use |
| --- | --- |
| `sample_data/screening_score_bins.csv` | Distribution of candidate cells and area by score |
| `examples/` | Calculate class totals or rebuild the score distribution from S1 |
| `published_results/` | Selected machine-readable reference outputs from the manuscript revision, including the ROC curve, model comparisons, coarse-grid tests, land-cover counts and wind-resource comparison |
| `spatial.py`, `models.py`, `regional.py`, `footprints.py`, `coarse.py` | Reproduce the spatial screen, classifier comparisons, validation and sensitivity analyses with S1 |
| `wind_resource_audit.py`, `new_candidates.py`, `figures.py` | Check the station and atlas, count land-cover classes outside project polygons, and generate the study figures |
| `historical_workflow/` | Earlier Python and PowerShell scripts for sample preparation, Weka fitting and map prediction in the original project structure |
| `requirements.txt` | Python environment used for the full revision analyses |
| `CITATION.cff` | Software citation metadata |

## Full manuscript analyses

Supporting Information S1 has the directory layout and study-specific inputs used by the reproduction scripts. Unpack the complete S1 archive in a separate working directory and run the commands **from its root**. Obtain S1 and the original inputs as described in [Data access and provenance](DATA_ACCESS.md). The public example above works immediately with the files in this repository.

```shell
python -m venv .venv
```

Activate the environment with `.venv\Scripts\activate.bat` in Windows Command Prompt or `source .venv/bin/activate` in a POSIX shell. Then run:

```shell
python -m pip install -r requirements.txt
python spatial.py
python models.py
python regional.py
python footprints.py
python coarse.py
python wind_resource_audit.py
python new_candidates.py
python figures.py
```

Run the commands in order. The scripts write to `results/` and S1 supplies reference outputs in `published_results/`. Values involving processing time depend on hardware. The optional historical Weka reproduction requires Java and Weka. Instructions for the Weka run, optional provider downloads and project-specific checks are in the S1 README. To derive a fresh copy of the public CSV, run the following command from this repository after installing the full dependencies. Replace the example input path with the unpacked S1 directory.

```shell
python examples/build_score_bins.py "/path/to/unpacked/S1" score_bins_rebuilt.csv
```

## Citation and reuse

Use the author list in [CITATION.cff](CITATION.cff) and record the release tag or commit you used. The contributors license their original Python and PowerShell code under the [MIT License](LICENSE). Third-party software and data products retain their own terms. [Data access and provenance](DATA_ACCESS.md) lists the sources, scope of the public sample and corresponding author contact.
