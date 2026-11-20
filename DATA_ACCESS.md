# Data access and provenance

Code repository for the study: [Kayseri Wind Farm Site Screening with GIS and SVM](https://github.com/msaitd/kayseri-wind-site-screening).

## Public repository files

The public sample is a province-wide frequency table derived from first-stage positive cells inside Kayseri. It contains 149 populated score bins. The separate reference outputs summarise study analyses. Coordinates, spatially indexed predictor matrices, project polygons, source GIS layers and editorial correspondence are outside this repository.

| Column in `sample_data/screening_score_bins.csv` | Definition |
| --- | --- |
| `weighted_sum` | Integer sum of the nine ordinal ranks weighted by 23, 14, 6, 4, 6, 3, 6, 3 and 4 in K1–K9 order. The value −1 means at least one criterion excluded the cell. |
| `cell_count` | Number of provincial first-stage positive cells in that bin |
| `area_km2` | Sum of WGS 84 ellipsoidal cell areas in that bin, in square kilometres |

For eligible cells, divide the weighted sum by 69. Scores at or below 1.75 are high suitability, scores above 1.75 and at or below 2.75 are suitable, and larger scores are moderate suitability. The exclusion bin is counted before applying these cutoffs. Areas are accumulated using the latitude of each source row. The public sample supports threshold recalculation at the baseline weights. Reweighting individual criteria requires the S1 inputs.

## Supporting Information S1 and original inputs

The full reproduction package is designated **Supporting Information S1** for the manuscript. Its README describes the input paths, source lineage and commands. During review, enquiries about S1 or access to study-specific inputs can be directed to the corresponding author, Mehmet Sait Dundar, at msaitdundar@erciyes.edu.tr. The applicable provider terms govern availability of the underlying institutional layers.

| Input or product | Access or provenance |
| --- | --- |
| Reference points, original split, Weka fit, binary prediction and nine classified criterion grids | The manuscript S1 contains the archived or derived study files. Table 4 gives the interval rules. |
| Sampled predictor arrays and coarse-resolution inputs | The S1 `inputs/` directory preserves row order and alignment for the full workflow. |
| Model comparison, ROC, wind, land-cover and coarse-grid result summaries | Selected reference outputs are in this repo. S1 contains the larger collection, including the spatial patch catalogue. |
| Wind speed and power | Global Wind Atlas v4, developed by the Technical University of Denmark with partner contributions. Country-level products were cropped to Kayseri for the study. [GIS downloads](https://globalwindatlas.info/en/download/gis-files) · [Dataset record](https://doi.org/10.11583/DTU.28955267) · [Provider terms](https://globalwindatlas.info/about/TermsOfUse) |
| Erkilet station | NOAA Global Surface Summary of the Day, station 17195099999, 2008–2017. Annual daily files are retrieved by `wind_resource_audit.py`. [NOAA dataset record](https://www.ncei.noaa.gov/access/metadata/landing-page/bin/iso?id=gov.noaa.ncdc%3AC00516) |
| Independent land-cover checks | ESA WorldCover 2020 v100 and 2021 v200. © ESA WorldCover project, CC BY 4.0. [2020 record](https://doi.org/10.5281/zenodo.5571936) · [2021 record](https://doi.org/10.5281/zenodo.7254221) |
| Original terrain, infrastructure, fault and conservation data | See Table 1 and its provider references in the manuscript. Some survey dates and the original composite land-cover code legend could not be recovered. ESA WorldCover was used for an independent check. |

The grouped score table is an author-derived analytical summary. The [MIT License](LICENSE) covers the contributors’ original software. Dataset distribution and reuse are governed separately by the applicable rights and provider terms.
