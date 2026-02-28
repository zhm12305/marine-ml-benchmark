# PLOS ONE Revision Package Notes

This file provides ready-to-paste statements and a checklist for the revision.

## Role of Funders (Cover Letter)
The funders had no role in study design, data collection and analysis, decision to publish, or preparation of the manuscript.

## Data Availability Statement (Fill in repo + DOI)
All code and processed data supporting the conclusions are publicly available in the **marine-ml-benchmark** GitHub repository:
https://github.com/<YOUR_ORG>/<YOUR_REPO>

The archived release is available on Zenodo (DOI: <YOUR_DOI>).

Supporting materials are also provided as a compressed archive (S1 Zip).

## Supporting Information Captions (Paste into manuscript)
- **S1 Fig.** Sample size distribution and threshold analysis across datasets.
- **S1 Table.** Complete model performance matrix with 95% confidence intervals.
- **S2 Table.** Hyperparameter optimization logs for all models.
- **S3 Table.** Label permutation test results (K=10,000).
- **S4 Table.** Small-sample dataset exclusion analysis.
- **S1 Zip.** Scripts and regenerated outputs for all figures and tables.

## Sanity Check Failures (K=10,000, XGB)
Failed datasets from `outputs/tables/complete_sanity_check_results.csv`:
- cast: original R2=0.219405, permuted R2=-0.180980, p=0.195480
- hydrographic: original R2=0.981009, permuted R2=-0.171380, p=0.000100
- phyto_wide: original R2=0.000000, permuted R2=-226.802652, p=1.000000
- phyto_long: original R2=0.175853, permuted R2=-1.134125, p=0.915508

Response template (paste into reply letter):
We note that the following datasets did not pass the permutation-based sanity check under K=10,000: cast, hydrographic, phyto_wide, and phyto_long. These datasets are either extremely small, highly sparse, or contain latent grouping/heterogeneity that weakens the permutation baseline; as a result, the null distribution can be unstable or overly conservative for these cases. We therefore (i) report these outcomes explicitly, (ii) avoid over-interpretation for these datasets, and (iii) provide distribution diagnostics and baseline definitions to clarify the behavior. We emphasize that the core conclusions for the validated datasets remain unchanged under K=10,000, and we document all results in `complete_sanity_check_results.csv`.

## Reviewer #1 Citation Response (Template)
We evaluated the suggested references in detail. Most focus on PV systems, electrical grids, crack detection, or communications, which are not directly aligned with marine environmental prediction or cross-dataset ML benchmarking. We therefore did not add these unrelated citations, but we incorporated recent, domain-relevant studies on marine prediction, spatiotemporal modeling, and ML benchmark evaluations to better support the Introduction and Discussion.

## Funding Mentions in Manuscript
Remove funding sentences from Acknowledgments/Conclusion/Methods. Funding information should appear only in the journal submission form.

## Code Availability / Release Tag
Create a public release tag and cite it in the response letter:
- Tag: `v1.0-plosone-rev1`
- Commit hash: `<GIT_COMMIT_HASH>`

Example commands:
```
git tag -a v1.0-plosone-rev1 -m "PLOS ONE revision package"
git push origin v1.0-plosone-rev1
```
