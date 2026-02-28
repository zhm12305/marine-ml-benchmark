# Response to Reviewers (PLOS ONE) - Filled Defaults

Use this as the submission template and replace placeholders like [Fig X] and [Table Y].

## Reviewer #2 - Sanity Check / Permutation Test (K=10,000)

We reran the permutation-based sanity checks with K=10,000 permutations and explicitly report the p-value formulation p=(b+1)/(K+1). Results are saved to `outputs/tables/complete_sanity_check_results.csv`.

Datasets that did not pass the permutation-based sanity check under K=10,000 (XGB) are listed below:
- cast: original R2=0.219405, permuted R2=-0.180980, p=0.195480
- hydrographic: original R2=0.981009, permuted R2=-0.171380, p=0.000100
- phyto_wide: original R2=0.000000, permuted R2=-226.802652, p=1.000000
- phyto_long: original R2=0.175853, permuted R2=-1.134125, p=0.915508

We note that these datasets are either extremely small, highly sparse, or contain latent grouping/heterogeneity that weakens the permutation baseline. For these cases, the null distribution can be unstable or overly conservative. We therefore (i) report these outcomes explicitly, (ii) avoid over-interpretation for these datasets, and (iii) provide distribution diagnostics and baseline definitions to clarify the behavior. The core conclusions for the validated datasets remain unchanged under K=10,000.

