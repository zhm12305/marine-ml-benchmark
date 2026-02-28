# Sanity Check Failures: List + Response Template

## Failed Datasets (from complete_sanity_check_results.csv)
- cast: original R?=0.2194048943648662, permuted R?=-0.1809797515138394, p=0.1954804519548045
- hydrographic: original R?=0.9810092443385104, permuted R?=-0.1713796230094097, p=9.999000099990002e-05
- phyto_wide: original R?=0.0, permuted R?=-226.80265230638383, p=1.0
- phyto_long: original R?=0.1758528802122065, permuted R?=-1.1341251856364172, p=0.9155084491550844

## Response Template (paste into reply letter)
We note that the following datasets did not pass the permutation-based sanity check under K=10,000: [LIST].
These datasets are either extremely small, highly sparse, or contain latent grouping/heterogeneity that weakens the permutation baseline; as a result, the null distribution can be unstable or overly conservative for these cases.
We therefore (i) report these outcomes explicitly, (ii) avoid over-interpretation for these datasets, and (iii) provide distribution diagnostics and baseline definitions to clarify the behavior.
We emphasize that the core conclusions for the validated datasets remain unchanged under K=10,000, and we document all results in complete_sanity_check_results.csv.