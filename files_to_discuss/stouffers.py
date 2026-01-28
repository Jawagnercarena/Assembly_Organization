import numpy as np
from scipy.stats import combine_pvalues
 
pvals = np.array([0.77, 0.011, 0.052])
 
# Example weights: effective sample size for 2-group comparisons
n_shared   = np.array([ 295, 1608, 178])
n_disjoint = np.array([1849, 2049, 1220])
weights = np.sqrt((n_shared * n_disjoint) / (n_shared + n_disjoint))
 
stat, p_meta = combine_pvalues(pvals, method="stouffer", weights=weights)
print("Z_meta:", stat, "p_meta:", p_meta)

print("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")

import numpy as np
from scipy.stats import norm
 
def stouffer_z(pvals, signs=None, weights=None, two_sided=True):
    """
    Signed (direction-aware) Stouffer meta-analysis.
 
    pvals: array-like of p-values
    signs: array-like of +1/-1 indicating direction of effect (required for signed)
           e.g., +1 if Disjoint>Shared, -1 if Shared>Disjoint
           If None, performs unsigned Stouffer (all signs=+1).
    weights: array-like weights (e.g., sqrt(n_eff)). If None, equal weights.
    two_sided: if True, interpret pvals as two-sided and convert using p/2
    """
    pvals = np.asarray(pvals, dtype=float)
    if np.any((pvals <= 0) | (pvals > 1)):
        raise ValueError("All p-values must be in (0, 1].")
 
    if signs is None:
        signs = np.ones_like(pvals)
    else:
        signs = np.asarray(signs, dtype=float)
        # allow any positive/negative values, but reduce to sign
        signs = np.sign(signs)
        if np.any(signs == 0):
            raise ValueError("signs must be nonzero (use +1 or -1).")
 
    if weights is None:
        weights = np.ones_like(pvals)
    else:
        weights = np.asarray(weights, dtype=float)
        if np.any(weights < 0):
            raise ValueError("weights must be nonnegative.")
        if np.all(weights == 0):
            raise ValueError("At least one weight must be > 0.")
 
    # Convert p -> |z|
    if two_sided:
        p_use = np.clip(pvals / 2.0, 1e-300, 1.0)  # avoid inf
    else:
        p_use = np.clip(pvals, 1e-300, 1.0)
 
    z_abs = norm.isf(p_use)          # = Φ^{-1}(1 - p_use)
    z_i = signs * z_abs              # apply direction
 
    Z = np.sum(weights * z_i) / np.sqrt(np.sum(weights**2))
 
    # two-sided meta p-value from combined Z
    p_meta = 2.0 * norm.sf(abs(Z))
    return Z, p_meta, z_i
 
# ---- Example with your three plots ----
pvals = [0.77, 0.011, 0.052]
 
# Direction from the mean±SEM panels:
# +1 means Disjoint > Shared; -1 means Shared > Disjoint
signs = [-1, +1, +1]  # change if any panel flips
 
n_shared   = np.array([ 295, 1608, 178])
n_disjoint = np.array([1849, 2049, 1220])
weights = np.sqrt((n_shared * n_disjoint) / (n_shared + n_disjoint))
 
Z, p_meta, z_each = stouffer_z(pvals, signs=signs, weights=weights, two_sided=True)
print("per-study z:", z_each)
print("Z_meta:", Z)
print("p_meta:", p_meta)

print("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")

import numpy as np
from scipy.stats import norm
 
# Example data: Z-scores and sample sizes for four two-sample t-tests
# Each row is a test: [Z_score, n1, n2]
tests = np.array([
    [-0.2923749, 295, 1849],
    [2.54269882, 1608, 2049],
    [1.94313375, 178, 1220],
])

# Calculate total sample size for each test
n_total = tests[:, 1] + tests[:, 2]
 
# Weights are sqrt(total sample size)
weights = np.sqrt(n_total)
 
# Stouffer's Z-score method
combined_z = np.sum(tests[:, 0] * weights) / np.sqrt(np.sum(weights**2))
 
# Two-tailed p-value for the combined Z-score
p_value = 2 * (1 - norm.cdf(np.abs(combined_z)))
 
print(f"Combined Z-score: {combined_z}")
print(f"Combined p-value: {p_value}")