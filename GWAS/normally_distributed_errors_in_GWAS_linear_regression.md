## **Why $\varepsilon_i$ $\sim$ N(0, $\sigma^2$) in GWAS Linear Regression?**
### 1. The Statistical Rationale
The assumption of **normally distributed errors** comes from the **Central Limit Theorem** and the **mathematical properties** we need for valid inference:
```math
\varepsilon_i \sim N(0, \sigma^2)
```
**Translation:** For each individual i, the difference between their **actual BMI** and their **predicted BMI** (from the model) follows a **normal distribution** with:
- **Mean = 0** → Predictions are unbiased on average
- **Variance = $\sigma^2$** → Constant spread of errors (homoscedasticity)
### 2. Why This Assumption is Needed
**Requirement**|**Why Normal Errors**|**What Breaks if Violated**
-|-|-
**Valid t-tests**|t-statistic = β̂/SE(β̂) follows exact t-distribution **only** if errors are normal|p-values become inaccurate
**Confidence Intervals**|95% CI = β̂ ± t* × SE assumes normality|CI coverage incorrect (e.g., nominal 95% might be 90% actual)
**Maximum Likelihood**|Normal errors → least squares = maximum likelihood estimator|Efficiency loss (higher variance)
**Prediction Intervals**|To predict BMI for new individuals|Prediction intervals too narrow/wide
### 3. The Reality check: BMI Might NOT Be Normally Distributed
**BMI itself** is often **not** normally distributed:
```r
# Real BMI distribution (skewed right)
hist(bmi_data, main = "BMI Distribution (Right-Skewed)")
# Shapiro-Wilk test often gives p < 0.001 → NOT normal!
```
**So why can we assume normal errors?** Because of these key points:  
**(A) Modeling Conditional Distributions**  
It not means:
```r
BMI_i ~ N(μ, σ²) ← WRONG!
```
It means:
```r
BMI_i | (genotype, age, sex, PCs) ~ N(β₀ + β₁*SNP + ..., σ²) ← CORRECT!
```
The **conditional distribution** (BMI **given** all predictors) is assumed normal, not the marginal distribution.  
**(B) Central Limit Theorem in Action**  
Even if the **raw phenotype** isn't normal, with large GWAS samples (n > 10,000):
```python
import numpy as np
import matplotlib.pyplot as plt

# Simulate: True relationship + non-normal "biological noise"
n = 10000
genotype = np.random.choice([0, 1, 2], n, p=[0.25, 0.5, 0.25])
true_effect = 0.15 * genotype

# NON-normal biological noise (mixture distribution)
biological_noise = np.random.exponential(scale=2, size=n) - 2   # Mean=0 but skewed

# Measurement error (normal)
measurement_error = np.random.normal(0, 1, n)

# Total error = biological + measurement
epsilon = biological_noise + measurement_error

# BMI = predicted + error
bmi = 25 + true_effect + epsilon

# Check distribution of errors
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.hist(bmi, bins=50, edgecolor='black')
plt.title("BMI Distribution (Skewed)")

plt.subplot(1, 2, 2)
plt.hist(epsilon, bins=50, edgecolor='black')
plt.title("Error Distribution (More Normal)")
plt.show()
```
**Key insight:** The **sum** of many small, independent biological effects tends toward normality by CLT, even if individual components aren't normal.
### 4. Mathematical Derivation
The normal assumption comes from the **linear regression model specification:**
```math
Y_i = X_i^T \beta + \varepsilon_i
```
Where:
- $Y_i$ = BMI for person i
- $X_i$ = [1, genotype, age, sex, PCs...]
- $\beta$ = parameters
- $\varepsilon_i$ = everything else affecting BMI that's **not** in our model

If assume:
1. Many small, independent biological factors affect BMI
2. No single factor dominates
3. Measurement error is normal (usually true)

Then by the **Central Limit Theorem:**
```math
\varepsilon_i = \sum_{k=1}^{K} \text{(small biological effects)}_k + \text{measurement error}
```
Approximately follows N(0, $\sigma^2$) for large K.
### 5. Practical Implications in GWAS
**Scenario 1: Mild Violations (Common)**
```r
# Most GWAS are robust to mild non-normality
# Why? Large sample sizes + focus on β₁, not predictions

# Check with residuals:
model <- lm(BMI ~ genotype + covariates, data = gwas_data)
residuals <- resid(model)

# Q-Q plot
qqnorm(residuals)
qqline(residuals, col = "red")

# Shapiro-Wilk test (but with n > 5000, it's too sensitive)
# shapiro.test(residuals[1:5000])   # Often significant even for minor deviations
```
**GWAS reality:** With n > 10,000, the **sampling distribution of β̂₁** is approximately normal even if errors aren't perfectly normal, thanks to CLT.  
**Scenario 2: Severe Violations (Need Action)**
```r
# Severe right-skew
hist(phenotype) # Heavily right-skewed

# Solution:
# 1. Transform the phenotype
log_phenotype <- log(phenotype + 1) # Log transform
sqrt_phenotype <- sqrt(phenotype)   # Square root transform

# 2. Use rank-based inverse normalization (RINT)
library(RNOmni)
rint_phenotype <- rankNorm(phenotype)   # Forces normal distribution

# 3. Use robust regression methods
library(MASS)
model_robust <- rlm(BMI ~ genotype + covariates)    # Less sensitive to outliers
```
### 6. Special Case: Binary Traits with Linear Regression
Sometimes researchers use linear regression for binary traits (called "linear probability model"):
```r
# For case-control (binary) trait
model_binary <- lm(disease ~ genotype + covariates)
# Here, εᵢ CANNOT be normal because:
# Yᵢ is binary (0/1), so εᵢ must be either:
# εᵢ = 1 - Ŷᵢ (if Yᵢ=1) or εᵢ = -Ŷᵢ (if Yᵢ=0)
# This violates normality assumption!
```
**Solution:** Use **logistic regression** for binary traits.
### 7. Checking the Assumption in Practice
```r
# Comprehensive diagnostic check
diagnose_normality <- function(model) {
    residuals <- resid(model)

    # 1. Histogram
    hist(residuals, breaks=50, main="Residual Distribution",
        xlab="Residuals", col="lightblue")

    # 2. Q-Q plot
    qqnorm(residuals, main="Normal Q-Q Plot")
    qqline(residuals, col="red")

    # 3. Statistical tests (for n < 5000)
    if(length(residuals) < 5000) {
        shapiro <- shapiro.test(residuals)
        cat("Shapiro-Wilk test p-value:", shapiro$p.value, "\n")
    }

    # 4. Skewness and kurtosis
    library(moments)
    skew <- skewness(residuals)
    kurt <- kurtosis(residuals)
    cat("Skewness:", round(skew, 3), "(Normal ≈ 0)\n")
    cat("Kurtosis:", round(kurt, 3), "(Normal ≈ 3)\n")

    # Rule of thumb:
    if(abs(skew) > 1 | abs(kurt - 3) > 1) {
        cat("Warning: Substantial departure from normality\n")
    }
}
```
### 8. What Happens if $\varepsilon_i$ is NOT Normal
Let's simulate the consequences:
```python
import numpy as np
import statsmodels.api as sm
from scipy import stats

def simulate_gwas_nonnormal(n=1000, effect_size=0.2, error_dist='normal', seed=None):
    """Simulate GWAS with different error distributions"""
    if seed is not None:
        np.random.seed(seed)

    # Genotype (MAF=0.3)
    maf = 0.3
    genotype = np.random.binomial(2, maf, n)

    # True effect
    true_beta = effect_size

    # Different error distributions
    if error_dist == 'normal':
        errors = np.random.normal(0, 1, n)
    elif error_dist == 'exponential':
        errors = np.random.exponential(1, n) - 1    # Mean=0
    elif error_dist == 't3':
        errors = stats.t(df=3).rvs(n)   # Heavy tails
    elif error_dist == 'uniform':
        errors = np.random.uniform(-1.732, 1.732, n)    # Var=1
    
    # Phenotype
    phenotype = true_beta * genotype + errors

    # Linear regression
    X = sm.add_constant(genotype)
    model = sm.OLS(phenotype, X).fit()

    return {
        'beta_hat': model.params[1],
        'se': model.bse[1],
        't': model.tvalues[1],
        'p': model.pvalues[1],
        'ci_lower': model.conf_int()[1, 0],
        'ci_upper': model.conf_int()[1, 1]
    }

# Compare different error distributions
distributions = ['normal', 'exponential', 't3', 'uniform']
results = {dist: [] for dist in distributions}
n_simulations = 1000

for dist in distributions:
    for i in range(n_simulations):
        results[dist].append(simulate_gwas_nonnormal(n=5000, error_dist=dist, seed=i))

# Check coverage of 95% CI
print("GWAS Simulation Results (n=5000, effect_size=0.2)")
print("=" * 50)
for dist in distributions:
    covers = [1 if (0.2 >= res['ci_lower'] and 0.2 <= res['ci_upper']) else 0 for res in results[dist]]
    coverage = np.mean(covers)
    print(f"{dist:12s}: 95% CI coverage = {coverage:.3f}")
```
**Typical output:**
```
GWAS Simulation Results (n=5000, effect_size=0.2)
==================================================
normal      : 95% CI coverage = 0.942
exponential : 95% CI coverage = 0.946
t3          : 95% CI coverage = 0.961
uniform     : 95% CI coverage = 0.943
```
The results are all close to 95% coverage even for non-normal errors. This demonstrates an important statistical principle: **For large sample sizes, the Central Limit Theorem ensures that OLS regression coefficients are approximately normally distributed regardless of the error distribution.**
### 9. Modern GWAS Approaches That Relax This Assumption
**(A) Rank-based Inverse Normal Transformation (RINT)**
```r
# Forces phenotype to be normal
library(RNOmni)
phenotype_normalized <- rankNorm(phenotype)

# Now εᵢ ~ N(0, σ²) by construction
model <- lm(phenotype_normalized ~ genotype + covariates)
```
**(B) Robust Standard Errors**
```r
# Sandwich estimator (Huber-White)
library(sandwich)
library(lmtest)

model <- lm(BMI ~ genotype + covariates)
# Regular SE (assumes normal errors)
summary(model)$coefficients["genotype", "std. Error"]

# Robust SE (relaxes normality)
coeftest(model, vcov = vcovHC(model, type = "HC3"))["genotype", "Std. Error"]
```
**(C) Quantile Regression**
```r
# Models median instead of mean (no normality assumption)
library(quantreg)
model_quantile <- rq(BMI ~ genotype + covariates, tau = 0.5)
summary(model_quantile)
```
### 10. Summary: Why Assume $\varepsilon_i$ ~ N(0, $\sigma^2$)
1. **Mathematical convenience:** Enables exact inference (t-tests, Cls)
2. **Theoretical justification:** CLT suggests approximate normality for large samples
3. **Practical robustness:** GWAS with n > 10,000 are fairly robust to violations
4. **Model checking:** We can (and should) check residuals
5. **Remedies exist:** Transformations, robust methods, nonparametric approaches

**Bottom line for GWAS:** The normal error assumption is a **working model** that's:
- Reasonable for many continuous traits (especially after transformation)
- Fairly robust with large samples
- Checkable via diagnostic plots
- Fixable when problematic

**Most importantly:** The primary goal in GWAS is **valid inference on $\beta_1$** (is SNP associated?), not perfect prediction of individual phenotypes. For this purpose, the normal error assumption often works well enough in practice, especially with the sample sizes in modern GWAS (n > 100,000).

---
**Contributor:** W.S.