## **Part 1: Quantitative Traits (Linear Regression)**
**Example: BMI**  
### Step 1: Data structure
We have 10,000 individuals with:
- Phenotype: Body Mass Index (BMI, continuous)
- Genotype: SNP rs12345, coded as 0, 1, 2 (additive model)
- Covariates: Age, sex, 10 genetic principal components
```r
# Simulated data structure
ID      BMI     rs12345 Age     Sex     PC1     ...     PC10
001     23.1    0       45      0       -0.02   ...     0.01
002     27.8    1       52      1       0.15    ...     -0.03
003     31.4    2       38      0       -0.07   ...     0.02
...
```
### Step 2: Model Specification
```math
BMI_i = \beta_0 + \beta_1 * (rs12345_i) + \beta_2 * Age_i + \beta_3 * Sex_i + \sum_{1}^{10} (\beta_{3+j} * PCj_i) + \varepsilon_i
```
where:
- $\beta_1$ is our parameter of interest (effect of SNP on BMI)
- $\varepsilon_i$ ~ N(0, $\sigma^2$) (normally distributed errors)
### Step 3: Null Hypothesis
$H_0$: $\beta_1$ = 0 (SNP has no effect on BMI)  
$H_1$: $\beta_1$ &ne; 0 (SNP affects BMI)
### Step 4: Run Linear Regression
Using R's `lm()` function:
```r
model <- lm(BMI ~ rs12345 + Age + Sex + PC1 + PC2 + ... + PC10,
            data = gwas_data)
summary(model)

# Output:
# Coefficients:
#               Estimate std. Error t value Pr(>|t|)
# (Intercept)   22.34567    0.12345 181.000 < 2e-16 ***
# rs12345       0.15678     0.02345   6.685 2.35e-11 ***
# Age           0.04567     0.00321  14.227 < 2e-16 ***
# Sex           0.78901     0.04567  17.275 < 2e-16 ***
```
### Step 5: p-value Calculation
From output: β̂₁ = 0.15678, SE(β̂₁) = 0.02345
#### Step 5.1: Calculate t-statistic
```
t = β̂₁ / SE(β̂₁) = 0.15678 / 0.02345 = 6.685
```
#### Step 5.2: Determine degrees of freedom
```
df = n - p - 1 = 10000 - 13 - 1 = 9986
```
(13 predictors: intercept + SNP + Age + Sex + 10 PCs)
#### Step 5.3: Calculate two-tailed p-value
```r
# In R:
p_value <- 2 * pt(-abs(6.685), df = 9986)
p_value # = 2.35e-11
```
Mathematically:
```r
p = 2 × P(T ≥ |6.685|) where T ~ t-distribution with 9986 df
p = 2 × [1 - F_t(6.685)]
  = 2 × [1 - 0.9999999998825]   # Very close to 1
  = 2 × 0.0000000001175
  = 2.35 × 10⁻¹¹
```
### Step 6: Interpretation
- **Effect Size:** Each additional effect allele increases BMI by **0.157 kg/m²** on average
- **Statistical significance:** p = 2.35 × $10^{-11}$ << 5 × $10^{-8}$
- **Conclusion:** Strong evidence against $H_0$. This SNP is significantly associated with BMI after adjusting for covariates.
### Step 7: Visualization
```r
# QQ-plot of test statistics
expected <- qchisq(ppoints(10000), df = 1)
observed <- (summary(model)$coefficients[-1, "t value"])^2
plot(expected, observed, main = "QQ-plot for BMI GWAS")
abline(0, 1, col = "red")
```
## **Part 2: Dichotomous Traits (Logistic Regression)**
**Example: Type 2 Diabetes**
### Step 1: Data structure
We have 5000 cases and 15000 controls:
- Phenotype: T2D (1 = case, 0 = control)
- Genotype: SNP rs67890, coded as 0, 1, 2
- Covariates: Age, Sex, BMI, 10 PCs
```r
# Data structure
ID  T2D rs67890 Age Sex BMI     PC1     ...
001  1     0     65  1  28.4    -0.02   ...
002  0     1     52  0  24.1     0.15   ...
003  1     2     58  1  31.2    -0.07   ...
...
```
### Step 2: Model Specification
```
log[P(T2D_i = 1) / (1 - P(T2D_i = 1))] = β₀ + β₁*(rs67890_i) + covariates
```
equivalently:
```
logit(P(T2D_i = 1)) = β₀ + β₁*(rs67890_i) + β₂*Age_i + ...
```
### Step 3: Null Hypothesis
```
H₀: β₁ = 0  (SNP not associated with T2D risk)
H₁: β₁ ≠ 0  (SNP associated with T2D risk)
```
### Step 4: Run Logistic Regression
```r
model <- glm(T2D ~ rs67890 + Age + Sex + BMI + PC1 + ... + PC10,
             family = binomial(link = "logit"),
             data = gwas_data)
summary(model)

# Output:
# Coefficients:
#               Estimate Std. Error z value Pr(>|z|)
# (Intercept)   -4.56789    0.23456 -19.474 < 2e-16 ***
# rs67890        0.12345    0.02345   5.265 1.41e-07 ***
# Age            0.04567    0.00321  14.227 < 2e-16 ***
# Sex            0.23456    0.04567   5.136 2.79e-07 ***
# BMI            0.12345    0.01234  10.002 < 2e-16 ***
```
### Step 5: p-value Calculation
From output: β̂₁ = 0.12345, SE(β̂₁) = 0.02345
#### Step 5.1: Calculate Wald statistic (Z-score)
```
Z = β̂₁ / SE(β̂₁) = 0.12345 / 0.02345 = 5.265
```
#### Step 5.2: Calculate two-tailed p-value
```r
# In R:
p_value <- 2 * pnorm(-abs(5.265))
p_value # = 1.41e-07
```
Mathematically:
```r
p = 2 × P(Z ≥ |5.265|) where Z ~ N(0,1)
p = 2 × [1 - Φ(5.265)]  # Φ = standard normal CDF
p = 2 × [1 - 0.9999999295]
p = 2 × 0.0000000705
p = 1.41 × 10⁻⁷
```
### Step 6: Calculate Odds Ratio (OR)
```
OR = exp(β̂₁) = exp(0.12345) = 1.1314
95% CI: exp(β̂₁ ± 1.96 × SE)
        = exp(0.12345 ± 1.96 × 0.02345)
        = (1.083, 1.182)
```
### Step 7: Interpretation
- **Effect Size:** Each additional effect allele increases **odds** of T2D by 13.14% (OR = 1.1314)
- **Precision:** 95% CI (1.083, 1.182) excludes 1 (no effect)
- **Statistical Significance:** p = 1.41 × $10^{-7}$ < 5 × $10^{-8}$
- **Conclusion:** Strong evidence that this SNP is associated with T2D risk
## **Part 3: Comparison & Key Differences**
### Side-by-Side Calculation
|**Aspect**|**Quantitative Traits**|**Dichotomous Traits**|
|-|-|-
|**Example Trait**|BMI|Type 2 Diabetes|
|**Model**|Linear Regression|Logistic Regression|
|**Dependent Variable**|Continuous BMI value|Binary T2D status (0/1)|
|**Parameter of Interest**|$\beta_1$ (slope)|$\beta_1 (log-odds)$|
|**Effect Interpretation**|Units change per allele|Log-odds change per allele|
|**Test Statistic**|t = β̂₁/SE(β̂₁)|Z = β̂₁/SE(β̂₁)|
|**Distribution Under $H_0$**|t-distribution (df = n-p-1)|Standard Normal N(0,1)|
|**p-value Formula**|`2 × pt(-\|t\|, df)`|`2 × pnorm(-\|Z\|)`|
|**Common GWAS Software**|PLINK, BOLT-LMM, fastGWA|SAIGE, REGENIE, PLINK|
|**Typical Sample Size**|50,000+|10,000+ cases, 50,000+ controls|
|**Multiple Testing Threshold**|5 × $10^{-8}$|5 × $10^{-8}$|
### Detailed Calculation Examples
**Example 1: Quantitative Trait with Small Sample**
```r
# Data for 500 individuals
set.seed(123)
n <- 500
genotype <- sample(0:2, n, replace = TRUE, prob = c(0.25, 0.5, 0.25))
phenotype <- 25 + 0.15*genotype + rnorm(n, 0, 3)

# Linear regression
model <- lm(phenotype ~ genotype)
summary_model <- summary(model)

# Extract values
beta <- coef(model)["genotype"] # 0.142
se <- summary_model$coefficients["genotype", "Std. Error"]  # 0.095
t_stat <- beta/se   # 1.495
df <- df.residual(model)    # 498
p_value <- 2 * pt(-abs(t_stat), df) # 0.135

cat(sprintf("β = %.3f, SE = %.3f, t = %.3f, df = %d, p = %.3f\n",
            beta, se, t_stat, df, p_value))
# Output: β = 0.142, SE = 0.095, t = 1.495, df = 498, p = 0.135
```
**Interpretation:** Not significant (p > 0.05). With only 500 individuals, we lack power to detect this small effect.  
**Example 2: Dichotomous Trait with Imbalanced Cases**
```r
# Data: 500 cases, 4500 controls
n_cases <- 500
n_controls <- 4500
n_total <- n_cases + n_controls

# Simulate genotypes (MAF = 0.3)
maf <- 0.3
genotype <- rbinom(n_total, 2, maf)

# Simulate disease status with OR = 1.2 per allele
log_odds <- -3 + log(1.2)*genotype  # Baseline risk ~4.7%
prob <- exp(log_odds)/(1 + exp(log_odds))
disease <- rbinom(n_total, 1, prob)

# Logistic regression
model <- glm(disease ~ genotype, family = binomial)
summary_model <- summary(model)

# Extract values
beta <- coef(model)["genotype"] # 0.167
se <- summary_model$coefficients["genotype", "Std. Error"]  # 0.087
z_stat <- beta/se # 1.920
p_value <- 2 * pnorm(-abs(z_stat))  # 0.055
or <- exp(beta) # 1.182

cat(sprintf("β = %.3f, SE = %.3f, Z = %.3f, p = %.3f, OR = %.3f\n",
            beta, se, z_stat, p_value, or))
# Output: β = 0.167, SE = 0.087, Z = 1.920, p = 0.055, OR = 1.182
```
**Interpretation:** Borderline significance (p = 0.055). Shows how case-control imbalance affects power.
## **Part 4: Advanced Considerations**
### 1. Multiple Testing Correction in GWAS
For genome-wide analysis with 1M SNPs:
```r
# Bonferroni correction
alpha_genomewide <- 0.05 / 1e6  # 5e-8

# Calculate adjusted p-value
p_raw <- 1e-7
p_bonferroni <- min(p_raw * 1e6, 1) # = 0.1

# But in practice, we compare:
if(p_raw < alpha_genomewide) {
  "Genome-wide significant"
} else if(p_raw < 1e-5) {
  "Suggestive association"
} else {
  "Not significant"
}
```
### 2. Lambda GC (Genomic Control) Calculation
```r
# Calculate genomic inflation factor
chi_sq <- qchisq(1 - p_values, df = 1)  # Convert p to χ²
lambda_gc <- median(chi_sq) / qchisq(0.5, 1)  # Divide by expected median

# Interpretation:
# λ ≈ 1.0: Minimal inflation (good)
# λ ≈ 1.05: Acceptable
# λ > 1.1: Significant inflation (needs correction)
```
### 3. Manhattan Plot Coordinates
```r
# For plotting -log10(p) by genomic position
minus_log10_p <- -log10(p_values)

# Genome-wide significance line
signif_line <- -log10(5e-8) # ≈ 7.3

# Suggestive line
suggestive_line <- -log10(1e-5) # ≈ 5.0
```
## **Part 5: Practical Implementation Pipeline**
### Complete R Pipeline for Quantitative Trait
```r
gwas_quantitative <- function(genotype, phenotype, covariates) {
  # Step 1: Data preparation
  data <- cbind(phenotype, genotypes, covariates)

  # Step 2: Loop through SNPs
  results <- data.frame(SNP = colnames(genotypes),
                        Beta = NA, SE = NA, t = NA, p = NA, N = NA)
  
  for(i in 1:ncol(genotypes)) {
    # Fit model
    formula <- as.formula(paste("phenotype ~",
                                colnames(genotypes)[i], "+",
                                paste(colnames(covariates), collapse = "+")))
    model <- lm(formula, data = data)

    # Extract results
    summ <- summary(model)
    results$Beta[i] <- coef(model)[2]
    results$SE[i] <- summ$coefficients[2, 2]
    results$t[i] <- summ$coefficients[2, 3]
    results$p[i] <- summ$coefficients[2, 4]
    results$N[i] <- nobs(model)
  }

  # Step 3: Multiple testing correction
  results$p_adj_bonf <- p.adjust(results$p, method = "bonferroni")
  results$p_adj_fdr <- p.adjust(results$p, method = "fdr")

  # Step 4: Add genomic coordinates (if available)
  # results$CHR <- chr_info$CHR
  # results$BP <- chr_info$BP

  return(results)
}
```
### Complete R Pipeline for Dichotomous Trait
```r
gwas_dichotomous <- function(genotypes, disease_status, covariates) {
  results <- data.frame(SNP = colnames(genotypes),
                        Beta = NA, SE = NA, OR = NA,
                        OR_lower = NA, OR_upper = NA,
                        Z = NA, p = NA, N_cases = NA, N_controls = NA)
  
  for(i in 1:ncol(genotypes)) {
    # Fit logistic model
    formula <- as.formula(paste("disease_status ~",
                                colnames(genotypes)[i], "+",
                                paste(colnames(covariates), collapse = "+")))
    model <- glm(formula, data = cbind(disease_status, genotypes, covariates), family = binomial)

    # Extract results
    summ <- summary(model)
    beta <- coef(model)[2]
    se <- summ$coefficients[2, 2]

    results$Beta[i] <- beta
    results$SE[i] <- se
    results$OR[i] <- exp(beta)
    results$OR_lower[i] <- exp(beta - 1.96*se)
    results$OR_upper[i] <- exp(beta + 1.96*se)
    results$Z[i] <- beta/se
    results$p[i] <- summ$coefficients[2, 4]

    # Count cases/controls
    results$N_cases[i] <- sum(disease_status == 1)
    results$N_controls[i] <- sum(disease_status == 0)
  }

  return(results)
}
```
## **Part 6: Interpretation Guide**
### Quantitative Trait Results Interpretation
```r
SNP: rs12345
β = 0.157 ± 0.023 (SE)
t = 6.685, p = 2.35e-11
N = 10,000
```
**Interpretation:**  
**1. Effect Direction:** Positive β → allele increases trait value  
**2. Effect Magnitude:** Each allele adds 0.157 BMI units  
**3. Precision:** Narrow CI (β ± 2SE ≈ 0.111-0.203)  
**4. Significance:** p << threshold → strong evidence  
**5. Practical Relevance:** Small effect (0.16 BMI ≈ 0.45 kg for avg height)
### Dichotomous Trait Results Interpretation
```r
SNP: rs67890
OR = 1.131 (95% CI: 1.083-1.182)
β = 0.123, SE = 0.023, Z = 5.265, p = 1.41e-07
N_cases = 5,000, N_controls = 15,000
```
**Interpretation:**  
**1. Effect Direction:** OR > 1 → risk allele  
**2. Effect Magnitude:** 13% increased odds per allele  
**3. Precision:** CI excludes 1 → statistically significant  
**4. Significance:** Genome-wide significant  
**5. Population Impact:** Small individual effect but potentially important at population level
## **Part 7: Common Pitfalls & Solutions**
**Pitfall**|**Solution**|**Code Check**
-|-|-
**Population Stratification**|Include PCs as covariates|`lambda_gc <- calculate_lambda(p_values)`
**Case-control Imbalance**|Use Firth correction or SAIGE|`library(logistf)` for Firth
**Relatedness**|Use linear mixed models|`library(GMMAT)`
**Low Minor Allele Frequency**|Use exact tests|`fisher.test()` for rare variants
**Missing Data**|Impute genotypes|`library(impute)` or MINIMAC
**Multiple Testing**|Use genome-wide threshold|`p < 5e-8` for significance
## **Summary**
The p-value calculation process differs fundamentally between quantitative and dichotomous traits:  
**For Quantitative Traits:**
- Use **linear regression** with continuous outcome
- Test statistic follows **t-distribution**
- p-value = `2 × pt(-|t|, df)`
- Report **β ± SE** with units

**For Dichotomous Traits:**
- Use **logistic regression** with binary outcome
- Test statistic follows **normal distribution** (Wald Z)
- p-value = `2 × pnorm(-|Z|)`
- Report **OR (95% CI)** for interpretability

**Universal GWAS Principles:**
1. Always adjust for covariates (age, sex, PCs)
2. Use genome-wide significance threshold (5 × $10^{-8}$)
3. Check for genomic inflation ($\lambda$ ~ 1.0)
4. Report effect sizes with p-values
5. Independent replication is essential

---
**Contributor:** W.S.