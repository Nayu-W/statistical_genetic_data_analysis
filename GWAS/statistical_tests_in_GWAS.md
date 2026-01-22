## **Statistical Tests in GWAS**
### 1. Comprehensive List of Statistical Tests in GWAS
#### CATEGORY 1: NORMALITY & DISTRIBUTION TESTS
**(A) Shapiro-Wilk Test**  
- **Purpose:** Test overall normality  
- **Usage:** Check residuals in linear regression
- **Limitation:** Too sensitive for large n (>5000)

**Formula:**
```math
W = \frac{(\sum_{i=1}^n a_i x_{(i)})^2}{\sum_{i=1}^n (x_i - \bar{x})^2}
```
Where:
- `x_i` = ith order statistic (sorted values)
- `a_i` = coefficients from normal distribution
- `x̄` = sample mean

**Null Hypothesis (H₀):** The data are normally distributed.  
**Interpretation:**
- **W close to 1** → More normal
- **p < 0.05** → Reject normality
- **p ≥ 0.05** → Fail to reject normality
```r
# Test residuals from GWAS model
model <- lm(BMI ~ genotype + covariates)
residuals <- resid(model)

shapiro_result <- shapiro.test(residuals[1:5000])  # Max 5000 observations
print(shapiro_result)

# Output:
# Shapiro-Wilk normality test
# data:  residuals[1:5000]
# W = 0.99234, p-value = 1.234e-08
```
**(B) Kolmogorov-Smirnov Test**
```r
# Test if residuals follow normal distribution
ks.test(residuals, "pnorm", mean=mean(residuals), sd=sd(residuals))
```
- **Compares:** Empirical vs theoretical CDF
- **Good for:** Large samples where Shapiro-Wilk fails
- **GWAS use:** Rarely used due to low power for normality

**(C) Anderson-Darling Test**
```r
library(nortest)
ad.test(residuals)  # More powerful than K-S for normality
```
- **Better** at detecting deviations in tails
- **Used** in some GWAS software for extreme value detection

**(D) Skewness & Kurtosis Tests**
```r
library(moments)
skewness(residuals)  # Normal ≈ 0
kurtosis(residuals)  # Normal ≈ 3

# Formal tests
agostino.test(residuals)  # D'Agostino test for skewness
anscombe.test(residuals)  # Anscombe-Glynn test for kurtosis
```
**GWAS interpretation:**
- **|Skewness| > 1:** Substantial skew
- **Kurtosis > 4:** Heavy tails (leptokurtic)
- **Kurtosis < 2:** Light tails (platykurtic)
#### CATEGORY 2: ASSOCIATION TESTS (Core GWAS Tests)
**(A) Linear Regression (Quantitative Traits)**
```r
# Basic model
model <- lm(phenotype ~ SNP + covariates)

# Test statistics extracted:
t_value <- summary(model)$coefficients["SNP", "t value"]
p_value <- summary(model)$coefficients["SNP", "Pr(>|t|)"]
```
**(B) Logistic Regression (Binary Traits)**
```r
# Wald test (default)
model <- glm(case_control ~ SNP + covariates, family=binomial)
z_value <- summary(model)$coefficients["SNP", "z value"]
p_value <- summary(model)$coefficients["SNP", "Pr(>|z|)"]

# Likelihood Ratio Test (LRT) - more accurate for small samples
library(lmtest)
full_model <- glm(case_control ~ SNP + covariates, family=binomial)
reduced_model <- glm(case_control ~ covariates, family=binomial)
lrt_result <- lrtest(full_model, reduced_model)
lrt_p <- lrt_result$"Pr(>Chisq)"[2]
```
**(C) Score Tests (Efficient for Large Datasets)**  
**Mathematical form:**
```math
\text{Score} = \frac{U(\beta)^2}{I(\beta)} \sim \chi^2_1
```
Where:
- `U(β)` = Score function (first derivative of log-likelihood)
- `I(β)` = Fisher information

**Implementation in PLINK:**
```bash
# Fastest for large GWAS
plink --bfile data --linear --covar covariates.txt
# Uses score test for efficiency
```
**(D) Firth's Bias-Reduced Logistic Regression**
```r
library(logistf)
model_firth <- logistf(case_control ~ SNP + covariates)
# Essential for case-control imbalance or separation
```
**(E) Cochran-Armitage Trend Test (CAT)**
```r
# For 2×3 contingency table (genotypes × case-control)
# Manual calculation:
table <- matrix(c(n_AA_case, n_AB_case, n_BB_case,
                  n_AA_ctrl, n_AB_ctrl, n_BB_ctrl), nrow=2)

library(DescTools)
CochranArmitageTest(table)
```
**Test statistic:**
```math
Z_{CAT} = \frac{\sum_{i=0}^2 w_i (r_i - E[r_i])}{\sqrt{\text{Var}(\sum w_i r_i)}}
```
#### CATEGORY 3: SPECIALIZED GWAS TESTS
**(A) Mixed Linear Models (for Relatedness)**
```r
# EMMAX, BOLT-LMM, GEMMA, GCTA
# Model: Y = Xβ + Zu + ε
# Where u ~ N(0, Kσ_g²) accounts for relatedness

library(GMMAT)  # For binary traits with relatedness
model <- glmmkin(fixed = phenotype ~ SNP + covariates,
                 random = ~ 1|ID, 
                 data = data,
                 kinship = kinship_matrix)
```
**(B) SAIGE (Scalable and Accurate Implementation of GEneralized mixed model)**
```r
# For binary traits with case-control imbalance
# Uses SPA (saddlepoint approximation)
library(SAIGE)
fitNULLGLMM <- fitNULLGLMM(fixed = phenotype ~ covariates,
                          data = data,
                          kinship = kinship_matrix)
assoc <- SPAGMMAT(fitNULLGLMM, genotype)
```
**(C) Burden Tests & SKAT (for Rare Variants)**
```r
# Burden test: Collapses multiple variants into single score
# SKAT: Variance-component test

library(SKAT)
# Create SNP set matrix
SNP_set <- as.matrix(genotype_data[, rare_snps])
obj <- SKAT_Null_Model(phenotype ~ covariates, out_type="C")
p_value <- SKAT(SNP_set, obj)$p.value
```
**(D) Meta-analysis Tests**
```r
# Inverse-variance weighted (fixed effects)
beta_meta <- sum(w_i * beta_i) / sum(w_i)
se_meta <- 1 / sqrt(sum(w_i))  # where w_i = 1/se_i²

# Cochran's Q test for heterogeneity
Q <- sum(w_i * (beta_i - beta_meta)^2)  # ~ χ²_{k-1}
I² <- max(0, (Q - (k-1))/Q) * 100  # % heterogeneity
```
#### CATEGORY 4: QUALITY CONTROL (QC) TESTS
**(A) Hardy-Weinberg Equilibrium (HWE) Test**
```r
# For each SNP, test if genotype frequencies follow HWE
library(HardyWeinberg)
HWExact(genotype_counts)  # Exact test (recommended)
# Or: HWTernaryExact() for multiple alleles

# In PLINK:
# plink --bfile data --hardy
```
**Test statistic (χ² test):**
```math
\chi^2 = \frac{(n_{AA} - E_{AA})^2}{E_{AA}} + \frac{(n_{AB} - E_{AB})^2}{E_{AB}} + \frac{(n_{BB} - E_{BB})^2}{E_{BB}}
```
Where expected counts: E_AA = n(p_A)², E_AB = 2n(p_A)(p_B), E_BB = n(p_B)²

**(B) Heterozygosity Test**
```r
# Identify sample outliers
het <- plink --bfile data --het
# F coefficient = (O(Het) - E(Het)) / (n - E(Het))
# Flag if |F| > 0.2
```
**(C) Sex Check Test**
```r
# Compare reported vs genetic sex
# Using X chromosome heterozygosity
# Males should have very low X chromosome heterozygosity
```
**(D) Mendelian Error Test (for Family Data)**
```r
# Check transmission consistency
plink --bfile data --mendel
# Expected: ~1% error rate max
```
#### CATEGORY 5: MULTIPLE TESTING CORRECTION TESTS
**(A) Bonferroni Correction**
```r
p_adj_bonf <- pmin(p_raw * n_tests, 1)
# Genome-wide threshold: 0.05/1e6 = 5e-8
```
**(B) Benjamini-Hochberg (FDR Control)**
```r
p_adj_fdr <- p.adjust(p_values, method="BH")
# Controls False Discovery Rate (expected % of false positives)
```
**(C) q-value (Storey's Method)**
```r
library(qvalue)
qobj <- qvalue(p_values)
q_values <- qobj$qvalues
pi0 <- qobj$pi0  # Estimated proportion of true nulls
```
**(D) Permutation-based p-values**
```r
# Empirical p-values
n_perm <- 10000
perm_pvals <- numeric(n_perm)

for(i in 1:n_perm){
  shuffled_pheno <- sample(phenotype)
  perm_model <- lm(shuffled_pheno ~ genotype + covariates)
  perm_pvals[i] <- summary(perm_model)$coefficients[2, 4]
}

empirical_p <- mean(perm_pvals <= original_p)
```
### 2. Practical GWAS Testing Pipeline
**Complete GWAS Association Testing Workflow**
```r
run_gwas_pipeline <- function(genotypes, phenotype, covariates, trait_type="quantitative"){
  
  results <- data.frame(SNP = colnames(genotypes),
                        Beta = NA, SE = NA, Stat = NA, P = NA, 
                        N = NA, Info = NA)
  
  for(i in 1:ncol(genotypes)){
    # 1. Basic QC on SNP
    maf <- mean(genotypes[, i])/2
    if(maf < 0.01) next  # Skip rare variants
    
    # 2. Choose test based on trait type
    if(trait_type == "quantitative"){
      # Linear regression with robust SE
      model <- lm(phenotype ~ genotypes[, i] + covariates)
      
      # Use HC3 robust standard errors
      library(sandwich)
      library(lmtest)
      robust_summary <- coeftest(model, vcov = vcovHC(model, type = "HC3"))
      
      results$Beta[i] <- robust_summary["genotypes[, i]", "Estimate"]
      results$SE[i] <- robust_summary["genotypes[, i]", "Std. Error"]
      results$Stat[i] <- robust_summary["genotypes[, i]", "t value"]
      results$P[i] <- robust_summary["genotypes[, i]", "Pr(>|t|)"]
      
    } else if(trait_type == "binary"){
      # Firth's bias-reduced logistic for rare events/case-control imbalance
      library(logistf)
      data_temp <- data.frame(Y = phenotype, 
                              SNP = genotypes[, i],
                              covariates)
      
      model <- logistf(Y ~ ., data = data_temp)
      results$Beta[i] <- model$coefficients["SNP"]
      results$SE[i] <- sqrt(diag(model$var))["SNP"]
      results$Stat[i] <- results$Beta[i]/results$SE[i]
      results$P[i] <- model$prob["SNP"]
    }
    
    # 3. Additional metrics
    results$N[i] <- sum(!is.na(phenotype) & !is.na(genotypes[, i]))
    results$Info[i] <- 1 - (results$SE[i]^2) / (var(genotypes[, i], na.rm=TRUE) * var(phenotype, na.rm=TRUE))
  }
  
  # 4. Multiple testing correction
  results <- results[!is.na(results$P), ]
  results$P_Bonferroni <- p.adjust(results$P, method="bonferroni")
  results$P_FDR <- p.adjust(results$P, method="BH")
  
  # 5. Genomic inflation factor
  chi_sq <- qchisq(1 - results$P, df=1)
  lambda_gc <- median(chi_sq, na.rm=TRUE) / qchisq(0.5, 1)
  cat(sprintf("Genomic inflation factor (λ_GC) = %.3f\n", lambda_gc))
  
  # 6. QQ plot
  expected <- -log10(ppoints(nrow(results)))
  observed <- -log10(sort(results$P))
  
  plot(expected, observed, 
       xlab = expression(paste("Expected -log"[10], "(p)")),
       ylab = expression(paste("Observed -log"[10], "(p)")),
       main = paste("QQ Plot (λ_GC =", round(lambda_gc, 3), ")"))
  abline(0, 1, col="red")
  
  return(results)
}
```
### 3. Key Test Statistics & Their Distributions
**Test**|**Statistic**|**Distribution Under H₀**|**GWAS Use Case**
-|-|-|-
**t-test**|`t = β̂/SE(β̂)`|t(df = n-p-1)|Linear regression
**Wald test**|`Z = β̂/SE(β̂)`|N(0,1)|Logistic regression
**Likelihood Ratio**|`-2[L(θ₀)-L(θ)]`|χ²(df = diff params)|Model comparison
**Score test**|`U(θ₀)²/I(θ₀)`|χ²(1)|Fast screening
**Cochran-Armitage**|`Z_CAT`|N(0,1)|Trend in 2×3 table
**Fisher's Exact**|Hypergeometric|Exact|Small samples, rare variants
**Cochran's Q**|`Σw_i(β_i-β̄)²`|χ²(k-1)|Meta-analysis heterogeneity
**Burden test**|`Σβ_i`|N(0,1)|Rare variant collapsing
**SKAT**|Quadratic form|Mixture χ²|Rare variant sets
### 4. GWAS-Specific Statistical Considerations
**(A) Genomic Control (λ_GC)**
```r
calculate_lambda_gc <- function(p_values){
  chi_sq <- qchisq(1 - p_values, df = 1)
  lambda <- median(chi_sq, na.rm = TRUE) / qchisq(0.5, 1)
  
  # Interpretation:
  if(lambda < 1.0) cat("Possible winner's curse or conservative test\n")
  if(lambda > 1.05) cat("Possible population stratification\n")
  if(lambda > 1.1) cat("Strong inflation - need correction\n")
  
  return(lambda)
}
```
**(B) Winner's Curse Correction**
```r
# Early GWAS effects are overestimated
library("pwr")
correct_winner_curse <- function(beta_obs, se_obs, alpha=5e-8){
  z_alpha <- qnorm(1 - alpha/2)
  # Zaykin correction
  beta_corrected <- beta_obs * (1 - (z_alpha/se_obs)^(-2))
  return(beta_corrected)
}
```
**(C) Power Calculation**
```r
library(pwr)
# For continuous trait
pwr.t.test(n = 10000, d = 0.1, sig.level = 5e-8)
# d = effect size (Cohen's d = β/sd)

# For binary trait
pwr.f2.test(u = 1, v = 9998, f2 = 0.0001, sig.level = 5e-8)
# f2 = effect size (R²/(1-R²))
```
### 5. Modern Best Practices
**For Quantitative Traits:**
```r
# 1. Rank-based inverse normal transformation
phenotype_rint <- rankNorm(phenotype)

# 2. Linear mixed model with robust SE
library(lme4)
model <- lmer(phenotype_rint ~ SNP + covariates + (1|FID), data=data)

# 3. Satterthwaite degrees of freedom
library(lmerTest)
summary(model, ddf="Satterthwaite")
```
**For Binary Traits:**
```r
# 1. Firth correction or SAIGE for imbalance
library(logistf)
model <- logistf(case_control ~ SNP + covariates)

# 2. Or SAIGE for large-scale with relatedness
library(SAIGE)
# (As shown earlier)
```
**For Rare Variants:**
```r
# Combine SKAT and burden tests
library(SKAT)
obj <- SKAT_Null_Model(phenotype ~ covariates, out_type="D")
p_skat <- SKAT(SNP_set, obj, method="optimal.adj")$p.value
```
### 6. Summary Table: Test Selection Guide
**Scenario**|**Recommended Test**|**Reason**
-|-|-
**Quantitative trait, unrelated**|Linear regression (HC3 SE)|Standard, robust
**Quantitative trait, related**|Linear mixed model (BOLT-LMM)|Accounts for relatedness
**Binary trait, balanced**|Logistic regression (Wald)|Standard
**Binary trait, imbalanced**|Firth or SAIGE|Reduces bias
**Binary trait, related**|SAIGE or GMMAT|Handles both
**Rare variants (MAF < 0.01)**|SKAT-O or burden test|Increased power
**Small sample size**|Exact tests (Fisher)|Accurate p-values
**Multiple cohorts**|Meta-analysis (IVW/FE)|Combines evidence
**Heterogeneous effects**|Meta-analysis (RE)|Allows variation

The choice of statistical test in GWAS depends on: trait type, sample size, relatedness, MAF, and computational resources. Modern GWAS increasingly uses mixed models and robust methods to handle the complexities of genetic data while maintaining statistical validity.

---
**Contributor:** W.S.