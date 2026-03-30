# AgentClinic Error Categorization Report

## Accuracy by Run

| Run | Dataset(s) | Total | Correct | Accuracy |
|-----|------------|-------|---------|----------|
| baseline_gpt4omini/medqa | MedQA | 107 | 20 | 18.7% |
| qwen72b | MedQA | 107 | 56 | 52.3% |
| qwen72b/medqa | MedQA | 107 | 56 | 52.3% |
| qwen72b/medqa_ext | MedQA_Ext | 108 | 64 | 59.3% |
| qwen72b/nejm_ext | NEJM_Ext | 56 | 11 | 19.6% |
| sdrp_gpt4omini/medqa | MedQA | 107 | 28 | 26.2% |
| sdrp_voyager | MedQA, MedQA_Ext, NEJM_Ext | 109 | 42 | 38.5% |
| sdrp_voyager/medqa | MedQA | 107 | 41 | 38.3% |
| sdrp_voyager/medqa_ext | MedQA_Ext | 214 | 80 | 37.4% |
| sdrp_voyager/nejm_ext | NEJM_Ext | 120 | 54 | 45.0% |
| sdrp_voyager_235b/medqa | MedQA | 107 | 6 | 5.6% |
| sdrp_voyager_235b/medqa_ext | MedQA_Ext | 214 | 5 | 2.3% |
| sdrp_voyager_235b/nejm_ext | NEJM_Ext | 120 | 7 | 5.8% |
| voyager/medqa | MedQA | 73 | 47 | 64.4% |
| voyager/medqa_ext | MedQA_Ext | 68 | 49 | 72.1% |
| voyager/nejm_ext | NEJM_Ext | 65 | 17 | 26.2% |

## Error Categories by Dataset

### MedQA  (n=822, accuracy=35.9%)

| Error Category | Count | % of total |
|----------------|-------|------------|
| correct | 295 | 35.9% |
| no_diagnosis | 200 | 24.3% |
| wrong_differential | 143 | 17.4% |
| anchor_bias | 12 | 1.5% |
| low_yield_testing | 9 | 1.1% |
| over_testing | 12 | 1.5% |
| incorrect_other | 151 | 18.4% |

### MedQA_Ext  (n=605, accuracy=32.9%)

| Error Category | Count | % of total |
|----------------|-------|------------|
| correct | 199 | 32.9% |
| no_diagnosis | 217 | 35.9% |
| wrong_differential | 100 | 16.5% |
| anchor_bias | 14 | 2.3% |
| low_yield_testing | 1 | 0.2% |
| over_testing | 3 | 0.5% |
| incorrect_other | 71 | 11.7% |

### NEJM_Ext  (n=362, accuracy=24.6%)

| Error Category | Count | % of total |
|----------------|-------|------------|
| correct | 89 | 24.6% |
| no_diagnosis | 128 | 35.4% |
| wrong_differential | 54 | 14.9% |
| anchor_bias | 20 | 5.5% |
| low_yield_testing | 6 | 1.7% |
| over_testing | 8 | 2.2% |
| knowledge_gap | 57 | 15.7% |

## Top 10 Wrong Diagnoses (Overall)

| Rank | Diagnosis | Count |
|------|-----------|-------|
| 1 | Unknown | 459 |
| 2 | (no diagnosis issued) | 74 |
| 3 | Lung Cancer | 10 |
| 4 | (Doctor did not issue DIAGNOSIS READY) | 7 |
| 5 | Basal Cell Carcinoma | 7 |
| 6 | Vestibular Neuritis | 6 |
| 7 | Gout | 5 |
| 8 | Multiple Sclerosis | 5 |
| 9 | Colorectal Cancer | 5 |
| 10 | Dengue Fever | 5 |

## Top Wrong Diagnoses by Error Category

### no_diagnosis  (n=545)

| Diagnosis | Count |
|-----------|-------|
| Unknown | 459 |
| (no diagnosis issued) | 74 |
| (Doctor did not issue DIAGNOSIS READY) | 7 |
| Unknown Presenting Condition | 5 |

### wrong_differential  (n=297)

| Diagnosis | Count |
|-----------|-------|
| Lung Cancer | 9 |
| Multiple Sclerosis | 5 |
| Dengue Fever | 5 |
| Cervical Radiculopathy | 5 |
| Placental Abruption | 4 |

### anchor_bias  (n=46)

| Diagnosis | Count |
|-----------|-------|
| Unexplained Chronic Fatigue Syndrome (CFS) or Idiopathic Chronic Fatigue. Furthe | 1 |
| Age-Related Macular Degeneration (AMD) | 1 |
| Advanced High-Grade Serous Ovarian Carcinoma with malignant ascites and probable | 1 |
| Syndrome of Inappropriate Antidiuretic Hormone Secretion (SIADH) secondary to se | 1 |
| Rocky Mountain Spotted Fever | 1 |

### low_yield_testing  (n=16)

| Diagnosis | Count |
|-----------|-------|
| Stevens-Johnson Syndrome (SJS). We will need to monitor you closely and start ap | 2 |
| Anemia due to bone marrow dysfunction, possibly contributing to episodes of apne | 2 |
| Viral Syndrome with Encephalopathy. | 2 |
| Suspected Lyme Disease with possible disseminated infection. We will confirm thi | 1 |
| Nontuberculous Mycobacterial Infection. | 1 |

### over_testing  (n=23)

| Diagnosis | Count |
|-----------|-------|
| Contact Dermatitis, likely due to an irritant or unknown allergen. | 2 |
| Insulinoma with Hypoglycemia | 2 |
| Neonatal Jaundice, possibly due to Polycythemia. | 2 |
| Multiple Sclerosis (MS) is strongly suspected based on your symptoms, MRI findin | 2 |
| Suspected endometrial or ovarian pathology requiring specialist evaluation. | 1 |

### knowledge_gap  (n=57)

| Diagnosis | Count |
|-----------|-------|
| Eosinophilic Lymphadenitis. We will proceed with a biopsy to confirm the diagnos | 1 |
| Idiopathic Pulmonary Fibrosis (IPF) | 1 |
| Angioimmunoblastic T-cell Lymphoma (AITL). We will need to start discussing trea | 1 |
| Systemic Sclerosis (Scleroderma) | 1 |
| Secondary Hyperparathyroidism leading to osteomalacia. | 1 |

### incorrect_other  (n=222)

| Diagnosis | Count |
|-----------|-------|
| Gout | 5 |
| Basal Cell Carcinoma | 5 |
| Primary Biliary Cholangitis | 4 |
| Rotator Cuff Tendinopathy | 4 |
| Acute Myeloid Leukemia | 4 |

## Secondary Flag Counts (Overall)

| Flag | Count |
|------|-------|
| never_requested_test | 742 |
| verifier_inconsistency | 621 |
| late_correct_mention | 38 |
| test_repetition | 28 |
| early_fixation | 4 |
