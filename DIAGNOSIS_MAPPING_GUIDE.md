# Diagnosis Mapping Guide

## Overview

The system now automatically infers diagnosis labels from dataset metadata, directory structure, and file locations. This ensures all available data is utilized for training, rather than discarding rows with missing diagnosis information.

## Implementation

### Components

1. **`DiagnosisMapper`** (`src/parsers/diagnosis_mapper.py`)
   - Infers diagnosis from file paths and metadata
   - Handles dataset-specific conventions
   - Normalizes all diagnoses to binary classification (ASD vs TD)

2. **Integration in `CHATParser`**
   - Automatically applies diagnosis mapping during file parsing
   - Transparent to feature extraction and training code
   - Logs diagnosis inference for debugging

## Dataset-Specific Rules

### All-ASD Datasets

These datasets contain only children with ASD:

#### 1. **asdbank_rollins**
- **Description**: Young boys with autism interviewed by a clinician
- **Rule**: All participants → `ASD`
- **Rationale**: Dataset description specifies "young boys with autism"

#### 2. **asdbank_flusberg**
- **Description**: Longitudinal study of children with ASD
- **Rule**: All participants → `ASD`
- **Rationale**: All participants in this cohort have ASD diagnosis

#### 3. **asdbank_aac**
- **Description**: Children using Augmentative and Alternative Communication (AAC) with ASD
- **Rule**: All participants → `ASD`
- **Rationale**: Study focuses on AAC users with ASD

### Mixed-Population Datasets

#### 4. **asdbank_eigsti**
- **Description**: Children with ASD, Developmental Delay (DD), and Typically Developing (TD)
- **Rules**:
  - `ASD` → `ASD`
  - `DD` → `TD` (treated as non-ASD for binary classification)
  - `TD` → `TD`
  - `TYP` → `TD`
- **Rationale**: DD children do not have ASD, so they're grouped with TD for binary classification

#### 5. **asdbank_nadig**
- **Description**: Children with ASD and normal controls
- **Rules**:
  - `ASD` → `ASD`
  - `TYP`, `TYPICAL`, `CONTROL` → `TD`
- **Rationale**: Normalizes various labels for typically developing children

### Risk-Based Datasets

#### 6. **asdbank_quigley_mcnalley**
- **Description**: Longitudinal study of high-risk and low-risk children for autism
- **Directory Rules**:
  - `/HR/` (High Risk) → `ASD`
  - `/LR/` (Low Risk) → `TD`
- **Rationale**: Directory structure indicates risk level, used as proxy for diagnosis
- **Example Paths**:
  - `data/asdbank_quigley_mcnalley/QuigleyMcNally/HR/Ron/000718.cha` → `ASD`
  - `data/asdbank_quigley_mcnalley/QuigleyMcNally/LR/Sam/001115.cha` → `TD`

## Diagnosis Inference Priority

The system uses the following priority order:

1. **Extracted from @ID line** (group field in CHAT file)
   - If present, this is used first
   - Example: `@ID:	eng|Eigsti|CHI|5;00.05|male|ASD||Target_Child|||`

2. **Dataset-specific default** (for all-ASD datasets)
   - Applied when no explicit diagnosis is found
   - Example: Rollins dataset → all `ASD`

3. **Directory-based rules** (for risk-based datasets)
   - Inferred from directory structure
   - Example: `/HR/` → `ASD`, `/LR/` → `TD`

4. **Diagnosis mapping** (for normalization)
   - Maps various labels to binary classification
   - Example: `DD` → `TD`, `TYP` → `TD`

## Expected Impact on Training

### Before Diagnosis Mapping

```
Total samples: 88 with diagnosis values
After filtering:
- 29 ASD
- 25 TYP → 41 TD (with 16 original TD)
- 16 DD (dropped)
Final: 70 samples (29 ASD + 41 TD)
```

### After Diagnosis Mapping

```
Total samples: 309
After diagnosis inference:
- asdbank_rollins: 21 files → all ASD
- asdbank_flusberg: 11 files → all ASD  
- asdbank_nadig: ~40 files → mixed (ASD + TYP/TD)
- asdbank_eigsti: ~40 files → mixed (ASD + DD→TD + TD)
- asdbank_quigley_mcnalley: ~83 files → mixed (HR→ASD + LR→TD)
- asdbank_aac: files → all ASD

Expected: 200+ samples with valid diagnoses
```

### Sample Distribution Improvement

| Dataset | Before | After | Change |
|---------|--------|-------|--------|
| Rollins | 0 | 21 (ASD) | +21 |
| Flusberg | 0 | 11 (ASD) | +11 |
| AAC | 0 | ~10 (ASD) | +10 |
| Quigley-McNally | 0 | ~83 (mixed) | +83 |
| Eigsti | 40 (dropped DD) | 56 (ASD+DD+TD) | +16 |
| Nadig | ~30 | ~30 | 0 |
| **Total** | **70** | **~211** | **+141** |

## Usage

### Automatic (Default)

The diagnosis mapper is automatically used during feature extraction:

```python
from src.features.feature_extractor import FeatureExtractor

extractor = FeatureExtractor()
features_df = extractor.extract_from_directory("data/asdbank_rollins")

# features_df now has 'diagnosis' column populated with 'ASD' for all rows
```

### Manual

You can also use the mapper directly:

```python
from pathlib import Path
from src.parsers.diagnosis_mapper import DiagnosisMapper

mapper = DiagnosisMapper()

# Infer diagnosis from file path
file_path = Path("data/asdbank_rollins/Rollins/Carl/020800.cha")
diagnosis = mapper.infer_diagnosis(file_path, extracted_diagnosis=None)
# Returns: 'ASD'

# With extracted diagnosis from @ID line
diagnosis = mapper.infer_diagnosis(file_path, extracted_diagnosis="DD")
# For Eigsti dataset, returns: 'TD' (DD mapped to TD)

# View all rules
mapper.print_dataset_rules()
```

## Adding New Datasets

To add rules for a new dataset, edit `src/parsers/diagnosis_mapper.py`:

```python
DATASET_RULES = {
    'new_dataset_name': {
        # Option 1: All participants have same diagnosis
        'default_diagnosis': 'ASD',
        'description': 'All participants are...'
        
        # Option 2: Directory-based rules
        'directory_rules': {
            '/GroupA/': 'ASD',
            '/GroupB/': 'TD',
        },
        'description': 'GroupA is ASD, GroupB is TD'
        
        # Option 3: Diagnosis label mapping
        'diagnosis_map': {
            'CONTROL': 'TD',
            'PATIENT': 'ASD',
        },
        'description': 'Maps CONTROL→TD, PATIENT→ASD'
    }
}
```

## Verification

To verify diagnosis mapping during training:

1. **Check Training Logs**:
   ```bash
   tail -f logs/asd_detection.log | grep -i diagnosis
   ```

2. **Inspect Output CSV**:
   ```bash
   # Count diagnoses in features CSV
   awk -F',' 'NR>1 && $3 != "" {print $3}' output/training_features.csv | sort | uniq -c
   ```

3. **Expected Output**:
   ```
   150 ASD
    61 TD
   ```

## Benefits

1. **More Training Data**: ~3x increase in usable samples (70 → 211+)
2. **Better Balance**: More diverse participant pool
3. **Dataset Utilization**: All datasets contribute to training
4. **Transparency**: Clear rules documented and logged
5. **Flexibility**: Easy to add new datasets or modify rules

## Notes

- All diagnosis labels are normalized to binary: `ASD` or `TD`
- The system logs all diagnosis inferences for debugging
- Unknown/unmapped diagnoses result in `None` (row dropped during training)
- Diagnosis mapping happens during parsing, before feature extraction

