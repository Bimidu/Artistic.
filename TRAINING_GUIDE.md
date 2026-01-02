# Training Guide - ASD Detection System

## Overview

The training system has been fully integrated into the frontend with real-time progress tracking, model management, and complete API support.

## Features

### ✅ Backend Features

1. **Real-time Training Progress Tracking**
   - Background task execution
   - Progress percentage updates
   - Current model being trained
   - Status messages (idle, training, completed, error)

2. **Model Management**
   - List all trained models with metrics
   - Delete models
   - Best model identification
   - Complete metadata (accuracy, F1, features, samples)

3. **Training API Endpoints**
   - `POST /training/train` - Start training
   - `GET /training/status` - Get training progress
   - `GET /models` - List all models
   - `DELETE /models/{model_name}` - Delete a model

### ✅ Frontend Features

1. **Training Interface**
   - Dataset selection with checkboxes
   - Model type selection (Random Forest, XGBoost, LightGBM, SVM)
   - Component selection
   - Real-time progress bar
   - Training status messages

2. **Model Display**
   - Cards showing all trained models
   - Metrics display (Accuracy, F1 Score, Features, Samples)
   - Best model highlighting
   - Delete functionality
   - Auto-refresh on training completion

3. **Visual Feedback**
   - Progress bar with percentage
   - Current model indicator
   - Success/error states
   - Loading animations
   - Smooth transitions

## How to Use

### Step 1: Start the API Server

```bash
cd /Users/bimidugunathilake/Documents/SE/Projects/Artistic.
python run_api.py
```

### Step 2: Open the Web Interface

Open `frontend/index.html` in your browser or navigate to `http://localhost:8000`

### Step 3: Switch to Training Mode

Click the "Training Mode" toggle at the top of the page

### Step 4: Select Datasets

1. Click "Refresh" to load available datasets
2. Check the datasets you want to use for training
3. Available datasets will show:
   - Dataset name
   - Number of CHAT files
   - Number of audio files

### Step 5: Configure Training

1. **Select Component**: Choose "Pragmatic & Conversational" (others are placeholders)
2. **Select Model Types**: Check one or more:
   - ✅ Random Forest (recommended)
   - ✅ XGBoost (high performance)
   - ✅ LightGBM (fast)
   - ✅ SVM (traditional ML)

### Step 6: Start Training

1. Click "Start Training"
2. Watch the progress bar update in real-time
3. See which model is currently being trained
4. Wait for completion (progress reaches 100%)

### Step 7: View Results

After training completes:
- ✅ Results show accuracy and F1 score for each model
- ✅ Models automatically appear in "Trained Models" section
- ✅ Best model is highlighted with a badge
- ✅ Each model shows detailed metrics

### Step 8: Use Trained Models

1. Switch to "User Mode"
2. Upload audio, text, or CHAT file
3. The best model will automatically be used for predictions

## Training Process Details

### What Happens During Training

1. **Feature Extraction** (Progress: 0-10%)
   - Loads datasets
   - Extracts 218 pragmatic/conversational features
   - Combines data from all selected datasets

2. **Preprocessing** (Progress: 10%)
   - Validates data quality
   - Handles missing values (median imputation)
   - Removes outliers (clipping)
   - Scales features (standard scaling)
   - Selects top 30 features

3. **Model Training** (Progress: 10-90%)
   - Trains each selected model type
   - Shows current model being trained
   - Progress divided equally among models
   - Each model trained on training set

4. **Evaluation** (Progress: 90-95%)
   - Tests on held-out test set (20% of data)
   - Calculates metrics:
     - Accuracy
     - F1 Score
     - Precision
     - Recall

5. **Saving Models** (Progress: 95-100%)
   - Registers models in registry
   - Saves model files
   - Saves preprocessor
   - Updates registry.json

## Model Management

### View All Models

1. Go to Training Mode
2. Scroll to "Trained Models" section
3. Click "Refresh" to load

Each model card shows:
- Model name and type
- Creation date
- "Best Model" badge (if applicable)
- Accuracy percentage
- F1 Score percentage
- Number of features used
- Training samples count

### Delete a Model

1. Click "Delete" button on model card
2. Confirm deletion
3. Model is removed from registry and disk

### Best Model Selection

- System automatically identifies best model by F1 score
- Highlighted with "Best Model" badge
- Used by default for predictions in User Mode

## API Details

### Start Training

```bash
POST /training/train
Content-Type: application/json

{
  "dataset_paths": ["asdbank_nadig", "asdbank_eigsti"],
  "model_types": ["random_forest", "xgboost"],
  "component": "pragmatic_conversational"
}
```

Response:
```json
{
  "status": "training_initiated",
  "component": "pragmatic_conversational",
  "model_types": ["random_forest", "xgboost"],
  "datasets": ["asdbank_nadig", "asdbank_eigsti"],
  "message": "Training started in background. Check /training/status for progress."
}
```

### Check Training Status

```bash
GET /training/status
```

Response (during training):
```json
{
  "status": "training",
  "component": "pragmatic_conversational",
  "model_types": ["random_forest", "xgboost"],
  "progress": 45,
  "current_model": "random_forest",
  "total_models": 2,
  "message": "Training random_forest...",
  "results": {},
  "error": null
}
```

Response (completed):
```json
{
  "status": "completed",
  "component": "pragmatic_conversational",
  "model_types": ["random_forest", "xgboost"],
  "progress": 100,
  "current_model": null,
  "total_models": 2,
  "message": "Training completed! Trained 2 models.",
  "results": {
    "random_forest": {
      "accuracy": 0.8571,
      "f1_score": 0.8333,
      "precision": 0.8333,
      "recall": 0.8333
    },
    "xgboost": {
      "accuracy": 0.9286,
      "f1_score": 0.9231,
      "precision": 0.9231,
      "recall": 0.9231
    }
  },
  "error": null
}
```

### List Models

```bash
GET /models
```

Response:
```json
{
  "models": [
    {
      "name": "random_forest",
      "type": "random_forest",
      "accuracy": 0.8571,
      "f1_score": 0.8333,
      "version": "1.0.0",
      "n_features": 30,
      "training_samples": 100,
      "created_at": "2026-01-01T20:00:00"
    }
  ],
  "count": 1,
  "best_model": "random_forest"
}
```

### Delete Model

```bash
DELETE /models/random_forest
```

Response:
```json
{
  "status": "success",
  "message": "Model 'random_forest' deleted successfully"
}
```

## Troubleshooting

### Training Stuck at 0%

- Check API server logs
- Ensure datasets have CHAT files
- Verify dataset paths are correct

### No Datasets Found

- Check `data/` folder has `asdbank_*` folders
- Each folder should contain `.cha` files
- Click "Refresh" to reload

### Training Failed

- Check error message in UI
- Check API logs: `logs/asd_detection.log`
- Common issues:
  - Not enough samples (need at least 10)
  - Missing diagnosis column
  - No valid CHAT files

### Models Not Appearing

- Click "Refresh" in Trained Models section
- Check `models/registry.json` exists
- Restart API server

## Performance Tips

1. **Start Small**: Train with 1-2 datasets first
2. **Use Random Forest**: Fastest and most reliable
3. **Monitor Progress**: Watch for stuck progress
4. **Check Logs**: API logs show detailed progress

## Next Steps

After training models:

1. ✅ Switch to User Mode
2. ✅ Upload CHAT files for prediction
3. ✅ View annotated transcripts
4. ✅ Compare model performance
5. ✅ Train more models with different datasets

## File Structure

```
models/
├── registry.json              # Model registry
├── random_forest/
│   ├── model.joblib          # Trained model
│   ├── preprocessor.joblib   # Feature preprocessor
│   └── metadata.json         # Model metadata
└── xgboost/
    ├── model.joblib
    ├── preprocessor.joblib
    └── metadata.json
```

## Credits

Training system designed and implemented for the ASD Detection System.
Supports multiple ML algorithms and real-time progress tracking.

