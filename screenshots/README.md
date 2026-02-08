# Project Screenshots

This directory contains screenshots demonstrating the ML Lifecycle project in action.

## 📸 Screenshots to Capture

### 1. **Web Dashboard Running** (`01_web_dashboard.png`)

**How to capture:**
1. Keep the Flask API running (`python app/api.py`)
2. Open browser: `http://localhost:5000`
3. Type a movie review (e.g., "This movie was absolutely amazing!")
4. Click "Analyze Sentiment"
5. Screenshot shows:
   - Dashboard with gradient background
   - Input box with text
   - Results showing sentiment and confidence score

**Example reviews to test:**
- Positive: "I absolutely loved this movie! Amazing performances and great plot."
- Negative: "Terrible movie. Waste of time. Bad acting and boring story."

---

### 2. **Test Results** (`02_pytest_results.png`)

**How to capture:**
```powershell
pytest tests/ -v
```

Screenshot shows:
- All 28 tests passing (green checkmarks)
- Test execution time
- No failures

---

### 3. **MLflow Dashboard** (`03_mlflow_ui.png`)

**How to capture:**
1. Open new PowerShell terminal
2. Run:
   ```powershell
   & 'C:/Users/KINLEY/OneDrive/Desktop/MLOP/ASS#/machine-learning-lifecycle-kinley-wa/venv/Scripts/python.exe' -m mlflow ui
   ```
3. Open browser: `http://localhost:5000`
4. Click on "sentiment-analysis" experiment
5. Screenshot shows:
   - List of experiment runs
   - Metrics (accuracy, F1, etc.)
   - Parameters (model name, batch size, etc.)
   - Timestamps and run IDs

---

### 4. **Model Predictions** (`04_api_predictions.png`)

**How to capture:**
1. Keep web dashboard open
2. Submit 2-3 different reviews (positive, negative, neutral)
3. Screenshot shows multiple results in dashboard
4. Shows sentiment badges and confidence bars

---

### 5. **Docker Build** (`05_docker_build.png`)

**How to capture:**
```powershell
docker build -t sentiment-api app/
```

Screenshot shows:
- Docker build progress
- Successfully built image
- Image ID

---

### 6. **File Structure** (`06_project_structure.png`)

**How to capture:**
1. Open File Explorer
2. Navigate to project root
3. Show folder structure with:
   - `src/` (source code)
   - `app/` (API + dashboard)
   - `tests/` (unit tests)
   - `notebooks/` (Jupyter notebooks)
   - `models/trained/` (trained models)
   - `data/` (dataset)

---

### 7. **GitHub Workflows** (`07_github_actions.png`)

**How to capture:**
1. Push to GitHub
2. Go to your repository
3. Click "Actions" tab
4. Screenshot shows:
   - Training workflow runs
   - Test workflow runs
   - All checks passing

---

### 8. **Model Metrics** (`08_model_metrics.png`)

**How to capture:**
```powershell
cat models/trained/model_info.json
```

Or open in VS Code and screenshot showing:
```json
{
  "model_type": "RandomForest",
  "accuracy": 1.0,
  "f1_score": 1.0,
  "training_samples": 400,
  "test_samples": 100
}
```

---

## 📋 Screenshot Checklist

- [ ] Web Dashboard (01_web_dashboard.png)
- [ ] Test Results (02_pytest_results.png)
- [ ] MLflow Dashboard (03_mlflow_ui.png)
- [ ] API Predictions (04_api_predictions.png)
- [ ] Docker Build (05_docker_build.png)
- [ ] Project Structure (06_project_structure.png)
- [ ] GitHub Actions (07_github_actions.png)
- [ ] Model Metrics (08_model_metrics.png)

---

## 🚀 Quick Commands to Run Everything

**Terminal 1 - API + Dashboard:**
```powershell
python app/api.py
```
Then open: `http://localhost:5000`

**Terminal 2 - MLflow:**
```powershell
python -m mlflow ui
```
Then open: `http://localhost:5000` (different tab)

**Terminal 3 - Tests:**
```powershell
pytest tests/ -v
```

**Terminal 4 - Docker (optional):**
```powershell
docker build -t sentiment-api app/
```

---

## 📝 How to Add Screenshots

1. Take screenshot (Windows: `Win + Shift + S` or `Snip & Sketch`)
2. Save to `screenshots/` folder
3. Name as shown above (01_web_dashboard.png, etc.)
4. Screenshots will appear in this README

---

## ✅ Evidence of Completion

These screenshots prove:
- ✅ Web dashboard working & responsive
- ✅ All tests passing (28/28)
- ✅ MLflow experiment tracking operational
- ✅ API predictions accurate
- ✅ Docker containerization successful
- ✅ Project structure complete
- ✅ CI/CD pipelines configured
- ✅ Model metrics documented
