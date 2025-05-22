# PowerShell script to automate the full crypto sentiment ML pipeline
# Save as run_daily_pipeline.ps1 in your backend/src directory

# Activate virtual environment
$venv = "../../venv310/Scripts/Activate.ps1"
. $venv

# Step 1: Fetch Exorde sentiment (cached, so only fetches if needed)
python exorde_sentiment_insights.py

# Step 2: Append today's Exorde sentiment to merged dataset
python append_realtime_exorde_to_merged.py

# Step 3: Run ML model for real-time prediction
python mock_ml_model.py

Write-Host "[INFO] Daily pipeline complete."
