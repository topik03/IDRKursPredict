#!/usr/bin/env python3
"""
run_pipeline.py - Master pipeline runner
Purpose: Run entire ML pipeline from start to finish
"""

import subprocess
import sys
from pathlib import Path

# Pipeline steps
PIPELINE_STEPS = [
    ("update_raw_data_yf.py", "Auto-updating USD/IDR raw data from Yahoo Finance..."),
    ("check_data.py", "Checking raw data..."),
    ("clean_data.py", "Cleaning USD/IDR data..."),
    ("download_external_data.py", "Downloading external market data..."),
    ("merge_external_features.py", "Merging external features..."),
    ("feature_engineering.py", "Creating features (USD/IDR only)..."),
    ("feature_engineering_external.py", "Creating features (with external)..."),
    ("train_baseline.py", "Training baseline model..."),
    ("train_xgboost.py", "Training XGBoost model..."),
    ("train_xgboost_external.py", "Training XGBoost with external..."),
    ("train_xgboost_improved.py", "Training XGBoost improved model..."),
    ("compare_models.py", "Comparing models..."),
    ("visualize_result.py", "Creating visualizations..."),
    ("visualize_external_result.py", "Creating external visualizations..."),
    ("predict_latest_improved.py", "Generating latest prediction..."),
    ("model_explainability.py", "Generating model explainability..."),
]

def run_script(script_name, description):
    """Run a single script with error handling"""
    print(f"\n{'='*60}")
    print(f"Step: {description}")
    print(f"{'='*60}")
    
    script_path = Path("src") / script_name
    
    if not script_path.exists():
        print(f"ERROR: Script not found: {script_path}")
        return False
    
    try:
        result = subprocess.run(
            ["python", str(script_path)],
            capture_output=False,
            text=True,
            shell=False
        )
        
        if result.returncode == 0:
            print(f"\n✓ {script_name} completed successfully")
            return True
        else:
            print(f"\n✗ {script_name} failed with exit code {result.returncode}")
            return False
    
    except Exception as e:
        print(f"ERROR running {script_name}: {e}")
        return False

def main():
    print("="*60)
    print("USD/IDR FORECASTING PIPELINE")
    print("Complete Machine Learning Pipeline")
    print("="*60)
    
    total_steps = len(PIPELINE_STEPS)
    successful = 0
    failed = []
    
    for i, (script, description) in enumerate(PIPELINE_STEPS, 1):
        print(f"\n[{i}/{total_steps}] Running {script}...")
        
        success = run_script(script, description)
        
        if success:
            successful += 1
        else:
            failed.append((script, description))
            print(f"WARNING: Pipeline will continue despite failure...")
    
    # Summary
    print(f"\n{'='*60}")
    print("PIPELINE COMPLETION SUMMARY")
    print(f"{'='*60}")
    print(f"Total steps: {total_steps}")
    print(f"Successful: {successful}")
    print(f"Failed: {len(failed)}")
    
    if failed:
        print(f"\nFailed steps:")
        for script, desc in failed:
            print(f"  - {script}: {desc}")
    
    if successful == total_steps:
        print("\n✓ Pipeline completed successfully!")
        print("\nNext steps:")
        print("  streamlit run app/streamlit_app.py")
    else:
        print("\n⚠ Pipeline completed with some failures.")
        print("Check individual scripts for details.")
    
    return successful == total_steps

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
