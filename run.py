import os
import sys
import time
import subprocess
from concurrent.futures import ThreadPoolExecutor

def check_models_exist():
    models_dir = os.path.join(os.path.dirname(__file__), "backend", "saved_models")
    required_files = [
        "approval_model.pkl",
        "pd_model.pkl",
        "fraud_model.pkl",
        "interest_model.pkl",
        "segment_model.pkl",
        "scaler_segment.pkl",
        "feature_names.pkl",
        "X_train.pkl"
    ]
    if not os.path.exists(models_dir):
        return False
    return all(os.path.exists(os.path.join(models_dir, f)) for f in required_files)

def run_pipeline():
    print("Required models are missing or incomplete.")
    print("Launching ML training pipeline (backend/pipeline.py)...")
    pipeline_script = os.path.join(os.path.dirname(__file__), "backend", "pipeline.py")
    result = subprocess.run([sys.executable, pipeline_script], check=True)
    if result.returncode == 0:
        print("ML training pipeline finished successfully.")
    else:
        print("Error running ML training pipeline.")
        sys.exit(1)

def start_backend():
    print("Starting FastAPI Backend at http://127.0.0.1:8000 ...")
    # Launch uvicorn
    cmd = [
        sys.executable, "-m", "uvicorn", "backend.app:app",
        "--host", "127.0.0.1", "--port", "8000"
    ]
    subprocess.run(cmd)

def start_frontend():
    # Wait for backend to start
    time.sleep(3)
    print("Starting Streamlit Dashboard...")
    dashboard_script = os.path.join(os.path.dirname(__file__), "frontend", "dashboard.py")
    cmd = [
        "streamlit", "run", dashboard_script
    ]
    subprocess.run(cmd)

def main():
    if not check_models_exist():
        run_pipeline()
    else:
        print("All required models exist on disk. Ready to start.")
        
    print("\n" + "="*50)
    print("Launching Credit Risk Intelligence Platform")
    print("="*50 + "\n")
    
    with ThreadPoolExecutor(max_workers=2) as executor:
        executor.submit(start_backend)
        executor.submit(start_frontend)

if __name__ == "__main__":
    main()
