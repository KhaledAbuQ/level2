import os
import json
import time
import glob
import subprocess
import sys
import tempfile
import shutil
import importlib.util
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_submission(user_submission_file):
    """
    Extracts and prepares the user's submission for evaluation.
    
    Args:
        user_submission_file: Path to the submitted file (.py, .zip, or .tar.gz)
        
    Returns:
        temp_dir: Path to the temporary directory where the submission is extracted
        model_script: Path to the main script to be executed
    """
    temp_dir = tempfile.mkdtemp()
    logger.info(f"Created temporary directory: {temp_dir}")
    
    # Get the file extension
    _, ext = os.path.splitext(user_submission_file)
    
    if ext.lower() == '.py':
        # For single Python file submissions
        model_script = os.path.join(temp_dir, 'model.py')
        shutil.copy(user_submission_file, model_script)
    elif ext.lower() == '.zip':
        # For ZIP archives
        import zipfile
        with zipfile.ZipFile(user_submission_file, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        # Look for main.py or predict.py
        for candidate in ['main.py', 'predict.py', 'model.py']:
            potential_script = os.path.join(temp_dir, candidate)
            if os.path.exists(potential_script):
                model_script = potential_script
                break
        else:
            # If no candidate files found, look for any Python file
            py_files = glob.glob(os.path.join(temp_dir, '*.py'))
            if py_files:
                model_script = py_files[0]
            else:
                raise ValueError("Could not find a Python script in the submitted ZIP file.")
    elif ext.lower() in ['.tar', '.gz', '.tgz'] or user_submission_file.endswith('.tar.gz'):
        # For TAR archives
        import tarfile
        with tarfile.open(user_submission_file, 'r:*') as tar:
            tar.extractall(temp_dir)
        # Look for main.py or predict.py
        for candidate in ['main.py', 'predict.py', 'model.py']:
            potential_script = os.path.join(temp_dir, candidate)
            if os.path.exists(potential_script):
                model_script = potential_script
                break
        else:
            # If no candidate files found, look for any Python file
            py_files = glob.glob(os.path.join(temp_dir, '*.py'))
            if py_files:
                model_script = py_files[0]
            else:
                raise ValueError("Could not find a Python script in the submitted TAR file.")
    else:
        raise ValueError(f"Unsupported file format: {ext}")
        
    logger.info(f"Using model script: {model_script}")
    return temp_dir, model_script

def load_reference_images(dataset_dir):
    """
    Loads reference images (one per person) from the dataset directory.
    
    Args:
        dataset_dir: Path to the dataset directory containing img_* folders
        
    Returns:
        reference_images: Dictionary mapping person name to reference image path
    """
    reference_images = {}
    person_dirs = glob.glob(os.path.join(dataset_dir, 'img_*'))
    
    for person_dir in person_dirs:
        person_name = os.path.basename(person_dir).replace('img_', '')
        # Take the first image as reference
        image_files = glob.glob(os.path.join(person_dir, '*'))
        if image_files:
            reference_images[person_name] = image_files[0]
    
    return reference_images

def load_test_images(dataset_dir, annotations_file=None):
    """
    Loads test images and their ground truth labels.
    
    Args:
        dataset_dir: Path to the dataset directory containing img_* folders
        annotations_file: Path to annotations file (optional)
        
    Returns:
        test_images: List of test image paths
        true_labels: List of corresponding true labels
    """
    # If annotations file is provided, use it
    if annotations_file and os.path.exists(annotations_file):
        with open(annotations_file, 'r') as f:
            annotations = json.load(f)
        
        test_images = []
        true_labels = []
        
        for item in annotations:
            image_path = os.path.join(dataset_dir, item['image_path'])
            test_images.append(image_path)
            true_labels.append(item['label'])
            
        return test_images, true_labels
    
    # Otherwise, use img_* folder structure
    test_images = []
    true_labels = []
    
    person_dirs = glob.glob(os.path.join(dataset_dir, 'img_*'))
    for person_dir in person_dirs:
        person_name = os.path.basename(person_dir).replace('img_', '')
        # Skip the first image (used as reference)
        image_files = glob.glob(os.path.join(person_dir, '*'))
        
        if len(image_files) > 1:
            for image in image_files[1:]:  # Skip the first reference image
                test_images.append(image)
                true_labels.append(person_name)
    
    return test_images, true_labels

def run_model(model_script, reference_images, test_images):
    """
    Runs the contestant's model on the test images.
    
    Args:
        model_script: Path to the contestant's model script
        reference_images: Dictionary of reference images (person_name -> image_path)
        test_images: List of test image paths
        
    Returns:
        predictions: List of predicted labels
        processing_time: Average processing time per image
    """
    # Prepare input JSON
    input_data = {
        "reference_images": reference_images,
        "test_images": test_images
    }
    
    input_file = tempfile.NamedTemporaryFile(delete=False, suffix='.json')
    input_file.close()
    
    with open(input_file.name, 'w') as f:
        json.dump(input_data, f)
    
    # Prepare output JSON path
    output_file = tempfile.NamedTemporaryFile(delete=False, suffix='.json')
    output_file.close()
    
    # Command to run the script
    cmd = [sys.executable, model_script, input_file.name, output_file.name]
    
    logger.info(f"Running model with command: {' '.join(cmd)}")
    
    # Measure execution time
    start_time = time.time()
    try:
        subprocess.run(cmd, check=True, timeout=600)  # 10-minute timeout
    except subprocess.TimeoutExpired:
        logger.error("Evaluation timed out after 10 minutes")
        # Return empty predictions if timeout
        return [], 600.0
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running model: {e}")
        # Return empty predictions if error
        return [], 0.0
        
    end_time = time.time()
    total_time = end_time - start_time
    
    # Calculate average time per image
    processing_time = total_time / len(test_images) if test_images else 0
    
    # Load predictions
    try:
        with open(output_file.name, 'r') as f:
            predictions = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        logger.error(f"Error loading predictions: {e}")
        predictions = []
    
    # Clean up temporary files
    os.unlink(input_file.name)
    os.unlink(output_file.name)
    
    return predictions, processing_time

def calculate_metrics(true_labels, predicted_labels):
    """
    Calculates performance metrics.
    
    Args:
        true_labels: List of true labels
        predicted_labels: List of predicted labels
        
    Returns:
        metrics: Dictionary of metrics
    """
    if not predicted_labels or len(predicted_labels) != len(true_labels):
        return {
            "Accuracy": 0.0,
            "F1 Score": 0.0,
            "Processing Time (s)": 0.0,
            "Total": 0.0
        }
    
    accuracy = accuracy_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels, average='weighted', zero_division=0)
    
    return {
        "Accuracy": accuracy,
        "F1 Score": f1
    }

def evaluate(test_annotation_file, user_submission_file, phase_codename, **kwargs):
    """
    Evaluates the submission for a particular challenge phase and returns score.
    
    Args:
        test_annotation_file: Path to test_annotation_file on the server
        user_submission_file: Path to file submitted by the user
        phase_codename: Phase to which submission is made
        **kwargs: Additional submission metadata
        
    Returns:
        output: Dictionary with evaluation results
    """
    logger.info(f"Starting evaluation for phase {phase_codename}...")
    
    # Get the dataset directory from environment or use default
    dataset_dir = os.environ.get('DATASET_DIR', '/dataset')
    
    try:
        # Set up the submission
        temp_dir, model_script = setup_submission(user_submission_file)
        
        # Load reference and test images
        reference_images = load_reference_images(dataset_dir)
        test_images, true_labels = load_test_images(dataset_dir, test_annotation_file)
        
        # Run the model
        predicted_labels, processing_time = run_model(model_script, reference_images, test_images)
        
        # Calculate metrics
        metrics = calculate_metrics(true_labels, predicted_labels)
        metrics["Processing Time (s)"] = processing_time
        
        # Calculate total score
        # Formula: 0.5*Accuracy + 0.3*F1 + 0.2*(1 - min(processing_time/10, 1))
        # This prioritizes accuracy but also rewards fast processing
        accuracy_weight = 0.5
        f1_weight = 0.3
        time_weight = 0.2
        
        # Normalize processing time (lower is better)
        time_score = 1.0 - min(processing_time / 10.0, 1.0)
        
        total_score = (
            accuracy_weight * metrics["Accuracy"] +
            f1_weight * metrics["F1 Score"] +
            time_weight * time_score
        )
        
        metrics["Total"] = total_score
        
        # Clean up
        shutil.rmtree(temp_dir)
        
        # Prepare output
        output = {"result": [], "submission_result": {}}
        
        if phase_codename == "dev_phase":
            output["result"].append({
                "dev_split": metrics
            })
            output["submission_result"] = metrics
        elif phase_codename == "test_phase":
            output["result"].append({
                "test_split": metrics
            })
            output["submission_result"] = metrics
        
        logger.info(f"Evaluation complete: {metrics}")
        return output
        
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}", exc_info=True)
        # Return zero scores on error
        error_metrics = {
            "Accuracy": 0.0,
            "F1 Score": 0.0,
            "Processing Time (s)": 0.0,
            "Total": 0.0,
            "Error": str(e)
        }
        
        output = {"result": [], "submission_result": error_metrics}
        
        if phase_codename == "dev_phase":
            output["result"].append({
                "dev_split": error_metrics
            })
        elif phase_codename == "test_phase":
            output["result"].append({
                "test_split": error_metrics
            })
            
        return output