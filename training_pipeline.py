# training_pipeline.py (Modified)

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
# New imports for MLflow and Hyperparameter Tuning
import mlflow
import mlflow.sklearn
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import numpy as np
import sys

# --- Configuration (Model Output Paths are now mostly ignored/removed) ---
# NOTE: DVC should only track data and DVC pipeline definitions (dvc.yaml), not models.

# --- Step 1: Define the Objective Function for Hyperopt ---
def objective(params, X_train, y_train):
    # Start a new MLflow run for each hyperparameter combination
    with mlflow.start_run(run_name=f"dt_run_max_depth_{params['max_depth']}"):
        
        # Log parameters specific to this run
        mlflow.log_params(params)
        
        # Instantiate and train model
        mod_dt = DecisionTreeClassifier(
            max_depth=int(params['max_depth']), 
            min_samples_leaf=int(params['min_samples_leaf']),
            random_state=1
        )
        
        # Use cross-validation for a more robust evaluation metric
        # Maximize accuracy (or minimize negative accuracy)
        accuracy = np.mean(cross_val_score(mod_dt, X_train, y_train, cv=5, scoring='accuracy'))
        
        # The Hyperopt objective function must minimize the metric
        loss = -accuracy
        
        # Log metrics to MLflow
        mlflow.log_metric("cv_accuracy", accuracy)
        mlflow.log_metric("loss", loss) # Log the minimized metric
        
        # If this is a good model, train it fully and log it
        if accuracy > 0.95: # Simple threshold to save models
            mod_dt.fit(X_train, y_train)
            
            # Log the model artifact to the MLflow server/registry
            mlflow.sklearn.log_model(
                sk_model=mod_dt, 
                artifact_path="model", 
                registered_model_name="IRIS_DecisionTree_Model"
            )
            
        return {'loss': loss, 'status': STATUS_OK, 'model_accuracy': accuracy}


# --- Step 2: Main Training and Optimization Loop ---
def run_hyperparameter_tuning(local_data_path):
    mlflow.set_experiment("IRIS_Classifier_Hyperopt")
    
    data_df = pd.read_csv(local_data_path)
    X = data_df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
    y = data_df.species
    
    # Define hyperparameter search space
    search_space = {
        'max_depth': hp.quniform('max_depth', 2, 10, 1),
        'min_samples_leaf': hp.quniform('min_samples_leaf', 5, 20, 1)
    }
    
    # Start optimization
    trials = Trials()
    best = fmin(
        fn=lambda p: objective(p, X, y),
        space=search_space,
        algo=tpe.suggest,
        max_evals=20, # Run 20 different hyperparameter combinations
        trials=trials
    )

    print(f"\nOptimization complete. Best run parameters: {best}")
    
    # Find the run with the best performance (lowest loss = highest accuracy)
    best_loss = trials.best_trial['result']['loss']
    best_acc = -best_loss
    
    print(f"Best cross-validation accuracy achieved: {best_acc:.3f}")
    
    # You can now find this best run in the MLflow UI
    # The best model is already registered if its accuracy was > 0.95

# --- DVC-Compliant Execution ---
if __name__ == "__main__":
    # Ensure local MLflow tracking is enabled (for a simple setup)
    # For a real deployment, replace with your MLflow Tracking Server URI:
    # mlflow.set_tracking_uri("http://your-mlflow-server:5000")
    
    if len(sys.argv) < 2:
        print("Usage: python training_pipeline.py <local_data_path>")
        sys.exit(1)
        
    local_data_path = sys.argv[1]
    run_hyperparameter_tuning(local_data_path)
