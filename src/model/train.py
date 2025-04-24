import os
import pandas as pd
import joblib
import argparse
import wandb
from sklearn.metrics import classification_report, accuracy_score

# --- Login to Weights & Biases ---
wandb.login(key="f505fc5086438e923a5d468c900a748275a8ff7c", relogin=True)

# --- Parse execution ID ---
parser = argparse.ArgumentParser()
parser.add_argument('--IdExecution', type=str, help='ID of the execution')
args = parser.parse_args()

if args.IdExecution:
    print(f"IdExecution: {args.IdExecution}")
else:
    args.IdExecution = "testing console"

# --- Load standardized data from wandb artifact ---
with wandb.init(project="EXPERIENCIAS", name=f"Train Model ExecId-{args.IdExecution}", job_type="train-model") as run:
    artifact = run.use_artifact("sgonzaleg2-universidad-eafit/EXPERIENCIAS/iris-tabular-standardized:latest", type="dataset")
    data_dir = artifact.download()

    df_train = pd.read_csv(os.path.join(data_dir, "train.csv"))
    df_test = pd.read_csv(os.path.join(data_dir, "test.csv"))

    # --- Split features and labels ---
    X_train = df_train.drop(columns="target")
    y_train = df_train["target"]
    X_test = df_test.drop(columns="target")
    y_test = df_test["target"]

    # --- Load the untrained model ---
    model_path = "./model/iris_logreg_model.joblib"
    model = joblib.load(model_path)

    # --- Train the model ---
    model.fit(X_train, y_train)

    # --- Evaluate the model ---
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    # --- Log metrics to wandb ---
    wandb.log({
        "accuracy": acc,
        "precision_setosa": report["0"]["precision"],
        "recall_setosa": report["0"]["recall"],
        "f1_setosa": report["0"]["f1-score"],
        "precision_versicolor": report["1"]["precision"],
        "recall_versicolor": report["1"]["recall"],
        "f1_versicolor": report["1"]["f1-score"],
        "precision_virginica": report["2"]["precision"],
        "recall_virginica": report["2"]["recall"],
        "f1_virginica": report["2"]["f1-score"],
    })

    # --- Save the trained model ---
    trained_model_path = "./model/iris_logreg_model_trained.joblib"
    joblib.dump(model, trained_model_path)
    print(f"✅ Model trained and saved to {trained_model_path}")
