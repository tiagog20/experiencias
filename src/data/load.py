import wandb
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
import argparse
import sys
import os

# --- Argumentos adaptables para Jupyter o script ---
def parse_args():
    if "ipykernel" in sys.argv[0]:  # Ejecutando en Jupyter
        class Args:
            IdExecution = "local-debug"
        return Args()
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument('--IdExecution', type=str, help='ID de la ejecución')
        return parser.parse_args()

args = parse_args()
# --- Configuración de WandB ---
wandb.login(key="f505fc5086438e923a5d468c900a748275a8ff7c", relogin=True)

# --- Cargar el dataset Iris y dividirlo ---
def load_and_split(train_size=0.8):
    iris = load_iris(as_frame=True)
    df = iris.frame
    df.columns = [*iris.feature_names, "target"]

    df_trainval, df_test = train_test_split(df, test_size=0.2, random_state=42, stratify=df["target"])
    df_train, df_val = train_test_split(df_trainval, test_size=1 - train_size, random_state=42, stratify=df_trainval["target"])
    return df_train, df_val, df_test

# --- Guardar y registrar los archivos como Artifact CSV ---
def log_to_wandb():
    with wandb.init(
        project="EXPERIENCIAS",
        name=f"Iris CSV ExecId-{args.IdExecution}",
        job_type="load-data") as run:

        df_train, df_val, df_test = load_and_split()

        os.makedirs("iris_data", exist_ok=True)
        df_train.to_csv("iris_data/train.csv", index=False)
        df_val.to_csv("iris_data/val.csv", index=False)
        df_test.to_csv("iris_data/test.csv", index=False)

        artifact = wandb.Artifact(
            name="iris-tabular",
            type="dataset",
            description="Iris dataset in tabular format (CSV)",
            metadata={"source": "sklearn.datasets.load_iris",
                      "train_size": len(df_train),
                      "val_size": len(df_val),
                      "test_size": len(df_test)}
        )

        artifact.add_file("iris_data/train.csv")
        artifact.add_file("iris_data/val.csv")
        artifact.add_file("iris_data/test.csv")

        run.log_artifact(artifact)
        print("✅ Dataset CSV cargado y registrado con éxito en wandb.")

# --- Ejecutar ---
log_to_wandb()
# --- Fin del script ---
# --- Notas ---