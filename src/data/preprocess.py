import argparse
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
import wandb

wandb.login(key="f505fc5086438e923a5d468c900a748275a8ff7c", relogin=True)
# --- Argumentos ---
parser = argparse.ArgumentParser()
parser.add_argument('--IdExecution', type=str, default="testing console", help='ID of the execution')
args = parser.parse_args()
print(f"IdExecution: {args.IdExecution}")

# --- Normalización estándar ---
def standardize(df):
    X = df.drop(columns="target")
    y = df["target"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    df_std = pd.DataFrame(X_scaled, columns=X.columns)
    df_std["target"] = y.values
    return df_std

# --- Exportar como CSV ---
def export_to_csv(df_train, df_val, df_test, output_dir="iris_data_standardized"):
    os.makedirs(output_dir, exist_ok=True)
    df_train.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    df_val.to_csv(os.path.join(output_dir, "val.csv"), index=False)
    df_test.to_csv(os.path.join(output_dir, "test.csv"), index=False)
    return output_dir

# --- Cargar dataset original desde wandb ---
def load_original_data():
    with wandb.init(project="EXPERIENCIAS", name=f"Load Original Iris - ExecId-{args.IdExecution}", job_type="load-original-data") as run:
        artifact = run.use_artifact("sgonzaleg2-universidad-eafit/EXPERIENCIAS/iris-tabular:latest", type="dataset")
        path = artifact.download()
        
        df_train = pd.read_csv(os.path.join(path, "train.csv"))
        df_val = pd.read_csv(os.path.join(path, "val.csv"))
        df_test = pd.read_csv(os.path.join(path, "test.csv"))
        
        return df_train, df_val, df_test

# --- Registrar nuevo artifact con datos estandarizados ---
def log_artifact(output_dir):
    with wandb.init(project="EXPERIENCIAS", name=f"Iris Standardized ExecId-{args.IdExecution}", job_type="preprocess-data") as run:
        artifact = wandb.Artifact(
            name="iris-tabular-standardized",
            type="dataset",
            description="Iris dataset standardized with StandardScaler",
            metadata={"source_artifact": "iris-tabular:latest"}
        )
        artifact.add_file(os.path.join(output_dir, "train.csv"))
        artifact.add_file(os.path.join(output_dir, "val.csv"))
        artifact.add_file(os.path.join(output_dir, "test.csv"))
        run.log_artifact(artifact)
        print("✅ Nuevo artifact registrado en wandb.")

# --- Main ---
def main():
    df_train, df_val, df_test = load_original_data()
    df_train_std = standardize(df_train)
    df_val_std = standardize(df_val)
    df_test_std = standardize(df_test)
    output_dir = export_to_csv(df_train_std, df_val_std, df_test_std)
    log_artifact(output_dir)

if __name__ == "__main__":
    main()
# --- Fin del script ---
# Este script carga el dataset Iris desde un artifact de wandb, lo estandariza usando StandardScaler,
#cabezón