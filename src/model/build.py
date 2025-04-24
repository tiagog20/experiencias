import os
import argparse
import wandb
from sklearn.linear_model import LogisticRegression
import joblib

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

# --- Create model directory if it doesn't exist ---
if not os.path.exists("./model"):
    os.makedirs("./model")

# --- Build the model ---
model = LogisticRegression(
    multi_class='multinomial',
    solver='lbfgs',
    max_iter=200
)

# --- Save the untrained model ---
model_path = "./model/iris_logreg_model.joblib"
joblib.dump(model, model_path)
print(f"✅ LogisticRegression model built and saved (untrained) at {model_path}")
