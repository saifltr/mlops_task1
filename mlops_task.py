# Import necessary libraries
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    balanced_accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
import optuna
import wandb

import warnings
warnings.filterwarnings('ignore')


def plot_confusion_matrix(cm):
    """Plot confusion matrix"""
    plt.figure(figsize=(9, 9))
    plt.imshow(cm, interpolation="nearest", cmap="Pastel1")
    plt.title("Confusion matrix", size=15)
    plt.colorbar()
    tick_marks = np.arange(10)
    plt.xticks(
        tick_marks,
        ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
        rotation=45,
        size=10,
    )
    plt.yticks(tick_marks, ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"], size=10)
    plt.tight_layout()
    plt.ylabel("Actual label", size=15)
    plt.xlabel("Predicted label", size=15)
    width, height = cm.shape
    for x in range(width):
        for y in range(height):
            plt.annotate(
                str(cm[x][y]),
                xy=(y, x),
                horizontalalignment="center",
                verticalalignment="center",
            )
    return plt


def sanity_checks(digits):
    """Perform sanity checks on the dataset"""
    wandb.log({"Image Data Shape": digits.data.shape})
    wandb.log({"Label Data Shape": digits.target.shape})

    plt.figure(figsize=(20, 4))
    for index, (image, label) in enumerate(zip(digits.data[0:5], digits.target[0:5])):
        plt.subplot(1, 5, index + 1)
        plt.imshow(np.reshape(image, (8, 8)), cmap=plt.cm.gray)
        plt.title("Training: %i\n" % label, fontsize=20)
    sanity_check_plot = plt
    plt.close()
    return sanity_check_plot


def visualize_test(x_test, y_test, predictions):
    """Visualize test results"""
    test_plots = []
    for idx, (image, label, prediction) in enumerate(zip(x_test, y_test, predictions)):
        plt.imshow(np.reshape(image, (8, 8)), cmap=plt.cm.gray)
        plt.title(f"Label: {label}, Prediction {prediction}")
        test_plot = plt.gcf()
        test_plots.append(test_plot)
        plt.close()
    return test_plots


def evaluate_model(logisticRegr, x_test, y_test, predictions):
    """Evaluate the model"""
    score = logisticRegr.score(x_test, y_test)
    balanced_accuracy = balanced_accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average=None)
    recall = recall_score(y_test, predictions, average=None)

    wandb.log({"Mean Accuracy": score, "Balanced Accuracy": balanced_accuracy, "Precision": precision, "Recall": recall})
    return score, balanced_accuracy, precision, recall


def show_confusion_matrix(y_test, predictions):
    """Show confusion matrix."""
    cm = confusion_matrix(y_test, predictions)
    wandb.log({"Confusion Matrix": cm})
    confusion_matrix_plot = plot_confusion_matrix(cm)
    return confusion_matrix_plot


def get_hyper_params_from_optuna(trial):
    """Get hyperparameters from Optuna trial"""
    penality = trial.suggest_categorical("penalty", ["l1", "l2", "elasticnet", "none"])

    if penality == "none":
        solver_choices = ["newton-cg", "lbfgs", "sag", "saga"]
    elif penality == "l1":
        solver = "liblinear"
    elif penality == "l2":
        solver_choices = ["newton-cg", "lbfgs", "liblinear", "sag", "saga"]
    elif penality == "elasticnet":
        solver = "saga"

    if not (penality == "l1" or penality == "elasticnet"):
        solver = trial.suggest_categorical("solver_" + penality, solver_choices)

    C = trial.suggest_float("C", 0.1, 1)

    fit_intercept = trial.suggest_categorical("fit_intercept", [True, False])

    intercept_scaling = trial.suggest_float("intercept_scaling", 0.1, 1.0)

    if penality == "elasticnet":
        l1_ratio = trial.suggest_float("l1_ratio", 0, 1)
    else:
        l1_ratio = None
    return penality, solver, C, fit_intercept, intercept_scaling, l1_ratio


def train_logistic_regression(x_train, y_train, hyperparameters):
    """Train logistic regression model"""
    logisticRegr = LogisticRegression(**hyperparameters)
    logisticRegr.fit(x_train, y_train)
    return logisticRegr


def objective(trial):
    """Objective function for Optuna optimization."""
    digits = datasets.load_digits()
    sanity_checks(digits)

    x_train, x_test, y_train, y_test = train_test_split(
        digits.data, digits.target, test_size=0.25, random_state=0
    )

    (
        penality,
        solver,
        C,
        fit_intercept,
        intercept_scaling,
        l1_ratio,
    ) = get_hyper_params_from_optuna(trial)

    hyperparameters = {
        "penalty": penality,
        "C": C,
        "fit_intercept": fit_intercept,
        "intercept_scaling": intercept_scaling,
        "solver": solver,
        "l1_ratio": l1_ratio,
    }

    logisticRegr = train_logistic_regression(x_train, y_train, hyperparameters)
    predictions = logisticRegr.predict(x_test)

    visualize_test(x_test, y_test, predictions)

    _, balanced_accuracy, _, _ = evaluate_model(
        logisticRegr, x_test, y_test, predictions
    )

    show_confusion_matrix(y_test, predictions)

    return balanced_accuracy


def main():
    """Main function"""
    # Initialize wandb
    wandb.init(project="mlops_task")

    # Increase the number of trials for better optimization
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=5)  # Increase number of trials

    trial = study.best_trial

    wandb.log({"Best Balanced Accuracy": trial.value})
    wandb.log({"Best Hyperparameters": trial.params})
    


if __name__ == "__main__":
    main()
