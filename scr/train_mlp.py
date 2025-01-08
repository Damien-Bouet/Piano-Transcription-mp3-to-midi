import json
import os
import sys
from typing import List, Optional

sys.path.append(os.path.abspath(".."))

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import scikitplot as skplt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import (
    DetCurveDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)
from scr.ts_utils import create_dataset_with_stft


def load_data_from_folder(folder, sample_rate, duration, ticks_per_beat, bpm, overlap_ratio, freq_resolution, fix_velocity=False, n_test = 3):
    """
    Given a folder with audio and MIDI files, this function will load the data
    into X and y arrays, split them into train and test sets (80% train, 20% test).
    """
    
    # List all .wav and .mid files in the folder
    mp3_files = [f for f in os.listdir(folder) if f.endswith('.wav')]
    midi_files = [f for f in os.listdir(folder) if f.endswith('.mid')]

    # Shuffle the list of files to ensure random splitting
    np.random.shuffle(midi_files)

    num_train = len(midi_files) - n_test
    train_files = midi_files[:num_train]
    test_files = midi_files[num_train:]
    print(f"{len(train_files)} Training songs")
    print(f"{len(test_files)} Testing songs")

    # Initialize lists to store the training and testing data
    X_train, y_train = [], []
    X_test, y_test = [], []

    # Load training data
    for midi_file in train_files:
        for mp3_file in mp3_files:
            if midi_file[:-4] in mp3_file.split("_shift")[0]:
                shift = int(mp3_file.split("_shift_")[1][:-4])
                mp3_path = os.path.join(folder, mp3_file)
                midi_path = os.path.join(folder, midi_file)
                
                X, y = create_dataset_with_stft(mp3_path, midi_path, shift, sample_rate, duration,
                                            ticks_per_beat=ticks_per_beat, bpm=bpm,
                                            overlap_ratio=overlap_ratio, freq_resolution=freq_resolution)
                # midi_to_image(midi_path, ticks_per_beat, bpm, duration, True, 0, str(midi_path).replace(".mid", "_img.png"))
                X_train.append(X)
                y_train.append(y)

    # Load testing data
    for midi_file in test_files:
        for mp3_file in mp3_files:
            if midi_file[:-4] in mp3_file.split("_shift")[0]:
                shift = int(mp3_file.split("_shift_")[1][:-4])
                mp3_path = os.path.join(folder, mp3_file)
                midi_path = os.path.join(folder, midi_file)
                
                X, y = create_dataset_with_stft(mp3_path, midi_path, shift, sample_rate, duration,
                                            ticks_per_beat=ticks_per_beat, bpm=bpm,
                                            overlap_ratio=overlap_ratio, freq_resolution=freq_resolution)
                # midi_to_image(midi_path, ticks_per_beat, bpm, duration, True, 0, str(midi_path).replace(".mid", "_img.png"))
                X_test.append(X)
                y_test.append(y)


    # Convert lists to numpy arrays or torch tensors (depending on your needs)
    X_train = np.concatenate(X_train, axis=0)  # Assuming the outputs are numpy arrays
    y_train = np.concatenate(y_train, axis=0)
    X_test = np.concatenate(X_test, axis=0)
    y_test = np.concatenate(y_test, axis=0)

    return X_train, y_train, X_test, y_test, len(train_files), len(test_files)


def roll(X, roll_depth=0):
    if roll_depth > 0:
        X_roll = np.zeros((X.shape[0], (1+2*roll_depth), X.shape[1]))
        for i in range(X.shape[0]):
            # Determine start and end indices for rolling window
            start = max(0, i - roll_depth)
            end = min(X.shape[0], i + roll_depth + 1)
            
            # Calculate offset in the rolled array
            offset_start = max(0, roll_depth - i)
            offset_end = offset_start + (end - start)
            
            # Copy the relevant slice
            X_roll[i, offset_start:offset_end, :] = X[start:end, :]
        return X_roll.reshape((X.shape[0], (1+2*roll_depth)* X.shape[1]))
    else:
        return X
    

class MLPModel(nn.Module):
    def __init__(self, roll_depth, input_shape):
        super(MLPModel, self).__init__()
        self.input_shape = input_shape
        self.roll_depth = roll_depth
        # Define convolutional layers
        self.linear_layers = nn.ModuleList([
            nn.Linear(88*(1 + 2*roll_depth), 128), 
            nn.Linear(128, 128),  
            nn.Linear(128, 88),
        ])
    
    def forward(self, x, max_batch = 5000):
        res = torch.zeros((len(x), *list(self._forward(x[:2]).detach().numpy().shape)[1:]))
        for i in range(0, len(x)//max_batch +1):
            res[i*max_batch: min((i+1)*max_batch, len(x))] = self._forward(x[i*max_batch: min((i+1)*max_batch, len(x))])
        return res

    def _forward(self, x):
        # Assuming x shape is (batch_size, 1, height, width)
        for i, l in enumerate(self.linear_layers):
            if i < len(self.linear_layers)-1:
                x = F.relu(l(x))
            else:
                x = l(x) 
        return x
    
    def predict_proba(self, x):
        preds = np.zeros((len(x), *list(self.forward(torch.Tensor(roll(x[:2], self.roll_depth))).detach().numpy().shape)[1:]))
        for i in range(0, len(x)//1000 +1):
            preds[i*1000: min((i+1)*1000, len(x))] = torch.sigmoid(self.forward(torch.Tensor(roll(x[i*1000: min((i+1)*1000, len(x))], self.roll_depth)))).detach().numpy()
        return preds
    
    def save_structure(self, folder, roll_depth, n_train, n_test):
        layer_params = {"roll_depth":roll_depth, "n_train":n_train, "n_test":n_test, "n_parameters": sum(p.numel() for p in self.parameters())}
        for name, module in self.named_modules():
            if len(list(module.children())) == 0:  # Exclude container modules
                if isinstance(module, nn.Conv2d):
                    layer_params[name] = {
                        "type": "Conv2d",
                        "in_channels": module.in_channels,
                        "out_channels": module.out_channels,
                        "kernel_size": module.kernel_size,
                        "stride": module.stride,
                        "padding": module.padding,
                    }
                elif isinstance(module, nn.Linear):
                    layer_params[name] = {
                        "type": "Linear",
                        "in_features": module.in_features,
                        "out_features": module.out_features,
                    }

        # Save to JSON
        with open(f"{folder}/model_layers.json", "w") as f:
            json.dump(layer_params, f, indent=4)


def train(X, y, X_test, y_test, roll_depth, folder, n_train, n_test):

    X = torch.Tensor(roll(X, roll_depth))
    X_test = torch.Tensor(roll(X_test, roll_depth))
    num_epochs = 25
    batch_size = 32

    # Model, loss, optimizer, and learning rate scheduler
    model = MLPModel(roll_depth, X.shape)
    model.save_structure(folder, roll_depth, n_train, n_test)

    criterion = nn.BCEWithLogitsLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4)  # Adam optimizer

    losses = []
    losses_test = []
    accuracies = []
    lrs = []

    # Training loop
    pbar = tqdm(range(num_epochs), desc="Training", dynamic_ncols=True)
    for epoch in pbar:
        model.train()  # Set model to training mode
        running_loss = 0.0
        correct = 0
        total = 0

        ind = torch.randperm(X.size(0))
        X = X[ind]
        y = y[ind]
        
        # Iterate over batches
        for i in range(0, len(X), batch_size):
            inputs = X[i:i+batch_size, :]
            targets = y[i:i+batch_size, :]
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)  # Remove extra dimension from the output (to match target shape)
            
            # Calculate loss
            loss = criterion(outputs, targets)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            predicted = (outputs > 0.5).float()  # Convert sigmoid output to binary (0 or 1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)
            lr = optimizer.param_groups[0]['lr']
            lrs.append(lr)
        

        # Print statistics
        # print(running_loss,  (len(X) // batch_size))
        epoch_loss = running_loss / (len(X) // batch_size)
        epoch_loss_test = criterion(model(X_test), y_test).item()
        # print(epoch_loss)
        accuracy = correct / total
        
        pbar.set_postfix(loss=epoch_loss, test_loss=epoch_loss_test, accuracy=accuracy, lr=lr)

        
        losses.append(epoch_loss)
        losses_test.append(epoch_loss_test)
        accuracies.append(accuracy)
        
        plt.plot(losses, label="train loss")
        plt.plot(losses_test, label="test loss")
        plt.yscale("log")
        plt.legend()
        plt.savefig(folder+"/loss.png")
        plt.close()

        plt.plot(lrs, label="learning rate")
        plt.yscale("log")
        plt.legend()
        plt.savefig(folder+"/lr.png")
        plt.close()

    return model, losses, losses_test, accuracies, lrs
# Optionally, you can save the model after training

def evaluate_binary(
    y_test,
    preds,
    labels: Optional[List[str]] = None,
    acc_baseline: Optional[float] = None,
    results_folder: Optional[str] = None,
    file_name_prefix: str = "",
    target_names: Optional[str] = None,
    overwrite: Optional[bool] = False,
    threshold: float = 0.5,
) -> None:
    """
    Evaluates the binary classification model performance on the provided test data by calculating and displaying various
    performance metrics and curves including Average Precision Score, ROC AUC Score, and Log Loss. Additionally, it plots
    Receiver Operating Characteristic (ROC) curves, Precision-Recall (PR) curves, Detection Error Tradeoff (DET) curves,
    and Accuracy depending on probability. It also calculates and plots the Kolmogorov-Smirnov (KS) statistic and the
    Calibration Curve.

    Args:
        X_test: Test set features.
        y_test: Test set labels.
        clfs: List of instances of binary classification models.
        acc_baseline: Baseline accuracy value to be plotted on the accuracy depending on probability plot.
        mlflow_log: Boolean indicating whether to log the results to MLflow or not.
        results_folder: Path to the folder where the results will be saved.
        overwrite: Boolean indicating whether to overwrite an existing folder in case the `results_folder` already exists.

    Returns:
        None
    """

    folder = results_folder if results_folder is not None else "results"

    try:
        if overwrite or not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)
        else:
            for n in range(2, 100):
                if overwrite or not os.path.exists(f"{folder}_{n}"):
                    os.makedirs(f"{folder}_{n}", exist_ok=True)
                    folder = f"{folder}_{n}"
                    break

    except Exception as exc:
        raise ValueError(
            "The folder creation raised an error. Check that the results_folder value is a valide directory name"
        ) from exc

    y_pred_proba_positive_class = preds.flatten()

    _average_precision_score = average_precision_score(
        y_test, y_pred_proba_positive_class
    )
    _roc_auc_score = roc_auc_score(y_test, y_pred_proba_positive_class)
    _log_loss = log_loss(y_test, y_pred_proba_positive_class)

    print(
        f"average_precision_score",
        _average_precision_score,
    )
    print(f"roc_auc_score", _roc_auc_score)
    print(f"log_loss", _log_loss)

    cm = confusion_matrix(
        y_test, np.array([1 if v > threshold else 0 for v in y_pred_proba_positive_class])
    )
    cm_log = np.log(cm + 1) / np.log(10)
    cm = (cm / np.max(cm) * 1000000).astype(int)
    # Create figure and axes
    plt.figure(figsize=(8, 6))
    
    norm = colors.LogNorm(vmin=max(cm_log.min(), 1e-10), vmax=cm_log.max())
    
    # Create heatmap with log scaling
    sns.heatmap(cm_log, annot=cm, fmt='d', cmap='YlGnBu', 
                norm=norm,
                xticklabels=['Neg', 'Pos'],
                yticklabels=['Neg', 'Pos'])
    
    # Add labels and title
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    plt.tight_layout()

    plt.savefig(
        f"{folder}/confusion_matrix.png"
    )
    plt.close()

    preds = (preds>threshold).astype(int)
    formatted_preds = preds.copy()
    for i in range(2, preds.shape[0] - 2):
        # Find "1"s in the current row
        ones = preds[i] == 1
        # Check the rows above and below for "0"
        # alone = ((preds[i - 1] == 0) & ((preds[i + 1] == 0) | (preds[i + 2] == 0))) | ((preds[i + 1] == 0) & ((preds[i - 1] == 0) | (preds[i - 2] == 0)))
        alone = (preds[i - 1] == 0) & (preds[i + 1] == 0)
        
        # # Replace "1"s that are "alone" with "0"
        formatted_preds[i][ones & alone] = 0

    cm = confusion_matrix(
        y_test, formatted_preds.flatten()
    )
    
    cm_log = np.log(cm + 1) / np.log(10)
    cm = (cm / np.max(cm) * 1000000).astype(int)
    # Create figure and axes
    plt.figure(figsize=(8, 6))
    
    norm = colors.LogNorm(vmin=max(cm_log.min(), 1e-10), vmax=cm_log.max())
    
    # Create heatmap with log scaling
    sns.heatmap(cm_log, annot=cm, fmt='d', cmap='YlGnBu', 
                norm=norm,
                xticklabels=['Neg', 'Pos'],
                yticklabels=['Neg', 'Pos'])
    
    # Add labels and title
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    plt.tight_layout()

    plt.savefig(
        f"{folder}/confusion_matrix_postprocessed.png"
    )
    plt.close()

    
    outer = [
        ["roc", "pr", "det"],
        ["acc", "ks", "cal"],
    ]

    fig, axd = plt.subplot_mosaic(outer, layout="constrained", figsize=(33, 20))

    min_acc = []
    

    RocCurveDisplay.from_predictions(
        y_test,
        y_pred_proba_positive_class,
        ax=axd["roc"],
        plot_chance_level=True,
    )

    DetCurveDisplay.from_predictions(
        y_test, y_pred_proba_positive_class, ax=axd["det"]
    )

    PrecisionRecallDisplay.from_predictions(
        y_test, y_pred_proba_positive_class, ax=axd["pr"]
    )

    cols = list(colors.TABLEAU_COLORS)

    y_pred_proba = np.array(
        [list(1 - y_pred_proba_positive_class), list(y_pred_proba_positive_class)]
    ).T

    
    skplt.metrics.plot_ks_statistic(
        y_test,
        y_pred_proba,
        ax=axd[f"ks"],
        title=f"KS plot",
        text_fontsize=matplotlib.rcParams["axes.labelsize"],
        title_fontsize=matplotlib.rcParams["axes.titlesize"],
    )
    for j, line in enumerate(axd[f"ks"].lines[-3:-1]):
        line.set_color(cols[j * 2])

    proba_threshold = np.linspace(0, 1, 100)

    list_acc = np.array(
        [
            accuracy_score(y_test, y_pred_proba_positive_class > p)
            for p in proba_threshold
        ]
    )
    min_acc.append(min(list_acc[len(list_acc) // 20 : -len(list_acc) // 20]))
    if acc_baseline and i == 0:
        axd["acc"].plot(
            [0, 1],
            [acc_baseline, acc_baseline],
            "k--",
            label=f"Baseline - acc = {acc_baseline:.3f}",
        )
    axd["acc"].plot(
        proba_threshold,
        list_acc,
        label=f"max acc = {max(list_acc):.3f}",
    )

    axd["acc"].grid(linestyle="--")
    axd["acc"].set_xlim(0, 1)
    axd["acc"].set_ylim(min(0.5, min(min_acc)), 1)
    axd["acc"].set_xlabel("proba threshold")
    axd["acc"].set_ylabel("Test Accuracy")

    axd["roc"].set_title("ROC curves")
    axd["pr"].set_title("Precision Recall curves")
    axd["det"].set_title("Detection Error Tradeoff curves")
    axd["acc"].set_title("Accuracy depending on proba")
    axd["cal"].set_title("Proba calibration")

    for ax in axd.values():
        ax.grid(linestyle="--")
        ax.legend()
    
    # fig.tight_layout()
    fig.savefig(f"{folder}/evaluation_metrics.png")

    plt.close(fig)

if __name__ == "__main__":
    np.random.seed(1)

    folder_path = 'simple_piano_fix_speed'
    sample_rate = 22050
    duration = None
    ticks_per_beat = 8
    bpm = 120
    overlap_ratio = 0.5
    freq_resolution = 2
    load = True
    n_test_songs = 4

    if not load:
        X_train, y_train, X_test, y_test, n_train, n_test = load_data_from_folder(
            folder_path, sample_rate, duration, ticks_per_beat, bpm, overlap_ratio, freq_resolution, True, n_test_songs
        )

        X_train, y_train, X_test, y_test = torch.Tensor(X_train), torch.Tensor(y_train), torch.Tensor(X_test), torch.Tensor(y_test)
    else:
    X_train = torch.load('X_train_full.pt', weights_only=True)
    y_train = torch.load('y_train_full.pt', weights_only=True)
    X_test = torch.load('X_test_full.pt', weights_only=True)
    y_test = torch.load('y_test_full.pt', weights_only=True)

    n_train = len([f for f in os.listdir(folder_path) if f.endswith('.mid')]) - n_test_songs
    n_test = n_test_songs

    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    mean, std = X_train.mean(dim=(0, 1)), X_train.std(dim=(0, 1))
    std[std == 0] = 1
    X_train = (X_train - mean)/std
    X_test = (X_test - mean)/std
    
    for roll_depth in [2]:
        i = 1
        while os.path.exists(f"results_mlp_{i}"):
            i += 1
        model_name = f"{i}"
        training = True

        os.makedirs(f"results_mlp_{model_name}", exist_ok=True)

        if training:
            model, losses, losses_test, accuracies, lrs = train(X_train, y_train, X_test, y_test, roll_depth, f"results_mlp_{model_name}", n_train, n_test)
            torch.save(model.state_dict(), f"results_mlp_{model_name}/{model_name}.pth")
            torch.save(
                {"losses":losses, "losses_test":losses_test, "accuracies":accuracies, "lrs":lrs},
                f"results_mlp_{model_name}/history.pth"
            )
        else:
            model = MLPModel(roll_depth, X_train[:1].shape)
            model.load_state_dict(torch.load(f"results_mlp_{model_name}/{model_name}.pth"))
            d = torch.load(f"results_mlp_{model_name}/history.pth")
            losses = d["losses"]
            losses_test = d["losses_test"]
            accuracies = d["accuracies"]
            lrs = d["lrs"]

        # Train
        from scr.display import midis_comparison
        preds = model.predict_proba(X_train)
        preds = (preds>0.5).astype(int)

        img = midis_comparison(
            y_train.numpy(), preds,
            target_color=(0, 0.7, 1), preds_color=(1, 0.9, 0.4), correct_color=(0.2, 0.9, 0.2),
            duration=600, ticks_per_beat = 4, bpm = 120,
        )
        img.save(f"results_mlp_{model_name}/train_preds.png")

        # Test
        preds = model.predict_proba(X_test)
        preds = (preds>0.5).astype(int)

        formatted_preds = preds.copy()
        for i in range(2, preds.shape[0] - 2):
            # Find "1"s in the current row
            ones = preds[i] == 1
            # Check the rows above and below for "0"
            # alone = ((preds[i - 1] == 0) & ((preds[i + 1] == 0) | (preds[i + 2] == 0))) | ((preds[i + 1] == 0) & ((preds[i - 1] == 0) | (preds[i - 2] == 0)))
            alone = (preds[i - 1] == 0) & (preds[i + 1] == 0)
            
            # # Replace "1"s that are "alone" with "0"
            formatted_preds[i][ones & alone] = 0

        img = midis_comparison(
            y_test.numpy(), formatted_preds,
            target_color=(0, 0.7, 1), preds_color=(1, 0.9, 0.4), correct_color=(0.2, 0.9, 0.2),
            duration=None, ticks_per_beat = 4, bpm = 120,
        )
        img.save(f"results_mlp_{model_name}/test_preds.png")


        preds = model.predict_proba(X_test)
        evaluate_binary(y_test.numpy().flatten(), preds, results_folder=f"results_mlp_{model_name}", overwrite=True)

