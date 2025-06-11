from model.model import SimpleMLP, SimpleMLPDeeper
from data_handling.dataset import EmbeddingsToAgeDataset, create_datasets, create_three_datasets
import pandas as pd
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import optuna
from functools import partial


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, cuda=False, early_stopping_patience=5, verbose=True):
    train_losses = []
    val_losses = []
    early_stopping_counter = 0
    early_stopping_min_loss = float('inf')
    if verbose:
        print(f"Starting training on {len(train_loader.dataset)} training samples and {len(val_loader.dataset)} validation samples.")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            if cuda:
                inputs = inputs.cuda()
                labels = labels.cuda()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        if verbose:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(val_loader):
                if cuda:
                    inputs = inputs.cuda()
                    labels = labels.cuda()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # early stopping
        if val_loss < early_stopping_min_loss:
            early_stopping_min_loss = val_loss
            early_stopping_counter = 0
            if verbose:
                print(f'\tValidation Loss: {val_loss:.4f}')
        else:
            early_stopping_counter += 1
            if verbose:
                print(f'\tValidation Loss: {val_loss:.4f}, Early stopping Counter: {early_stopping_counter}/{early_stopping_patience}')
            if early_stopping_counter >= early_stopping_patience:
                if verbose:
                    print("Early stopping triggered.")
                break
    return train_losses, val_losses

def test_model(model, test_loader, criterion, verbose=True):
    test_losses = []
    for i, (inputs, labels) in enumerate(test_loader):
        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_losses.append(loss.item())
    avg_test_loss = sum(test_losses) / len(test_losses)
    if verbose:
        print(f'Average Test Loss: {avg_test_loss:.4f}')
    return test_losses

def run_param_training(trial, datasets):
    
    criterion = trial.suggest_categorical('criterion', [torch.nn.L1Loss(), torch.nn.MSELoss()])
    optimizer = trial.suggest_categorical('optimizer', [optim.Adam, optim.SGD])
    BATCHSIZE = trial.suggest_int('batch_size', 16, 64, step=16)
    NUM_EPOCHS = trial.suggest_int('num_epochs', 10, 100, step=10)
    HIDDEN_DIM = trial.suggest_int('hidden_dim', 256, 2048, step=256)
    OUTPUT_DIM = 1
    LR = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    
    verbose = False
    cuda = torch.cuda.is_available()
    
    print(f"Running trial with parameters: BATCHSIZE={BATCHSIZE}, NUM_EPOCHS={NUM_EPOCHS}, HIDDEN_DIM={HIDDEN_DIM}")
    if verbose:
        print("Loading data from:", train_path)
    train_set, val_set, test_set = datasets # 

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=BATCHSIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=BATCHSIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=BATCHSIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

    INPUT_DIM = train_set.X.shape[1]
    if verbose:
        print(f"Data loaded successfully.\n\tInput dimension: {INPUT_DIM}, Hidden dimension: {HIDDEN_DIM}, Output dimension: {OUTPUT_DIM}")
    
    model = SimpleMLPDeeper(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
    optimizer = optimizer(model.parameters(), lr=LR)

    train_losses, val_losses = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=NUM_EPOCHS, cuda=cuda, verbose=verbose)
    torch.save(model.state_dict(), 'model/simple_mlp.pth')
    test_losses = test_model(model, test_loader, criterion)
    avg_test_loss = sum(test_losses) / len(test_losses)

    if verbose:
        print(f"Average Test Loss: {avg_test_loss:.4f}")
    return avg_test_loss


if __name__ == "__main__":
    print("Starting training process...")
    train_path = "../data/radiomics_embeddings_fat.csv" 
    datasets = create_three_datasets(train_path, val_size=0.3)
    study = optuna.create_study(direction='minimize')
    partial_run_param_training = partial(run_param_training, datasets=datasets)

    study.optimize(partial_run_param_training, n_trials=10)
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    

    print("FINISHED")
