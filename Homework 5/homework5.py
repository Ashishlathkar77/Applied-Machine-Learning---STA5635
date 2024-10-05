import sys
import os
import numpy as np
import torch
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from typing import Literal


class FSA:
    def __init__(self, k: int, mu=300, s=0.001, Niter=300, lr=0.01, h=0.1):
        self.mu = mu
        self.k = k
        self.Niter = Niter
        self.lr = lr
        self.s = s
        self.h = h


    def predict(self, X: torch.Tensor) -> torch.Tensor:
        xw = X[:, self.idx].float() @ self.w.view(-1, 1) + self.w0
        return xw


    def fit(self, X: torch.Tensor, y: torch.Tensor, device: torch.device):
        p = X.shape[1]
        y = y.clone()
        y[y == 0] = -1
        self.idx = torch.arange(0, p).long().to(device)
        self.w = torch.zeros((p, 1), device=device, requires_grad=True)
        self.w0 = torch.zeros(1, device=device, requires_grad=True)
        optimizer = optim.SGD([self.w, self.w0], lr=self.lr)
        losses = []
        for i in range(1, self.Niter + 1):
            optimizer.zero_grad()
            xw = self.predict(X)
            z = y * xw.squeeze()
            loss = torch.zeros_like(z)
            idx1 = (z >= 1 + self.h)
            idx2 = (torch.abs(1 - z) <= self.h)
            idx3 = (z <= 1 - self.h)
            loss[idx1] = 0
            loss[idx2] = ((1 + self.h - z[idx2]) ** 2) / (4 * self.h)
            loss[idx3] = 1 - z[idx3]
            loss1 = torch.mean(loss) + self.s * (torch.sum(self.w ** 2) + self.w0 ** 2)
            loss1.backward()
            optimizer.step()
            m = int(self.k + (p - self.k) * max(0, (self.Niter - 2 * i) / (2 * i * self.mu + self.Niter)))
            if m < self.w.shape[0]:
                sw = torch.abs(self.w.view(-1))
                sw_sorted, indices = torch.sort(sw, descending=True)
                thr = sw_sorted[m - 1].item()
                j = torch.where(sw >= thr)[0]
                self.idx = self.idx[j]
                self.w = self.w[j].detach().clone().requires_grad_(True)
                optimizer = optim.SGD([self.w, self.w0], lr=self.lr)
            losses.append(loss1.item())
        return losses


def load_data(dataset_name: Literal['Gisette', 'dexter', 'Madelon']):
    if dataset_name == 'Gisette':
        X_train = pd.read_csv('gisette_train.data', sep='\s+', header=None).to_numpy()
        y_train = pd.read_csv('gisette_train.labels', header=None).to_numpy().squeeze()
        X_test = pd.read_csv('gisette_valid.data', sep='\s+', header=None).to_numpy()
        y_test = pd.read_csv('gisette_valid.labels', header=None).to_numpy().squeeze()
    elif dataset_name == 'dexter':
        X_train = np.genfromtxt("dexter_train.csv", delimiter=',')
        y_train = np.loadtxt("dexter_train.labels")
        X_test = np.genfromtxt("dexter_valid.csv", delimiter=',')
        y_test = np.loadtxt("dexter_valid.labels")
    elif dataset_name == 'Madelon':
        X_train = np.loadtxt('madelon_train.data')
        y_train = np.loadtxt('madelon_train.labels')
        X_test = np.loadtxt('madelon_valid.data')
        y_test = np.loadtxt('madelon_valid.labels')
    else:
        raise ValueError("Unsupported dataset name.")
   
    return X_train, y_train, X_test, y_test


def main(DATASET: Literal['Gisette', 'dexter', 'Madelon']):
    output_dir = f'output_{DATASET}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    X_train, y_train, X_test, y_test = load_data(DATASET)


    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1  
    X_train_norm = (X_train - mean) / std
    X_test_norm = (X_test - mean) / std


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_train_tensor = torch.tensor(X_train_norm).float().to(device)
    y_train_tensor = torch.tensor(y_train).float().to(device)
    X_test_tensor = torch.tensor(X_test_norm).float().to(device)
    y_test_tensor = torch.tensor(y_test).float().to(device)


    k_list = [10, 30, 100, 300, 500]
    train_errors = []
    test_errors = []
    train_losses_k30 = None
    model_k30 = None


    for k in k_list:
        model = FSA(k=k, mu=300, s=0.001, Niter=300, lr=0.01, h=0.1)
        losses = model.fit(X_train_tensor, y_train_tensor.clone(), device)
        if k == 30:
            train_losses_k30 = losses
            model_k30 = model
            plt.figure()
            plt.plot(range(1, len(losses) + 1), losses)
            plt.xlabel('Iteration')
            plt.ylabel('Training Loss')
            plt.title(f'Training Loss vs Iteration (k=30) on {DATASET}')
            plt.savefig(os.path.join(output_dir, f'Training_Loss_k30_{DATASET}.png'))
            plt.close()


        with torch.no_grad():
            y_train_pred = model.predict(X_train_tensor)
            y_train_pred_labels = (y_train_pred.squeeze() >= 0).cpu().numpy().astype(int)
            train_error = np.mean(y_train_pred_labels != (y_train_tensor.cpu().numpy() == -1).astype(int))
            train_errors.append(train_error)


            y_test_pred = model.predict(X_test_tensor)
            y_test_pred_labels = (y_test_pred.squeeze() >= 0).cpu().numpy().astype(int)
            test_error = np.mean(y_test_pred_labels != (y_test_tensor.cpu().numpy() == -1).astype(int))
            test_errors.append(test_error)


        print(f'Completed training for k={k}')


    print('k\tTraining Error\tTest Error')
    for i, k in enumerate(k_list):
        print(f'{k}\t{train_errors[i]:.4f}\t\t{test_errors[i]:.4f}')


    error_df = pd.DataFrame({
        'k': k_list,
        'Training Error': train_errors,
        'Test Error': test_errors
    })
    error_df.to_csv(os.path.join(output_dir, f'Misclassification_Errors_{DATASET}.csv'), index=False)


    with open(os.path.join(output_dir, f'Misclassification_Errors_{DATASET}.tex'), 'w') as f:
        f.write(error_df.to_latex(index=False, float_format="%.4f"))


    plt.figure()
    plt.plot(k_list, train_errors, label='Training Error', marker='o')
    plt.plot(k_list, test_errors, label='Test Error', marker='s')
    plt.xlabel('Number of Features (k)')
    plt.ylabel('Misclassification Error')
    plt.title(f'Misclassification Error vs Number of Features on {DATASET}')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f'Misclassification_Error_vs_k_{DATASET}.png'))
    plt.close()


    if model_k30 is not None:
       
        y_train_pred_scores = model_k30.predict(X_train_tensor).squeeze().cpu().detach().numpy()
        fpr_train, tpr_train, _ = roc_curve((y_train_tensor.cpu().numpy() == -1).astype(int), y_train_pred_scores)
        roc_auc_train = auc(fpr_train, tpr_train)


       
        y_test_pred_scores = model_k30.predict(X_test_tensor).squeeze().cpu().detach().numpy()
        fpr_test, tpr_test, _ = roc_curve((y_test_tensor.cpu().numpy() == -1).astype(int), y_test_pred_scores)
        roc_auc_test = auc(fpr_test, tpr_test)


       
        plt.figure()
        plt.plot(fpr_train, tpr_train, color='blue', label=f'Train ROC curve (area = {roc_auc_train:.2f})')
        plt.plot(fpr_test, tpr_test, color='orange', label=f'Test ROC curve (area = {roc_auc_test:.2f})')
        plt.plot([0, 1], [0, 1], color='black', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc='lower right')
        plt.savefig(os.path.join(output_dir, f'ROC_Curve_k30_{DATASET}.png'))
        plt.close()


if __name__ == '__main__':
    # Choose your dataset here: 'Gisette', 'dexter', or 'Madelon'
    main('Gisette')  
