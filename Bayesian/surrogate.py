import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
from torch.utils.data import DataLoader
from Bayesian.BayesianDataset import BayesianDataset
import pickle
import gpytorch




class MLPEncoder(nn.Module):
    def __init__(self, input_size=8192, hidden_size=256, latent_size=10 ,output_size=1, device=torch.device('cpu')):
        super(MLPEncoder, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)  # 输入层到第一个隐藏层
        self.drop1 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(hidden_size, hidden_size)    # 第一个隐藏层到第二个隐藏层
        self.drop2 = nn.Dropout(0.1)
        self.fc3 = nn.Linear(hidden_size, hidden_size)      # 第二个隐藏层到第三个隐藏层
        self.drop3 = nn.Dropout(0.1)
        self.fc4 = nn.Linear(hidden_size,latent_size)         # 输出层
        self.fc5 = nn.Linear(latent_size,output_size)
        self.device = device

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.drop1(x)
        x = F.relu(self.fc2(x))
        x = self.drop2(x)
        x = F.relu(self.fc3(x))
        x = self.drop3(x)
        # latent_x = self.fc4(x)
        latent_x = F.relu(self.fc4(x))
        output = self.fc5(latent_x)
        return output, latent_x

    def predict(self, x):
        self.eval()
        return self.forward(x)

    def train_mlp(self, dataset, normalizer=None, epochs=100, batch_size=50, learning_rate=0.001, weight_decay=0.001):
        device = self.device
        with torch.no_grad():
            self.reinitialize_weights_he()

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)  # 使用Adam优化器
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.train()
        for epoch in range(epochs):
            running_loss = 0.0

            for embeddings, scores in dataloader:
                if normalizer is not None:
                    scores = normalizer.normalize_one(scores)
                optimizer.zero_grad()
                embeddings = embeddings.to(device)
                scores = scores.to(device, torch.bfloat16).squeeze()

                outputs, _ = self.forward(embeddings)
                outputs = outputs.squeeze()

                loss = criterion(outputs, scores)


                loss.backward()

                optimizer.step()

                running_loss += loss.item()
                pass



        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(dataloader):.4f}")
        print("MLP Encoder Training Complete")
        return running_loss / len(dataloader)

    def reinitialize_weights_he(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                # kaiming he
                init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                if layer.bias is not None:
                    init.constant_(layer.bias, 0)

class GPModel(gpytorch.models.ExactGP):

    def __init__(self, train_x, train_y, likelihood, device = torch.device('cuda:0'), latent_size=10):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        # rbf_kernel = gpytorch.kernels.RBFKernel(ard_num_dims=latent_size)
        # rbf_kernel.lengthscale = torch.ones(latent_size) * 0.5
        # rbf_kernel.register_constraint("raw_lengthscale", gpytorch.constraints.Interval(0.01, 2.0))

        matern_15_kernel = gpytorch.kernels.MaternKernel(nu=1.5, ard_num_dims=latent_size)
        matern_15_kernel.lengthscale = torch.ones(latent_size) * 0.5
        matern_15_kernel.register_constraint("raw_lengthscale", gpytorch.constraints.Interval(0.01, 2.0))

        matern_25_kernel = gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=latent_size)
        matern_25_kernel.lengthscale = torch.ones(latent_size) * 0.5
        matern_25_kernel.register_constraint("raw_lengthscale", gpytorch.constraints.Interval(0.01, 2.0))
        self.covar_module = gpytorch.kernels.ScaleKernel(matern_15_kernel)+ gpytorch.kernels.ScaleKernel(matern_25_kernel)


    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def predict(self, x):
        self.eval()
        self.likelihood.eval()
        preds = self(x)
        return preds.mean, preds.stddev

    def fit(self,training_iterations=100, lr=0.1):

        self.train()
        self.likelihood.train()

        # init
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)

        # train
        for i in range(training_iterations):
            optimizer.zero_grad()
            output = self.forward(*self.train_inputs)
            loss = -mll(output, self.train_targets)
            loss.backward()
            optimizer.step()

        print(f"Iter {i}/{training_iterations} - Loss: {loss.item():.3f}")
        print("GP Fitting Complete")

class Surrogate(nn.Module):
    def __init__(self, dataset, normalizer, device = torch.device('cuda:0'), data_type=torch.bfloat16, epoch_mlp=100, lr_mlp=0.001,epoch_gp=100, lr_gp=0.1):
        super(Surrogate, self).__init__()
        self.latent_size = 20
        self.mlp = MLPEncoder(input_size=dataset.embeddings[0].shape[1], hidden_size=512, latent_size=self.latent_size,output_size=1, device=device).to(device, data_type)
        self.device = device
        self.data_type = data_type
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device, data_type)
        self.normalizer = None
        self.embeddings = None
        self.y = None
        self.gp = None
        self.update_gp(dataset,normalizer, epoch_mlp, lr_mlp, epoch_gp, lr_gp)
    def update_gp(self, dataset, normalizer, epoch_mlp=100, lr_mlp=0.001,epoch_gp=100, lr_gp=0.1):
        self.normalizer = normalizer

        self.mlp.train_mlp(dataset, normalizer=normalizer,epochs=epoch_mlp,learning_rate=lr_mlp)

        self.embeddings = torch.cat(dataset.embeddings, dim=0)
        self.y = normalizer.normalize_one(torch.tensor(dataset.scores))
        self.embeddings, self.y = self.embeddings.to(self.device), self.y.to(self.device)
        with torch.no_grad():
            _, reduced_embeddings = self.mlp(self.embeddings)
        self.gp = GPModel(reduced_embeddings, self.y, self.likelihood, self.device, self.latent_size).to(self.device)
        self.gp.fit(training_iterations=epoch_gp, lr=lr_gp)
    def forward(self,x):
        # reduce
        with torch.no_grad():
            _, reduced_embeddings = self.mlp.predict(x)
            return self.gp.predict(reduced_embeddings)
    def batch_predict(self,x,batch_size=300):
        mean = []
        std = []
        scores = [0 for _ in range(len(x))]
        temp_dataset = BayesianDataset(None, x, scores)
        data_loader = DataLoader(temp_dataset, batch_size=batch_size, shuffle=False)
        for batch in data_loader:
            emb, y = batch
            emb = emb.to(self.device)
            batch_mean, batch_std = self(emb)
            mean.extend(batch_mean.cpu().tolist())
            std.extend(batch_std.cpu().tolist())
        return torch.tensor(mean, dtype=torch.float32), torch.tensor(std, dtype=torch.float32)
