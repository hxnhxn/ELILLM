import torch
import gpytorch
import matplotlib.pyplot as plt
import pandas as pd
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from Bayesian.bayesianLLM import BayesianLLM
from sklearn.preprocessing import StandardScaler

from Bayesian.BayesianDataset import BayesianDataset


# 设置设备
exp_path = 'results/crossdocked/0'
seed = 1
output_file_path = os.path.join(exp_path, str(seed) + '_result.csv')
mol_output_file_path = os.path.join(exp_path, str(seed) + '_mol_record.csv')

# Loading Llama-3.1
checkpoint = "/home/hxnbo/meta-llama/Llama-3.1-8B-Instruct/"
# checkpoint = "/data/hxn2022/DeepSeek-R1-Distill-Llama-8B/"
device_num = 0
device = f"cuda:{device_num}"  # for GPU usage or "cpu" for CPU usage
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype=torch.bfloat16).to(device)
# Init bayesianLLM
repairer_prompt = open("prompt/repairer_prompt.txt", "r").read()
explorer_prompt = open("prompt/explorer_prompt.txt", "r").read()
# init_score = pd.read_csv(os.path.join(exp_path, 'init_score.csv'))
init_score = pd.read_csv(os.path.join(exp_path, '1_result.csv'))
init_docking_score = init_score["Docking Score"].tolist()
init_smiles = init_score["Molecule"].tolist()
for i in range(len(init_docking_score)):
    if init_docking_score[i] < 0:
        init_docking_score[i] = -2
        print(f"{init_smiles[i]} exceeding the docking score threshold, setting to -2")

bayesianLLM = BayesianLLM(tokenizer, model, device, agent_mode="multi",
                          repairer_prompt=repairer_prompt, explorer_prompt=explorer_prompt)
tempLLM = BayesianLLM(tokenizer, model, device, agent_mode="multi",
                          repairer_prompt=repairer_prompt, explorer_prompt=explorer_prompt)
train_num = 80
bayesianLLM.init_dataset(init_smiles[:train_num], init_docking_score[:train_num])
bayesianLLM.init_surrogate()

tempLLM.init_dataset(init_smiles[train_num:], init_docking_score[train_num:])

gp = bayesianLLM.surrogate
normalizer = bayesianLLM.normalizer
test_dataset = tempLLM.dataset
embeddings_test = torch.cat(test_dataset.embeddings,dim=0).to(device)
y_test = torch.tensor(test_dataset.scores)
y_test = normalizer.normalize_one(y_test)

mean, std = gp(embeddings_test)
mean, std = mean.cpu(), std.cpu()
lower = mean - 2 * std
upper = mean + 2 * std


plt.figure(figsize=(10, 5))
plt.plot(y_test.cpu().numpy(), 'k*', label='True y (Test)')
plt.plot(mean.numpy(), 'b', label='Predicted Mean')
plt.fill_between(range(len(y_test)), lower.cpu(), upper.cpu(), alpha=0.3, label='Confidence')
plt.title("GP Prediction on Test Data")
plt.xlabel("Test Sample Index")
plt.ylabel("Target Value")
plt.legend()
plt.show()
