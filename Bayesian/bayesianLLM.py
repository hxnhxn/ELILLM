import numpy as np
from transformers import pipeline
from transformers.pipelines.text_generation import Chat
import torch
import time
import random
from Bayesian.surrogate import Surrogate
from Bayesian.BayesianDataset import BayesianDataset
from Bayesian.acquisition_function import AcquisitionFunction
from Bayesian.normalizer import Normalizer



def divide_smiles_from_prompt(prompt):
    begin = prompt.find('"')+1
    end = prompt.find('"', begin)
    return [begin, end]


def divide_smiles_from_tokens(prompt, token_list, tokenizer):
    string_list = []
    char_index_to_token = []
    for i in range(token_list.shape[1]):
        string_list.append(tokenizer.decode(token_list[0][i]))
        for j in range(len(string_list[i])):
            char_index_to_token.append(i)
    smiles_index_list = divide_smiles_from_prompt(prompt)
    for i in range(len(smiles_index_list)):
        smiles_index_list[i] = char_index_to_token[smiles_index_list[i]]
    return smiles_index_list


def aggregate_embeddings(embeddings, max_length=80):
    step = 1.0 / max_length
    weight = []
    embeddings = embeddings.clone().squeeze()
    length = embeddings.shape[0]
    for i in range(length):
        weight.append(step*i)
    weight = torch.tensor(weight, device=embeddings.device, dtype=embeddings.dtype).unsqueeze(1)
    reversed_weight = 1 - weight
    # reversed_weight = torch.tensor(reversed_weight, device=embeddings.device, dtype=embeddings.dtype)
    final_embeddings = torch.cat((torch.sum(reversed_weight*embeddings, dim=0),torch.sum(weight*embeddings, dim=0)), dim=-1)/length
    return final_embeddings.unsqueeze(0)



class BayesianLLM:
    def __init__(self,tokenizer,model,device,repairer_prompt=None,epoch_mlp=150, lr_mlp=0.001, epoch_gp=100, lr_gp=0.1):
        self.max_new_tokens = 80
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.pipe = pipeline(
            "text-generation",
            model=model,
            torch_dtype=torch.bfloat16,
            tokenizer=tokenizer,
            device=0,
        )
        self.surrogate = None

        self.dataset = BayesianDataset()
        self.acquisition_function = None
        self.normalizer = Normalizer()
        self.repairer_prompt = repairer_prompt
        self.epoch_mlp = epoch_mlp
        self.lr_mlp = lr_mlp
        self.epoch_gp = epoch_gp
        self.lr_gp = lr_gp

    def update_normalizer(self):
        scores = self.dataset.scores
        if scores is None:
            return
        mean = np.mean(scores)
        std = np.std(scores)
        self.normalizer.update(mean, std)

    def init_dataset(self, smiles, scores):
        self.normalizer.update(-5,5)
        for i in range(len(smiles)):
            if scores[i] is None:
                continue
            current_score = scores[i]
            if current_score >= 10:
                mean, std = self.normalizer.mean, self.normalizer.std
                if current_score == 1000:
                    current_score = 0
                if current_score > mean + 3 * std:
                    current_score = mean + 3 * std
            current_score = float(current_score)
            smiles_token = self.tokenizer.encode(smiles[i],return_tensors="pt").to(self.device)
            smiles_embedding = self.model.base_model.embed_tokens(smiles_token)
            smiles_embedding = smiles_embedding.cpu().detach().clone()
            # smile_avg_embedding = torch.sum(smiles_embedding, dim=1) / smiles_embedding.size(1)
            smile_avg_embedding = aggregate_embeddings(smiles_embedding, max_length=self.max_new_tokens)
            self.dataset.add_item(smiles[i],smile_avg_embedding, current_score)
        self.update_normalizer()

    def init_surrogate(self):
        self.surrogate = Surrogate(self.dataset, self.normalizer,self.device,epoch_mlp=self.epoch_mlp, lr_mlp=self.lr_mlp,
                                   epoch_gp=self.epoch_gp, lr_gp=self.lr_gp)
        self.acquisition_function = AcquisitionFunction(self.surrogate, self.device)

    def update(self, molecules, scores):
        # cnt = 0
        try:
            assert len(molecules) == len(scores)
            for i in range(len(molecules)):
                if scores[i] is None:
                    continue
                # cnt += 1
                current_score = scores[i]
                if current_score >= 10:
                    mean, std = self.normalizer.mean, self.normalizer.std
                    if current_score == 1000:
                        current_score = 0
                    if current_score > mean + 3 * std:
                        current_score = mean + 3 * std
                current_score = float(current_score)
                smiles = molecules[i]
                smiles_token = self.tokenizer.encode(smiles,return_tensors="pt").to(self.device)
                smiles_embedding = self.model.base_model.embed_tokens(smiles_token)
                smiles_embedding = smiles_embedding.cpu().detach().clone()
                # smile_avg_embedding = torch.sum(smiles_embedding, dim=1) / smiles_embedding.size(1)
                smile_avg_embedding = aggregate_embeddings(smiles_embedding, max_length=self.max_new_tokens)
                self.dataset.add_item(smiles, smile_avg_embedding, current_score)
            self.update_normalizer()
        except AssertionError:
            print("Length of molecules and scores do not match. Failed update.")


    def sample(self, num_perturbation=5):
        max_new_tokens = self.max_new_tokens
        param_dict = {'max_new_tokens': self.max_new_tokens}
        chat = Chat([{"role": "system", "content": self.repairer_prompt},
                     {"role": "assistant", "content": 'OK.'},
            {"role": "user", "content": '"candidate embedding"'}])
        model_inputs = self.pipe.preprocess(chat, **param_dict)
        inputs = model_inputs['input_ids'].to(self.device)
        attention_mask = model_inputs['attention_mask'].to(self.device)
        st_time = time.time()
        smiles_index_list = divide_smiles_from_tokens(self.tokenizer.decode(inputs[0]), inputs, self.tokenizer)
        begin, end = smiles_index_list


        smiles_embed_list = []
        smiles_avg_embed_list = []
        # self.dataset.sort()
        # good_smiles = self.dataset.smiles[:int(len(self.dataset.smiles)/2)]
        # bad_smiles = self.dataset.smiles[int(len(self.dataset.smiles)/2):]
        # sampled_bad_smiles = random.sample(bad_smiles, max(min(10,len(bad_smiles)),int(0.2*len(bad_smiles))))
        # candidate_smiles = good_smiles + sampled_bad_smiles
        candidate_smiles = self.dataset.smiles
        for smiles in candidate_smiles:
            smiles_token = self.tokenizer.encode(smiles, return_tensors="pt").to(self.device)
            smiles_embed = self.model.base_model.embed_tokens(smiles_token).detach().clone()
            pass
            for i in range(100):
                perturbation_smiles_embed = torch.normal(1, 0.4, size=smiles_embed.shape, device=smiles_embed.device,
                                                         dtype=smiles_embed.dtype) * smiles_embed
                smiles_embed_list.append(perturbation_smiles_embed.clone().cpu())
                # smiles_avg_embed_list.append(
                #     torch.sum(perturbation_smiles_embed, dim=1) / perturbation_smiles_embed.size(1))
                avg_emb = aggregate_embeddings(perturbation_smiles_embed, max_length=max_new_tokens)
                smiles_avg_embed_list.append(avg_emb.cpu())


        # Llama 3.1 8B [1, lines, 4096]
        embedding = self.model.base_model.embed_tokens(inputs)
        # maxy = self.dataset.maxy()
        # self.acquisition_function.update(maxy)
        predict_st_time = time.time()
        # scores_pre = self.acquisition_function.EI(smiles_avg_embed_list).squeeze()
        scores_pre, mu, std = self.acquisition_function.LCB(smiles_avg_embed_list, len(self.dataset)+1)
        scores_pre, mu, std = scores_pre.squeeze(), mu.squeeze(), std.squeeze()
        # print(f"predict_time: {time.time() - predict_st_time}s")
        # best_id = torch.argmax(scores_pre, dim=0)
        top_value, top_id = torch.topk(scores_pre, k=num_perturbation, largest=False)
        embedding = embedding.to(self.device)
        perturbation_embeddings = []
        for i in range(num_perturbation):
            perturbation_embedding = torch.cat([embedding[:,:begin], smiles_embed_list[top_id[i].item()].to(self.device),embedding[:,end:]], dim=1)
            perturbation_embeddings.append(perturbation_embedding.clone())
        # print(f"perturbation time: {time.time() - st_time}s")
        # perturbation_embedding[0, begin:end] = smiles_embed_list[best_id.item()].to(self.device)
        attention_masks = [torch.ones(perturbation_embeddings[i].shape[:2]).to(self.device) for i in range(num_perturbation)]

        del smiles_avg_embed_list, smiles_embed_list
        torch.cuda.empty_cache()
        perturbation_smileses = []
        for i in range(num_perturbation):
            with torch.no_grad():
                outputs = self.model.generate(inputs_embeds=perturbation_embeddings[i], attention_mask=attention_masks[i],
                                         max_new_tokens=max_new_tokens, output_hidden_states=True,temperature=0.4,
                                     return_dict_in_generate=True, pad_token_id=self.tokenizer.eos_token_id)
            perturbation_smiles = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True,
                                       clean_up_tokenization_spaces=True)
            perturbation_smileses.append(perturbation_smiles)
            print(f"perturbation_smiles: {perturbation_smiles} id:{top_id[i]} surrogate mean: {mu[top_id[i].item()].item()} std: {std[top_id[i].item()].item()}")

            # new_smileses = []
            # chat = Chat([{"role": "system", "content": self.explorer_prompt},
            #              {"role": "assistant", "content": 'OK.'},
            #              {"role": "user", "content": ' '.join([f'[{x}]' for x in perturbation_smileses])}])
            # model_inputs = self.pipe.preprocess(chat, **param_dict)
            #
            # inputs = model_inputs['input_ids'].to(self.device)
            # final_attention_mask = model_inputs['attention_mask'].to(self.device)
            # num_explore = 1
            # for _ in range(num_explore):
            #     final_outputs = self.model.generate(inputs, attention_mask=final_attention_mask,max_new_tokens=max_new_tokens, output_hidden_states=True,temperature=0.8,
            #                              return_dict_in_generate=True,pad_token_id=self.tokenizer.eos_token_id)
            #     new_smiles = self.tokenizer.decode(final_outputs.sequences[0][inputs.shape[1]:], skip_special_tokens=True,
            #                                        clean_up_tokenization_spaces=True)
            #     new_smileses.append(new_smiles)
            #     print(f"explore_smiles: {new_smiles}")
        return perturbation_smileses


    def train_surrogate(self):
        self.surrogate.update_gp(self.dataset, self.normalizer, self.epoch_mlp, self.lr_mlp, self.epoch_gp, self.lr_gp)

    # def train_judger(self, epochs=150, batch_size=64, learning_rate=0.001, weight_decay=0.001):
    #     """
    #     surrogate: 神经网络模型
    #     dataset: 训练数据集，应该是 Program_embedding_dataset 类型的实例
    #     epochs: 训练的轮数
    #     batch_size: 每个批次的数据量
    #     learning_rate: 学习率
    #     """
    #     dataset = self.judge_dataset
    #     judger = self.judger
    #     device = self.device
    #     # 使用 DataLoader 加载数据
    #     with torch.no_grad():
    #         judger.reinitialize_weights_he()
    #
    #     # 定义损失函数和 Adam 优化器
    #     criterion = nn.CrossEntropyLoss()  # 使用均方误差损失
    #     optimizer = optim.Adam(judger.parameters(), lr=learning_rate, weight_decay=weight_decay)  # 使用Adam优化器
    #     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    #     # 开始训练循环
    #     for epoch in range(epochs):
    #         running_loss = 0.0
    #
    #         for embeddings, labels in dataloader:
    #             optimizer.zero_grad()
    #             embeddings = embeddings.to(device)
    #             # 前向传播
    #             outputs = judger(embeddings).squeeze()
    #             labels = labels.to(device,torch.long).squeeze()
    #             # 计算损失
    #             loss = criterion(outputs, labels)
    #
    #             # 反向传播
    #             loss.backward()
    #             # 更新权重
    #             optimizer.step()
    #             # with torch.no_grad():
    #             running_loss += loss.item()
    #             pass
    #             # 累计损失
    #
    #         # 打印每个 epoch 的平均损失
    #     print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(dataloader):.4f}")
    #     print("判别模型训练完成")
    #     return running_loss / len(dataloader)

