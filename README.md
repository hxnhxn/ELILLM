# ELILLM

A framework for performing explicit BO exploration in LLM latent space.


## Environment Setup

We recommend using `conda` to manage the environment.

### Step 1: Create environment from file

```bash
conda env create -f environment.yaml
conda activate your_env_name
```
### Step 2: Download LLaMA 3.1 8B
Download the LLaMA 3.1 8B model manually from Meta's official release or Hugging Face (if available under their terms).
Change the path setting in main file(e.g., elillm_diff.py)
```python
# Loading Llama-3.1
  checkpoint = "yourpath/Llama-3.1-8B-Instruct/"
```
### Step3: Smina
Make sure smina is installed and accessible.

Then, set the path to the smina executable in tools/docking.py, e.g.:
```python
if output:
    launch_args = ['your_smina_path/smina', '-r', file_protein, '-l', file_output, '--autobox_ligand', file_lig_ref,
                   '--autobox_add', autobox_add, '--seed', seed, '--exhaustiveness', exhaustiveness, '-o', prefix + f'{name_protein}_result.pdb',
                   '--cpu', '30', '>>', smina_cmd_output]
else:
    launch_args = ['your_smina_path/smina', '-r', file_protein, '-l', file_output, '--autobox_ligand', file_lig_ref,
                   '--autobox_add', autobox_add, '--seed', seed, '--exhaustiveness', exhaustiveness, '--cpu','30',
                   '>>', smina_cmd_output]
```

## Data Preparation

We follow the data split and preprocessing protocol from [TamGen](https://github.com/microsoft/TamGen/tree/main/data), based on the CrossDocked2020 dataset.

We evaluate on the **test set only**. The training set is used solely to randomly sample 100 initial ligand molecules per target.

To reproduce our results, you can directly use the preprocessed data we provide:

- For **ELILLM-rand**, use data in: "results/rand/"
- For **ELILLM-diff**, use data in: "results/diff"
The data for ELILLM-diff is constructed by re-evaluating the publicly available experimental results from ALIDIFF ([https://github.com/MinkaiXu/AliDiff](https://github.com/MinkaiXu/AliDiff)).

> No additional preprocessing is required if using the above folders.

### Custom Initialization (Optional)

If you wish to select your own initial molecules:

1. First, obtain the **training and test splits** following TamGen's instructions.  
 *Note: we only provide the processed test set.*

2. Then, run the following script to select initial molecules:

```bash
python tools/select_init.py
```
Finally, compute docking scores for the selected molecules:
```bash
python init.py 
```
## Generation

To run molecule generation, simply call the corresponding Python script for each experimental setting.

For example:

```bash
python elillm_diff.py
```
or
```bash
python elillm_rand.py
```

## Test

To analyze the generated results, run:

```bash
python test_crossdocked.py
```

## Visualization

All figures in the paper can be reproduced using the Python scripts under the `draw/` directory.