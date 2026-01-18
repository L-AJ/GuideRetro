# GuideRetro ： Toward Synthesizability-Aware Multi-Step Retrosynthetic Planning
## 📝 Abstract 
Multi-step retrosynthetic planning aims to decompose target molecules into available starting materials by iteratively invoking single-step prediction models within a search algorithm. 
The success of retrosynthetic planning depends on the joint guidance of single-step reasoning and global search across steps. 
However, most existing frameworks make step-wise decisions based only on the current molecular state, without explicitly modeling synthesizability signals that reflect long-range reachability. 
In this work, we propose \textbf{GuideRetro}, a synthesizability-aware framework for multi-step retrosynthetic planning that integrates global synthesizability knowledge into step-wise retrosynthetic prediction. GuideRetro learns transferable knowledge from large-scale reaction networks by modeling the evolution of synthetic complexity along reaction pathways. 
During planning, a route-aware synthesis state modeling module combines the evolving retrosynthetic route with retrieved global signals to guide reactant generation at each step. 
Experiments on benchmark datasets show that GuideRetro achieves state-of-the-art performance. The integration of global synthesizability knowledge and route-aware modeling improves planning accuracy and search efficiency under realistic retrosynthetic settings. 
## 🛠️ Requirements
All the required packages can be installed by running `pip install -r requirements.txt`.
## 📂 Data PreparationData 
### 1. Download Steps

1.  **RetroBench & Zinc Stock**: Download **[RetroBench and zinc_stock_17_04_20.hdf5](https://github.com/SongtaoLiu0823/FusionRetro)** and place the files into `Data/`.
2.  **Building Blocks & Models**: Download **[retro_plan data](https://www.dropbox.com/scl/fi/cchn0wjz8j0dqxhr0qrom/retro_data.zip?rlkey=kqz60ec7vx7087vg1o63nucyo&e=1&dl=0)**. Unzip it and move the `Data/` and `retro_star/one_step_model/` folders to the project root directory (merge with existing folders).
3.  **USPTO-Full**: Download **[USPTO-Full](https://github.com/Hanjun-Dai/GLN)** and place the files into `Data/Train/for embedding/`.

### 2. Directory Structur
Organize the data structure as follows:
```text
Data/
├──Test/
│   ├──chembl_1000.pkl
│   └──gdb17_1000.pkl
│   └──routes_possible_test_hard.pkl
│   └──test_dataset.json
├── Train/
│   ├── for embedding
│       └──raw_test.csv
│       └──raw_train.csv
│       └──raw_val.csv
│   └── for model
│       └──train_canolize_dataset.json
│       └──valid_canolize_dataset.json
├── zinc_stock_17_04_20.hdf5
├── origin_dict.csv
```
### 3. Data preprocessing
```text
# Canolize RetroBench (eg. train_dataset.json --> train_canolize_dataset.json)
python Dataprocess/to_canolize.py --dataset train  
python Dataprocess/to_canolize.py --dataset valid
 
# Get training data and ensure that it does not contain test molecules
python Dataprocess/get_clear_emb_node.py
python Dataprocess/get_clear_train_data.py

```
## 🚀 Usage (运行)
### 1. Training
```text
# Global

# Single-step model train
python model_train.py --batch_size 32 --epochs 300
```
### 2. Evaluation 
```text
##For Exact Match Results
# Greedy dfs
python greedy_dfs.py --beam_size 5 --temperature 2.2
#Retro* seach
python retro_search.py --use_value --beam_size 5 ----temperature 2.2
#Retro*-0 seach
python retro_search.py --beam_size 5 --temperature 2.2

##For Success Rate Results
python retro_star/retro_plan.py --temperature 1.5
```

