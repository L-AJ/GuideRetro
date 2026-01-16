# GuideRetro ： Toward Synthesizability-Aware Multi-Step Retrosynthetic Planning
## 📝 Abstract 
Multi-step retrosynthetic planning aims to decompose target molecules into available starting materials by combining single-step predictions with search algorithms. The success of multi-step retrosynthesis depends on the joint guidance of single-step reasoning and global search across steps. While recent single-step models have achieved strong performance, most existing approaches still rely primarily on local chemical context and lack explicit global synthesizability signals to steer planning toward reachable starting materials. In this work, we propose GuideRetro, a synthesizability-aware framework for multi-step retrosynthetic planning that integrates global synthesizability knowledge into step-wise retrosynthetic prediction. GuideRetro learns transferable global synthesizability knowledge from large-scale reaction networks by modeling how synthetic difficulty evolves along reaction sequences, providing directional signals of synthetic accessibility. During planning, a route-aware state modeling module combines the evolving retrosynthetic route with retrieved global synthesizability signals to guide reactant generation at each step. Experiments on standard benchmarks show that GuideRetro consistently improves multi-step planning, achieving higher success rates and more efficient search under realistic planning settings.
## 🛠️ Requirements
All the required packages can be installed by running **pip install -r requirements.txt.**
## 📂 Data PreparationData 
### download 
Please download the [RetroBench and zinc_stock_17_04_20.hdf5](https://github.com/SongtaoLiu0823/FusionRetro)**  and put the file (`Data/`).  
Please download the retro_plan **[building block molecules, pretrained models](https://www.dropbox.com/scl/fi/cchn0wjz8j0dqxhr0qrom/retro_data.zip?rlkey=kqz60ec7vx7087vg1o63nucyo&e=1&dl=0)** and put all the folders (`Data/`, `retro_star/one_step_model/`) into the root directory.  
Please download the **[USPTO-Full](https://github.com/Hanjun-Dai/GLN)** and put the file(`Data/Train/for embedding`)  
#### Organize the data structure as follows:
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
### Data preprocessing
```text
# Canolize RetroBench (eg. train_dataset.json --> train_canolize_dataset.json)
python Dataprocess/to_canolize.py --dataset train  
python Dataprocess/to_canolize.py --dataset valid
 
# Remove the data containing the test molecules
python Dataprocess/get_clear_train_data.py 
```
## 🚀 Usage (运行)
### Training
```text
python model_train.py --batch_size 32 --epochs 300
```
### Evaluation 
  #### For Exact Match Results
```text
# Greedy dfs
python greedy_dfs.py --beam_size 5 --temperature 2.2

#Retro* seach
python retro_search.py --use_value --beam_size 5 ----temperature 2.2

#Retro*-0 seach
python retro_search.py --beam_size 5 --temperature 2.2
```
  #### For Success Rate Results
```text
python retro_star/retro_plan.py --temperature 1.5
```

