# GuideRetro ： Toward Synthesizability-Aware Multi-Step Retrosynthetic Planning
## 📝 Abstract 
Multi-step retrosynthetic planning aims to decompose target molecules into available starting materials by combining single-step predictions with search algorithms. The success of multi-step retrosynthesis depends on the joint guidance of single-step reasoning and global search across steps. While recent single-step models have achieved strong performance, most existing approaches still rely primarily on local chemical context and lack explicit global synthesizability signals to steer planning toward reachable starting materials. In this work, we propose GuideRetro, a synthesizability-aware framework for multi-step retrosynthetic planning that integrates global synthesizability knowledge into step-wise retrosynthetic prediction. GuideRetro learns transferable global synthesizability knowledge from large-scale reaction networks by modeling how synthetic difficulty evolves along reaction sequences, providing directional signals of synthetic accessibility. During planning, a route-aware state modeling module combines the evolving retrosynthetic route with retrieved global synthesizability signals to guide reactant generation at each step. Experiments on standard benchmarks show that GuideRetro consistently improves multi-step planning, achieving higher success rates and more efficient search under realistic planning settings.
## 🛠️ Requirements
All the required packages can be installed by running **pip install -r requirements.txt.**
## 📂 Data PreparationData 
### download 
Please download the starting material file **zinc_stock_17_04_20.hdf5** from https://www.dropbox.com/scl/fi/j3kh641irxtpbrnjnmoop/zinc_stock_17_04_20.hdf5?rlkey=zqbymj13skpdqlswu2uvji1sq&st=c1805gz0&dl=0  （it come from FusionRetro）
Organize the data structure as follows:
```text
data/
├──Test/
│   ├──chembl_1000.pkl
│   └──gdb17_1000.pkl
│   └──routes_possible_test_hard.pkl
│   └──test_dataset.json
├── Train/
│   ├── for embedding
│   └── for model
│       └──train_dataset.json
│       └──valid_dataset.json
├── zinc_stock_17_04_20.hdf5
├── origin_dict.csv
```
### Data preprocessing
python Dataprocess/to_canolize.py --dataset train  
python Dataprocess/to_canolize.py --dataset valid  
python Dataprocess/to_canolize.py --dataset test
python Dataprocess/get_clear_train_data.py 

## 🚀 Usage (运行)
### Training
To train the model on [Dataset Name], run:
### Evaluation / Testing
To evaluate the pre-trained model:
