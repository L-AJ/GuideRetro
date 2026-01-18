import os
import json
import numpy as np
from collections import defaultdict
from collections import defaultdict
from Similar_search import SimilaritySearcher

chars = " ^#%()+-./0123456789=@ABCDEFGHIKLMNOPRSTVXYZ[\\]abcdefgilmnoprstuy$"
vocab_size = len(chars)

char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }


def get_chars():
    return chars

def get_vocab_size():
    return vocab_size

def get_char_to_ix():
    return char_to_ix

def get_ix_to_char():
    return ix_to_char

def get_dataset_all_feature(file_name, entity_file, features_path, cache_file):
    
    depth_products_list = defaultdict(list)
    depth_reactants_list = defaultdict(list)
    depth_product_ids_list = defaultdict(list)

    
    custom_feat_file = cache_file.replace('.npz', '_custom_features.npy')
    custom_map_file = cache_file.replace('.npz', '_custom_mapping.json')

   
    custom_mapping = {}
    custom_features_matrix = None

    # == check new smi ===
    if os.path.exists(custom_map_file) and os.path.exists(custom_feat_file):
        print(f"Loading existing custom mapping from {custom_map_file}")
        with open(custom_map_file, 'r', encoding='utf-8') as f:
            custom_mapping = json.load(f)
        
        print(f"Loading existing custom features from {custom_feat_file}")
        custom_features_matrix = np.load(custom_feat_file)
    else:
        print("No existing custom storage found. Will create new.")
        
        custom_features_matrix = np.empty((0, 0)) 

    
    print(f"Scanning {file_name} for required products...")
    required_products = set()
    
    with open(file_name, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
        for _, reaction_tree in dataset.items():
            retro_routes = reaction_tree['retro_routes']
            for retro_route in retro_routes:
                for reaction in retro_route:
                    product = reaction.split('>')[0].strip()
                    required_products.add(product)

    
    new_products = [p for p in required_products if p not in custom_mapping]
    print(f"Found {len(required_products)} unique products. {len(new_products)} are NEW.")

    # == query new smi ===
    if new_products:
        print("Initializing SimilaritySearcher to fetch features for new products...")
        
        searcher = SimilaritySearcher(
            entity_file=entity_file,
            feature_file=features_path,
            cache_file=cache_file,
            use_gpu=False  
        )

        print(f"Fetching features for {len(new_products)} new molecules...")

        #batch seach top5 mean pooling feat
        batch_results = searcher.batch_get_mean_pooling_result(new_products, top_k=5, verbose=True)
        new_feats_list = []
        valid_new_products = []
        
        current_idx = len(custom_features_matrix) if custom_features_matrix.ndim > 1 else 0
        for i, res in enumerate(batch_results):
            prod = new_products[i]
            if res is None:
                
                feat_dim = custom_features_matrix.shape[1] if custom_features_matrix.ndim > 1 else searcher.features.shape[1]
                feat = np.zeros(feat_dim, dtype=np.float32)
            else:
                feat = res['feature']

            
            new_feats_list.append(feat)
            
            # SMILES -> id
            custom_mapping[prod] = current_idx
            current_idx += 1

        
        new_feats_block = np.stack(new_feats_list).astype(np.float32)

        
        if custom_features_matrix.ndim > 1 and len(custom_features_matrix) > 0:
            custom_features_matrix = np.vstack([custom_features_matrix, new_feats_block])
        else:
            custom_features_matrix = new_feats_block

        
        print(f"Saving updated custom features to {custom_feat_file}...")
        np.save(custom_feat_file, custom_features_matrix)
        
        print(f"Saving updated mapping to {custom_map_file}...")
        with open(custom_map_file, 'w', encoding='utf-8') as f:
            json.dump(custom_mapping, f, ensure_ascii=False, indent=2)

    else:
        print("No new products found. Using existing cache.")

    
    print("Constructing final dataset with Custom IDs...")
    
    
    
    for _, reaction_tree in dataset.items():
        retro_routes = reaction_tree['retro_routes']
        depth = reaction_tree['depth']

        for retro_route in retro_routes:
            products, reactants, product_ids = [], [], []

            for reaction in retro_route:
                parts = reaction.split('>')
                product = parts[0].strip()
                reactant_str = parts[-1].strip()

                products.append(product)
                reactants.append(reactant_str)

                
                pid = custom_mapping.get(product)
                
                if pid is not None:
                    product_ids.append(str(pid))
                else:
                    
                    product_ids.append("-1")

            depth_products_list[depth].append(products)
            depth_reactants_list[depth].append(reactants)
            depth_product_ids_list[depth].append(product_ids)

    print(f"Done. Custom Feature Matrix Shape: {custom_features_matrix.shape}")
    
    return depth_products_list, depth_reactants_list, depth_product_ids_list

def convert_symbols_to_inputs(products_list, reactants_list, product_ids_list, max_depth, max_length):
    # products
    products_input = np.zeros((len(products_list), max_depth, max_length))
    products_input_mask = np.zeros((len(products_list), max_depth, max_length))

    # reactants
    reactants_input = np.zeros((len(products_list), max_depth, max_length))
    reactants_input_mask = np.zeros((len(products_list), max_depth, max_length))

    # for output
    label_input = np.zeros((len(products_list), max_depth, max_length))

    # memory
    memory_input_mask = np.zeros((len(products_list), max_depth))
    
    # 新增产物ID矩阵
    products_id_matrix = np.zeros((len(products_list), max_depth), dtype=np.int64)
    
    for index, (products, product_ids) in enumerate(zip(products_list, product_ids_list)):
        reactants = reactants_list[index]
        memory_input_mask[index, :len(products)] = 1
        
        for i, (product, product_id) in enumerate(zip(products, product_ids)):
            # 处理产物ID
            products_id_matrix[index, i] = int(product_id)
            
            # 原有的产物和反应物处理
            reactant = reactants[i]
            product = '^' + product + '$'
            reactant = '^' + reactant + '$'
        
            for j, symbol in enumerate(product):
                products_input[index, i, j] = char_to_ix[symbol]
            products_input_mask[index, i, :len(product)] = 1

            for j in range(len(reactant)-1):
                reactants_input[index, i, j] = char_to_ix[reactant[j]]
                label_input[index, i, j] = char_to_ix[reactant[j+1]]
            reactants_input_mask[index, i, :len(reactant)-1] = 1
            
    return (products_input, products_input_mask, reactants_input, 
            reactants_input_mask, memory_input_mask, label_input, 
            products_id_matrix)

 