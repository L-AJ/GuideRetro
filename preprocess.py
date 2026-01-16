import numpy as np
import json
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


def get_dataset(file_name):
    # file_name = "train_canolize_dataset.json"
    product_id_file = "/root/AKG/Data/entities_replaced.txt"
    depth_products_list = defaultdict(list)
    depth_reactants_list = defaultdict(list)
    depth_product_ids_list = defaultdict(list)
    
    
    product_to_id = {}
    with open(product_id_file, 'r') as f:
        for line in f:
            pid, product = line.strip().split('\t')
         
            product_to_id[product] = pid
    print(f"总共读取了 {len(product_to_id)} 个产物-ID映射")

    total_products = 0
    missing_count = 0
    with open(file_name, 'r') as f:
        dataset = json.load(f)
        for _, reaction_tree in dataset.items():
            retro_routes = reaction_tree['retro_routes']
            depth = reaction_tree['depth']
            for retro_route in retro_routes:
                products, reactants = [], []
                product_ids = []
                
                # 获取第一个反应的产物ID
                first_product = retro_route[0].split('>')[0]
                route_product_id = product_to_id.get(first_product, '-1')
                
                for reaction in retro_route:
                    product = reaction.split('>')[0]
                    products.append(product)
                    reactants.append(reaction.split('>')[-1])
                    
                    total_products += 1
                    # 使用路径第一个反应的产物ID
                    product_ids.append(route_product_id)
                    if route_product_id == '-1':
                        missing_count += 1
                
                depth_products_list[depth].append(products)
                depth_reactants_list[depth].append(reactants)
                depth_product_ids_list[depth].append(product_ids)
            
    print(f"总共读取了 {total_products} 个产物")
    print(f"总共缺失 {missing_count} 个产物ID")
    print(f"ID匹配率: {((total_products - missing_count) / total_products * 100):.2f}%")

    return depth_products_list, depth_reactants_list, depth_product_ids_list



from collections import defaultdict
import json
import os
from pathlib import Path
import os

def get_dataset_all_feature(file_name, entity_file, features_path, cache_file):
    
    depth_products_list = defaultdict(list)
    depth_reactants_list = defaultdict(list)
    depth_product_ids_list = defaultdict(list)

    # ==================== 永久缓存路径 ====================
    mapping_cache = cache_file.replace('.npz', '_product_to_id.json')
    similar_cache = cache_file.replace('.npz', '_missing_similar.json')

    # ==================== 1. 加载精确映射 ====================
    product_to_id = {}
    if os.path.exists(mapping_cache):
        print(f"Loading exact product→id mapping from cache: {mapping_cache}")
        with open(mapping_cache, 'r', encoding='utf-8') as f:
            product_to_id = json.load(f)
    else:
        print(f"Building exact product→id mapping from {entity_file}")
        product_to_id = {}
        with open(entity_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if '\t' not in line:
                    continue
                pid, smiles = line.split('\t', 1)
                product_to_id[smiles] = pid
        with open(mapping_cache, 'w', encoding='utf-8') as f:
            json.dump(product_to_id, f, ensure_ascii=False, indent=2)
        print(f"Exact mapping saved to {mapping_cache}")

    print(f"Loaded {len(product_to_id):,} exact product→id mappings")

    # ==================== 2. 加载相似搜索缓存 ====================
    missing_to_similar = {}
    if os.path.exists(similar_cache):
        print(f"Loading similar search cache: {similar_cache}")
        with open(similar_cache, 'r', encoding='utf-8') as f:
            missing_to_similar = json.load(f)
    else:
        print(f"No similar cache found, will build it at {similar_cache}")

    # 初始化 searcher
    print("Initializing SimilaritySearcher...")
    searcher = SimilaritySearcher(
        entity_file=entity_file,
        feature_file=features_path,
        cache_file=cache_file,
        use_gpu=False  # 如有 GPU 可设 True，批量查询收益更大
    )

    # ==================== 3. 第一遍扫描：收集所有缺失的 product SMILES ====================
    print("First pass: collecting all missing products...")
    all_missing_products = set()   # 用 set 去重，避免重复查询
    total_products = 0

    with open(file_name, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

        for _, reaction_tree in dataset.items():
            retro_routes = reaction_tree['retro_routes']

            for retro_route in retro_routes:
                for reaction in retro_route:
                    product = reaction.split('>')[0].strip()
                    total_products += 1

                    if product not in product_to_id and product not in missing_to_similar:
                        all_missing_products.add(product)

    print(f"Total products: {total_products:,}")
    print(f"Need to search for {len(all_missing_products):,} unique missing molecules")

    # ==================== 4. 批量相似搜索（核心优化） ====================
    new_similar_results = {}  # product -> best_id

    if all_missing_products:
        print(f"Performing batch similarity search for {len(all_missing_products)} molecules...")
        missing_list = list(all_missing_products)

        # 批量查询：一次返回所有分子的 Top-1 结果
        # 注意：这里我们需要扩展 searcher 支持 batch query
        # 如果你还没实现，我在下面提供一个简单高效的 batch_query 方法
        batch_results = searcher.batch_query(missing_list, top_k=1, verbose=True)

        for product, results in zip(missing_list, batch_results):
            if results and len(results) > 0:
                best = results[0]
                sim_id = str(best['index'])
                similarity = best['similarity']

                # 可选：设置最低相似度阈值（如 0.4）
                if similarity >= 0:
                    new_similar_results[product] = sim_id
                    print(f"  [Batch] {product[:60]}... → ID {sim_id} (sim={similarity:.3f})")
                else:
                    new_similar_results[product] = "-1"
            else:
                new_similar_results[product] = "-1"

        # 更新缓存
        missing_to_similar.update(new_similar_results)
        with open(similar_cache, 'w', encoding='utf-8') as f:
            json.dump(missing_to_similar, f, ensure_ascii=False, indent=2)
        print(f"Batch search completed. Updated cache with {len(new_similar_results)} entries.")

    # ==================== 5. 第二遍：正式构建数据集（使用完整映射） ====================
    print("Second pass: building final dataset...")
    missing_count = 0
    sim_count = 0

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

                
                # 优先精确匹配 → 相似匹配 → -1
                pid = product_to_id.get(product)
                if pid is not None:
                    product_ids.append(pid)
                else:
                    missing_count += 1
                    sim_id = missing_to_similar.get(product, "-1")
                    product_ids.append(sim_id)
                    if sim_id != "-1":
                        sim_count += 1

            depth_products_list[depth].append(products)
            depth_reactants_list[depth].append(reactants)
            depth_product_ids_list[depth].append(product_ids)

    # ==================== 统计 ====================
    effective_count = total_products - missing_count + sim_count
    print(f"Total products: {total_products:,}")
    print(f"Missing in exact dict: {missing_count:,}")
    print(f"Recovered by similarity: {sim_count:,}")
    print(f"Final ID coverage rate: {effective_count / total_products:.4%}")

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

 