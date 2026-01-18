import argparse
import os
import torch
import sys


parser = argparse.ArgumentParser()

# =================== transformer ================== #

parser.add_argument('--max_length', type=int, default=200, help='The max length of a molecule.')
parser.add_argument('--max_depth', type=int, default=14, help='The max depth of a synthesis route.')
parser.add_argument('--embedding_size', type=int, default=64, help='The size of embeddings')
parser.add_argument('--hidden_size', type=int, default=640, help='The size of hidden units')
parser.add_argument('--num_hidden_layers', type=int, default=3, help='Number of layers in encoder\'s module. Default 3.')
parser.add_argument('--num_attention_heads', type=int, default=10, help='Number of attention heads. Default 10.')
parser.add_argument('--intermediate_size', type=int, default=512, help='The size of hidden units of position-wise layer.')
parser.add_argument('--hidden_dropout_prob', type=float, default=0.1, help='Dropout rate (1 - keep probability).')
parser.add_argument('--temperature', type=float, default=1.5, help='Temperature for decoding. Default 1.5')
parser.add_argument('--beamsize', type=int, default=10, help='Beam size for decoding. Default 10')
parser.add_argument('--step_k', type=int, default=8, help='beam branching factor at each step. Default 8')
parser.add_argument('--maxreturn', type=int, default=10, help='Max number of complete beam to return for select. Default 10')
parser.add_argument('--alpha', type=float, default=1, help='Length penalty coefficient. Default 1')
parser.add_argument('--sim_feat_seach_topk', type=int, default=5, help='top k for sim feat searcher')

# =================== random seed ================== #
parser.add_argument('--seed', type=int, default=1234)

# ==================== dataset ===================== #
parser.add_argument('--test_routes',default='Data/Test/retro*_190.pkl')
parser.add_argument('--starting_molecules', default='FusionRetro/retro_star/dataset/origin_dict.csv')

# ================== value dataset ================= #
parser.add_argument('--value_root', default='dataset')
parser.add_argument('--value_train', default='train_mol_fp_value_step')
parser.add_argument('--value_val', default='val_mol_fp_value_step')

# ================== one-step model ================ #
parser.add_argument('--mlp_model_dump',
                    default='one_step_model/saved_rollout_state_1_2048.ckpt')
parser.add_argument('--mlp_templates',
                    default='one_step_model/template_rules_1.dat')

# ===================== all algs =================== #
parser.add_argument('--iterations', type=int, default=500)
parser.add_argument('--expansion_topk', type=int, default=50)
parser.add_argument('--viz', action='store_true')
parser.add_argument('--viz_dir', default='viz')

# ===================== model ====================== #
parser.add_argument('--fp_dim', type=int, default=2048)
parser.add_argument('--n_layers', type=int, default=1)
parser.add_argument('--latent_dim', type=int, default=128)

# ==================== training ==================== #
parser.add_argument('--n_epochs', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--save_epoch_int', type=int, default=1)
parser.add_argument('--save_folder', default='FusionRetro/retro_star/saved_models')

# ==================== evaluation =================== #
parser.add_argument('--use_value_fn', action='store_true')
parser.add_argument('--value_model', default='best_epoch_final_4.pt')
parser.add_argument('--result_folder', default='FusionRetro/retro_star/results')

args = parser.parse_args()

# # setup device
# os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
