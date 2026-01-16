import pickle
import pandas as pd
import logging

from retro_star.alg import molstar



def prepare_starting_molecules(filename):
    logging.info('Loading starting molecules from %s' % filename)

    if filename[-3:] == 'csv':
        starting_mols = set(list(pd.read_csv(filename)['mol']))
    else:
        assert filename[-3:] == 'pkl'
        with open(filename, 'rb') as f:
            starting_mols = pickle.load(f)

    logging.info('%d starting molecules loaded' % len(starting_mols))
    return starting_mols

# def prepare_mlp(templates, model_dump):
#     logging.info('Templates: %s' % templates)
#     logging.info('Loading trained mlp model from %s' % model_dump)
#     one_step = MLPModel(model_dump, templates, device=-1)
#     return one_step

###addition


def prepare_molstar_planner(value_fn, starting_mols,
                            iterations, viz=False, viz_dir=None, searcher=None):
    # expansion_handle = lambda x: one_step.run(x, topk=expansion_topk)

    plan_handle = lambda x, y=0: molstar(
        target_mol=x,
        target_mol_id=y,
        starting_mols=starting_mols,
        # expand_fn=expansion_handle,
        value_fn=value_fn,
        iterations=iterations,
        viz=viz,
        viz_dir=viz_dir,
        searcher=searcher
    )
    return plan_handle
