
import torch

ensemble_model_path = 'best_painn_ensemble_model'

n_models = 3

for i in range(n_models):
    ensemble_model = torch.load(ensemble_model_path, 'cpu')
    model = ensemble_model
    model.postprocessors[1]._property = 'energy'
    model.model_outputs = ['energy', 'forces']

    model.output_modules[0+2*i].output_key = 'energy'
    model.output_modules[1+2*i].energy_key = 'energy'
    model.output_modules[1+2*i].force_key = 'forces'
    model.output_modules[1+2*i].model_outputs = ['forces']

    extra_model_indices = [idx for idx in range(2*n_models) if idx not in [0+2*i, 1+2*i]]
    for idx in sorted(extra_model_indices, reverse=True):
        del model.output_modules[idx]

    torch.save(model, f'best_painn_model_{i}')
