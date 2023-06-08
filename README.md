# Accounting For Informative Sampling When Learning to Forecast Treatment Outcomes Over Time </br><sub><sub>T. Vanderschueren*, A. Curth*, W. Verbeke & M. van der Schaar (ICML 2023)</sub></sub>
This repository provides the code for the paper *"Accounting For Informative Sampling When Learning to Forecast Treatment Outcomes Over Time"* (ICML 2023).

The structure of the code is as follows:
```
TESAR-CDE/
|_ data/
|_ experiments/
  |_ config.yml                     # Training config
|_ src/
  |_ models/
    |_ CDE_model.py                 # Code for TE-CDE and TESAR-CDE
  |_ utils/
    |_ cancer_simulation.py         # Generate the latent data
    |_ data_utils.py                # Preprocess observational data
    |_ losses.py                    # Loss functions 
    |_ process_irregular_data.py    # Informatively sample latent data to get observational data
    |_ training_tools.py            # Early stopping, dropout
main.py                             # Main executable to run experiment
trainer.py                          # Process data as tensors, fit model, make predictions
```

## Installation
The ```requirements.txt``` provides the necessary packages.
All code was written for ```python 3.7.9```.

Weights and Biases ([W&B](https://wandb.com)) is required to log the experiments. 

## Usage
The experiments can be run through the ```main.py```file using the following arguments:
```
python main.py
[--obs_coeff [informativeness coefficient]]
[--intensity_cov [number of covariates influencing the intensity, but not the outcome]]
[--intensity_cov_only [only include the intensity covariates]]
[--max_intensity [maximal intensity, i.e., observation probability]]
[--results_dir [path to directory to store results]]
[--model_name [final model name]]
[--load_dataset [boolean whether to load a saved version of the dataset from file]]
[--use_transformed [boolean whether to sample data]]
[--experiment [name of experiment yml]]
[--data_path [path to experiment data if loading one from a location]]
[--kappa [kappa parameter for the Hawkes process]]
[--iterations [number of iterations for the experiment]]
[--save_raw_datapath [path to save the raw dataset, so it can be reused to speed things up]]
[--save_transformed_datapath [path to save the transformed dataset, so it can be reused to speed things up]]
```
Example usage to train TESAR-CDE (Multitask) for informativeness $\gamma=6$:

```python main.py --obs_coeff=6 --intensity_cov_only=False --num_patients=200 --importance_weighting=True --multitask=False```

## Acknowledgements
Our code builds upon the code for TE-CDE ([Seedat et al. 2022](https://github.com/seedatnabeel/TE-CDE)). 

Seedat, N., Imrie, F., Bellot, A., Qian, Z., & van der Schaar, M. (2022, June). Continuous-Time Modeling of Counterfactual Outcomes Using Neural Controlled Differential Equations. In *International Conference on Machine Learning* (pp. 19497-19521). PMLR.

## Citing
Please cite our paper and/or code as follows:
```tex

@InProceedings{tesarcde2023,
  title = 	 {{Accounting For Informative Sampling When Learning to Forecast Treatment Outcomes Over Time}},
  author =       {Vanderschueren, Toon and Alicia, Curth and Verbeke, Wouter and van der Schaar, Mihaela},
  booktitle = 	 {Proceedings of the 40th International Conference on Machine Learning},
  year = 	 {2023},
  volume = 	 {202},
  series = 	 {Proceedings of Machine Learning Research},
  publisher =    {PMLR},
}
```
