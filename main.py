import argparse
import os
import traceback
import wandb
import numpy as np
import torch

from scipy.stats import sem
from src.utils.cancer_simulation import get_cancer_sim_data
from src.utils.data_utils import process_data, read_from_file, write_to_file
from src.utils.process_irregular_data import *
from trainer import trainer

os.environ["WANDB_API_KEY"] = ""        # Fill in wandb API key
wandb_entity = ""                       # Fill in wandb username


def init_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chemo_coeff", default=0, type=int)          # Not used
    parser.add_argument("--radio_coeff", default=0, type=int)          # Not used
    parser.add_argument("--obs_coeff", default=6, type=float)
    parser.add_argument("--intensity_cov", default=0, type=int)        # 10 for final experiment (Figure 6)
    parser.add_argument("--intensity_cov_only", default=False)         # True for final experiment (Figure 6)
    parser.add_argument("--max_intensity", default=1, type=float)      # Max intensity (1 / S_\lambda in paper)
    parser.add_argument("--num_patients", default=200, type=int)
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--model_name", default="te_cde_test")
    parser.add_argument("--load_dataset", default=False)               # True to skip data generation
    parser.add_argument("--use_transformed", default=False)            # True to skip data transformation
    parser.add_argument("--experiment", type=str, default="default")   # Add other experiments as yml files
    parser.add_argument("--data_path", type=str, default="data/transformed/new_cancer_sim_0_0_kappa_10.p")
    parser.add_argument("--importance_weighting", default=False)       # Use importance weighting (TESAR-CDE)
    parser.add_argument("--ground_truth_iw", default=False)            # Use "ground truth" importance weights
    parser.add_argument("--multitask", default=False)                  # Use multi-task setup, if not two-step is used
    parser.add_argument("--kappa", type=int, default=5)
    parser.add_argument("--max_samples", type=int, default=1)          # Not used
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--save_raw_datapath", type=str, default="data/raw")
    parser.add_argument("--save_transformed_datapath", type=str, default="data/transformed")
    return parser.parse_args()


if __name__ == "__main__":

    # Setup project:
    args = init_arg()

    load_dataset = str(args.load_dataset) == "True"
    use_transformed = str(args.use_transformed) == "True"

    logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)
    logging.getLogger().setLevel(logging.INFO)

    strategy = "all"

    logging.info("WANDB init...")
    # start a new run
    run = wandb.init(
        project="InformativeObservation",
        entity=wandb_entity,
        config=f"./experiments/{args.experiment}.yml",
    )

    config = wandb.config
    wandb.config.update(args)

    rmses = []
    rmses1 = []
    rmses2 = []
    rmses3 = []
    rmses4 = []
    rmses5 = []
    mses_int = []

    for i in range(args.iterations):

        print('Starting iteration ', i)

        # Set random seed
        random_seed = i
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

        # Generate or load raw data -- latent paths of X_t, A_t, Y_t, lambda_t
        if not(load_dataset) or args.data_path == None:
            logging.info("Generating dataset")
            pickle_map = get_cancer_sim_data(
                chemo_coeff=args.chemo_coeff,
                radio_coeff=args.radio_coeff,
                obs_coeff=args.obs_coeff,
                intensity_cov=args.intensity_cov,
                intensity_cov_only=bool(args.intensity_cov_only),
                max_intensity=args.max_intensity,
                num_patients=args.num_patients,
                b_load=True,
                b_save=False,
                model_root=args.results_dir,
            )
        else:
            logging.info(f"Loading dataset from: {args.data_path}")
            pickle_map = read_from_file(args.data_path)

        wandb.log({"chemo_coeff": args.chemo_coeff})
        wandb.log({"radio_coeff": args.radio_coeff})

        kappa = int(args.kappa)
        wandb.log({"kappa": kappa})

        importance_weighting = str(args.importance_weighting) == "True"
        wandb.log({"importance_weighting": importance_weighting})

        ground_truth_iw = str(args.ground_truth_iw) == "True"
        wandb.log({"ground_truth_iw": ground_truth_iw})

        multitask = str(args.multitask) == "True"
        wandb.log({"multitask": multitask})

        max_samples = int(args.max_samples)
        wandb.log({"max_samples": max_samples})

        wandb.log({"strategy": strategy})

        coeff = int(args.radio_coeff)

        if args.save_raw_datapath != None:
            logging.info(f"Writing raw data to {args.save_raw_datapath}")
            write_to_file(
                pickle_map,
                f"{args.save_raw_datapath}/new_cancer_sim_{coeff}_{coeff}.p",
            )

        # Transformed data (or load) -- apply observation process
        if bool(use_transformed) == False:
            logging.info("Transforming dataset")
            pickle_map = transform_data(
                data=pickle_map,
                interpolate=False,
                strategy=strategy,
                sample_prop=config["sample_proportion"],
                kappa=kappa,
                max_samples=max_samples,
            )
        else:
            transformed_datapath = f"data/transformed/new_cancer_sim_{coeff}_{coeff}_kappa_{kappa}.p"
            logging.info(f"Loading transformed data from {transformed_datapath}")
            pickle_map = read_from_file(transformed_datapath)

        if args.save_transformed_datapath != None:
            logging.info(f"Writing transformed data to {args.save_transformed_datapath}")
            write_to_file(
                pickle_map,
                f"{args.save_transformed_datapath}/new_cancer_sim_{coeff}_{coeff}_kappa_{kappa}.p",
            )

        # Process data (train-val-test + normalisation)
        logging.info("Processing dataset")
        training_processed, validation_processed, test_f_processed, test_cf_processed = process_data(pickle_map)

        use_time = config["use_time"]

        # Train model
        logging.info("Training model...")
        cde_trainer = trainer(
            run=run,
            hidden_channels_x=config["hidden_channels_x"],
            hidden_channels_enc=config["hidden_channels_enc"],
            hidden_layers_enc=config["hidden_layers_enc"],
            hidden_channels_dec=config["hidden_channels_dec"],
            hidden_layers_dec=config["hidden_layers_dec"],
            hidden_channels_map=config["hidden_channels_map"],
            hidden_layers_map=config["hidden_layers_map"],
            alpha=config["alpha"],
            cutoff=config["cutoff"],
            output_channels=config["output_channels"],
            sample_proportion=config["sample_proportion"],
            use_time=config["use_time"],
            window=config["window"],
            importance_weighting=importance_weighting,
            ground_truth_iw=ground_truth_iw,
            multitask=multitask,
        )

        wandb.log({"proportion": config["sample_proportion"]})
        cde_trainer.fit(
            train_data=training_processed,
            validation_data=validation_processed,
            epochs=config["epochs"],
            patience=config["patience"],
            batch_size=config["batch_size"],
        )

        # Test model (factual scenarios + counterfactual scenarios):
        logging.info("Testing model (factual)...")
        rmse_factual, rmse1_factual, rmse2_factual, rmse3_factual, rmse4_factual, rmse5_factual, mse_intensities = \
            cde_trainer.predict(test_data=test_f_processed, label="Factual")

        logging.info("Testing model (counterfactual)...")
        rmse_counterfactual, rmse1_counterfactual, rmse2_counterfactual, rmse3_counterfactual, rmse4_counterfactual, rmse5_counterfactual, mse_intensities_counterfactual = cde_trainer.predict(
            test_data=test_cf_processed, label="Counterfactual")

        # Log average RMSE
        rmse_total = (rmse_factual + rmse_counterfactual) / 2
        rmse1_total = (rmse1_factual + rmse1_counterfactual) / 2
        rmse2_total = (rmse2_factual + rmse2_counterfactual) / 2
        rmse3_total = (rmse3_factual + rmse3_counterfactual) / 2
        rmse4_total = (rmse4_factual + rmse4_counterfactual) / 2
        rmse5_total = (rmse5_factual + rmse5_counterfactual) / 2
        mse_int_total = (mse_intensities + mse_intensities_counterfactual) / 2
        run.log({f"RMSE Outcome Loss Test Total": rmse_total})
        run.log({f"RMSE Outcome Loss Test Total 1": rmse1_total})
        run.log({f"RMSE Outcome Loss Test Total 2": rmse2_total})
        run.log({f"RMSE Outcome Loss Test Total 3": rmse3_total})
        run.log({f"RMSE Outcome Loss Test Total 4": rmse4_total})
        run.log({f"RMSE Outcome Loss Test Total 5": rmse5_total})
        run.log({f"MSE Intensity Loss Test Total": mse_int_total})
        print({f"RMSE Outcome Loss Test Total": rmse_total})
        print({f"RMSE Outcome Loss Test Total 1": rmse1_total})
        print({f"RMSE Outcome Loss Test Total 2": rmse2_total})
        print({f"RMSE Outcome Loss Test Total 3": rmse3_total})
        print({f"RMSE Outcome Loss Test Total 4": rmse4_total})
        print({f"RMSE Outcome Loss Test Total 5": rmse5_total})
        print({f"MSE Intensity Loss Test Total": mse_int_total})
        rmses.append(rmse_total)
        rmses1.append(rmse1_total)
        rmses2.append(rmse2_total)
        rmses3.append(rmse3_total)
        rmses4.append(rmse4_total)
        rmses5.append(rmse5_total)
        mses_int.append(mse_int_total)

    run.log({f"RMSE Outcome Average": np.mean(rmses)})
    run.log({f"RMSE Outcome Std": sem(rmses)})
    run.log({f"RMSE Outcome Avg 1": np.mean(rmses1)})
    run.log({f"RMSE Outcome Std 1": sem(rmses1)})
    run.log({f"RMSE Outcome Avg 2": np.mean(rmses2)})
    run.log({f"RMSE Outcome Std 2": sem(rmses2)})
    run.log({f"RMSE Outcome Avg 3": np.mean(rmses3)})
    run.log({f"RMSE Outcome Std 3": sem(rmses3)})
    run.log({f"RMSE Outcome Avg 4": np.mean(rmses4)})
    run.log({f"RMSE Outcome Std 4": sem(rmses4)})
    run.log({f"RMSE Outcome Avg 5": np.mean(rmses5)})
    run.log({f"RMSE Outcome Std 5": sem(rmses5)})
    run.log({f"MSE Intensity Avg": np.mean(mses_int)})
    run.log({f"MSE Intensity Std": sem(mses_int)})
    print(f"RMSE Outcome Avg: {np.mean(rmses)}")
    print(f"RMSE Outcome Std: {sem(rmses)}")
    print(f"RMSE Outcome Avg 1: {np.mean(rmses1)}")
    print(f"RMSE Outcome Std 1: {sem(rmses1)}")
    print(f"RMSE Outcome Avg 2: {np.mean(rmses2)}")
    print(f"RMSE Outcome Std 2: {sem(rmses2)}")
    print(f"RMSE Outcome Avg 3: {np.mean(rmses3)}")
    print(f"RMSE Outcome Std 3: {sem(rmses3)}")
    print(f"RMSE Outcome Avg 4: {np.mean(rmses4)}")
    print(f"RMSE Outcome Std 4: {sem(rmses4)}")
    print(f"RMSE Outcome Avg 5: {np.mean(rmses5)}")
    print(f"RMSE Outcome Std 5: {sem(rmses5)}")
    print(f"MSE Intensity Avg: {np.mean(mses_int)}")
    print(f"MSE Intensity Std: {sem(mses_int)}")
