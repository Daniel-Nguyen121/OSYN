import os
import sdv
import pandas as pd
import numpy as np
import torch
from sdv.single_table import TVAESynthesizer, CopulaGANSynthesizer, GaussianCopulaSynthesizer, CTGANSynthesizer
from sdv.metadata import Metadata

from sdv.evaluation.single_table import run_diagnostic
from sdv.evaluation.single_table import evaluate_quality

def train_generator(data, type_model='tvae', epochs=1000, batch_size=500, embedding_dim=128, type_data='oracle', save_dir="GAN_models"):
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Automatically detect column types
    metadata = Metadata.detect_from_dataframe(data)
    metadata.validate()
    meta_json = os.path.join(save_dir, f"metadata_bank_{type_data}.json")
    
    # Remove existing metadata file if it exists
    if os.path.exists(meta_json):
        os.remove(meta_json)
        print(f"Removed existing metadata file: {meta_json}")
    
    metadata.save_to_json(meta_json)
    print(f"Metadata detected and saved as metadata_bank_{type_data}.json")

    #Declare generator
    if type_model == 'tvae':
        generator = TVAESynthesizer(
            metadata=metadata,
            epochs=epochs,
            batch_size=batch_size,
            embedding_dim=embedding_dim,
            verbose=True,
            cuda=torch.cuda.is_available()
            )
    elif type_model == 'copula':
        generator = CopulaGANSynthesizer(
            metadata=metadata, 
            epochs=epochs, 
            batch_size=batch_size,
            embedding_dim=embedding_dim,
            verbose=True,
            cuda=torch.cuda.is_available()
            )
    elif type_model == 'gaussiancop':
        generator = GaussianCopulaSynthesizer(
            metadata=metadata
            )
    elif type_model == 'ctgan':
        generator = CTGANSynthesizer(
            metadata=metadata,
            epochs=epochs,
            batch_size=batch_size,
            embedding_dim=embedding_dim,
            verbose=True,
            cuda=torch.cuda.is_available()
            )
    else:
        print("Only except keywords in 'ctgan', 'tvae', 'copula', and 'gaussiancop'.")

    print("Training Generator...")
    generator.fit(data)
    print("Training complete.")

    gen_model = os.path.join(save_dir, f"{type_model}_bank_{type_data}_ep_{epochs}.pt")
    
    # Remove existing model file if it exists
    if os.path.exists(gen_model):
        os.remove(gen_model)
        print(f"Removed existing model file: {gen_model}")
    
    generator.save(gen_model)
    print(f"Generator saved successfully at {gen_model}.")
    return generator, metadata

def load_generator(model_dir, type_model='tvae'):
    if type_model == 'tvae':
        generator = TVAESynthesizer.load(filepath=model_dir)
    elif type_model == 'copula':
        generator = CopulaGANSynthesizer.load(filepath=model_dir)
    elif type_model == 'gaussiancop':
        generator = GaussianCopulaSynthesizer.load(filepath=model_dir)
    elif type_model == 'ctgan':
        generator = CTGANSynthesizer.load(filepath=model_dir)
    else:
        print("Only except keywords in 'ctgan', 'tvae', 'copula', and 'gaussiancop'.")
    
    return generator

def load_metadata(metadata_path):
    """Load metadata from a JSON file."""
    return Metadata.load_from_json(metadata_path)

def evaluate_generator(generator, oracle_data, metadata):
    synthetic_data = generator.sample(num_rows=len(oracle_data))
    diagnostic_report = run_diagnostic(
        real_data=oracle_data,
        synthetic_data=synthetic_data,
        metadata=metadata,
    )

    quality_report = evaluate_quality(
        real_data=oracle_data,
        synthetic_data=synthetic_data,
        metadata=metadata
    )

    # # Save reports
    # quality_report.save(filepath='quality_report.pkl')
    # from sdmetrics.reports.single_table import QualityReport
    # quality = QualityReport.load('quality_report.pkl')

    return diagnostic_report, quality_report

def sample_synthetic_data(generator, num_rows=1000, save_dir="GAN_models"):
    synthetic_data = generator.sample(num_rows=num_rows)
    return synthetic_data

def get_data(data_path, data_type='oracle'):
    if data_type == 'oracle':
        return pd.read_csv(os.path.join(data_path, "df_oracle.csv"))
    elif data_type == 'train':
        return pd.read_csv(os.path.join(data_path, "df_train.csv"))
    elif data_type == 'small':
        return pd.read_csv(os.path.join(data_path, "df_small.csv"))

def run():
    data_path = "Datasets/Bank_Marketing/300_200"

    gan_models = ["tvae", "copula", "gaussiancop", "ctgan"]
    data_types = ["oracle", "train", "small"]
    epochs = 1000
    batch_size = 500
    embedding_dim = 128
    save_dir = "GAN_models"

    # Load oracle data
    oracle_data = get_data(data_path, "oracle")

    # Train and evaluate each GAN model for each data type
    for data_type in data_types:
        print("="*50)
        print(f"---------- Working on {data_type} data ----------")

        data_df = get_data(data_path, data_type)
        print(f"{data_type} data loaded successfully.")

        for gan_model in gan_models:
            print(f"Training {gan_model}...")
            generator, metadata = train_generator(data_df, type_model=gan_model, epochs=epochs, batch_size=batch_size, embedding_dim=embedding_dim, type_data=data_type, save_dir=save_dir)
            print(f"Training {gan_model} for {data_type} data completed.")

            print(f"Evaluating {gan_model}...")
            diagnostic_report, quality_report = evaluate_generator(generator, oracle_data, metadata)

            print(f"Finished {gan_model} for {data_type} data")
        print("="*50)
    
    print("DONE!!!")

if __name__ == "__main__":
    run()
    # # Train generator
    # generator, metadata = train_generator(oracle_data, type_model='tvae', epochs=1000, batch_size=500, embedding_dim=128, type_data='oracle', save_dir="GAN_models")

    # # Evaluate generator
    # diagnostic_report, quality_report = evaluate_generator(generator, oracle_data, metadata)

    # # Sample synthetic data
    # synthetic_data = sample_synthetic_data(generator, num_rows=1000, save_dir="GAN_models")
    # synthetic_data.head()