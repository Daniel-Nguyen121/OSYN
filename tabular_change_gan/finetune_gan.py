from ctgan import CTGAN
import torch
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.serialization import add_safe_globals
import pandas as pd
import time
from config import *
cfg = get_cfg_defaults()

from get_data import *

def finetune_gan(df_gan, save_dir=cfg.GAN.SAVE_DIR):
    discrete_columns, num_columns = find_columns(df_gan)

    print("Fine tune the CTGAN")
    start_time = time.time()
    gan_wrapper = CTGAN(
        epochs=cfg.GAN.EPOCHS,
        cuda=True,
        batch_size=cfg.GAN.BATCH_SIZE,  # Increased batch size for faster training
        generator_dim=cfg.GAN.GENERATOR_DIM,  # Optimized network architecture
        discriminator_dim=cfg.GAN.DISCRIMINATOR_DIM,
        generator_lr=cfg.GAN.GENERATOR_LR,  # Adjusted learning rates
        discriminator_lr=cfg.GAN.DISCRIMINATOR_LR,
        discriminator_steps=cfg.GAN.DISCRIMINATOR_STEPS,  # Reduced discriminator steps
        pac=cfg.GAN.PAC,  # Increased pac for better stability
        log_frequency=cfg.GAN.LOG_FREQUENCY,
        verbose=cfg.GAN.VERBOSE
    )
    # First fit the model to initialize the networks
    gan_wrapper.fit(df_gan, discrete_columns=discrete_columns)
    print("Fitted the CTGAN")
    print(f"GAN training time: {time.time() - start_time:.2f} seconds")
    save_gan(gan_wrapper, save_dir)
    print(f"GAN model saved in {save_dir}")
    return gan_wrapper

def eval_synthetic(real_df_test, synth_df):
    # Prepare X/y
    X_real = pd.get_dummies(real_df_test.drop(columns='y'), drop_first=True)
    y_real = (real_df_test['y']=='yes').astype(int)
    X_syn = pd.get_dummies(synth_df.drop(columns='y'), drop_first=True)
    y_syn = (synth_df['y']=='yes').astype(int)
    
    # Align columns
    X_syn = X_syn.reindex(columns=X_real.columns, fill_value=0)

    # Train on synthetic, test on real
    clf = RandomForestClassifier(n_estimators=100, random_state=0)
    clf.fit(X_syn, y_syn)
    preds = clf.predict(X_real)
    return {
        'accuracy': round(accuracy_score(y_real, preds), 4),
        'roc_auc': round(roc_auc_score(y_real, clf.predict_proba(X_real)[:,1]), 4)
    }

def save_gan(gan_wrapper, path):
    # Move model to CPU before saving
    if hasattr(gan_wrapper, '_generator'):
        gan_wrapper._generator.cpu()
    torch.save(gan_wrapper, path)
    # Move model back to GPU if available
    if torch.cuda.is_available():
        gan_wrapper._generator.cuda()

def load_gan(path, device='cuda'):
    
    # Add all necessary CTGAN components to safe globals
    from ctgan.synthesizers.ctgan import CTGAN
    from ctgan.data_transformer import DataTransformer
    from ctgan.synthesizers.base import BaseSynthesizer
    
    # Add all required classes to safe globals
    add_safe_globals([
        CTGAN,
        DataTransformer,
        BaseSynthesizer,
        torch.nn.Module,
        torch.nn.Linear,
        torch.nn.BatchNorm1d,
        torch.nn.LeakyReLU,
        torch.nn.Sigmoid,
        torch.nn.Tanh
    ])
    
    # Convert device string to torch.device object
    if isinstance(device, str):
        device = torch.device(device)
    
    # Load model with weights_only=False since we trust our own saved model
    model = torch.load(path, map_location=device, weights_only=False)
    model._generator.to(device)
    model._generator.eval()
    return model