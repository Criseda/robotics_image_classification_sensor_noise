# --- HPC CONDA ENVIRONMENT SETUP ---
# You will need the following packages to run this script safely on the cluster.
# Run these commands step-by-step on the cluster before submitting your job:
#
# 1. module load apps/binapps/conda/miniforge3/25.9.1
# 2. conda create -n cogrob_env python=3.10 numpy pandas matplotlib scikit-learn
# 3. conda activate cogrob_env
# 4. conda install -c conda-forge tensorflow optuna
# 5. pip install optuna-integration[tfkeras]
# -----------------------------------

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import optuna
import pandas as pd
import matplotlib
# Use Agg backend for matplotlib on headless HPC servers to prevent Display crashes
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# HPC script will output locally to its own directory / outputs
OUTPUT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "outputs"))
os.makedirs(OUTPUT_DIR, exist_ok=True)
print("Saving outputs to:", OUTPUT_DIR)

print("TensorFlow version:", tf.__version__)
print("Optuna version:", optuna.__version__)

print("Loading CIFAR-100 dataset...")
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data(label_mode='fine')
x_train, x_test = x_train / 255.0, x_test / 255.0

def add_gaussian_noise(images, severity=0.1):
    noise = tf.random.normal(shape=tf.shape(images), mean=0.0, stddev=severity, dtype=tf.float64)
    noisy_images = images + noise
    return tf.clip_by_value(noisy_images, 0.0, 1.0).numpy()

print("Generating Sim-to-Real Dataset...")
x_test_noisy_low = add_gaussian_noise(x_test, severity=0.1)
x_test_noisy_med = add_gaussian_noise(x_test, severity=0.25)
x_test_noisy_high = add_gaussian_noise(x_test, severity=0.4)

BATCH_SIZE = 64
base_train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000)
val_dataset_clean = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_dataset_low = tf.data.Dataset.from_tensor_slices((x_test_noisy_low, y_test)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_dataset_med = tf.data.Dataset.from_tensor_slices((x_test_noisy_med, y_test)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_dataset_high = tf.data.Dataset.from_tensor_slices((x_test_noisy_high, y_test)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

def build_model(conv_blocks, filters, dropout_rate):
    inputs = keras.Input(shape=(32, 32, 3))
    x = inputs
    for i in range(conv_blocks):
        x = layers.Conv2D(filters * (2**i), (3, 3), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(filters * (2**i), (3, 3), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = layers.Dropout(dropout_rate)(x)
        
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(100, activation='softmax')(x)
    return keras.Model(inputs=inputs, outputs=outputs)

# --- HYPERPARAMETERS FOR FINAL HPC RUN ---
EPOCHS = 40 # 40 gives the CNN enough time to actually separate good hyperparameters from bad ones
N_TRIALS = 50
SEEDS = [42, 123, 999, 8888] # 4 seeds specifically to match the professor's strict marking criteria example
# -----------------------------------------

def objective(trial):
    conv_blocks = trial.suggest_int("conv_blocks", 2, 4)
    filters = trial.suggest_categorical("filters", [32, 64])
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    augmentation_factor = trial.suggest_float("augmentation_factor", 0.0, 0.3)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
    
    seed_accuracies = []
    pruning_callback = optuna.integration.TFKerasPruningCallback(trial, "val_accuracy")
    
    for i, seed in enumerate(SEEDS):
        print(f"  -> Trial {trial.number} | Training with seed {seed}...")
        keras.utils.set_random_seed(seed)
        
        model = build_model(conv_blocks, filters, dropout_rate)
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate, weight_decay=weight_decay)
        model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        
        def augment(image, label):
            image = tf.image.random_flip_left_right(image)
            noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=augmentation_factor, dtype=tf.float64)
            return tf.clip_by_value(image + noise, 0.0, 1.0), label
            
        trial_dataset = base_train_dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        trial_dataset = trial_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        callbacks = [pruning_callback] if i == 0 else []
        
        history = model.fit(
            trial_dataset,
            epochs=EPOCHS,
            validation_data=val_dataset_med,
            verbose=2, # Verbose 2 prints one line per epoch (important to prevent 10GB HPC log dumps)
            callbacks=callbacks
        )
        final_acc = history.history["val_accuracy"][-1]
        seed_accuracies.append(final_acc)
        
    # Log the exact standard deviation and variation across the seeds into the CSV for your report!
    trial.set_user_attr("std_dev", float(np.std(seed_accuracies)))
    trial.set_user_attr("seed_scores", str([float(acc) for acc in seed_accuracies]))
    
    return np.mean(seed_accuracies)

print(f"Starting Bayesian Optimization with {N_TRIALS} trials over {EPOCHS} epochs each...")
study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

print("Best Trial:")
print("  Average Accuracy:", study.best_trial.value)
print("  Best Hyperparameters:", study.best_trial.params)

# Export Data into CSV safely
optuna_results_path = os.path.join(OUTPUT_DIR, 'optuna_results.csv')
study.trials_dataframe().to_csv(optuna_results_path, index=False)
print(f"Saved Optuna trials data to {optuna_results_path}")

try:
    fig1 = optuna.visualization.matplotlib.plot_optimization_history(study)
    plt.tight_layout()
    optimization_history_path = os.path.join(OUTPUT_DIR, 'optimization_history.png')
    plt.savefig(optimization_history_path)
    
    fig2 = optuna.visualization.matplotlib.plot_param_importances(study)
    plt.tight_layout()
    hyperparameter_importances_path = os.path.join(OUTPUT_DIR, 'hyperparameter_importances.png')
    plt.savefig(hyperparameter_importances_path)
    print(
        f"Saved visualization plots ({optimization_history_path}, {hyperparameter_importances_path})"
    )
except Exception as e:
    print(f"Failed to generate plots: {e}")

# FINAL EVALUATION MODEL
best_params = study.best_trial.params

def final_augment(image, label):
    image = tf.image.random_flip_left_right(image)
    noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=best_params["augmentation_factor"], dtype=tf.float64)
    return tf.clip_by_value(image + noise, 0.0, 1.0), label

final_batch_size = best_params.get("batch_size", 64)
final_train_dataset = base_train_dataset.map(final_augment, num_parallel_calls=tf.data.AUTOTUNE).batch(final_batch_size).prefetch(tf.data.AUTOTUNE)

acc_clean_list, acc_low_list, acc_med_list, acc_high_list = [], [], [], []

FINAL_EPOCHS = 100

for seed in SEEDS:
    print(f"\nTraining final optimal benchmark model with seed {seed}... (Evaluating Robustness Variance)")
    keras.utils.set_random_seed(seed)
    
    final_model = build_model(
        conv_blocks=best_params["conv_blocks"],
        filters=best_params["filters"],
        dropout_rate=best_params["dropout_rate"]
    )
    final_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=best_params["learning_rate"], weight_decay=best_params.get("weight_decay", 0.0)), 
        loss="sparse_categorical_crossentropy", 
        metrics=["accuracy"]
    )
    
    final_model.fit(final_train_dataset, epochs=FINAL_EPOCHS, validation_data=val_dataset_clean, verbose=2)
    
    acc_clean_list.append(final_model.evaluate(val_dataset_clean, verbose=0)[1])
    acc_low_list.append(final_model.evaluate(val_dataset_low, verbose=0)[1])
    acc_med_list.append(final_model.evaluate(val_dataset_med, verbose=0)[1])
    acc_high_list.append(final_model.evaluate(val_dataset_high, verbose=0)[1])

means = [np.mean(acc_clean_list), np.mean(acc_low_list), np.mean(acc_med_list), np.mean(acc_high_list)]
stds = [np.std(acc_clean_list), np.std(acc_low_list), np.std(acc_med_list), np.std(acc_high_list)]

print(f"\nFINAL EVALUATION STATS (Averaged across {len(SEEDS)} seeds):")
print(f"Clean:     {means[0]:.4f} ± {stds[0]:.4f}")
print(f"Low Noise: {means[1]:.4f} ± {stds[1]:.4f}")
print(f"Med Noise: {means[2]:.4f} ± {stds[2]:.4f}")
print(f"High Noise:{means[3]:.4f} ± {stds[3]:.4f}")

plt.figure(figsize=(8,5))
plt.errorbar(['Clean', 'Low Noise', 'Med Noise', 'High Noise'], means, yerr=stds, marker='o', linestyle='-', color='b', capsize=5)
plt.title('Final Model: Accuracy vs. Sim-to-Real Sensor Noise (with StdDev)')
plt.ylabel('Average Accuracy')
plt.grid(True)
final_accuracy_dropoff_path = os.path.join(OUTPUT_DIR, 'final_accuracy_dropoff.png')
plt.savefig(final_accuracy_dropoff_path)
print(f"Saved final accuracy drop-off chart to {final_accuracy_dropoff_path}")

print("HPC RUN COMPLETED SUCCESSFULLY.")
