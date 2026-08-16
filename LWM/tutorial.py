import subprocess
import os
import shutil

def clone_dataset_scenario(repo_url, model_repo_dir="./LWM", scenarios_dir="scenarios"):
    """
    Clones all scenarios from a repository, ensuring all files (small and large) are downloaded.

    Args:
        repo_url (str): URL of the Git repository
        model_repo_dir (str): Path to the model repository
        scenarios_dir (str): Directory name for storing scenarios
    """
    current_dir = os.path.basename(os.getcwd())
    if current_dir == "LWM":
        model_repo_dir = "."

    scenarios_path = os.path.join(model_repo_dir, scenarios_dir)
    os.makedirs(scenarios_path, exist_ok=True)

    original_dir = os.getcwd()

    try:
        if os.path.exists(scenarios_path):
            shutil.rmtree(scenarios_path)

        print("Cloning entire repository into temporary directory ...")
        subprocess.run([
            "git", "clone",
            repo_url,
            scenarios_path
        ], check=True)

        os.chdir(scenarios_path)

        print("Pulling all files using Git LFS ...")
        subprocess.run(["git", "lfs", "install"], check=True) 
        subprocess.run(["git", "lfs", "pull"], check=True) 

        print(f"Successfully cloned all scenarios into {scenarios_path}")

    except subprocess.CalledProcessError as e:
        print(f"Error cloning scenarios: {str(e)}")
    finally:
        if os.path.exists(scenarios_path):
            shutil.rmtree(scenarios_path)
        os.chdir(original_dir)
#%%
model_repo_url = "https://huggingface.co/wi-lab/lwm"
model_repo_dir = "./LWM"

if not os.path.exists(model_repo_dir):
    print(f"Cloning model repository from {model_repo_url}...")
    subprocess.run(["git", "clone", model_repo_url, model_repo_dir], check=True)
#%%
import numpy as np
dataset_repo_url = "https://huggingface.co/datasets/wi-lab/lwm" 
clone_dataset_scenario(dataset_repo_url, model_repo_dir)
#%%
if os.path.exists(model_repo_dir):
    os.chdir(model_repo_dir)
    print(f"Changed working directory to {os.getcwd()}")
else:
    print(f"Directory {model_repo_dir} does not exist. Please check if the repository is cloned properly.")
#%%
from input_preprocess import tokenizer
from lwm_model import lwm
import torch

scenario_names = np.array([
    "city_18_denver", "city_15_indianapolis", "city_19_oklahoma", 
    "city_12_fortworth", "city_11_santaclara", "city_7_sandiego"
])
scenario_idxs = np.array([0, 1, 2, 3, 4, 5])[3] 
selected_scenario_names = scenario_names[scenario_idxs]

snr_db = None

preprocessed_chs = tokenizer(
    selected_scenario_names=selected_scenario_names,
    manual_data=None,
    gen_raw=True,
    snr_db=snr_db
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Loading the LWM model on {device} ...")
model = lwm.from_pretrained(device=device)
#%%
from inference import lwm_inference, create_raw_dataset
input_types = ['cls_emb', 'channel_emb', 'raw']
selected_input_type = input_types[1] 

if selected_input_type in ['cls_emb', 'channel_emb']:
    dataset = lwm_inference(preprocessed_chs, selected_input_type, model, device)
else:
    dataset = create_raw_dataset(preprocessed_chs, device)
#%%
from input_preprocess import create_labels
n_beams = 16
tasks = ['LoS/NLoS Classification', 'Beam Prediction']
task = tasks[0]
labels = create_labels(task, selected_scenario_names, n_beams=n_beams) 
# %% Dimensionality Reduction Visualization

# Import the dimensionality reduction plotting function
from utils import plot_dimensionality_reduction

# Iterate over tasks (e.g., LoS/NLoS Classification, Beam Prediction)
for task in tasks:
    
    # Create labels for the current task
    labels = create_labels(task, selected_scenario_names, n_beams=n_beams)
    
    # Iterate over input types (e.g., raw data or embeddings)
    for input_type_idx, input_type in enumerate(input_types):
        
        # Select the current input type
        selected_input_type = input_types[input_type_idx]
        
        # Prepare dataset based on input type
        if selected_input_type in ['cls_emb', 'channel_emb']:
            dataset = lwm_inference(
                preprocessed_chs, 
                selected_input_type, 
                model, 
                device
            )
        else:
            dataset = create_raw_dataset(preprocessed_chs, device)
        
        # Plot dimensionality reduction for the dataset
        plot_dimensionality_reduction(
            dataset, 
            method='all',  # Use all available dimensionality reduction methods
            labels=labels,  # Labels for visualization
            task=task,  # Current task (for title or labeling)
            input_type=input_type  # Current input type (for title or labeling)
        )
        
#%% TRAINING
#%% TRAINING PARAMETERS
task = ['LoS/NLoS Classification', 'Beam Prediction'][0] # Select the task
n_trials = 1  # Number of trials for each configuration
num_classes = 2 if task == 'LoS/NLoS Classification' else n_beams  # Set number of classes based on the task
input_types = ['raw', 'cls_emb']  # Types of input data
split_ratios = np.array([.005, .0075, .01, .015, .02, .03, 
                         .05, .1, .25, .5, .8])  # Dataset split ratios
f1_scores = np.zeros((n_trials, len(input_types), len(split_ratios)))  # Store F1 scores for each trial, input type, and split ratio
labels = create_labels(task, selected_scenario_names, n_beams=n_beams)  # Create labels for the selected task

#%% TRAINING
from utils import get_data_loaders, FCN, train_model, plot_metrics

# Iterate over input types (e.g., raw data or embeddings)
for input_type_idx, input_type in enumerate(input_types):
    
    # Prepare dataset based on input type
    if input_type in ['cls_emb', 'channel_emb']:
        dataset = lwm_inference(preprocessed_chs, input_type, model, device)
    else:
        dataset = create_raw_dataset(preprocessed_chs, device)
    
    # Reshape dataset for training
    dataset = dataset.view(dataset.size(0), -1)
    input_dim = dataset.shape[-1]  # Get input dimension for the model
    
    # Iterate over different dataset split ratios
    for split_ratio_idx, split_ratio in enumerate(split_ratios):
        
        n_train = int(split_ratio * dataset.shape[0])  # Calculate number of training samples
        
        # Run multiple trials for each split ratio
        for trial in range(n_trials):
            
            print(f"\ninput type: {input_type}, \nnumber of training samples: {int(split_ratio*len(dataset))}, \ntrial: {trial}\n")
            
            torch.manual_seed(trial)  # Set seed for reproducibility
            
            if snr_db is not None:
                preprocessed_chs = tokenizer(
                    selected_scenario_names=selected_scenario_names,
                    manual_data=None,
                    gen_raw=True,
                    snr_db=snr_db
                )
                if input_type in ['cls_emb', 'channel_emb']:
                    dataset = lwm_inference(preprocessed_chs, input_type, model, device)
                else:
                    dataset = create_raw_dataset(preprocessed_chs, device)
                dataset = dataset.view(dataset.size(0), -1)
                
            train_loader, test_loader = get_data_loaders(
                dataset, 
                labels, 
                batch_size=128, 
                split_ratio=split_ratio
            )
            
            # Initialize the Fully Connected Network (FCN) model
            FCN_model = FCN(input_dim=input_dim, num_classes=num_classes)
            
            # Train the model and retrieve losses and F1 scores
            train_losses, test_f1_scores = train_model(
                FCN_model, 
                train_loader, 
                test_loader, 
                epochs=120, 
                lr=0.0001 if input_type == "raw" else 0.001,  # Learning rate depends on input type
                device=device, 
                decay_step=30, 
                decay_rate=0.5
            )
            
            # Store the final F1 score for this trial
            f1_scores[trial, input_type_idx, split_ratio_idx] = test_f1_scores[0, -1]
            
            # Plot metrics for the current trial
            # plot_metrics(test_f1_scores, [input_type])

# Plot average F1 scores across all trials for each input type and split ratio
plot_metrics(
    np.mean(f1_scores, axis=0),  # Average F1 scores across trials
    input_types, 
    np.asarray(split_ratios * dataset.shape[0], dtype=int),  # Convert split ratios to actual sample counts
    flag=1
)

# %% Few-Shot Learning with Pretrained Embeddings

# Initialize array to store F1 scores for KNN classification
f1_scores_knn = np.zeros((n_trials, len(input_types), len(split_ratios)))

# Import the classification function
from utils import classify_by_euclidean_distance

# Iterate over input types (e.g., raw data or embeddings)
for input_type_idx, input_type in enumerate(input_types):
    
    # Prepare dataset based on input type
    if input_type in ['cls_emb', 'channel_emb']:
        dataset = lwm_inference(preprocessed_chs, input_type, model, device)
    else:
        dataset = create_raw_dataset(preprocessed_chs, device)
    
    # Reshape dataset for compatibility
    dataset = dataset.view(dataset.size(0), -1)
    input_dim = dataset.shape[-1]  # Get input dimension
    
    # Iterate over different dataset split ratios
    for split_ratio_idx, split_ratio in enumerate(split_ratios):
        
        n_train = int(split_ratio * dataset.shape[0])  # Calculate number of training samples
        
        # Run multiple trials for each split ratio
        for trial in range(n_trials):
            
            torch.manual_seed(trial)  # Set seed for reproducibility
            
            if snr_db is not None:
                preprocessed_chs = tokenizer(
                    selected_scenario_names=selected_scenario_names,
                    manual_data=None,
                    gen_raw=True,
                    snr_db=snr_db
                )
                if input_type in ['cls_emb', 'channel_emb']:
                    dataset = lwm_inference(preprocessed_chs, input_type, model, device)
                else:
                    dataset = create_raw_dataset(preprocessed_chs, device)
                dataset = dataset.view(dataset.size(0), -1)
            
            train_loader, test_loader = get_data_loaders(
                dataset, 
                labels, 
                batch_size=128, 
                split_ratio=split_ratio
            )
            
            # Perform classification using Euclidean distance
            f1 = classify_by_euclidean_distance(
                train_loader, 
                test_loader, 
                device="cpu"
            )
            
            # Store the F1 score for this trial
            f1_scores_knn[trial, input_type_idx, split_ratio_idx] = f1

# Plot average F1 scores across all trials for each input type and split ratio
plot_metrics(
    np.mean(f1_scores_knn, axis=0),  # Average F1 scores across trials
    input_types, 
    np.asarray(split_ratios * dataset.shape[0], dtype=int),  # Convert split ratios to actual sample counts
    flag=1
)









