import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, SubsetRandomSampler, Subset
from sklearn.metrics import average_precision_score, accuracy_score, recall_score, precision_score, f1_score
import torchvision.models as models
from efficientnet_pytorch import EfficientNet
import pandas as pd
import time
import matplotlib.pyplot as plt
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Set the paths
Train_path = r"C:\ThesisMedia\CroppedPralines\Grouped_dataset\Thesis_Dataset_Size\New_trained"
Test_path = r"C:\ThesisMedia\CroppedPralines\Grouped_dataset\Thesis_Dataset_Size\Test"
output_dir = r"C:\ThesisMedia\Trained_Models"

batch_size = 64
num_epochs = 10
num_classes = 2
window_size = 3   #Smoothing plot curve

# Set the device (CUDA if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the dataset sizes to study
dataset_sizes = [50, 100, 250, 500, 1000, 2000]#, 5000, 10000]

# Define the models to test
models_to_test = [
    ('vgg11_bn', models.vgg11_bn(pretrained=True)),
    ('ResNet_bn', models.resnet50(pretrained=True)),
    # ('Inception_bn', models.inception_v3(pretrained=True)),
    # ('EfficientNet_bn', EfficientNet.from_pretrained('efficientnet-b0')),
    ('MobileNet_bn', models.mobilenet_v2(pretrained=True))
]

def get_transform(model_name, data_augmentation):
    if model_name == 'Inception_bn':
        image_size = (299, 299)
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        image_size = (224, 224)
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    if data_augmentation:
        angle = random.choice([0, 90, 180, 270])
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(angle),
            transform
        ])

    return transform

def set_worker_seed(worker_id):
    torch.manual_seed(worker_id)

def create_data_loader(dataset, batch_size, num_samples, shuffle):

    good_indices_test = [i for i, (_, label) in enumerate(dataset.imgs) if label == 0]
    bad_indices_test = [i for i, (_, label) in enumerate(dataset.imgs) if label == 1]

    # Create a balanced subset with equal samples from both classes
    balanced_subset = list(SubsetRandomSampler(good_indices_test[:num_samples])) + list(SubsetRandomSampler(bad_indices_test[:num_samples]))
    balanced_dataset = torch.utils.data.Subset(dataset, balanced_subset)

    return DataLoader(
        balanced_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers = 8,
        pin_memory=False,
        worker_init_fn=set_worker_seed
    )

def smooth_curve(data, window_size):
    # Apply a moving average to the data using a specified window size
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')
    
# Create an empty DataFrame to store the results
results_df = pd.DataFrame(columns=['Model', 'Dataset Size', 'Training Duration',
                                   'Train-Test Splits', 'Train mAP', 'Train Accuracy','Test mAP',
                                   'Test Accuracy', 'Test Recall', 'Test Precision', 'Test F1 Score'])

def train_model(model, train_loader, test_loader, model_name, dataset_size, num_epochs):
    
    global results_df
    model.to(device)

    # Define your loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    train_map_values = []
    test_map_values = []
    test_accuracy_values = []
    train_accuracy_values = []
    y_true = []
    y_pred = []

    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        y_true = []
        y_pred = []

        # Create a progress bar for the training loop
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False)

        for images, labels in progress_bar:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # Forward pass
            if model_name == 'Inception_bn':
                outputs, _ = model(images)
            else:
                outputs = model(images)

            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            # Update the progress bar with the current loss
            progress_bar.set_postfix({"Loss": loss.item()})

            # Store the true labels and predicted labels for mAP calculation
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

        epoch_loss = running_loss / len(train_loader)
        epoch_train_accuracy = train_correct / train_total
        train_map = average_precision_score(y_true, y_pred)
        train_map_values.append(train_map)
        train_accuracy_values.append(epoch_train_accuracy)
        print(f"Epoch [{epoch + 1}/{num_epochs}] - Training Loss: {epoch_loss:.4f} - Training Accuracy: {epoch_train_accuracy:.4f}")

        # Print GPU memory usage during training
        memory_used = torch.cuda.memory_allocated(device) / 1024 / 1024  # Convert to MiB
        print(f"Epoch [{epoch + 1}/{num_epochs}] - GPU Memory Used: {memory_used:.2f} MiB")

        # Save the best model based on training mAP
        if epoch == 0 or train_map > max(train_map_values[:-1]):
            torch.save(model.state_dict(), f'{output_dir}/Model_{model_name}_{dataset_size}.pt')

        # Test the model on the test dataset
        model.eval()  # Set model to evaluation mode
        test_y_true = []
        test_y_pred = []

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)

                _, predicted = torch.max(outputs.data, 1)

                # Store the true labels and predicted labels for mAP calculation
                test_y_true.extend(labels.cpu().numpy())
                test_y_pred.extend(predicted.cpu().numpy())

        # Calculate mAP for test set
        test_accuracy = accuracy_score(test_y_true, test_y_pred)
        test_recall = recall_score(test_y_true, test_y_pred)
        test_precision = precision_score(test_y_true, test_y_pred)
        test_f1_score = f1_score(test_y_true, test_y_pred)
        test_map = average_precision_score(test_y_true, test_y_pred)
        test_map_values.append(test_map)
        test_accuracy_values.append(test_accuracy)

        print(f"Test Accuracy: {test_accuracy:.4f}", f"Test Recall: {test_recall:.4f}", f"Test Precision: {test_precision:.4f}")
        print(f"Test F1 Score: {test_f1_score:.4f}", f"Test mAP: {test_map:.4f}")

        # Clear GPU cache to free up memory
        torch.cuda.empty_cache()

    # Load the best model based on training mAP
    model.load_state_dict(torch.load(f'{output_dir}/Model_{model_name}_{dataset_size}.pt'))

    # Compute training duration
    end_time = time.time()
    training_duration = end_time - start_time

    # Store the results in the DataFrame
    results_df = results_df.append({
        'Model': model_name,
        'Dataset Size': dataset_size*2,
        'Training Duration': training_duration,
        'Train-Test Splits': f'{len(train_loader.dataset)}-{len(test_loader.dataset)}',
        'Train Accuracy': train_accuracy_values[-1],
        'Train mAP': train_map_values[-1],
        'Test mAP': test_map_values[-1],
        'Test Accuracy': test_accuracy,
        'Test Recall': test_recall,
        'Test Precision': test_precision,
        'Test F1 Score': test_f1_score
    }, ignore_index=True)
    

    # Smooth the mAP curves using a moving average with a window size of 5
    smoothed_train_map_values = smooth_curve(train_map_values, window_size)
    smoothed_test_map_values = smooth_curve(test_map_values, window_size)
    smoothed_test_accuracy_values = smooth_curve(test_accuracy_values, window_size)
    smoothed_train_accuracy_values = smooth_curve(train_accuracy_values, window_size)

    # Plot the training and testing mAP graphs
    plt.plot(range(window_size, num_epochs + 1), smoothed_train_map_values, label='Train mAP', color='purple')
    plt.plot(range(window_size, num_epochs + 1), smoothed_test_map_values, label='Test mAP', color='magenta')
    plt.plot(range(window_size, num_epochs + 1), smoothed_train_accuracy_values, label='Train Accuracy', color='blue') 
    plt.plot(range(window_size, num_epochs + 1), smoothed_test_accuracy_values, label='Test Accuracy', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('mAP & Accuracy')
    plt.title(f'{model_name} - Dataset Size: {dataset_size*2}', loc='left')
    plt.text(1.0, 1.02, f'Training Duration: {training_duration:.2f} sec', transform=plt.gca().transAxes, ha='right')
    plt.legend(loc='upper left')
    plt.savefig(f'{output_dir}/mAP_Plot_Dataset_training/{model_name}_dataset_size_{dataset_size}_mAP_Accuracy_graph.png')
    plt.close()

def main():
    global results_df

    for model_name, model in models_to_test:
        print(f"\n--- Model: {model_name} ---")

        # Modify the classification layer for the model
        if model_name == 'EfficientNet_bn':
            model._fc = nn.Sequential(
                nn.Linear(in_features=1280, out_features=512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.5),  # Add dropout layer with dropout rate of 0.5
                nn.Linear(in_features=512, out_features=num_classes)
            )
        elif model_name == 'MobileNet_bn':
            num_features = model.classifier[-1].in_features
            model.classifier[-1] = nn.Sequential(
                nn.Linear(in_features=num_features, out_features=512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.5),  # Add dropout layer with dropout rate of 0.5
                nn.Linear(in_features=512, out_features=num_classes)
            )
        elif model_name == 'ResNet_bn' or model_name == 'Inception_bn':
            num_features = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Linear(in_features=num_features, out_features=512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.5),  # Add dropout layer with dropout rate of 0.5
                nn.Linear(in_features=512, out_features=num_classes)
            )
        else:
            num_features = model.classifier[-1].in_features
            model.classifier[-1] = nn.Sequential(
                nn.Linear(in_features=num_features, out_features=4096),
                nn.BatchNorm1d(4096),
                nn.ReLU(True),
                nn.Dropout(0.5),
                nn.Linear(4096, 4096),
                nn.BatchNorm1d(4096),
                nn.ReLU(True),
                nn.Dropout(0.5),
                nn.Linear(4096, 2)  # 2 classes: Good and Bad
            )

        for size in dataset_sizes:
            print(f"\n--- Dataset Size: {size} ---")

            # Create DataLoader for training, validation, and testing with the respective transformations
            train_dataset = ImageFolder(root=Train_path, transform=get_transform(model_name, data_augmentation=(size > 2000)))
            test_dataset = ImageFolder(root=Test_path, transform=get_transform(model_name, data_augmentation=False))
            train_loader = create_data_loader(train_dataset, batch_size, size, shuffle=True)
            test_loader = create_data_loader(test_dataset, batch_size, 300, shuffle=False)

            # Train the model and evaluate on the test set
            train_model(model, train_loader, test_loader, model_name, size, num_epochs)

    # Save the results to an Excel file
    results_df.to_excel(f'{output_dir}/DataSet_Training_Table.xlsx', index=False)

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    main()