import torch
import torch.nn as nn
import time
import os
import numpy as np
from torchvision.transforms.transforms import RandomCrop
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt
from torchvision import transforms as T
from shutil import copyfile
from net import FeatureNetwork1HeadWithHandcrafted, FeatureNetwork1HeadNoHandcrafted, FeatureNetwork1HeadWithHandcraftedV2, FeatureNetwork1HeadWithHandcraftedV3
from helper import Mode
import logging
from util import ImageDataset
import math
import pandas as pd
from collections import Counter


logging.basicConfig(level=logging.INFO)

config = {
  "base": {
    "base_network": "ResNet 50",
    "img_size": 224,
    "problem_type": "DME",
    "feature_scaler": "standardscaler",
    "use_handcrafted_features": True,
    "pretrained": False,
    "gpu_id": 0,
    "mode": "training",
    "hyper_params": {
      "batch_size": 32
    }
  },
  "datasets": {
    "training": {
      "dir_inputs": "data/CV/1/training/DME",
      "dir_features": "data/training/features"
    },
    "validation": {
      "dir_inputs": "data/CV/1/validation/DME",
      "dir_features": "data/validation/features"
    },
    "test": {
      "dir_inputs": "data/test/data/DR",
      "dir_features": "data/test/features"
    }
  },
  "modes": {
    "hptuning": {
      "learning_rate": True,
      "batch_size": False,
      "epoch": False
    },
    "training": {
      "hyper_params": {
        "epochs": 150,
        "lr": 0.0001
      },
      "optimizer": "adam",
      "checkpoints": {
        "saving_frequency": 25,
        "saving_directory": "network_checkpoints"
      }
    },
    "test": {
      "checkpoint": "experiments/ResNet 50_224_1629901828/networks/network_final_1629914966.pth",
      "saving_directory": "test_results",
      "tag": "final"
    }
  }
}

base_network = config["base"]["base_network"]
img_size = config["base"]["img_size"]
batch_size = config["base"]["hyper_params"]["batch_size"]
problem_type = config["base"]["problem_type"]
feature_scaler_type = config["base"]["feature_scaler"]
pretrained = config["base"]["pretrained"]

if problem_type == "DR":
    logging.info("Using DR dataset.")
    CLASSES = ('dr0', 'dr1', 'dr2', 'dr3', 'dr4')
elif problem_type == "DME":
    logging.info("Using DME dataset.")
    CLASSES = ('grade0', 'grade1', 'grade2')
else:
    logging.error(f"Unknown problem_type '{problem_type}'.")
    exit(-1)
    
num_classes = len(CLASSES)
device = torch.device(f'cuda:{config["base"]["gpu_id"]}' if torch.cuda.is_available() else "cpu")
mode = config["base"]["mode"]
use_handcrafted_features = config["base"]["use_handcrafted_features"]

mpl.style.use("seaborn")

transforms_main_train = T.Compose([
        T.Resize((int(img_size * 1.1), int(img_size * 1.1))),
        T.ToTensor(),
        T.Normalize((0,0,0), (1,1,1))
    ])

transform_augment_train = T.Compose([
    T.RandomCrop((img_size, img_size)),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip()
])

transforms_main_eval = T.Compose([
    T.Resize((img_size, img_size)),
    T.ToTensor(),
    T.Normalize((0,0,0), (1,1,1))
])

if use_handcrafted_features:
    logging.info("Using hand-crafted features.")
    #model = FeatureNetwork1HeadWithHandcrafted(base_network=base_network, num_classes=num_classes, img_size=img_size, pretrained=pretrained)
    model = FeatureNetwork1HeadWithHandcraftedV2(base_network=base_network, num_classes=num_classes, img_size=img_size, pretrained=pretrained)
    #model = FeatureNetwork1HeadWithHandcraftedV3(base_network=base_network, num_classes=num_classes, img_size=img_size, pretrained=pretrained)
else:
    logging.info("Not using hand-crafted features.")
    model = FeatureNetwork1HeadNoHandcrafted(base_network=base_network, num_classes=num_classes, img_size=img_size, pretrained=pretrained)

model.to(device)

def hyperparamtuning(num_epochs=100):
    train_dir_in = config["datasets"]["training"]["dir_inputs"]
    train_features_in = config["datasets"]["training"]["dir_features"]

    tune_lr = config["modes"]["hptuning"]["learning_rate"]

    best_lr = 0.0001
    if tune_lr:
        logging.info("Tuning learning rate.")
        def schedule_lr(epoch):
            return 1e-5 * (10 ** (epoch / 20))

        def update_learning_rate(optimizer, new_lr):
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr

        epochs = list(range(num_epochs))
        lrs = [schedule_lr(e) for e in epochs]

        plt.plot(epochs, lrs)
        plt.xlabel("epoch")
        plt.ylabel("lr")
        plt.title("LR scheduling")
        plt.savefig("lrschedulingpng")
        plt.clf()

        train_dataset = ImageDataset(img_dir=train_dir_in, class_names=CLASSES, features_dir=train_features_in, feature_scaler_path=f"data/training/feature_scaler_{feature_scaler_type}_{problem_type}.pkl", transform_main=transforms_main_train, transform_augment=transform_augment_train)

        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                batch_size=batch_size, 
                                                shuffle=True)
    
        len_train_dataset = len(train_dataset)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=best_lr)

        losses = np.zeros(num_epochs)
        losses_per_batch = np.zeros(num_epochs)
        losses_avg = np.zeros(num_epochs)

        model.train()

        for epoch in range(num_epochs):
            logging.info(f"Learning rate: {lrs[epoch]}")
            update_learning_rate(optimizer, lrs[epoch])

            # Training part
            total_loss = 0
            
            progress_bar_train = tqdm(total=len(train_loader))
        
            for images, features, labels in train_loader:
                images = images.float().to(device)
                features = features.float().to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
        
                outputs = model(images, features)
        
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                total_loss += loss.item() * images.size(0)
        
                progress_bar_train.update(1)
            
            progress_bar_train.close()

            avg_loss = total_loss / len_train_dataset
            
            print('Epoch [{}/{}], Total Training Loss: {:.4f}, Avg Loss: {:.4f}'.format(epoch+1, num_epochs, total_loss, avg_loss))

            losses[epoch] = total_loss
            losses_avg[epoch] = avg_loss

        epochs = list(range(num_epochs))

        logging.info("Training DONE.")

        plt.plot(lrs, losses, "-")
        plt.xlabel("lr")
        plt.ylabel("loss")
        plt.xscale("log")

        plt.savefig("losses.png")
        plt.clf()

        plt.plot(lrs, losses_per_batch, "-")
        plt.xlabel("lr")
        plt.ylabel("loss per batch")
        plt.xscale("log")

        plt.savefig("lossesperbatch.png")
        plt.clf()

        plt.plot(lrs, losses_avg, "-")
        plt.xlabel("lr")
        plt.ylabel("loss avg")
        plt.xscale("log")

        plt.savefig("lossesavg.png")
        plt.clf()

def train():

    if "last_checkpoint" in config["modes"]["training"].keys():
        checkpoint = config["modes"]["training"]["last_checkpoint"]

        checkpoint_path = checkpoint["path"]
        start_epoch = checkpoint["epoch"]
        min_val_loss = checkpoint["best_val_loss"]

        logging.info(f"Using last checkpoint '{checkpoint_path}'")
        
        model.load_state_dict(torch.load(checkpoint_path))
    else:
        logging.info("Starting from scracth (no checkpoint).")
        start_epoch = 0
        min_val_loss = math.inf

    optimizer_name = config["modes"]["training"]["optimizer"]

    train_dir_in = config["datasets"]["training"]["dir_inputs"]
    train_features_in = config["datasets"]["training"]["dir_features"]
    val_dir_in = config["datasets"]["validation"]["dir_inputs"]
    val_features_in = config["datasets"]["validation"]["dir_features"]
    
    lr = config["modes"]["training"]["hyper_params"]["lr"]
    num_epochs = config["modes"]["training"]["hyper_params"]["epochs"]
    save_frequency = config["modes"]["training"]["checkpoints"]["saving_frequency"]
    saving_directory_networks = "networks"

    if not os.path.exists(saving_directory_networks):
        logging.info(f"Saving directory '{saving_directory_networks}' created.")
        os.makedirs(saving_directory_networks, exist_ok=True)
    
    train_dataset = ImageDataset(img_dir=train_dir_in, class_names=CLASSES, features_dir=train_features_in, feature_scaler_path=f"data/training/feature_scaler_{feature_scaler_type}_{problem_type}.pkl", transform_main=transforms_main_train, transform_augment=transform_augment_train)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=batch_size, 
                                            shuffle=True)

    validation_dataset = ImageDataset(img_dir=val_dir_in, class_names=CLASSES, features_dir=val_features_in, feature_scaler_path=f"data/validation/feature_scaler_{feature_scaler_type}_{problem_type}.pkl", transform_main=transforms_main_eval)
    validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset,
                                            batch_size=batch_size, 
                                            shuffle=False)

    criterion = nn.CrossEntropyLoss()

    if optimizer_name == "SGD":
        logging.info("Using SGD optimizer.")
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    elif optimizer_name == "adam":
        logging.info("Using Adam optimizer.")
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        logging.error(f"Unknown optimizer name {optimizer_name}.")
        exit(-1)

    if "last_checkpoint" in config["modes"]["training"].keys():
        logging.info(f"Loading optimizer from {checkpoint['optimizer']}")
        optimizer.load_state_dict(torch.load(checkpoint["optimizer"]))
    
    # Train the model
    total_steps_training = len(train_dataset)
    total_steps_val = len(validation_dataset)
    total_steps_training_loader = len(train_loader)
    total_steps_val_loader = len(validation_loader)
    
    logging.info(f"Total steps training: {total_steps_training}.")

    logging.info(f"Total steps validation: {total_steps_val}.")
    logging.info(f"Total steps training loader: {total_steps_training_loader}.")

    logging.info(f"Total steps validation loader: {total_steps_val_loader}.")

    losses = np.zeros(num_epochs)
    validation_losses = np.zeros(num_epochs)

    for epoch in range(start_epoch, num_epochs):
        total_loss = 0

        label_output_matches = 0
        label_output_matches_val = 0
        
        progress_bar_train = tqdm(total=len(train_loader))
        
        model.train()
        
        for images, features, labels in train_loader:
            images = images.float().to(device)
            features = features.float().to(device)

            labels = labels.to(device)
            
            optimizer.zero_grad()
    
            outputs = model(images, features)
    
            loss = criterion(outputs, labels)
    
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)

            with torch.no_grad():
                _, predictions = torch.max(outputs, 1)
                
                label_output_matches += (predictions == labels).sum().item()
            
            progress_bar_train.update(1)
        
        progress_bar_train.close()
        
        acc = label_output_matches / total_steps_training
        avg_loss = total_loss / total_steps_training_loader

        print ('Epoch [{}/{}], Total Training Loss: {:.4f}, Avg. Loss: {:.4f}, ACC: {:.4f}'.format(epoch+1, num_epochs, total_loss, avg_loss, acc))
        
        losses[epoch] = total_loss

        # Validation part
        total_loss_val = 0
        
        progress_bar_val = tqdm(total=len(validation_loader))
        
        model.eval()
        
        with torch.no_grad():
            for images, features, labels in validation_loader:
                images = images.float().to(device)
                features = features.float().to(device)
                labels = labels.to(device)
        
                outputs =  model(images, features)
        
                loss = criterion(outputs, labels)
        
                total_loss_val += loss.item() * images.size(0)

                outputs_softmax =  torch.softmax(outputs, dim=1)

                _, predictions = torch.max(outputs_softmax, 1)
                
                label_output_matches_val += (predictions == labels).sum().item()

                progress_bar_val.update(1)
        
        progress_bar_val.close()

        acc_val = label_output_matches_val / total_steps_val
        avg_loss_val = total_loss_val / total_steps_val_loader
        
        print ('Epoch [{}/{}], Total Validation Loss: {:.4f}, Avg. Val Loss: {:.4f}, VAL ACC: {:.4f}'.format(epoch+1, num_epochs, total_loss_val, avg_loss_val, acc_val))
        
        validation_losses[epoch] = total_loss_val
    
        if total_loss_val < min_val_loss:
            logging.info("Saving new best model.")
            
            torch.save(model.state_dict(), f"{saving_directory_networks}/network_val_{int(time.time())}__loss_{total_loss}__val_loss_{total_loss_val}.pth")
            
            min_val_loss = total_loss_val

        if save_frequency != -1 and ((epoch + 1) % save_frequency == 0):
            logging.info("Saving model.")
            
            torch.save(model.state_dict(), f"{saving_directory_networks}/network_{int(time.time())}.pth")
            torch.save(optimizer.state_dict(), f"{saving_directory_networks}/optimizer_{int(time.time())}.pth")

    logging.info("Training DONE.")

    logging.info("Saving model.")
            
    torch.save(model.state_dict(), f"{saving_directory_networks}/network_final_{int(time.time())}.pth")
    torch.save(optimizer.state_dict(), f"{saving_directory_networks}/network_final_{int(time.time())}_optimizer.pth")

    logging.info("Plotting results.")

    epochs = range(1, num_epochs + 1)

    plt.title("Losses normed")
    plt.plot(epochs, [l / total_steps_training for l in losses], "c-")
    plt.plot(epochs, [l / total_steps_val for l in validation_losses], "-", color="orange")
    plt.legend(["Loss", "Val Loss"])
    
    plt.savefig("training_losses_normed.png")
    plt.clf()

    plt.title("Losses")
    plt.plot(epochs, losses, "c-")
    plt.plot(epochs, validation_losses, "-", color="orange")
    plt.legend(["Loss", "Val Loss"])
    
    plt.savefig("training_losses.png")
    #plt.clf()

def test():
    logging.info("Entering test mode.")
    
    test_dir_in = config["datasets"]["test"]["dir_inputs"]
    test_features_in = config["datasets"]["test"]["dir_features"]
    model_checkpoint_path = config["modes"]["test"]["checkpoint"]
    tag = config["modes"]["test"]["tag"]
    
    logging.info(f"Loading model '{model_checkpoint_path}'")
    model.load_state_dict(torch.load(model_checkpoint_path))

    test_dataset = ImageDataset(img_dir=test_dir_in, class_names=CLASSES, features_dir=test_features_in, feature_scaler_path=f"data/training/feature_scaler_{feature_scaler_type}_{problem_type}.pkl", transform_main=transforms_main_eval)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=batch_size, 
                                            shuffle=False)
    
    logging.info("Running test.")

    progress_bar_val = tqdm(total=len(test_loader))
        
    model.eval()

    total_steps = len(test_dataset)
    
    results_table = pd.DataFrame([], columns=["label", "prediction"])
    table_idx = 0

    with torch.no_grad():
        for images, features, labels in test_loader:
            images = images.float().to(device)
            features = features.float().to(device)
            labels = labels.to(device)
            
            outputs =  torch.softmax(model(images, features), dim=1)

            _, predictions = torch.max(outputs, 1)

            for label, prediction in zip(labels.data.cpu().numpy(), predictions.data.cpu().numpy()):
                results_table.loc[table_idx] = [label, prediction]
                table_idx += 1

            progress_bar_val.update(1)
        
    progress_bar_val.close()

    y_true = results_table.label.values.astype(int)
    y_pred = results_table.prediction.values.astype(int)

    results_table.to_json(f"results_table.json", orient="records", force_ascii=False)

    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true=y_true, y_pred=y_pred)

    FP = cm.sum(axis=0) - np.diag(cm)
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)

    print("FP", FP)
    print("FN", FN)
    print("TP", TP)
    print("TN", TN)

    ACC2 = (TP + TN) / (TP + FP + FN + TN) 
    weigths_test = [(y_true == idx).sum() / len(y_true) for idx, c in enumerate(CLASSES)]

    print("ACC weighted", sum([a * weigths_test[idx] for idx, a in enumerate(ACC2)]))

    results_table.to_json("results_table.json")

    logging.info("DONE.")

if mode == Mode.HPTUNING.value:
    logging.info("Entering hyperparameter tuning mode.")

    hyperparamtuning()
elif mode == Mode.TRAIN.value:
    logging.info("Entering training mode.")
    
    train()
elif mode == Mode.TEST.value:
    logging.info("Entering test mode.")

    test()
else:
    logging.error(f"Unknown mode '{mode}'")
