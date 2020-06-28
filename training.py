import math
import numpy as np
import os
import datetime
import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, datasets
from net import Net
from itertools import takewhile
import matplotlib.pyplot as plt
from whitening import Whitening
from random import randint
from config import Config
from torchvision.datasets import ImageFolder
from helper import calcMeanStdWhitenMatrixMeanVec


class Training:
    def __init__(self, learning_rate=0.1, momentum=0.0, weight_decay=0.0, data_dir="data", savepoint_dir="savepoints", no_cuda=False, prep_dir="cache", batch_size=20, max_savepoints=20, num_workers=2,  sp_serial=-1, no_save_savepoints=False, no_save_prep=False):
        self.prep_dir = prep_dir
        self.savepoint_dir = savepoint_dir
        self.no_save_prep = no_save_prep
        self.no_save_savepoint = no_save_savepoint
        self.sp_serial = sp_serial
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_savepoints = max_savepoints
        self.num_workers = num_workers
        self.data_dir = data_dir
        self.dataset = ImageFolder(root=self.data_dir)
        # sample_idx = 5000
        # plt.imshow(self.dataset[sample_idx][0])
        plt.show()
        mean, std, whiten_matrix, mean_vec = self._loadOrCalcTransformValues()
        self.transforms = transforms.Compose([
            transforms.Grayscale(1),
            transforms.RandomAffine(0, translate=(.1, .1)),
            transforms.ToTensor(),
            transforms.Normalize((mean,), (std,)),
            transforms.LinearTransformation(whiten_matrix, mean_vec)
        ])
        self.dataset.transform = self.transforms
        # (img, label) = self.dataset[sample_idx]
        # label = self.dataset.classes[label]
        # plt.title(label)
        # plt.imshow(transforms.functional.to_pil_image(img*130), "gray")
        # plt.show()
        # exit()
        self.net = Net(num_classes=len(self.dataset.classes))

        if (not no_cuda) and torch.cuda.is_available():
            self.net.cuda()
            self.device = "cuda"
            print(f"Device :: CUDA {torch.cuda.get_device_name()}")
        else:
            self.device = "cpu"
            print(f"Device :: CPU")

        # TODO: dynamic learning rate

        # Define optimizer AFTER device is set
        self.optimizer = optim.RMSprop(self.net.parameters(
        ), lr=self.learning_rate, momentum=self.momentum, weight_decay=self.weight_decay)
        self.criterion = torch.nn.CrossEntropyLoss()

        # load savepoints if available
        savepoints = os.listdir(self.savepoint_dir) if os.path.isdir(
            self.savepoint_dir) else []
        if not savepoints == []:
            self._loadSavepoint(savepoints)
        else:
            self.epoch = 0
            self.current_loss = None
            print("No savepoints found!")

        self.trainloader = torch.utils.data.DataLoader(
            self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        self.testloader = torch.utils.data.DataLoader(
            self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def train(self, epochs=1, evaluate=True, print_per_batches=None):
        print("Starting training!")
        self.net.train()

        target_epoch = epochs+self.epoch
        # for each epoch
        while self.epoch <= target_epoch:
            self.epoch += 1
            print(f"Epoch: {self.epoch} / {target_epoch}.")
            running_loss = 0.0

            # for each batch
            for i, data in enumerate(self.trainloader):
                inputs, targets = data

                if self.device == "cuda":
                    inputs = inputs.cuda()
                    targets = targets.cuda()

                # run batch through net and calculate loss
                outputs = self.net(inputs)
                loss = self.criterion(outputs, targets)

                # FOR DEBUGGING
                if math.isnan(loss.item()):
                    print(" ############# Loss is NaN #############")
                    print("Outputs: ")
                    print(outputs)
                    print("Loss: ")
                    print(loss)
                    exit(-1)

                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                if print_per_batches != None and i % print_per_batches == print_per_batches-1:
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / self.print_per))
                    running_loss = 0.0
            self.current_loss = running_loss
            self._makeSavepoint()
        print("Finished training!")
        if evaluate:
            self.evaluate()

    def _loadSavepoint(self, savepoints):
        if not os.path.isdir(self.savepoint_dir):
            return
        target_file = None
        ser_files = self._getSavepointList()
        if len(ser_files) == 0:
            print("No existing savepoints!")
            return

        if self.sp_serial > -1:
            for n, f in ser_files:
                if n == self.sp_serial:
                    target_file = f
        else:
            self.sp_serial, target_file = ser_files[-1]

        print(f"Loading progress from {target_file}!")
        checkpoint = torch.load(os.path.join(self.savepoint_dir, target_file))
        self.net.load_state_dict(checkpoint["net_state_dict"])
        self.optimizer.load_tate_dict(checkpoint["optimizer_state_dict"])
        self.epoch = checkpoint["epoch"]
        self.current_loss = checkpoint["current_loss"]
        self.net.eval()

    def _makeSavepoint(self):
        if self.no_save_savepoint:
            return
        if not os.path.isdir(self.savepoint_dir):
            os.mkdir(self.savepoint_dir)
        target_path = os.path.join(
            self.savepoint_dir, self._getNextSavepointPath())
        print(f"Saving progress in {target_path}!")
        torch.save({
            "net_state_dict": self.net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": self.epoch,
            "current_loss": self.current_loss
        }, target_path)
        self._removeOldSavepoints()

    def _getSavepointList(self):
        # only look @ .pt and .pth files
        path_files = [f for f in os.listdir(self.savepoint_dir) if f[-4:]
                      == ".pth" or f[-3:] == ".pt"]
        # parse serial number
        ser_files = [(int(''.join([t for t in takewhile(lambda x: x != '_', f)])), f)
                     for f in path_files]
        # sort in place
        ser_files.sort()
        return ser_files

    def _getNextSavepointPath(self):
        sn = self.sp_serial + 1
        fn = "%03d_savepoint.pth" % sn
        current_files = os.listdir(self.savepoint_dir)
        while fn in current_files:
            sn = sn + 1
            fn = "%03d_savepoint.pth" % sn
        self.sp_serial = sn
        return fn

    def _removeOldSavepoints(self):
        files = self._getSavepointList()
        # files :: [(sn :: Int, path :: String)] sorted
        while len(files) > self.max_savepoints:
            t = files[0][1]
            os.remove(os.path.join(self.savepoint_dir, t))
            print(
                f"Removing old savepoint: {t}")
            files = files[1:]

    def evaluate(self):
        self.net.eval()
        correct = 0
        total = 0
        l = len(self.dataset)
        correct_class = [0 for _ in range(l)]
        total_class = [0 for _ in range(l)]
        with torch.no_grad():
            for data in self.testloader:
                # for data in tqdm(self.testloader):
                inputs, targets = data
                if self.device == "cuda":
                    inputs = inputs.cuda()
                    targets = targets.cuda()
                outputs = self.net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                for i, t in enumerate(targets):
                    total_class[t] += 1
                    if predicted[i] == t:
                        correct_class[t] += 1
                correct += (predicted == targets).sum().item()

        print("Accuracy of the network on %d test images: %d %%  -  %d / %d" %
              (total, 100 * correct / total, correct, total))
        for i, c in enumerate(correct_class):
            t = total_class[i]
            print("Accuracy of class %s is %d %%  -   %d/%d" %
                  (self.dataset.classes[i], c / t * 100, c, t))

    def _loadOrCalcTransformValues(self):
        filepath = os.path.join(self.data_dir, "prep.pt")
        if os.path.isfile(filepath):
            tf_val_dict = torch.load(filepath)
            return tf_val_dict["mean"], tf_val_dict["std"], tf_val_dict["whiten_matrix"], tf_val_dict["mean_vector"]
        else:
            mean, std, whiten_matrix, mean_vector = calcMeanStdWhitenMatrixMeanVec(
                self.dataset)
            if not self.no_save_prep:
                torch.save({
                    "mean": mean,
                    "std": std,
                    "whiten_matrix": whiten_matrix,
                    "mean_vector": mean_vector
                }, filepath)
                print(f"Saved prep values to {filepath}")
            return mean, std, whiten_matrix, mean_vector
