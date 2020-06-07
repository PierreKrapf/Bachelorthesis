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
# from tqdm.notebook import tqdm

MAX_SAVEPOINTS = 10
CLASSES = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


class Training:
    def __init__(self, lr=0.0001, momentum=0.0, weight_decay=0.0, savepoint_dir="savepoints", sp_serial=-1, no_cuda=False, batch_size=10, num_workers=2, print_per=2000, eval_only=False):
        self.eval_only = eval_only
        self.sp_serial = sp_serial
        self.savepoint_dir = savepoint_dir
        self.print_per = print_per
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.net = Net(classes=len(CLASSES))
        if (not no_cuda) and torch.cuda.is_available():
            self.net.cuda()
            self.device = "cuda"
            print(f"Device :: CUDA {torch.cuda.get_device_name()}")
        else:
            self.device = "cpu"
            print(f"Device :: CPU")

        # TODO: dynamic learning rate

        # Define optimizer AFTER device is set
        self.optimizer = optim.RMSprop(
            self.net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        self.criterion = torch.nn.CrossEntropyLoss()

        whiten_matrix = torch.load(os.path.join(
            "Drive", "My Drive", "zca_matrix.pt"))
        mean_vector = torch.load(os.path.join("Drive", "My Drive", "mean.pt"))

        self.transforms = transforms.Compose([
            transforms.Grayscale(1),
            transforms.RandomAffine(0, translate=(.1, .1)),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,)),
            transforms.LinearTransformation(
                transformation_matrix=whiten_matrix, mean_vector=mean_vector)
        ])

        # load savepoints if available
        savepoints = os.listdir(self.savepoint_dir) if os.path.isdir(
            self.savepoint_dir) else []
        if not savepoints == []:
            self._loadSavepoint(savepoints)
        else:
            print("No savepoints found!")

        # TODO: Use actual dataset
        # Using CIFAR10 to test
        self.trainset = datasets.CIFAR10(
            os.path.join("drive", "My Drive", "data"), train=True, download=True, transform=self.transforms)
        self.trainloader = torch.utils.data.DataLoader(
            self.trainset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

        self.testset = datasets.CIFAR10(
            os.path.join("drive", "My Drive", "data"), train=False, download=True, transform=self.transforms)
        self.testloader = torch.utils.data.DataLoader(
            self.testset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def run(self, epochs=1):
        if self.eval_only:
            return self.evaluate()
        while True:
            print("Starting training!")
            self.net.train()
            # for each epoch
            for epoch in range(epochs):
                print(f"Epoch {epoch+1} of {epochs}:")
                running_loss = 0.0

                # for each batch
                # for i, data in tqdm(enumerate(self.trainloader)):
                for i, data in enumerate(self.trainloader):
                    inputs, targets = data

                    if self.device == "cuda":
                        inputs = inputs.cuda()
                        targets = targets.cuda()
                    # Show first image for testing transforms
                    # for index, i in enumerate(inputs):
                    #     img = i.numpy()[0]
                    #     plt.imshow(img, cmap="gray")
                    #     plt.title(CLASSES[labels[index]])
                    #     plt.show()
                    # exit()

                    # run batch through net and calculate loss
                    outputs = self.net(inputs)
                    loss = self.criterion(outputs, targets)

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
                    if i % self.print_per == self.print_per-1:
                        print('[%d, %5d] loss: %.3f' %
                              (epoch + 1, i + 1, running_loss / self.print_per))
                        running_loss = 0.0
                self._makeSavepoint()
            print("Finished training!")
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
        self.net.load_state_dict(torch.load(
            os.path.join(self.savepoint_dir, target_file)))
        self.net.eval()

    def _makeSavepoint(self):
        if not os.path.isdir(self.savepoint_dir):
            os.mkdir(self.savepoint_dir)
        target_path = os.path.join(
            self.savepoint_dir, self._getNextSavepointPath())
        print(f"Saving progress in {target_path}!")
        torch.save(self.net.state_dict(), target_path)
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
        while len(files) > MAX_SAVEPOINTS:
            t = files[0][1]
            os.remove(os.path.join(self.savepoint_dir, t))
            print(
                f"Removing old savepoint: {t}")
            files = files[1:]

    def evaluate(self):
        self.net.eval()
        correct = 0
        total = 0
        correct_class = [0 for _ in range(len(CLASSES))]
        total_class = [0 for _ in range(len(CLASSES))]
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
                  (CLASSES[i], c / t * 100, c, t))
