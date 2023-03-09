from customModel import CNNClassifier, save_model, load_model
from utils import accuracy, load_data
import torch
from torchvision import models
from torch.utils.tensorboard import SummaryWriter
import os


def train(args):
    model = CNNClassifier()
    dirname = os.path.join("runs", args.tbdir)
    tb = SummaryWriter(log_dir=dirname)

    # --- Initializations ---
    model = CNNClassifier() #models.resnet18(pretrained=True)
    model = load_model()

    # # --- Freeze layers and replace FC layer ---
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         if "fc" not in name:
    #             param.requires_grad = False
    # model.fc = torch.nn.Linear(512, 3)

    # Potential GPU optimization.
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.to(device)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lrate)
    criterion = torch.nn.MSELoss()
    train_loader = load_data("/home/asf170004/data/customData/train")
    validation_loader = load_data("/home/asf170004/data/customData/valid")

    # --- SGD Iterations ---
    for epoch in range(args.epochs):
        
        print("Starting Epoch: ", epoch)
        if (epoch%50 == 0):
            save_model(model)

        # Per epoch train loop.
        model.train()
        for _, (rgb_input, sem_input, yhat) in enumerate(train_loader):
            yhat = yhat.view(-1, 3, 1)
            yhat = yhat.cuda()
            optimizer.zero_grad()
            sem_input.to(device)
            ypred = model(sem_input.cuda())
            loss = criterion(ypred, yhat)
            loss.backward()
            optimizer.step()

            # Record training loss and accuracy
            tb.add_scalar("Train Loss", loss, epoch)
            steer, throttle, brake, average = accuracy(ypred, yhat)
            tb.add_scalar("Train Accuracy", average, epoch)

            tb.add_scalar("Steer Accuracy", steer, epoch)
            tb.add_scalar("Throttle Accuracy", throttle, epoch)
            tb.add_scalar("Brake Accuracy", brake, epoch)

        # After each train epoch, do validation before starting next train epoch.
        model.eval()
        for _, (rgb_input, sem_input, yhat) in enumerate(validation_loader):
            yhat = yhat.view(-1, 3, 1)
            yhat = yhat.cuda()
            with torch.no_grad():
                sem_input.to(device)
                ypred = model(sem_input.cuda())
                loss = criterion(ypred, yhat)

            # Record validation loss and accuracy
            tb.add_scalar("Validation Loss", loss, epoch)
            steer, throttle, brake, average = accuracy(ypred, yhat)
            tb.add_scalar("Validation Accuracy", average, epoch)
            tb.add_scalar("Steer Accuracy", steer, epoch)
            tb.add_scalar("Throttle Accuracy", throttle, epoch)
            tb.add_scalar("Brake Accuracy", brake, epoch)

    save_model(model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--lrate', type=float, default=0.001)
    parser.add_argument('-e', '--epochs', type=int, default=450)
    parser.add_argument('-t', '--tbdir', type=str, default="")

    args = parser.parse_args()
    train(args)
