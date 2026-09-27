import torch
from model import ResNet
from DataLoading import load_data
from utils import eval_step

if __name__ == "__main__":
    EPOCHS = 8

    train_loader, test_loader = load_data(32)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = ResNet(3, 64, classes=10).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optim = torch.optim.Adam(params=model.parameters(), lr=0.00011)


    for epoch in range(EPOCHS):
        model.train()
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)


            optim.zero_grad()

            logits = model(images)

            loss = criterion(logits, labels)

            loss.backward()

            optim.step()

            if batch_idx % 100 == 0:
                print(f"Batch: {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")

        test_loss, test_acc = eval_step(model, test_loader, criterion, device)
        print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")

    torch.save(model.state_dict(), 'model.pth')


