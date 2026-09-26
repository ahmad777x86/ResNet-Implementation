import torch
import torchvision.transforms as transforms

def eval_step(model, test_loader, criterion, device):
    model.eval()

    total = 0
    correct = 0
    test_loss = 0

    for (images, labels) in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        logits = model(images)

        loss = criterion(logits, labels)

        test_loss += loss.item() * images.size(0)

        total += labels.size(0)

        preds = torch.argmax(logits, dim=1)

        correct += torch.eq(preds, labels).sum().item()

    test_loss = test_loss / total
    test_acc = correct / total * 100

    return test_loss, test_acc

def preprocess_image(img):
    tf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(32),
        transforms.CenterCrop((32,32)),
        transforms.ToTensor()
        ])
    img = tf(img)
    return img