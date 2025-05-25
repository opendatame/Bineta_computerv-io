from torchvision import transforms

def get_train_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),  # taille modifiée
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

def get_test_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),  # taille modifiée
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
