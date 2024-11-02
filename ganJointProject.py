import os
import glob
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
import matplotlib.pyplot as plt

# Ana klasörün dosya yolu - File path of the main folder
baseFolder = "C:\\Users\\Sena\\Desktop\\flowers"  

# Creating lists to store images and labels - Görselleri ve etiketleri depolamak için listeler oluşturulur
images = [] 
labels = []

# Her çiçek türü klasöründeki tüm .jpg dosyaları açılır, yeniden boyutlandırılır ve listeye eklenir
# Open all .jpg files in each flower type folder, resize them, and add to the list
for flowerType in os.listdir(baseFolder):
    folderPath = os.path.join(baseFolder, flowerType)
    if os.path.isdir(folderPath):  
        for imgFile in glob.glob(os.path.join(folderPath, "*.jpg")):
            img = Image.open(imgFile).resize((128, 128))  
            imgArray = np.array(img) / 255.0  # Normalleştirme - Normalization
            
            images.append(imgArray)
            labels.append(flowerType)

# Listeler NumPy dizilerine dönüştürülür - Convert lists to NumPy arrays
images = np.array(images)
labels = np.array(labels)

print("Total number of images:", len(images))
print("Image size:", images[0].shape)

# Görselleri tensöre dönüştürmek ve DataLoader'da kullanmak için Dataset sınıfı tanımlanır
# Define Dataset class to convert images to tensors and use them in DataLoader
class FlowerDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Transform object to convert images to tensors - Görselleri tensöre dönüştüren transform nesnesi
transform = transforms.Compose([
    transforms.ToTensor(),  
])

# Dataset ve DataLoader nesneleri oluşturulur - Creating Dataset and DataLoader objects
flowerDataset = FlowerDataset(images, labels, transform=transform)
dataLoader = DataLoader(flowerDataset, batchSize=32, shuffle=True)

# İlk batch'i kontrol etmek için - To check the first batch
for batchImages, batchLabels in dataLoader:
    print("Batch size:", batchImages.size())
    break

# Generator sınıfı tanımlanır - Define Generator class
class Generator(nn.Module):
    def __init__(self, z_dim=256, img_dim=128*128*3): 
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 256),         
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),          
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, img_dim),      
            nn.Tanh(),  
        )

    def forward(self, x):
        return self.gen(x)

# Discriminator sınıfı tanımlanır - Define Discriminator class
class Discriminator(nn.Module):
    def __init__(self, img_dim=128*128*3):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            nn.Linear(img_dim, 512),       
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),           
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),            
            nn.Sigmoid(),
        ) 

    def forward(self, x):
        return self.disc(x)


# Cihaz ayarlama - Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
lrGen = 0.0001  
lrDis = 0.00005  
z_dim = 256 
img_dim = 128*128*3  

# Model örnekleri - Create model instances
gen = Generator(z_dim, img_dim).to(device)
disc = Discriminator(img_dim).to(device)

# Optimizasyon işlemleri - Optimization processes
genOpt = torch.optim.Adam(gen.parameters(), lr=lrGen, betas=(0.5, 0.999))
discOpt = torch.optim.Adam(disc.parameters(), lr=lrDis, betas=(0.5, 0.999))

# Kayıp fonksiyonu - Loss function
criterion = nn.BCELoss()

# Eğitim döngüsü parametreleri - Training loop parameters
epochs = 400

def show_generated_images(images):
    images = (images + 1) / 2  
    plt.figure(figsize=(10, 10))
    for i in range(min(9, images.size(0))):  
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].cpu().detach().numpy().transpose(1, 2, 0))  
        plt.axis('off')
    plt.show()

for epoch in range(epochs):
    for real, _ in dataLoader:
        real = real.view(-1, img_dim).float().to(device)

        batchSize = real.size(0)
        
        # Gerçek ve sahte etiketleri oluşturma - Generate real and fake labels
        realLabels = torch.ones(batchSize, 1).to(device)
        fakeLabels = torch.zeros(batchSize, 1).to(device)
        
        # Discriminator Eğitimi - Train Discriminator
        noise = torch.randn(batchSize, z_dim).to(device)
        fakeImages = gen(noise)
        
        discReal = disc(real)
        discFake = disc(fakeImages.detach())
        
        lossReal = criterion(discReal, realLabels)
        lossFake = criterion(discFake, fakeLabels)
        
        discLoss = (lossReal + lossFake) / 2
        discOpt.zero_grad()
        discLoss.backward()
        discOpt.step()
        
        # Generator Eğitimi - Train Generator
        output = disc(fakeImages)
        genLoss = criterion(output, realLabels)
        
        genOpt.zero_grad()
        genLoss.backward()
        genOpt.step()
        
    print(f"Epoch [{epoch+1}/{epochs}] | Generator Loss: {genLoss:.4f} | Discriminator Loss: {discLoss:.4f}")

    # Her 10 epoch'ta oluşturulan görüntüleri göster - Show generated images every 10 epochs
    if (epoch + 1) % 10 == 0:
        show_generated_images(fakeImages.view(-1, 3, 128, 128)) 