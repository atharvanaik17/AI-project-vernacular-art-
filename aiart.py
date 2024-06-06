import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import itertools
import multiprocessing


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_features, in_features, 3, stride=1, padding=1),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_features, in_features, 3, stride=1, padding=1),
            nn.InstanceNorm2d(in_features)
        )

    def forward(self, x):
        return x + self.block(x)

class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        super(Generator, self).__init__()
        
        model = [
            nn.Conv2d(input_nc, 64, 7, padding=3),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        ]
        
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features * 2
      
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]
        
        out_features = in_features // 2
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features // 2
     
        model += [nn.Conv2d(64, output_nc, 7, padding=3), nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()
        model = [
            nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        model += [
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        model += [
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        model += [
            nn.Conv2d(256, 512, 4, stride=1, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        model += [nn.Conv2d(512, 1, 4, padding=1)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, mode='train'):
        self.transform = transforms.Compose(transforms_)
        self.files_A = sorted(os.listdir(os.path.join(root, f'{mode}/A')))
        self.files_B = sorted(os.listdir(os.path.join(root, f'{mode}/B')))
        self.root = root
        self.mode = mode

    def __getitem__(self, index):
        item_A = self.transform(Image.open(os.path.join(self.root, f'{self.mode}/A/{self.files_A[index % len(self.files_A)]}')))
        item_B = self.transform(Image.open(os.path.join(self.root, f'{self.mode}/B/{self.files_B[index % len(self.files_B)]}')))
        return {'A': item_A, 'B': item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
num_epochs = 200
batch_size = 1
learning_rate = 0.0002
size = 256
n_residual_blocks = 9
lambda_cyc = 10.0
lambda_id = 5.0

def train():
    
    transform = [
        transforms.Resize(int(size * 1.12), transforms.InterpolationMode.BICUBIC),
        transforms.RandomCrop(size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ]

    dataloader = DataLoader(ImageDataset(root='data', transforms_=transform), batch_size=batch_size, shuffle=True, num_workers=4)

    
    G_AB = Generator(3, 3, n_residual_blocks).to(device)
    G_BA = Generator(3, 3, n_residual_blocks).to(device)
    D_A = Discriminator(3).to(device)
    D_B = Discriminator(3).to(device)

   
    G_AB.apply(weights_init_normal)
    G_BA.apply(weights_init_normal)
    D_A.apply(weights_init_normal)
    D_B.apply(weights_init_normal)


    criterion_GAN = nn.MSELoss()
    criterion_cycle = nn.L1Loss()
    criterion_identity = nn.L1Loss()

   
    optimizer_G = optim.Adam(itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=learning_rate, betas=(0.5, 0.999))
    optimizer_D_A = optim.Adam(D_A.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    optimizer_D_B = optim.Adam(D_B.parameters(), lr=learning_rate, betas=(0.5, 0.999))

    # Training
    for epoch in range(num_epochs):
        for i, batch in enumerate(dataloader):
            real_A = batch['A'].to(device)
            real_B = batch['B'].to(device)

            
            output_shape = D_A(real_A).shape[1:]

            
            valid = torch.ones((real_A.size(0), *output_shape)).to(device)
            
            output_shape = D_A(real_A).shape[1:]
            fake = torch.zeros((real_A.size(0), *output_shape)).to(device)
            
           

            optimizer_G.zero_grad()

           
            loss_id_A = criterion_identity(G_BA(real_A), real_A)
            loss_id_B = criterion_identity(G_AB(real_B), real_B)
            loss_identity = (loss_id_A + loss_id_B) / 2

            
            fake_B = G_AB(real_A)
            loss_GAN_AB = criterion_GAN(D_B(fake_B), valid)
            fake_A = G_BA(real_B)
            loss_GAN_BA = criterion_GAN(D_A(fake_A), valid)
            loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2

            
            recovered_A = G_BA(fake_B)
            loss_cycle_A = criterion_cycle(recovered_A, real_A)
            recovered_B = G_AB(fake_A)
            loss_cycle_B = criterion_cycle(recovered_B, real_B)
            loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

            # Total loss
            loss_G = loss_GAN + lambda_cyc * loss_cycle + lambda_id * loss_identity
            loss_G.backward()
            optimizer_G.step()

           

            optimizer_D_A.zero_grad()

            loss_real = criterion_GAN(D_A(real_A), valid)
            loss_fake = criterion_GAN(D_A(fake_A.detach()), fake)
            loss_D_A = (loss_real + loss_fake) / 2

            loss_D_A.backward()
            optimizer_D_A.step()

           

            optimizer_D_B.zero_grad()

            loss_real = criterion_GAN(D_B(real_B), valid)
            loss_fake = criterion_GAN(D_B(fake_B.detach()), fake)
            loss_D_B = (loss_real + loss_fake) / 2

            loss_D_B.backward()
            optimizer_D_B.step()

         
            print(f"Epoch [{epoch}/{num_epochs}] Batch [{i}/{len(dataloader)}] "
                  f"Loss D_A: {loss_D_A.item()}, Loss D_B: {loss_D_B.item()}, "
                  f"Loss G: {loss_G.item()}")

 
    torch.save(G_AB.state_dict(), 'G_AB.pth')
    torch.save(G_BA.state_dict(), 'G_BA.pth')
    torch.save(D_A.state_dict(), 'D_A.pth')
    torch.save(D_B.state_dict(), 'D_B.pth')

if __name__ == "__main__":
    train()
