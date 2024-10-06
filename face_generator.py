import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def train_discriminator(discriminator, real_data, fake_data, optimizer, loss_fn):
    optimizer.zero_grad()

    real_data = real_data.view(-1, 1, 32, 32)
    real_preds = discriminator(real_data)
    real_loss = loss_fn(real_preds, torch.ones_like(real_preds) * 0.9)

    fake_preds = discriminator(fake_data)
    fake_loss = loss_fn(fake_preds, torch.zeros_like(fake_preds) + 0.1)

    total_loss = real_loss + fake_loss
    total_loss.backward()
    optimizer.step()

    return total_loss.item()


def train_generator(generator, discriminator, optimizer, loss_fn):
    optimizer.zero_grad()

    noise = torch.randn(64, 1, 32, 32).to(device)
    fake_data = generator(noise)

    fake_preds = discriminator(fake_data)

    loss = loss_fn(fake_preds, torch.ones_like(fake_preds))

    loss.backward()
    optimizer.step()

    return loss.item()


def evaluate_generator(generator, discriminator):
    noise = torch.randn(64, 1, 32, 32).to(device)
    fake_data = generator(noise)
    fake_preds = discriminator(fake_data)
    return fake_preds.mean().item()


def show_images(generator):
    noise = torch.randn(16, 1, 32, 32).to(device)
    fake_images = generator(noise)

    # Tworzenie siatki obrazów
    grid_img = torchvision.utils.make_grid(fake_images, nrow=4, normalize=True)

    grid_img = grid_img.cpu().numpy()  # Kopiowanie do pamięci hosta i konwersja

    # Wyświetlanie obrazu
    plt.imshow(grid_img.transpose(1, 2, 0))
    plt.axis('off')  # Ukrycie osi
    plt.show()


transform = transforms.Compose([
    transforms.Grayscale(),  # Konwersja do skali szarości
    transforms.Resize((32, 32)),  # Zmiana rozmiaru obrazów do 32x32
    transforms.ToTensor(),  # Konwersja do tensora
    transforms.Normalize((0.5,), (0.5,))  # Normalizacja do zakresu [-1, 1]
])

dataset = datasets.ImageFolder(root='data/yale', transform=transform)
data_loader = DataLoader(dataset, batch_size=64, shuffle=True)

generator = Generator().to(device)

# Inicjalizacja dyskryminatora
discriminator = Discriminator().to(device)

d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))
g_optimizers = optim.Adam(generator.parameters(), lr=0.0001, betas=(0.5, 0.999))

loss_fn = nn.BCELoss()
g_losses = []
d_losses = []

for epoch in range(1000):  # Liczba epok treningowych
    g_epoch_loss = 0.0
    d_epoch_loss = 0.0
    for real_images, _ in data_loader:
        real_images = real_images.to(device)
        # Trenuj dyskryminator
        if epoch % 2 == 0:
            noise = torch.randn(64, 1, 32, 32).to(device)
            fake_images = generator(noise).detach()
            d_loss = train_discriminator(discriminator, real_images, fake_images, d_optimizer, loss_fn)
            d_epoch_loss += d_loss  # Zaktualizuj stratę dyskryminatora

        # Trenuj generator
        fitness_scores = []
        g_loss = train_generator(generator, discriminator, g_optimizers, loss_fn)
        g_epoch_loss += g_loss  # Zaktualizuj stratę generatora
        fitness_score = evaluate_generator(generator, discriminator)
        fitness_scores.append(fitness_score)

        #print(fitness_scores)

    # Zapisz średnie straty dla generatora i dyskryminatora w tej epoce
    g_losses.append(g_epoch_loss / len(data_loader))
    d_losses.append(d_epoch_loss / len(data_loader))

    print(f'Epoch {epoch + 1} has ended.')
    if epoch > 300 and epoch % 20 == 0:
        show_images(generator)
