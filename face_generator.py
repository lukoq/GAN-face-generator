import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class Generator(nn.Module):
    def __init__(self, input_dim):
        super(Generator, self).__init__()
        self.input_dim = input_dim

        self.network = nn.Sequential(
            # Warstwa w pełni połączona: przekształcenie wejścia do 7x7x256
            nn.Linear(input_dim, 7 * 7 * 256),
            # Przekształcenie tensora do wymiarów (batch_size, 256, 7, 7)
            nn.Unflatten(dim=1, unflattened_size=(256, 7, 7)),

            # Warstwa transponowanej konwolucji: (256, 7, 7) -> (128, 14, 14)
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),

            # Warstwa transponowanej konwolucji: (128, 14, 14) -> (64, 28, 28)
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),

            # Piąta warstwa konwolucji: Dodatkowa konwolucja dla większej złożoności (64, 28, 28) -> (32, 28, 28)
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32, 0.8),
            nn.LeakyReLU(0.2, inplace=True),


            # Ostatnia warstwa transponowanej konwolucji: (32, 28, 28) -> (1, 28, 28)
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=1, padding=1),
            nn.Tanh()  # Normalizacja do zakresu [-1, 1]
        )

    def forward(self, x):
        x = self.network(x)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),  # 1 kanał (szarość) -> 64 kanały
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 64 -> 128
            nn.LeakyReLU(0.2),

            # Dodana nowa warstwa konwolucyjna
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 128 -> 256 kanałów
            nn.LeakyReLU(0.2),

            # Czwarta warstwa konwolucyjna (opcjonalna dla zwiększenia głębokości)
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),  # 4x4 -> 2x2
            nn.LeakyReLU(0.2),

            nn.Flatten(),  # Spłaszczenie wyjścia
            nn.Linear(512 * 2 * 2, 1),  # 7x7 po konwolucjach
            nn.Sigmoid()  # Wartość prawdopodobieństwa
        )

    def forward(self, x):
        return self.network(x)


def train_discriminator(discriminator, real_data, fake_data, optimizer, loss_fn):
    optimizer.zero_grad()

    # Przekonwertuj rzeczywiste dane do formatu 4D
    real_data = real_data.view(-1, 1, 28, 28)  # (batch_size, channels, height, width)

    # Prawdziwe dane
    real_preds = discriminator(real_data)
    real_loss = loss_fn(real_preds, torch.ones_like(real_preds) * 0.9)  # Label smoothing dla prawdziwych danych

    # Fałszywe dane
    fake_preds = discriminator(fake_data)
    fake_loss = loss_fn(fake_preds, torch.zeros_like(fake_preds) + 0.1)  # Label smoothing dla fałszywych danych

    # Zsumowanie strat i optymalizacja
    total_loss = real_loss + fake_loss
    total_loss.backward()
    optimizer.step()

    return total_loss.item()


def train_generator(generator, discriminator, optimizer, loss_fn):
    optimizer.zero_grad()

    # Wygenerowanie fałszywych danych
    noise = torch.randn(64, 100).to(device)  # Losowy wektor o wymiarach 100
    fake_data = generator(noise)  # Fałszywe dane o wymiarach (64, 1, 28, 28)

    # Dyskryminator ocenia fałszywe dane
    fake_preds = discriminator(fake_data)

    # Chcemy, by dyskryminator uznał fałszywe dane za prawdziwe
    loss = loss_fn(fake_preds, torch.ones_like(fake_preds))

    # Optymalizacja generatora
    loss.backward()
    optimizer.step()

    return loss.item()


# Funkcja ewaluacji generatora (fitness)
def evaluate_generator(generator, discriminator):
    noise = torch.randn(64, 100).to(device)
    fake_data = generator(noise)
    fake_preds = discriminator(fake_data)

    return fake_preds.mean().item()


def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)


def show_images(generator):
    noise = torch.randn(16, 100).to(device)
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
    transforms.Resize((28, 28)),  # Zmiana rozmiaru obrazów do 28x28
    transforms.ToTensor(),  # Konwersja do tensora
    transforms.Normalize((0.5,), (0.5,))  # Normalizacja do zakresu [-1, 1]
])

dataset = datasets.ImageFolder(root='data/faces', transform=transform)
data_loader = DataLoader(dataset, batch_size=64, shuffle=True)

generator = Generator(100).to(device)

# Inicjalizacja dyskryminatora
discriminator = Discriminator().to(device)

d_optimizer = optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.9, 0.999))
g_optimizers = optim.Adam(generator.parameters(), lr=1e-4, betas=(0.9, 0.999))

loss_fn = nn.BCELoss()
g_losses = []
d_losses = []

for epoch in range(150):  # Liczba epok treningowych
    g_epoch_loss = 0.0
    d_epoch_loss = 0.0
    for real_images, _ in data_loader:
        real_images = real_images.to(device)
        # Trenuj dyskryminator
        if epoch % 3 == 0:
            noise = torch.randn((64, 100)).to(device)
            fake_images = generator(noise).detach()
            d_loss = train_discriminator(discriminator, real_images, fake_images, d_optimizer, loss_fn)
            d_epoch_loss += d_loss  # Zaktualizuj stratę dyskryminatora

        # Trenuj generator
        fitness_scores = []
        g_loss = train_generator(generator, discriminator, g_optimizers, loss_fn)
        g_epoch_loss += g_loss  # Zaktualizuj stratę generatora
        fitness_score = evaluate_generator(generator, discriminator)
        fitness_scores.append(fitness_score)

        # print(fitness_scores)

    # Zapisz średnie straty dla generatora i dyskryminatora w tej epoce
    g_losses.append(g_epoch_loss / len(data_loader))
    d_losses.append(d_epoch_loss / len(data_loader))

    print(f'Epoch {epoch + 1} has ended.')

# Wyświetlanie najlepszego generatora
show_images(generator)
