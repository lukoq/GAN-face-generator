import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

CHANNEL = 1
SHAPE = 32
BATCH_SIZE = 64
EPOCH = 420
DATASET_PATH = 'data/yale'


def train_discriminator(discriminator, real_data, fake_data, optimizer, loss_fn):
    optimizer.zero_grad()

    real_data = real_data.view(-1, CHANNEL, SHAPE, SHAPE)
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

    noise = torch.randn(BATCH_SIZE, CHANNEL, SHAPE, SHAPE).to(device)
    fake_data = generator(noise)

    fake_preds = discriminator(fake_data)

    loss = loss_fn(fake_preds, torch.ones_like(fake_preds))

    loss.backward()
    optimizer.step()

    return loss.item()


def evaluate_generator(generator, discriminator):
    noise = torch.randn(BATCH_SIZE, CHANNEL, SHAPE, SHAPE).to(device)
    fake_data = generator(noise)
    fake_preds = discriminator(fake_data)
    return fake_preds.mean().item()


def wasserstein_loss(y_pred, y_true):
    return torch.mean(y_true * y_pred)


def show_images(generator):
    for _ in range(1):
        noise = torch.randn(16, CHANNEL, SHAPE, SHAPE).to(device)
        fake_images = generator(noise)

        grid_img = torchvision.utils.make_grid(fake_images, nrow=4, normalize=True)
        grid_img = grid_img.cpu().numpy()

        plt.imshow(grid_img.transpose(1, 2, 0))
        plt.axis('off')
        plt.show()


def plot_losses(g_losses, d_losses):
    epochs = range(1, len(g_losses) + 1)

    plt.plot(epochs, g_losses, label='Generator Loss')
    plt.plot(epochs, d_losses, label='Discriminator Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Generator and Discriminator Loss')
    plt.legend()

    plt.show()


transform = transforms.Compose([
    # transforms.Grayscale(),
    transforms.Resize((SHAPE, SHAPE)),
    transforms.ToTensor(),
])

dataset = datasets.ImageFolder(root=DATASET_PATH, transform=transform)
data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

if SHAPE > 32:
    generator = EnhancedGenerator(CHANNEL).to(device)
    discriminator = EnhancedDiscriminator(CHANNEL).to(device)
else:
    generator = Generator(CHANNEL).to(device)
    discriminator = Discriminator(CHANNEL).to(device)

d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))
g_optimizers = optim.Adam(generator.parameters(), lr=0.0001, betas=(0.5, 0.999))

loss_fn = nn.BCELoss()
g_losses = []
d_losses = []

for epoch in range(EPOCH):
    g_epoch_loss = 0.0
    d_epoch_loss = 0.0
    for real_images, _ in data_loader:
        real_images = real_images.to(device)
        if epoch % 2 == 0:
            noise = torch.randn(BATCH_SIZE, CHANNEL, SHAPE, SHAPE).to(device)
            fake_images = generator(noise).detach()
            d_loss = train_discriminator(discriminator, real_images, fake_images, d_optimizer, loss_fn)
            d_epoch_loss += d_loss

        g_loss = train_generator(generator, discriminator, g_optimizers, loss_fn)
        g_epoch_loss += g_loss

    g_losses.append(g_epoch_loss / len(data_loader))
    d_losses.append(d_epoch_loss / len(data_loader))

    fitness_scores = []
    fitness_score = evaluate_generator(generator, discriminator)
    fitness_scores.append(fitness_score)


    print(f'Epoch {epoch + 1} has ended.')
    if epoch % 50 == 0:  # Monitor progress after every 50 epochs
        plot_losses(g_losses, d_losses)
        show_images(generator)
