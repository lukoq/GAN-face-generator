import torch
import torch.nn as nn

channel = 1


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # Warstwy enkodera
        self.encoder1 = self.encoder_block(channel, 64)  # 32x32 -> 16x16
        self.encoder2 = self.encoder_block(64, 128)  # 16x16 -> 8x8
        self.encoder3 = self.encoder_block(128, 256)  # 8x8 -> 4x4

        self.bottleneck = self.bottleneck_block(256, 512)  # 4x4 -> 2x2

        self.decoder3 = self.decoder_block(512, 256)  # 2x2 -> 4x4
        self.decoder2 = self.decoder_block(256, 128)  # 4x4 -> 8x8
        self.decoder1 = self.decoder_block(128, 64)  # 8x8 -> 16x16

        self.skip_conv3 = nn.Conv2d(512, 256, kernel_size=1)
        self.skip_conv2 = nn.Conv2d(256, 128, kernel_size=1)
        self.skip_conv1 = nn.Conv2d(128, 64, kernel_size=1)

        self.final_conv = nn.ConvTranspose2d(64, channel, kernel_size=4, stride=2, padding=1)  # 16x16 -> 32x32

    def encoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def bottleneck_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        # Enkoder
        enc1 = self.encoder1(x)  # 32x32 -> 16x16
        enc2 = self.encoder2(enc1)  # 16x16 -> 8x8
        enc3 = self.encoder3(enc2)  # 8x8 -> 4x4

        # Bottleneck
        bottleneck = self.bottleneck(enc3)  # 4x4 -> 2x2

        # Dekoder
        dec3 = self.decoder3(bottleneck)  # 2x2 -> 4x4
        dec3 = torch.cat((dec3, enc3), dim=1)  # Skip connection z enc3
        dec3 = self.skip_conv3(dec3)

        dec2 = self.decoder2(dec3)  # 4x4 -> 8x8
        dec2 = torch.cat((dec2, enc2), dim=1)  # Skip connection z enc2
        dec2 = self.skip_conv2(dec2)

        dec1 = self.decoder1(dec2)  # 8x8 -> 16x16
        dec1 = torch.cat((dec1, enc1), dim=1)  # Skip connection z enc1
        dec1 = self.skip_conv1(dec1)

        # Warstwa wyjściowa
        output = self.final_conv(dec1)  # 16x16 -> 32x32

        return output


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # Warstwy konwolucyjne
        self.conv1 = self.discriminator_block(channel, 64, batch_norm=False)  # 32x32 -> 16x16
        self.conv2 = self.discriminator_block(64, 128)  # 16x16 -> 8x8
        self.conv3 = self.discriminator_block(128, 256)  # 8x8 -> 4x4
        self.conv4 = self.discriminator_block(256, 512)  # 4x4 -> 2x2

        # Warstwa w pełni połączona
        self.fc = nn.Linear(512 * 2 * 2, 1)
        self.sigmoid = nn.Sigmoid()

    def discriminator_block(self, in_channels, out_channels, batch_norm=True):
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)  # 32x32 -> 16x16
        x = self.conv2(x)  # 16x16 -> 8x8
        x = self.conv3(x)  # 8x8 -> 4x4
        x = self.conv4(x)  # 4x4 -> 2x2

        # Flattenowanie i klasyfikacja
        x = x.view(x.size(0), -1)  # Flatten 512*2*2
        x = self.fc(x)
        x = self.sigmoid(x)

        return x


class EnhancedGenerator(nn.Module):
    def __init__(self):
        super(EnhancedGenerator, self).__init__()

        # Warstwy enkodera
        self.encoder1 = self.encoder_block(channel, 64)
        self.encoder2 = self.encoder_block(64, 128)
        self.encoder3 = self.encoder_block(128, 256)
        self.encoder4 = self.encoder_block(256, 512)

        self.bottleneck = self.bottleneck_block(512, 1024)

        self.decoder4 = self.decoder_block(1024, 512)
        self.decoder3 = self.decoder_block(512, 256)
        self.decoder2 = self.decoder_block(256, 128)
        self.decoder1 = self.decoder_block(128, 64)

        self.skip_conv3 = nn.Conv2d(1024, 512, kernel_size=1)
        self.skip_conv3 = nn.Conv2d(512, 256, kernel_size=1)
        self.skip_conv2 = nn.Conv2d(256, 128, kernel_size=1)
        self.skip_conv1 = nn.Conv2d(128, 64, kernel_size=1)

        self.final_conv = nn.ConvTranspose2d(64, channel, kernel_size=4, stride=2, padding=1)

    def encoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def bottleneck_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        # Enkoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)

        # Bottleneck
        bottleneck = self.bottleneck(enc4)

        # Dekoder
        dec4 = self.decoder4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.skip_conv4(dec4)

        dec3 = self.decoder3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.skip_conv3(dec3)

        dec2 = self.decoder2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.skip_conv2(dec2)

        dec1 = self.decoder1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.skip_conv1(dec1)

        output = self.final_conv(dec1)

        return output



class EnhancedDiscriminator(nn.Module):
    def __init__(self):
        super(EnhancedDiscriminator, self).__init__()

        # Warstwy konwolucyjne
        self.conv1 = self.discriminator_block(channel, 64, batch_norm=False)  # 128x128 -> 64x64
        self.conv2 = self.discriminator_block(64, 128)  # 64x64 -> 32x32
        self.conv3 = self.discriminator_block(128, 256)  # 32x32 -> 16x16
        self.conv4 = self.discriminator_block(256, 512)  # 16x16 -> 8x8
        self.conv5 = self.discriminator_block(512, 1024)  # 8x8 -> 4x4
        self.conv6 = self.discriminator_block(1024, 2048)  # 4x4 -> 2x2

        # Warstwa w pełni połączona
        self.fc = nn.Linear(2048 * 2 * 2, 1)
        self.sigmoid = nn.Sigmoid()

    def discriminator_block(self, in_channels, out_channels, batch_norm=True):
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)  # 128x128 -> 64x64
        x = self.conv2(x)  # 64x64 -> 32x32
        x = self.conv3(x)  # 32x32 -> 16x16
        x = self.conv4(x)  # 16x16 -> 8x8
        x = self.conv5(x)  # 8x8 -> 4x4
        x = self.conv6(x)  # 4x4 -> 2x2


        x = x.view(x.size(0), -1)  # Flatten 2048*2*2
        x = self.fc(x)
        x = self.sigmoid(x)

        return x
