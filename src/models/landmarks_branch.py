from torch import nn


class LandmarksBranch(nn.Module):
    def __init__(self, num_landmarks=21, num_coordinates=3, out_dim=64):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Flatten(),  # flattening (21, 3) shaped landmarks to (63,)
            nn.Linear(in_features=num_landmarks * num_coordinates, out_features=128),
            nn.LayerNorm(128),
            nn.ReLU()
            # nn.LeakyReLU()
        )

        self.dropout1 = nn.Dropout(p=0.5)

        self.block2 = nn.Sequential(
            nn.Linear(in_features=128, out_features=256),
            nn.LayerNorm(256),
            nn.ReLU()
            # nn.LeakyReLU()
        )

        self.dropout2 = nn.Dropout(p=0.5)

        self.block3 = nn.Sequential(
            nn.Linear(in_features=256, out_features=out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU()
            # nn.LeakyReLU()
        )

    def forward(self, x):
        x = self.dropout1(self.block1(x))
        x = self.dropout2(self.block2(x))

        return self.block3(x)  # (batch_size, out_dim)-shaped landmark features