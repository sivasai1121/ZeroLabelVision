
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import UnlabeledDataset
from model import ZeroLabelModel

DATA_DIR = "data/unlabeled"
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = UnlabeledDataset(DATA_DIR)

if len(dataset) < 2:
    raise ValueError("Add at least two images to data/unlabeled.")

loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True,
    num_workers=0,
    drop_last=False
)

model = ZeroLabelModel(pretrained=True).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
temperature = 0.2
epochs = 10

for epoch in range(epochs):
    model.train()
    total_loss = 0.0

    for view1, view2 in loader:
        view1 = view1.to(device)
        view2 = view2.to(device)

        z1 = model(view1)
        z2 = model(view2)
        z = torch.cat([z1, z2], dim=0)

        similarity = torch.matmul(z, z.T) / temperature
        similarity.fill_diagonal_(float("-inf"))

        batch_size = view1.size(0)
        targets = torch.cat([
            torch.arange(batch_size, 2 * batch_size),
            torch.arange(0, batch_size)
        ]).to(device)

        loss = F.cross_entropy(similarity, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(
        f"Epoch {epoch + 1}/{epochs}, "
        f"Loss: {total_loss / len(loader):.4f}"
    )

torch.save(model.state_dict(), MODEL_DIR / "zerolabel_model.pth")
print("Training complete. Model saved successfully.")
