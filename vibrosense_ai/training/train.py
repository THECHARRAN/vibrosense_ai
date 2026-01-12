import torch
from vibrosense_ai.models.temporal_autoencoder import VibroSenseModel
from vibrosense_ai.utils.synthetic_data import create_dataset
device = 'cuda' if torch.cuda.is_available() else "cpu"
model=VibroSenseModel(features=96).to(device)
optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)
data=create_dataset().to(device)
data=data.float()
epochs=30
for epoch in range(epochs):
    model.train()
    recon, forecast=model(data)
    recon_loss=torch.mean((recon-data)**2)
    forecast_target=recon_loss.detach()*torch.ones_like(forecast)
    forecast_loss=torch.mean((forecast-forecast_target)**2)
    loss=recon_loss+0.3*forecast_loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1} | loss:{loss.item(): .4f}")