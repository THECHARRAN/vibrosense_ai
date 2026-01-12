import torch
import torch.nn as nn
class encoder(nn.Module):
    def __init__(self, input_features, latent_dim):
        super().__init__()
        self.conv=nn.Sequential(
            nn.Conv1d(input_features, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(64,128,kernel_size=5,padding=2),
            nn.ReLU(),
            nn.Conv1d(128,latent_dim, kernel_size=3, padding=1),
            nn.ReLU())
    def forward(self,x):
        x=x.permute(0,2,1)
        z=self.conv(x)
        z=z.permute(0,2,1)
        return z
class TemporalCore(nn.Module):
    def __init__(self, latent_dim, hidden_dim):
        super().__init__()
        self.lstm=nn.LSTM(
            latent_dim,
            hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True)
    def forward(self,z):
        out,_=self.lstm(z)
        return out
class Decoder(nn.Module):
    def __init__(self,hidden_dim, output_features):
        super().__init__()
        self.deconv = nn.Sequential(nn.Conv1d(hidden_dim*2,128,kernel_size=3,padding=1),nn.ReLU(),
                                    nn.Conv1d(128,64,kernel_size=3,padding=1),nn.ReLU(),
                                    nn.Conv1d(64,output_features,kernel_size=3,padding=1))
    def forward(self,h):
        h=h.permute(0,2,1)
        x_hat=self.deconv(h)
        x_hat=x_hat.permute(0,2,1)
        return x_hat
class ForecastHead(nn.Module):
    def __init__(self,hidden_dim,horizon=5):
        super().__init__()
        self.fc=nn.Sequential(
            nn.Linear(hidden_dim*2,64),
            nn.ReLU(),
            nn.Linear(64,horizon)
        )
    def forward(self,h):
        last_state=h[:,-1,:]
        forecast =self.fc(last_state)
        return forecast
class VibroSenseModel(nn.Module):
    def __init__(self, features,latent_dim=64, hidden_dim=64):
        super().__init__()
        self.encoder=encoder(features,latent_dim)
        self.temporal=TemporalCore(latent_dim,hidden_dim)
        self.decoder=Decoder(hidden_dim,features)
        self.forecaster=ForecastHead(hidden_dim)
    def forward(self,x):
        z=self.encoder(x)
        h=self.temporal(z)
        recon=self.decoder(h)
        forecast=self.forecaster(h)
        return recon, forecast