import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim, nn, utils, Tensor
import pytorch_lightning as pl

from hyperparameters import BATCH_SIZE, DATA_LEN, WAVELET_FEATURE_LEN

dropout = 0.5
model_name = "model_rnn_big"
base_lr = 3e-3

class NewGELUActivation(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
    the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """

    def forward(self, input: Tensor) -> Tensor:
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        
    def forward(self, hidden, encoder_outputs):
        timestep = encoder_outputs.size(1)
        h = hidden.repeat(timestep, 1, 1).transpose(0, 1)
        # encoder_outputs = encoder_outputs.transpose(0, 1)
        attn_energies = self.score(h, encoder_outputs)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)
    
    def score(self, hidden, encoder_outputs):
        energy = self.attn(torch.cat([hidden, encoder_outputs], 2))
        energy = energy.transpose(1, 2)
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)
        energy = torch.bmm(v, energy)
        return energy.squeeze(1)


class SEMGRNN(pl.LightningModule):
    def __init__(self, onnx_export=False):
        super(SEMGRNN, self).__init__()

        self.input_dim = 8
        self.output_dim = 8
        self.hidden_dim = 128
        self.n_layers = 2

        self.cuda_device = torch.device("cpu") if onnx_export else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.act = NewGELUActivation()

        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.n_layers, batch_first=True, dropout=dropout, bidirectional=True)
        self.attention = Attention(self.hidden_dim * 2)
        self.dropout = nn.Dropout(dropout)

        self.fc_hard = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            self.act,
            nn.Linear(self.hidden_dim, self.output_dim)
        )
        self.fc_features = nn.Sequential(
            nn.Linear(self.input_dim * 6, self.input_dim * 6),
            self.act,
            nn.Linear(self.input_dim * 6, self.output_dim * 6),
            self.act,
            nn.Linear(self.input_dim * 6, self.output_dim)
        )
        self.fc_wavelet = nn.Sequential(
            nn.Linear(self.input_dim * WAVELET_FEATURE_LEN, self.input_dim * 6),
            self.act,
            nn.Linear(self.input_dim * 6, self.output_dim * 6),
            self.act,
            nn.Linear(self.input_dim * 6, self.output_dim)
        )
        self.fc_out = nn.Sequential(
            nn.Linear(self.output_dim * 4, self.input_dim * 3),
            # nn.Linear(self.input_dim, self.input_dim * 4),
            self.act,
            nn.Linear(self.input_dim * 3, self.output_dim * 3),
            self.act,
            nn.Linear(self.input_dim * 3, self.output_dim)
        )
        self.fc_test = nn.Sequential(
            nn.Linear(self.output_dim * 50, self.output_dim * 50),
            self.act,
            nn.Linear(self.output_dim * 50, self.output_dim * 16),
            self.act,
            nn.Linear(self.output_dim * 16, self.output_dim)
        )

        # Lightning stuff
        self.training_step_outputs = []
        # Print validation output once per epoch
        self.should_val_print = True
        self.loss_fn = nn.MSELoss(reduction='sum')

        self.scheduler = None
        self.plateau_scheduler = None
        self.best_epoch_average = 1000000

    def forward(self, x):
        batch_size = x.size(0)

        # Time / freq domain features
        features = x[:, :6, :]
        # Wavelet features
        wavelet = x[:, 6 : 6 + WAVELET_FEATURE_LEN, :]
        
        # Full EMG
        emg_full = x[:, -DATA_LEN * 2 : -DATA_LEN, :]
        # Take only last 50 samples of EMG for processing
        emg = emg_full[:, -50:, :]
        # Wrist angles are last DATA_LEN samples, but only the first 3 dimensions
        # last_wrist_angle = x[:, -1:, :3]
        # Concat emg features with wrist angles
        
        h0 = torch.zeros(self.n_layers * 2, batch_size, self.hidden_dim).to(self.cuda_device)
        c0 = torch.zeros(self.n_layers * 2, batch_size, self.hidden_dim).to(self.cuda_device)
        
        out, (hidden, cell) = self.lstm(emg, (h0, c0))
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        
        attn_weights = self.attention(hidden, out)
        context = attn_weights.bmm(out)
        context_out = self.fc_hard(context)

        # Features
        out_features = self.fc_features(features.view(batch_size, -1)).unsqueeze(1)
        out_features = self.dropout(out_features)

        # Wavelet features
        out_wavelet = self.fc_wavelet(wavelet.view(batch_size, -1)).unsqueeze(1)
        out_wavelet = self.dropout(out_wavelet)

        out_test = self.fc_test(emg.view(batch_size, -1)).unsqueeze(1)
        out_test = self.dropout(out_test)

        # Cat everything
        final_x = torch.cat((context_out, out_features, out_wavelet, out_test), dim=-1)
        out = self.fc_out(final_x)

        return out

    def loss_function(self, x, y):
        loss = self.loss_fn(x, y)
        return loss

    def training_step(self, batch, batch_idx):
        x = batch['sample']
        y = batch['label']

        y_expected = y[:, -1:, :]
        
        z = self.forward(x)
        loss = self.loss_function(z, y_expected)
        self.log("train_loss", loss, on_step=False, on_epoch=True, sync_dist=True)

        diffs = []
        z_comp = z.cpu().detach().numpy()[:, -1:]
        y_comp = y_expected.cpu().detach().numpy()[:, -1:]

        np.set_printoptions(precision=3)
        diffs = np.abs(z_comp - y_comp)
        
        self.log("avd_mean_train", np.mean(diffs), on_step=False, on_epoch=True, sync_dist=True)
        self.log("avd_std_train", np.std(diffs), on_step=False, on_epoch=True, sync_dist=True)

        self.training_step_outputs.append(loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch['sample']
        y = batch['label']

        # y_expected = y
        y_expected = y[:, -1:, :]

        z = self.forward(x)
        loss = self.loss_function(z, y_expected)
        self.log("val_loss", loss, on_epoch=True, sync_dist=True)

        diffs = []
        z_comp = z.cpu().detach().numpy()[:, -1:]
        y_comp = y_expected.cpu().detach().numpy()[:, -1:]

        np.set_printoptions(precision=3)
        diffs = np.abs(z_comp - y_comp)
        
        self.log("avd_mean_test", np.mean(diffs), on_step=False, on_epoch=True, sync_dist=True)
        self.log("avd_std_test", np.std(diffs), on_step=False, on_epoch=True, sync_dist=True)

        if self.should_val_print:
            print("Validation")
            print("z_comp:")
            print(z_comp[:5, -1].flatten())
            print("y_comp:")
            print(y_comp[:5, -1].flatten())
            self.should_val_print = False

        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=base_lr)
        self.optimizer = optimizer
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.98)
        # self.scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=base_lr, max_lr=base_lr * 2, step_size_up=2, step_size_down=2, mode='triangular2', cycle_momentum=False)
        self.plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
            mode='min',
            factor=0.5,
            patience=3,
            min_lr=base_lr / 30)
        return optimizer
    
    def on_train_epoch_end(self):
        epoch_average = torch.stack(self.training_step_outputs).mean()
        self.log("training_epoch_average", epoch_average, sync_dist=True)
        self.log("learning_rate", self.optimizer.param_groups[0]['lr'], sync_dist=True)
        self.training_step_outputs.clear()  # free memory

        if self.trainer.current_epoch < 100:
            self.scheduler.step()
        
        # if self.trainer.current_epoch > 100:
            # self.plateau_scheduler.step(epoch_average)

        self.should_val_print = True

        # Save model if it's the best so far
        if epoch_average < self.best_epoch_average:
            self.best_epoch_average = epoch_average
            torch.save(self.state_dict(), model_name + ".pt")


if __name__ == "__main__":
    model = SEMGRNN(onnx_export=True)
    input_tensor = torch.rand(
        64, 
        6                # 6 time/freq domain features 
        + WAVELET_FEATURE_LEN             # 36 wavelet features
        + DATA_LEN       # DATA_LEN EMG samples
        # Note: the wrist angles were part of an experiment and are not actually used in the model.
        + DATA_LEN,      # DATA_LEN wrist angle samples
        8)
    y = model(input_tensor)

    print(y.shape)

    # Print number of trainable parameters
    print("Number of trainable parameters: " + str(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    # torch.save(model.state_dict(), model_name + ".pt")