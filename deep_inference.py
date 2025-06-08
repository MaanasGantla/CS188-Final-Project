import torch
import torch.nn as nn
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")




# policy querying the MLP
class MLPPolicy(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=128, output_dim=7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )
    def forward(self, x):
        return self.net(x)

class DeepEEPolicy:
    """
    at a high level:
        1) grab eef position and pass it through to model to get action vector
        2) do action
        3) grab new eef position and repeat this cycle until success
    """
    def __init__(self, model_path="policy.pth", norm_stats_path="norm_stats.npz"):

        stats = np.load(norm_stats_path)

        self.mean_X = stats["mean_X"].reshape(-1)   
        self.std_X  = stats["std_X"].reshape(-1)    


        self.model = MLPPolicy(input_dim=10, hidden_dim=128, output_dim=7).to(DEVICE)
        self.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        self.model.eval()

    def get_action(self, ee_pos, square_pos, square_quat):

        s_raw = np.concatenate([ee_pos, square_pos, square_quat], axis=0).astype(np.float32)


        s_norm = (s_raw - self.mean_X) / self.std_X  


        s_t = torch.from_numpy(s_norm).unsqueeze(0).to(DEVICE)  
        with torch.no_grad():
            a_t = self.model(s_t).cpu().numpy().flatten()  
        return a_t
