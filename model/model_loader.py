import os
import torch
import json


class ModelLoader:
    def __init__(self, base_dir: str | None = None):
        self.base_dir = base_dir or "models"


    def load_dense_MC(self, model: torch.nn.Module, criteria_num: int,  criterion: str) -> tuple[torch.nn.Module, dict[str, list]]:
        model_type = "dense_monte_carlo"
        dir_path = os.path.join(self.base_dir, model_type, f"criteria{criteria_num}")
    
        model_path = os.path.join(dir_path, "model", f"{criterion}.pth")
        history_path = os.path.join(dir_path, "history", f"{criterion}.json")
        
        if not os.path.exists(model_path) or not os.path.exists(history_path):
            raise FileNotFoundError(f"There are no files in {dir_path} for {criterion}")

        with open(history_path, "r", encoding="utf-8") as file:
            history = json.load(file)

        model.load_state_dict(torch.load(model_path, weights_only=True))
        print(f"History and model: {criterion} loaded from {dir_path}")
        
        return model, history
