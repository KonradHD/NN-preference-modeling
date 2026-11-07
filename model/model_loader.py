import os
import torch
import json


class ModelLoader:
    def __init__(self, base_dir: str | None = None, device: torch.device | str | None = None):
        self.base_dir = base_dir or "models"
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def load_dense_MC(self, model: torch.nn.Module, criteria_num: int,  criterion: str, device: torch.device | str | None = None) -> tuple[torch.nn.Module, dict[str, list]]:
        target_device = device or self.device
        model_type = "dense_monte_carlo"
        dir_path = os.path.join(self.base_dir, model_type, f"criteria{criteria_num}")
    
        model_path = os.path.join(dir_path, "model", f"{criterion}.pth")
        history_path = os.path.join(dir_path, "history", f"{criterion}.json")
        
        if not os.path.exists(model_path) or not os.path.exists(history_path):
            raise FileNotFoundError(f"There are no files in {dir_path} for {criterion}")

        with open(history_path, "r", encoding="utf-8") as file:
            history = json.load(file)

        model.load_state_dict(torch.load(model_path, map_location=target_device, weights_only=True))
        model.to(target_device)
        print(f"History and model: {criterion} loaded from {dir_path}")
        
        return model, history

    def load_model(
        self,
        model: torch.nn.Module | None = None,
        params: dict | None = None,
        model_str: str | None = None,
        model_type: str = "dense_hiperparameters",
        device: torch.device | str | None = None,
        n_criteria: int | None = None,
        **kwargs,
    ) -> tuple[torch.nn.Module, dict]:
        from model.trainers.trainer import load_model as _load_model
        return _load_model(
            model=model,
            params=params,
            model_str=model_str,
            model_type=model_type,
            base_dir=self.base_dir,
            device=device or self.device,
            n_criteria=n_criteria,
            **kwargs,
        )
