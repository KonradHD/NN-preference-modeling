import os
import torch
import json


class ModelLoader:
    def __init__(self, base_dir: str | None = None, device: torch.device | str | None = None):
        self.base_dir = base_dir or "models"
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def _build_dense_mc_model(
        self,
        state_dict: dict,
        criteria_num: int,
        model_type: str,
        **kwargs,
    ) -> torch.nn.Module:
        from model.architecture.dense_monte_carlo import DenseMonteCarloNet

        has_bn = any("running_mean" in k for k in state_dict.keys())
        input_dim = state_dict["feature_extractor.0.weight"].shape[1]
        neurons = state_dict["feature_extractor.0.weight"].shape[0]
        expansion_factor = neurons // input_dim
        layers = len([k for k, v in state_dict.items() if k.startswith("feature_extractor.") and k.endswith(".weight") and v.dim() == 2])
        slope = kwargs.get("negative_slope", 0.5 if model_type == "dense_optimized" else 0.01)
        drop = kwargs.get("dropout_rate", 0.2)

        return DenseMonteCarloNet(
            n_criteria=criteria_num,
            neurons=neurons,
            layers=layers,
            expansion_factor=expansion_factor,
            dropout_rate=drop,
            negative_slope=slope,
            use_batch_norm=has_bn,
        )

    def load_dense_MC(
        self,
        model: torch.nn.Module | None = None,
        criteria_num: int = 5,
        criterion: str = "",
        device: torch.device | str | None = None,
        model_type: str = "dense_monte_carlo",
        **kwargs,
    ) -> tuple[torch.nn.Module, dict[str, list]]:
        target_device = device or self.device
        dir_path = os.path.join(self.base_dir, model_type, f"criteria{criteria_num}")
    
        model_path = os.path.join(dir_path, "model", f"{criterion}.pth")
        history_path = os.path.join(dir_path, "history", f"{criterion}.json")
        
        if not os.path.exists(model_path) or not os.path.exists(history_path):
            raise FileNotFoundError(f"There are no files in {dir_path} for {criterion}")

        with open(history_path, "r", encoding="utf-8") as file:
            history = json.load(file)

        state_dict = torch.load(model_path, map_location=target_device, weights_only=True)

        if model is None:
            model = self._build_dense_mc_model(state_dict, criteria_num, model_type, **kwargs)
            model.load_state_dict(state_dict)
        else:
            try:
                model.load_state_dict(state_dict)
            except RuntimeError:
                model = self._build_dense_mc_model(state_dict, criteria_num, model_type, **kwargs)
                model.load_state_dict(state_dict)

        model.to(target_device)
        print(f"History and model: {criterion} loaded from {dir_path}")
        
        return model, history

    def load_model(
        self,
        model: torch.nn.Module | None = None,
        params: dict | None = None,
        model_str: str | None = None,
        model_type: str = "dense_hiperparameters_exp",
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
