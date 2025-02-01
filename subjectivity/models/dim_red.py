import pacmap
import umap
import numpy as np
from typing import Any, Optional
import logging

class DimRed:

    def __init__(self, model: str = "pacmap", model_params: Optional[dict[str, Any]] = None) -> None:
        self.model = model
        self.model_params = model_params

    def reduce(self, data: np.ndarray) -> np.ndarray:
        if self.model_params is None:
            self.model_params = {} # for unpacking dict
        
        if self.model == "pacmap":
            logging.info(f"Reducing with '{self.model}' and params '{self.model_params}'.")
            model = pacmap.PaCMAP(**self.model_params)
            result = model.fit_transform(data, init="pca")
        elif self.model == "umap":
            model = umap.UMAP(**self.model_params)
            result = model.fit_transform(data)

        else:
            raise NotImplementedError(f"Model '{self.model}' not supported.")
        return result