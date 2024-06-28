from typing import Optional

import carb
import numpy as np
import torch
from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage


class iris(Robot):
    def __init__(
        self,
        prim_path: str,
        name: Optional[str] = "iris",
        usd_path: Optional[str] = None,
        translation: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
        scale: Optional[np.array] = None
    ) -> None:
        """[summary]"""
        
        from omni.isaac.core.utils.nucleus import get_server_path
        self.server_path = get_server_path()
        if self.server_path is None:
            print("Could not find Isaac Sim server path")
            return
        
        self._usd_path = usd_path
        self._name = name

        if self._usd_path is None:
            assets_root_path = get_assets_root_path()
            if assets_root_path is None:
                carb.log_error("Could not find Isaac Sim assets folder")
            self._usd_path = self.server_path + "/Users/jaramy/Thesis-asset/iris.usd"
            
        add_reference_to_stage(self._usd_path, prim_path)
        
        scale = torch.tensor([1, 1, 1])
        super().__init__(prim_path=prim_path, name=name, translation=translation, orientation=orientation, scale=scale)