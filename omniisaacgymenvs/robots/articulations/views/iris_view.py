from typing import Optional

from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.prims import RigidPrimView
from omni.isaac.sensor import IMUSensor
import numpy as np

class irisView(ArticulationView):
    def __init__(
        self,
        prim_paths_expr: str,
        name: Optional[str] = "irisView"
    ) -> None:
        """[summary]
        """
        super().__init__(
            prim_paths_expr=prim_paths_expr,
            name=name,
        )
        self.physics_bodys = RigidPrimView(prim_paths_expr=f"/World/envs/.*/iris/body",name=f"body_view")
        self.physics_rotors = [RigidPrimView(prim_paths_expr=f"/World/envs/.*/iris/rotor{i}",
                                            name=f"rotor{i}_prop_view") for i in range(0, 4)]
        
        # self._imu_sensor_interface = IMUSensor(prim_path="/World/envs/.*/iris/body", name="imu", frequency=60, translation=np.array([0, 0, 0]), orientation=np.array([1, 0, 0, 0]))
        
        

