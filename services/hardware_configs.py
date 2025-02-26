from interfaces.hardware_detector import HardwareDetector
from models.kernel import HardwareSpecs
from typing import Optional

class NvidiaDetector(HardwareDetector):
    def detect(self) -> Optional[HardwareSpecs]:
        try:
            import pycuda.autoinit # type: ignore
            import pycuda.driver as cuda # type: ignore
            
            device = cuda.Device(0)  # Get primary GPU
            attrs = device.get_attributes()
            
            return HardwareSpecs(
                vendor="nvidia",
                device_name=device.name(),
                compute_capability=f"{device.compute_capability()[0]}.{device.compute_capability()[1]}",
                total_memory=device.total_memory() // (1024*1024),  # Convert to MB
                clock_speed=device.get_attribute(cuda.device_attribute.CLOCK_RATE),
                num_sm=attrs[cuda.device_attribute.MULTIPROCESSOR_COUNT],
                cores_per_sm=self._get_cores_per_sm(device.compute_capability())
            )
        except:
            return None
            
    def _get_cores_per_sm(self, compute_capability: tuple) -> int:
        """Get cores per SM based on compute capability"""
        cores_by_cc = {
            (7, 0): 64,  # Volta
            (7, 5): 64,  # Turing
            (8, 0): 64,  # Ampere
            (8, 6): 128, # Ada Lovelace
            (9, 0): 128, # Hopper
        }
        return cores_by_cc.get(compute_capability, 32)
        
    def is_available(self) -> bool:
        try:
            import pycuda.driver as cuda # type: ignore
            cuda.init()
            return cuda.Device.count() > 0
        except:
            return False