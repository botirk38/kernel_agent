from services.hardware_configs import NvidiaDetector
from typing import Optional
from interfaces.hardware_detector import HardwareDetector

class HardwareDetectorFactory:
    _detectors = {
        "nvidia": NvidiaDetector,
        # Add other vendors here
    }
    
    @classmethod
    def create_detector(cls) -> Optional[HardwareDetector]:
        """Create appropriate detector for available hardware"""
        for detector_class in cls._detectors.values():
            detector = detector_class()
            if detector.is_available():
                return detector
        return None