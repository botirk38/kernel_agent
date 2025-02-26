from abc import ABC, abstractmethod
from typing import Optional
from models.kernel import HardwareSpecs


class HardwareDetector(ABC):
    @abstractmethod
    def detect(self) -> Optional[HardwareSpecs]:
        """Detect and return hardware specifications"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if this hardware type is available"""
        pass