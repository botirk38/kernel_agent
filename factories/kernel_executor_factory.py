from typing import Dict, Optional, Type
from interfaces.kernel_executor import KernelExecutor
from services.kernel_executors import NvidiaExecutor

# Import other vendor executors as they're implemented
# from amd_executor import AmdExecutor
# from intel_executor import IntelExecutor

class ExecutorFactory:
    """Factory for creating vendor-specific kernel executors."""
    
    # Registry of available executors
    _executors: Dict[str, Type[KernelExecutor]] = {
        "nvidia": NvidiaExecutor,
    }
    
    @classmethod
    def create_executor(cls, vendor: str) -> Optional[KernelExecutor]:
        """
        Create an appropriate kernel executor for the specified vendor.
        
        Args:
            vendor: The hardware vendor name (e.g., "nvidia", "amd", "intel")
            
        Returns:
            KernelExecutor if a suitable executor is available, None otherwise
        """
        executor_class = cls._executors.get(vendor)
        
        if executor_class:
            executor = executor_class()
            if executor.is_available():
                return executor
        return None
    
    @classmethod
    def get_available_executor(cls) -> Optional[KernelExecutor]:
        """
        Find the first available executor from any supported vendor.
        
        Returns:
            KernelExecutor if any suitable executor is available, None otherwise
        """
        for vendor in cls._executors:
            executor = cls.create_executor(vendor)
            if executor:
                return executor
        return None