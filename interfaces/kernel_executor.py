from abc import ABC, abstractmethod
from typing import Dict, Optional, Any
from pydantic import BaseModel, Field

from models.kernel import PerformanceMetrics

class KernelExecutionError(Exception):
    """Exception raised when kernel execution fails."""
    pass

class KernelExecutor(ABC):
    """Interface for hardware-specific kernel execution."""
    
    @abstractmethod
    def execute(self, code: str, task_description: str) -> PerformanceMetrics:
        """
        Compiles and executes kernel code, collecting performance metrics.
        
        Args:
            code: The source code to compile and execute
            task_description: Description of the computation task
            
        Returns:
            PerformanceMetrics: Performance data collected during execution
            
        Raises:
            KernelExecutionError: If compilation or execution fails
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """
        Checks if this executor's hardware and tools are available.
        
        Returns:
            bool: True if this executor can be used, False otherwise
        """
        pass