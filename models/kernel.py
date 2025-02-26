from pydantic import BaseModel, Field

class PerformanceMetrics(BaseModel):
    """Performance metrics collected from hardware profiling tools."""
    execution_time: float = Field(description="Kernel execution time in milliseconds")
    memory_bandwidth: float = Field(description="Memory bandwidth utilization in GB/s")
    sm_efficiency: float = Field(description="Streaming multiprocessor efficiency percentage")
    occupancy: float = Field(description="Kernel occupancy (0.0-1.0)")
    memory_usage: float = Field(description="Peak GPU memory usage in MB")


class HardwareSpecs(BaseModel):
    """Hardware specifications for GPU devices"""
    vendor: str = Field(description="GPU vendor name (e.g. nvidia, amd)")
    device_name: str = Field(description="Name of the GPU device")
    compute_capability: str = Field(description="Compute capability version")
    total_memory: int = Field(description="Total GPU memory in MB")
    clock_speed: int = Field(description="GPU clock speed in MHz") 
    num_sm: int = Field(description="Number of streaming multiprocessors")
    cores_per_sm: int = Field(description="CUDA cores per streaming multiprocessor")
