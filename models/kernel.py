from pydantic import BaseModel, Field

class PerformanceMetrics(BaseModel):
    """Performance metrics collected from hardware profiling tools."""
    execution_time: float = Field(description="Kernel execution time in milliseconds")
    memory_bandwidth: float = Field(description="Memory bandwidth utilization in GB/s")
    sm_efficiency: float = Field(description="Streaming multiprocessor efficiency percentage")
    occupancy: float = Field(description="Kernel occupancy (0.0-1.0)")
    memory_usage: float = Field(description="Peak GPU memory usage in MB")