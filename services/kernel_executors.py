
import os
import subprocess
import tempfile
import re
import uuid
from typing import Dict, Tuple, Optional
from interfaces.kernel_executor import KernelExecutor,  KernelExecutionError
from models.kernel import PerformanceMetrics

class NvidiaExecutor(KernelExecutor):
    """Kernel executor for NVIDIA GPUs using CUDA."""
    
    def is_available(self) -> bool:
        """Check if NVIDIA tools are available."""
        try:
            # Check for nvcc compiler
            subprocess.run(
                ["nvcc", "--version"], 
                capture_output=True, 
                check=True
            )
            
            # Check for either nvprof (older) or nsys (newer)
            has_profiler = False
            try:
                subprocess.run(
                    ["nvprof", "--version"],
                    capture_output=True,
                    check=True
                )
                has_profiler = True
            except (subprocess.CalledProcessError, FileNotFoundError):
                try:
                    subprocess.run(
                        ["nsys", "--version"],
                        capture_output=True,
                        check=True
                    )
                    has_profiler = True
                except (subprocess.CalledProcessError, FileNotFoundError):
                    pass
            
            return has_profiler
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def execute(self, code: str) -> PerformanceMetrics:
        """Execute CUDA code and collect performance metrics using NVIDIA tools."""
        # Create unique identifier for this run
        run_id = str(uuid.uuid4())[:8]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Set up file paths
            cuda_file = os.path.join(temp_dir, f"kernel_{run_id}.cu")
            executable = os.path.join(temp_dir, f"kernel_{run_id}")
            profile_output = os.path.join(temp_dir, f"profile_{run_id}.csv")
            
            # Write code to file
            with open(cuda_file, 'w') as f:
                f.write(code)
            
            # Compile the code
            try:
                subprocess.run(
                    ["nvcc", cuda_file, "-o", executable, "-lineinfo"],
                    capture_output=True,
                    text=True,
                    check=True
                )
            except subprocess.CalledProcessError as e:
                raise KernelExecutionError(f"CUDA compilation failed:\n{e.stderr}")
            
            # Check if executable runs without profiling
            try:
                subprocess.run(
                    [executable],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=True
                )
            except subprocess.CalledProcessError as e:
                raise KernelExecutionError(f"Kernel execution failed:\n{e.stderr}")
            except subprocess.TimeoutExpired:
                raise KernelExecutionError("Kernel execution timed out")
            
            # Determine which profiler to use
            try:
                subprocess.run(["nvprof", "--version"], 
                              capture_output=True, check=True)
                return self._profile_with_nvprof(executable)
            except (subprocess.CalledProcessError, FileNotFoundError):
                try:
                    subprocess.run(["nsys", "--version"], 
                                  capture_output=True, check=True)
                    return self._profile_with_nsys(executable, profile_output)
                except (subprocess.CalledProcessError, FileNotFoundError):
                    raise KernelExecutionError(
                        "No NVIDIA profiling tools found. Please install CUDA toolkit with profiling tools."
                    )
    
    def _profile_with_nvprof(self, executable: str) -> PerformanceMetrics:
        """Profile with nvprof (for older CUDA versions)."""
        try:
            # Run with nvprof
            result = subprocess.run(
                [
                    "nvprof", 
                    "--metrics", "achieved_occupancy,dram_read_throughput,dram_write_throughput,sm_efficiency",
                    executable
                ],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            # Parse the output
            output = result.stderr  # nvprof outputs to stderr
            
            # Extract execution time
            time_match = re.search(r'GPU activities:\s+(\d+\.\d+)%\s+(\d+\.\d+)ms', output)
            execution_time = float(time_match.group(2)) if time_match else 100.0
            
            # Extract occupancy
            occupancy_match = re.search(r'achieved_occupancy\s+(\d+\.\d+)', output)
            occupancy = float(occupancy_match.group(1)) if occupancy_match else 0.5
            
            # Extract memory bandwidth (combine read and write)
            read_bw_match = re.search(r'dram_read_throughput\s+(\d+\.\d+)', output)
            write_bw_match = re.search(r'dram_write_throughput\s+(\d+\.\d+)', output)
            
            read_bw = float(read_bw_match.group(1)) if read_bw_match else 0.0
            write_bw = float(write_bw_match.group(1)) if write_bw_match else 0.0
            memory_bandwidth = read_bw + write_bw
            
            # Extract SM efficiency
            sm_eff_match = re.search(r'sm_efficiency\s+(\d+\.\d+)', output)
            sm_efficiency = float(sm_eff_match.group(1)) if sm_eff_match else 0.0
            
            # Get memory usage
            memory_match = re.search(r'(\d+\.\d+)MiB', output)
            memory_usage = float(memory_match.group(1)) if memory_match else 0.0
            
            return PerformanceMetrics(
                execution_time=execution_time,
                memory_bandwidth=memory_bandwidth,
                sm_efficiency=sm_efficiency,
                occupancy=occupancy,
                memory_usage=memory_usage
            )
            
        except subprocess.TimeoutExpired:
            raise KernelExecutionError("Profiling with nvprof timed out")
        except Exception as e:
            raise KernelExecutionError(f"Error during nvprof profiling: {str(e)}")
    
    def _profile_with_nsys(self, executable: str, output_file: str) -> PerformanceMetrics:
        """Profile with Nsight Systems (for newer CUDA versions)."""
        try:
            # Run with nsys
            subprocess.run(
                [
                    "nsys", "profile",
                    "--stats=true",
                    "--force-overwrite=true", 
                    "--export=csv",
                    "--output", output_file,
                    executable
                ],
                capture_output=True,
                text=True,
                timeout=30,
                check=True
            )
            
            # Run additional command to get GPU metrics
            subprocess.run(
                [
                    "nsys", "stats",
                    "--report", "gputrace",
                    "--format", "csv",
                    output_file
                ],
                capture_output=True,
                text=True,
                check=True
            )
            
            # Parse the CSV output
            metrics_data = {}
            
            # Read stats output
            stats_file = f"{output_file}.csv"
            if os.path.exists(stats_file):
                with open(stats_file, 'r') as f:
                    csv_content = f.read()
                    
                    # Extract execution time
                    time_match = re.search(r'CUDA Kernel,(\d+\.\d+)', csv_content)
                    if time_match:
                        metrics_data['execution_time'] = float(time_match.group(1))
                    
                    # Extract occupancy (approximate from nsys data)
                    occupancy_match = re.search(r'SM Occupancy,(\d+\.\d+)', csv_content)
                    if occupancy_match:
                        metrics_data['occupancy'] = float(occupancy_match.group(1)) / 100.0
                    
                    # Extract memory bandwidth
                    bw_match = re.search(r'Memory Throughput,(\d+\.\d+)', csv_content)
                    if bw_match:
                        metrics_data['memory_bandwidth'] = float(bw_match.group(1))
                    
                    # Extract SM efficiency (approximation)
                    sm_match = re.search(r'SM Activity,(\d+\.\d+)', csv_content)
                    if sm_match:
                        metrics_data['sm_efficiency'] = float(sm_match.group(1))
                    
                    # Extract memory usage
                    mem_match = re.search(r'Memory Used,(\d+\.\d+)', csv_content)
                    if mem_match:
                        metrics_data['memory_usage'] = float(mem_match.group(1))
            
            # Fill in defaults for missing metrics
            return PerformanceMetrics(
                execution_time=metrics_data.get('execution_time', 100.0),
                memory_bandwidth=metrics_data.get('memory_bandwidth', 0.0),
                sm_efficiency=metrics_data.get('sm_efficiency', 0.0),
                occupancy=metrics_data.get('occupancy', 0.5),
                memory_usage=metrics_data.get('memory_usage', 0.0)
            )
            
        except subprocess.TimeoutExpired:
            raise KernelExecutionError("Profiling with nsys timed out")
        except Exception as e:
            raise KernelExecutionError(f"Error during nsys profiling: {str(e)}")