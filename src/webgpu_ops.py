try:
    import wgpu
    import numpy as np
    from typing import List, Tuple, Optional, Union
    from .math_ops import Matrix
    
    # Check if wgpu is properly initialized
    try:
        # Test if the required APIs are available
        adapter = wgpu.gpu.request_adapter(power_preference="high-performance")
        if adapter is None:
            raise ImportError("WebGPU adapter not available on this system")
        test_device = adapter.request_device()
        WEBGPU_AVAILABLE = True
    except (AttributeError, ImportError) as e:
        print(f"WebGPU initialization failed: {str(e)}")
        WEBGPU_AVAILABLE = False

    class WebGPUMatrix:
        def __init__(self, rows: int, cols: int, data: Optional[List[List[float]]] = None):
            if not WEBGPU_AVAILABLE:
                raise ImportError("WebGPU is not available on this system.")
            
            self.rows = rows
            self.cols = cols
            self.adapter = wgpu.gpu.request_adapter(power_preference="high-performance")
            self.device = self.adapter.request_device()
            
            # Convert data to numpy array if provided
            if data is not None:
                self.data = np.array(data, dtype=np.float32)
            else:
                self.data = np.zeros((rows, cols), dtype=np.float32)
            
            # Create GPU buffer
            self.buffer = self.device.create_buffer(
                size=self.data.nbytes,
                usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC
            )
            self.device.queue.write_buffer(self.buffer, 0, self.data.tobytes())
        
        def __getitem__(self, key):
            i, j = key
            return self.data[i, j]
        
        def __setitem__(self, key, value):
            i, j = key
            self.data[i, j] = float(value)
            # Update GPU buffer
            self.device.queue.write_buffer(
                self.buffer,
                (i * self.cols + j) * 4,
                np.array([float(value)], dtype=np.float32).tobytes()
            )
        
        def to_cpu(self) -> Matrix:
            """Convert to CPU Matrix"""
            return Matrix(self.rows, self.cols, self.data.tolist())
        
        def copy(self) -> 'WebGPUMatrix':
            """Create a deep copy of the matrix"""
            return WebGPUMatrix(self.rows, self.cols, self.data.copy())

    def webgpu_matrix_multiply(a: WebGPUMatrix, b: WebGPUMatrix) -> WebGPUMatrix:
        """Matrix multiplication using WebGPU"""
        if a.cols != b.rows:
            raise ValueError(f"Matrix dimensions don't match: {(a.rows, a.cols)} and {(b.rows, b.cols)}")
        
        # Create shader for matrix multiplication
        shader_source = """
        struct Dimensions {
            a_rows: u32,
            a_cols: u32,
            b_cols: u32,
        };
        
        @group(0) @binding(0) var<storage, read> a: array<f32>;
        @group(0) @binding(1) var<storage, read> b: array<f32>;
        @group(0) @binding(2) var<storage, read_write> c: array<f32>;
        @group(0) @binding(3) var<uniform> dims: Dimensions;
        
        @compute @workgroup_size(8, 8, 1)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
            let row = global_id.x;
            let col = global_id.y;
            
            if (row >= dims.a_rows || col >= dims.b_cols) {
                return;
            }
            
            var sum = 0.0;
            for (var i = 0u; i < dims.a_cols; i++) {
                sum += a[row * dims.a_cols + i] * b[i * dims.b_cols + col];
            }
            c[row * dims.b_cols + col] = sum;
        }
        """
        
        # Create dimensions buffer
        dims = np.array([a.rows, a.cols, b.cols], dtype=np.uint32)
        dims_buffer = a.device.create_buffer(
            size=dims.nbytes,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST
        )
        a.device.queue.write_buffer(dims_buffer, 0, dims.tobytes())
        
        # Create result buffer
        result = WebGPUMatrix(a.rows, b.cols)
        
        # Create bind group layout
        bind_group_layout = a.device.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "buffer": {"type": "read-only-storage"}
                },
                {
                    "binding": 1,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "buffer": {"type": "read-only-storage"}
                },
                {
                    "binding": 2,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "buffer": {"type": "storage"}
                },
                {
                    "binding": 3,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "buffer": {"type": "uniform"}
                }
            ]
        )
        
        # Create pipeline layout
        pipeline_layout = a.device.create_pipeline_layout(
            bind_group_layouts=[bind_group_layout]
        )
        
        # Create shader module
        shader_module = a.device.create_shader_module(
            code=shader_source
        )
        
        # Create compute pipeline
        pipeline = a.device.create_compute_pipeline(
            layout=pipeline_layout,
            compute={"module": shader_module, "entry_point": "main"}
        )
        
        # Create bind group
        bind_group = a.device.create_bind_group(
            layout=bind_group_layout,
            entries=[
                {"binding": 0, "resource": {"buffer": a.buffer}},
                {"binding": 1, "resource": {"buffer": b.buffer}},
                {"binding": 2, "resource": {"buffer": result.buffer}},
                {"binding": 3, "resource": {"buffer": dims_buffer}}
            ]
        )
        
        # Create command encoder
        command_encoder = a.device.create_command_encoder()
        compute_pass = command_encoder.begin_compute_pass()
        compute_pass.set_pipeline(pipeline)
        compute_pass.set_bind_group(0, bind_group)
        compute_pass.dispatch_workgroups(
            (a.rows + 7) // 8,
            (b.cols + 7) // 8,
            1
        )
        compute_pass.end()
        
        # Submit command
        a.device.queue.submit([command_encoder.finish()])
        
        return result

    def webgpu_matrix_add(a: WebGPUMatrix, b: WebGPUMatrix) -> WebGPUMatrix:
        """Matrix addition using WebGPU"""
        if a.rows != b.rows or a.cols != b.cols:
            raise ValueError(f"Matrix dimensions don't match: {(a.rows, a.cols)} and {(b.rows, b.cols)}")
        
        result = WebGPUMatrix(a.rows, a.cols)
        result.data = a.data + b.data
        result.device.queue.write_buffer(result.buffer, 0, result.data.tobytes())
        return result

    def webgpu_matrix_subtract(a: WebGPUMatrix, b: WebGPUMatrix) -> WebGPUMatrix:
        """Matrix subtraction using WebGPU"""
        if a.rows != b.rows or a.cols != b.cols:
            raise ValueError(f"Matrix dimensions don't match: {(a.rows, a.cols)} and {(b.rows, b.cols)}")
        
        result = WebGPUMatrix(a.rows, a.cols)
        result.data = a.data - b.data
        result.device.queue.write_buffer(result.buffer, 0, result.data.tobytes())
        return result

    def webgpu_matrix_hadamard(a: WebGPUMatrix, b: WebGPUMatrix) -> WebGPUMatrix:
        """Element-wise multiplication using WebGPU"""
        if a.rows != b.rows or a.cols != b.cols:
            raise ValueError(f"Matrix dimensions don't match: {(a.rows, a.cols)} and {(b.rows, b.cols)}")
        
        result = WebGPUMatrix(a.rows, a.cols)
        result.data = a.data * b.data
        result.device.queue.write_buffer(result.buffer, 0, result.data.tobytes())
        return result

    def webgpu_matrix_scalar_multiply(a: WebGPUMatrix, scalar: float) -> WebGPUMatrix:
        """Scalar multiplication using WebGPU"""
        result = WebGPUMatrix(a.rows, a.cols)
        result.data = a.data * scalar
        result.device.queue.write_buffer(result.buffer, 0, result.data.tobytes())
        return result

    def webgpu_matrix_sum_axis(a: WebGPUMatrix, axis: Optional[int] = None) -> WebGPUMatrix:
        """Sum along axis using WebGPU"""
        if axis is None:
            result = WebGPUMatrix(1, 1)
            result.data = np.array([[a.data.sum()]], dtype=np.float32)
        elif axis == 0:
            result = WebGPUMatrix(1, a.cols)
            result.data = a.data.sum(axis=0, keepdims=True)
        else:  # axis == 1
            result = WebGPUMatrix(a.rows, 1)
            result.data = a.data.sum(axis=1, keepdims=True)
        
        result.device.queue.write_buffer(result.buffer, 0, result.data.tobytes())
        return result

    def webgpu_matrix_transpose(a: WebGPUMatrix) -> WebGPUMatrix:
        """Matrix transpose using WebGPU"""
        result = WebGPUMatrix(a.cols, a.rows)
        result.data = a.data.T
        result.device.queue.write_buffer(result.buffer, 0, result.data.tobytes())
        return result

    def webgpu_matrix_reshape(a: WebGPUMatrix, new_rows: int, new_cols: int) -> WebGPUMatrix:
        """Reshape matrix using WebGPU"""
        if a.rows * a.cols != new_rows * new_cols:
            raise ValueError("New dimensions must have same total size as original matrix")
        
        result = WebGPUMatrix(new_rows, new_cols)
        result.data = a.data.reshape(new_rows, new_cols)
        result.device.queue.write_buffer(result.buffer, 0, result.data.tobytes())
        return result

    def webgpu_matrix_random(rows: int, cols: int, low: float = -1.0, high: float = 1.0) -> WebGPUMatrix:
        """Create random matrix using WebGPU"""
        result = WebGPUMatrix(rows, cols)
        result.data = np.random.uniform(low, high, (rows, cols)).astype(np.float32)
        result.device.queue.write_buffer(result.buffer, 0, result.data.tobytes())
        return result

    def webgpu_matrix_zeros(rows: int, cols: int) -> WebGPUMatrix:
        """Create zero matrix using WebGPU"""
        return WebGPUMatrix(rows, cols)

    def webgpu_matrix_ones(rows: int, cols: int) -> WebGPUMatrix:
        """Create matrix of ones using WebGPU"""
        result = WebGPUMatrix(rows, cols)
        result.data = np.ones((rows, cols), dtype=np.float32)
        result.device.queue.write_buffer(result.buffer, 0, result.data.tobytes())
        return result

    def webgpu_matrix_eye(size: int) -> WebGPUMatrix:
        """Create identity matrix using WebGPU"""
        result = WebGPUMatrix(size, size)
        result.data = np.eye(size, dtype=np.float32)
        result.device.queue.write_buffer(result.buffer, 0, result.data.tobytes())
        return result
except (ImportError, ModuleNotFoundError, AttributeError) as e:
    # When wgpu is not available, create stub implementations
    import numpy as np
    from typing import List, Tuple, Optional, Union
    from .math_ops import Matrix
    
    print(f"WebGPU not available: {str(e)}")
    WEBGPU_AVAILABLE = False
    
    # Create a stub WebGPUMatrix class that raises an error when used
    class WebGPUMatrix:
        def __init__(self, *args, **kwargs):
            raise ImportError("WebGPU is not available. Please install the 'wgpu' package.")
    
    # Create stub functions that raise errors when called
    def _webgpu_not_available(*args, **kwargs):
        raise ImportError("WebGPU is not available. Please install the 'wgpu' package.")
    
    # Assign the stub function to all webgpu operations
    webgpu_matrix_multiply = _webgpu_not_available
    webgpu_matrix_add = _webgpu_not_available
    webgpu_matrix_subtract = _webgpu_not_available
    webgpu_matrix_hadamard = _webgpu_not_available
    webgpu_matrix_scalar_multiply = _webgpu_not_available
    webgpu_matrix_sum_axis = _webgpu_not_available
    webgpu_matrix_transpose = _webgpu_not_available
    webgpu_matrix_reshape = _webgpu_not_available
    webgpu_matrix_random = _webgpu_not_available
    webgpu_matrix_zeros = _webgpu_not_available
    webgpu_matrix_ones = _webgpu_not_available
    webgpu_matrix_eye = _webgpu_not_available 