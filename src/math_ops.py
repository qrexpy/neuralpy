class Matrix:
    def __init__(self, rows, cols, data=None):
        self.rows = rows
        self.cols = cols
        if data is None:
            self.data = [[0.0 for _ in range(cols)] for _ in range(rows)]
        else:
            if len(data) != rows or any(len(row) != cols for row in data):
                raise ValueError("Data dimensions don't match specified size")
            self.data = [[float(val) for val in row] for row in data]

    def __getitem__(self, key):
        i, j = key
        return self.data[i][j]

    def __setitem__(self, key, value):
        i, j = key
        self.data[i][j] = float(value)

    def __str__(self):
        return "\n".join([" ".join(f"{x:8.4f}" for x in row) for row in self.data])

    def shape(self):
        return (self.rows, self.cols)
    
    def copy(self):
        """Create a deep copy of the matrix"""
        return Matrix(self.rows, self.cols, [row[:] for row in self.data])
    
    def sum(self):
        """Sum all elements in the matrix"""
        return sum(sum(row) for row in self.data)
    
    def mean(self):
        """Calculate the mean of all elements in the matrix"""
        return self.sum() / (self.rows * self.cols)
    
    def max(self):
        """Find the maximum element in the matrix"""
        return max(max(row) for row in self.data)
    
    def min(self):
        """Find the minimum element in the matrix"""
        return min(min(row) for row in self.data)
    
    def apply(self, func):
        """Apply a function to each element of the matrix"""
        result = Matrix(self.rows, self.cols)
        for i in range(self.rows):
            for j in range(self.cols):
                result[i, j] = func(self[i, j])
        return result

def matrix_multiply(a: Matrix, b: Matrix) -> Matrix:
    if a.cols != b.rows:
        raise ValueError(f"Matrix dimensions don't match: {a.shape()} and {b.shape()}")
    
    result = Matrix(a.rows, b.cols)
    for i in range(a.rows):
        for j in range(b.cols):
            sum_val = 0.0
            for k in range(a.cols):
                sum_val += a[i, k] * b[k, j]
            result[i, j] = sum_val
    return result

def matrix_add(a: Matrix, b: Matrix) -> Matrix:
    if a.rows != b.rows or a.cols != b.cols:
        raise ValueError(f"Matrix dimensions don't match: {a.shape()} and {b.shape()}")
    
    result = Matrix(a.rows, a.cols)
    for i in range(a.rows):
        for j in range(a.cols):
            result[i, j] = a[i, j] + b[i, j]
    return result

def matrix_subtract(a: Matrix, b: Matrix) -> Matrix:
    if a.rows != b.rows or a.cols != b.cols:
        raise ValueError(f"Matrix dimensions don't match: {a.shape()} and {b.shape()}")
    
    result = Matrix(a.rows, a.cols)
    for i in range(a.rows):
        for j in range(a.cols):
            result[i, j] = a[i, j] - b[i, j]
    return result

def matrix_transpose(a: Matrix) -> Matrix:
    result = Matrix(a.cols, a.rows)
    for i in range(a.rows):
        for j in range(a.cols):
            result[j, i] = a[i, j]
    return result

def matrix_hadamard(a: Matrix, b: Matrix) -> Matrix:
    """Element-wise multiplication"""
    if a.rows != b.rows or a.cols != b.cols:
        raise ValueError(f"Matrix dimensions don't match: {a.shape()} and {b.shape()}")
    
    result = Matrix(a.rows, a.cols)
    for i in range(a.rows):
        for j in range(a.cols):
            result[i, j] = a[i, j] * b[i, j]
    return result

def matrix_scalar_multiply(a: Matrix, scalar: float) -> Matrix:
    result = Matrix(a.rows, a.cols)
    for i in range(a.rows):
        for j in range(a.cols):
            result[i, j] = a[i, j] * scalar
    return result

def matrix_scalar_add(a: Matrix, scalar: float) -> Matrix:
    """Add a scalar to each element of the matrix"""
    result = Matrix(a.rows, a.cols)
    for i in range(a.rows):
        for j in range(a.cols):
            result[i, j] = a[i, j] + scalar
    return result

def matrix_sum_axis(a: Matrix, axis: int = None) -> Matrix:
    """
    Sum elements along the specified axis.
    If axis is None, sum all elements and return a 1x1 matrix.
    """
    if axis is None:
        return Matrix(1, 1, [[a.sum()]])
    
    if axis == 0:  # Sum along columns
        result = Matrix(1, a.cols)
        for j in range(a.cols):
            sum_val = 0.0
            for i in range(a.rows):
                sum_val += a[i, j]
            result[0, j] = sum_val
        return result
    
    elif axis == 1:  # Sum along rows
        result = Matrix(a.rows, 1)
        for i in range(a.rows):
            sum_val = 0.0
            for j in range(a.cols):
                sum_val += a[i, j]
            result[i, 0] = sum_val
        return result
    
    else:
        raise ValueError("Axis must be 0, 1, or None")

def exp(x: float) -> float:
    """
    Compute e^x using Taylor series approximation
    """
    result = 1.0
    term = 1.0
    for i in range(1, 100):  # 100 terms for good precision
        term *= x / i
        result += term
        if abs(term) < 1e-10:  # Stop if term becomes very small
            break
    return result

def sigmoid(x: float) -> float:
    """
    Compute sigmoid function 1/(1 + e^(-x))
    """
    if x < -100:
        return 0.0
    if x > 100:
        return 1.0
    return 1.0 / (1.0 + exp(-x))

def matrix_sigmoid(a: Matrix) -> Matrix:
    result = Matrix(a.rows, a.cols)
    for i in range(a.rows):
        for j in range(a.cols):
            result[i, j] = sigmoid(a[i, j])
    return result

def relu(x: float) -> float:
    """
    Compute ReLU function max(0, x)
    """
    return max(0.0, x)

def matrix_relu(a: Matrix) -> Matrix:
    result = Matrix(a.rows, a.cols)
    for i in range(a.rows):
        for j in range(a.cols):
            result[i, j] = relu(a[i, j])
    return result

def leaky_relu(x: float, alpha: float = 0.01) -> float:
    """
    Compute Leaky ReLU function max(αx, x)
    """
    return x if x > 0 else alpha * x

def matrix_leaky_relu(a: Matrix, alpha: float = 0.01) -> Matrix:
    result = Matrix(a.rows, a.cols)
    for i in range(a.rows):
        for j in range(a.cols):
            result[i, j] = leaky_relu(a[i, j], alpha)
    return result

def sqrt(x: float) -> float:
    """
    Compute square root using Newton's method
    """
    if x < 0:
        raise ValueError("Cannot compute square root of negative number")
    if x == 0:
        return 0.0
    
    guess = x / 2.0
    for _ in range(50):  # 50 iterations should be enough for convergence
        new_guess = (guess + x / guess) / 2.0
        if abs(new_guess - guess) < 1e-10:
            return new_guess
        guess = new_guess
    return guess

def random_uniform(low: float, high: float) -> float:
    """
    Generate a random number between low and high
    Using linear congruential generator
    """
    # Using some prime numbers for good randomness
    a = 1597
    c = 51749
    m = 244944
    
    # Get current time in microseconds as seed
    import time
    seed = int(time.time() * 1000000) % m
    
    # Generate random number
    random_val = ((a * seed + c) % m) / m
    return low + (high - low) * random_val

def matrix_random(rows: int, cols: int, low: float, high: float) -> Matrix:
    result = Matrix(rows, cols)
    for i in range(rows):
        for j in range(cols):
            result[i, j] = random_uniform(low, high)
    return result

def matrix_zeros(rows: int, cols: int) -> Matrix:
    """Create a matrix filled with zeros"""
    return Matrix(rows, cols)

def matrix_ones(rows: int, cols: int) -> Matrix:
    """Create a matrix filled with ones"""
    return Matrix(rows, cols, [[1.0 for _ in range(cols)] for _ in range(rows)])

def matrix_eye(size: int) -> Matrix:
    """Create an identity matrix of the given size"""
    result = Matrix(size, size)
    for i in range(size):
        result[i, i] = 1.0
    return result

def matrix_reshape(a: Matrix, rows: int, cols: int) -> Matrix:
    """Reshape a matrix to the given dimensions"""
    if a.rows * a.cols != rows * cols:
        raise ValueError(f"Cannot reshape matrix of size {a.shape()} to ({rows}, {cols})")
    
    result = Matrix(rows, cols)
    flat_data = [a[i, j] for i in range(a.rows) for j in range(a.cols)]
    
    for i in range(rows):
        for j in range(cols):
            result[i, j] = flat_data[i * cols + j]
    
    return result

def matrix_concatenate(a: Matrix, b: Matrix, axis: int = 0) -> Matrix:
    """
    Concatenate two matrices along the specified axis.
    axis=0 means concatenate vertically (stacking rows)
    axis=1 means concatenate horizontally (stacking columns)
    """
    if axis == 0:
        if a.cols != b.cols:
            raise ValueError(f"Cannot concatenate matrices with different column counts: {a.shape()} and {b.shape()}")
        
        result = Matrix(a.rows + b.rows, a.cols)
        
        # Copy first matrix
        for i in range(a.rows):
            for j in range(a.cols):
                result[i, j] = a[i, j]
        
        # Copy second matrix
        for i in range(b.rows):
            for j in range(b.cols):
                result[i + a.rows, j] = b[i, j]
        
        return result
    
    elif axis == 1:
        if a.rows != b.rows:
            raise ValueError(f"Cannot concatenate matrices with different row counts: {a.shape()} and {b.shape()}")
        
        result = Matrix(a.rows, a.cols + b.cols)
        
        # Copy first matrix
        for i in range(a.rows):
            for j in range(a.cols):
                result[i, j] = a[i, j]
        
        # Copy second matrix
        for i in range(b.rows):
            for j in range(b.cols):
                result[i, j + a.cols] = b[i, j]
        
        return result
    
    else:
        raise ValueError("Axis must be 0 or 1") 