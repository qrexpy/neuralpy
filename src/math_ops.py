from typing import List

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

def matrix_scalar_subtract(a: Matrix, scalar: float) -> Matrix:
    """
    Subtract a scalar from each element of a matrix
    """
    result = Matrix(a.rows, a.cols)
    for i in range(a.rows):
        for j in range(a.cols):
            result[i, j] = a[i, j] - scalar
    return result

def matrix_sum_axis(a: Matrix, axis: int = None) -> Matrix:
    """
    Sum matrix elements along specified axis
    axis: 0 for columns, 1 for rows, None for all elements
    """
    if axis is None:
        # Sum all elements
        total = sum(sum(row) for row in a.data)
        return Matrix(1, 1, [[total]])
    elif axis == 0:
        # Sum along columns
        sums = [sum(row[j] for row in a.data) for j in range(a.cols)]
        return Matrix(1, a.cols, [sums])
    else:  # axis == 1
        # Sum along rows
        sums = [[sum(row)] for row in a.data]
        return Matrix(a.rows, 1, sums)

def exp(x: float) -> float:
    """
    Compute exponential using Taylor series
    """
    if x > 100:
        return float('inf')
    if x < -100:
        return 0.0
    
    result = 1.0
    term = 1.0
    for i in range(1, 20):  # 20 terms should be enough for good precision
        term *= x / i
        result += term
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
    
    # Initial guess
    guess = x / 2.0
    
    # Newton's method iteration
    for _ in range(10):  # 10 iterations should be enough for good precision
        guess = (guess + x / guess) / 2.0
    
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
    return Matrix(rows, cols, [[0.0 for _ in range(cols)] for _ in range(rows)])

def matrix_ones(rows: int, cols: int) -> Matrix:
    """Create a matrix filled with ones"""
    return Matrix(rows, cols, [[1.0 for _ in range(cols)] for _ in range(rows)])

def matrix_eye(size: int) -> Matrix:
    """Create an identity matrix of the given size"""
    data = [[1.0 if i == j else 0.0 for j in range(size)] for i in range(size)]
    return Matrix(size, size, data)

def matrix_reshape(a: Matrix, new_rows: int, new_cols: int) -> Matrix:
    """
    Reshape matrix to new dimensions
    """
    if a.rows * a.cols != new_rows * new_cols:
        raise ValueError("New dimensions must have same total size as original matrix")
    
    # Flatten the matrix
    flat_data = [val for row in a.data for val in row]
    
    # Reshape to new dimensions
    new_data = []
    for i in range(new_rows):
        new_data.append(flat_data[i * new_cols:(i + 1) * new_cols])
    
    return Matrix(new_rows, new_cols, new_data)

def matrix_concatenate(a: Matrix, b: Matrix, axis: int = 0) -> Matrix:
    """
    Concatenate two matrices along specified axis
    axis: 0 for vertical concatenation, 1 for horizontal
    """
    if axis == 0:
        if a.cols != b.cols:
            raise ValueError("Matrices must have same number of columns for vertical concatenation")
        new_data = a.data + b.data
        return Matrix(a.rows + b.rows, a.cols, new_data)
    else:
        if a.rows != b.rows:
            raise ValueError("Matrices must have same number of rows for horizontal concatenation")
        new_data = [row_a + row_b for row_a, row_b in zip(a.data, b.data)]
        return Matrix(a.rows, a.cols + b.cols, new_data)

def matrix_split(a: Matrix, indices: List[int], axis: int = 0) -> List[Matrix]:
    """
    Split a matrix into multiple matrices along the specified axis.
    
    Args:
        a: Input matrix to split
        indices: List of indices where to split the matrix
        axis: Axis along which to split (0 for rows, 1 for columns)
    
    Returns:
        List of matrices after splitting
    """
    if not indices:
        return [a]
    
    # Sort indices and remove duplicates
    indices = sorted(list(set(indices)))
    
    # Validate indices
    if axis == 0:
        if any(i <= 0 or i >= a.rows for i in indices):
            raise ValueError("Split indices must be between 0 and number of rows")
    else:
        if any(i <= 0 or i >= a.cols for i in indices):
            raise ValueError("Split indices must be between 0 and number of columns")
    
    # Add start and end indices
    indices = [0] + indices + [a.rows if axis == 0 else a.cols]
    
    # Split the matrix
    result = []
    for i in range(len(indices) - 1):
        if axis == 0:
            # Split along rows
            data = [a.data[j] for j in range(indices[i], indices[i + 1])]
            result.append(Matrix(len(data), a.cols, data))
        else:
            # Split along columns
            data = [[a.data[j][k] for k in range(indices[i], indices[i + 1])] 
                   for j in range(a.rows)]
            result.append(Matrix(a.rows, len(data[0]), data))
    
    return result 