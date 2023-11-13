class Matrix:
    def __init__(self, input_data):
        self.data = []
        self.shape = ()

        if not isinstance(input_data, (tuple, list)):
            raise TypeError("Unexpected data type, must be a list or a tuple.")
        if isinstance(input_data, list):
            if any([not isinstance(l, list) for l in input_data]):
                raise TypeError("Incorrect type of data.")
            if len(input_data) == 0:
                raise ValueError("Incorrect length.")
            for l_line in input_data:
                if any([not isinstance(elem, (int, float)) for elem in l_line]):
                    raise TypeError("Incorrect type of data.")
                if len(l_line) != len(input_data[0]):
                    raise ValueError("Incorrect length.")
            self.data = input_data
            self.shape = (len(input_data), len(input_data[0]))
        if isinstance(input_data, tuple):
            if len(input_data) != 2:
                raise ValueError("Incorrect length.")
            if not isinstance(input_data[0], (int, float)) or not isinstance(input_data[1], (int, float)):
                raise TypeError("Incorrect type of data.")
            if input_data[0] <= 0 or input_data[1] <= 0:
                raise ValueError("Incorrect data.")
            self.data = [[0.0 for i in range(input_data[1])] for i in range(input_data[0])]
            self.shape = input_data

    def __add__(self, other):
        if not isinstance(other, (Matrix, Vector)):
            raise ArithmeticError("Right member of the addition operator is not a Matrix.")
        if self.shape != other.shape:
            raise ValueError('Shape must be the same for addition')
        result = Matrix(self.shape)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                result.data[i][j] = self.data[i][j] + other.data[i][j]
        return result

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        if not isinstance(other, (Matrix, Vector)):
            raise ArithmeticError("Right member of the addition operator is not a Matrix.")
        if self.shape != other.shape:
            raise ValueError('Shape must be the same for subtraction')
        result = Matrix(self.shape)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                result.data[i][j] = self.data[i][j] - other.data[i][j]
        return result

    def __rsub__(self, other):
        return other - self

    def __truediv__(self, scalar):
        if not isinstance(scalar, (int, float)):
            raise TypeError("Division can only be performed with a scalar")
        if scalar == 0:
            raise ZeroDivisionError("Cannot divide by zero")
        result = Matrix(self.shape)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                result.data[i][j] = self.data[i][j] / scalar
        return result

    def __rtruediv__(self, scalar):
        raise TypeError("Cannot divide a scalar by a Matrix")

    def __mul__(self, other):
        if not isinstance(other, (Matrix, int, float, Vector)):
            raise ValueError("Right member of the multiplication operator is incorrect.")

        if isinstance(other, (int, float)):
            result = Matrix(self.shape)
            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    result.data[i][j] = self.data[i][j] * other
            return result

        elif isinstance(other, Matrix):
            if self.shape[1] != other.shape[0]:
                raise ValueError("Incompatible dimensions for matrix multiplication")
            result = Matrix((self.shape[0], other.shape[1]))
            for i in range(self.shape[0]):
                for j in range(other.shape[1]):
                    for k in range(self.shape[1]):
                        result.data[i][j] += self.data[i][k] * other.data[k][j]
            return result

        elif isinstance(other, Vector):
            if self.shape[1] != other.shape[0]:
                raise ValueError("Incompatible dimensions for matrix-vector multiplication")
            result = Vector((self.shape[0], 1))
            for i in range(self.shape[0]):
                for j in range(other.shape[0]):  # Usually, Vector shape[1] should be 1
                    result.data[i][0] += self.data[i][j] * other.data[j][0]
            return result

        raise ValueError("Multiplication can only be performed with a scalar, vector, or another Matrix.")

    def __rmul__(self, other):
        return self * other

    def __str__(self):
        return '\n'.join(['\t'.join([str(cell) for cell in row]) for row in self.data])

    def __repr__(self):
        return f'Matrix({self.data})'

    def T(self):
        result = Matrix((self.shape[1], self.shape[0]))
        for i in range(self.shape[1]):
            for j in range(self.shape[0]):
                result.data[i][j] = self.data[j][i]
        return result


class Vector(Matrix):
    def __init__(self, *args):
        super().__init__(*args)
        if all([l != 1 for l in self.shape]):
            raise ValueError("Data should have a dimension with size 1: shape == (n, 1) or (1, m).")

    def dot(self, v):
        if not isinstance(v, Vector):
            raise TypeError("Dot product can only be performed with another Vector")
        if self.shape != v.shape or self.shape[0] != 1 and self.shape[1] != 1:
            raise ValueError("Vectors must have the same dimensions for dot product")
        result = 0
        if self.shape[0] == 1:
            for i in range(self.shape[1]):
                result += self.data[0][i] * v.data[0][i]
        else:
            for i in range(self.shape[0]):
                result += self.data[i][0] * v.data[i][0]
        return result

    def __add__(self, other):
        result_matrix = super().__add__(other)
        return Vector(result_matrix.data)

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        result_matrix = super().__sub__(other)
        return Vector(result_matrix.data)

    def __rsub__(self, other):
        return other - self

    def __mul__(self, other):
        if not isinstance(other, (Vector, int, float)):
            raise TypeError("Incorrect type of data.")
        if isinstance(other, Vector):
            if self.shape != other.shape:
                raise ValueError("Can only multiply Vectors of the same shape element-wise")

        result_matrix = super().__mul__(other)
        return Vector(result_matrix.data)

    def __rmul__(self, other):
        if isinstance(other, Matrix):
            return other.__mul__(self)
        result = super().__rmul__(other)
        return Vector(result.data)

    def __truediv__(self, scalar):
        result_matrix = super().__truediv__(scalar)
        return Vector(result_matrix.data)

    def __rtruediv__(self, scalar):
        raise TypeError("Cannot divide a scalar by a Vector")
