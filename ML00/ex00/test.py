import unittest
from matrix import Matrix, Vector


class TestMatrix(unittest.TestCase):

    def test_matrix_init(self):
        m1 = Matrix([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
        m2 = Matrix((3, 2))
        self.assertEqual(m1.data, [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
        self.assertEqual(m1.shape, (3, 2))
        self.assertEqual(m2.data, [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
        self.assertEqual(m2.shape, (3, 2))

    def test_matrix_add_ok(self):
        m1 = Matrix([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
        m2 = Matrix([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
        m3 = m1 + m2
        self.assertEqual(m3.data, [[0.0, 2.0], [4.0, 6.0], [8.0, 10.0]])

    def test_matrix_add_fail(self):
        m1 = Matrix([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
        m3 = Matrix([[0.0, 2.0], [4.0, 6.0]])
        self.assertRaises(ValueError, m1.__add__, m3)

    def test_matrix_sub_ok(self):
        m1 = Matrix([[0.0, 2.0], [1.0, 4.0], [4.0, 10.0]])
        m2 = Matrix([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
        m3 = m1 - m2
        self.assertEqual(m3.data, [[0.0, 1.0], [-1.0, 1.0], [0.0, 5.0]])

    def test_matrix_sub_fail(self):
        m1 = Matrix([[0.0, 2.0], [1.0, 4.0], [4.0, 10.0]])
        m2 = Matrix([[0.0, 1.0], [2.0, 3.0]])
        self.assertRaises(ValueError, m1.__sub__, m2)

    def test_matrix_div_ok(self):
        m1 = Matrix([[0.0, 2.0], [1.0, 4.0], [4.0, 10.0]])
        m2 = m1 / 2
        self.assertEqual(m2.data, [[0.0, 1.0], [0.5, 2.0], [2.0, 5.0]])

    def test_matrix_div_fail(self):
        m1 = Matrix([[0.0, 2.0], [1.0, 4.0], [4.0, 10.0]])
        m2 = Matrix([[0.0, 1.0], [2.0, 3.0]])
        self.assertRaises(TypeError, m1.__truediv__, m2)

    def test_matrix_rdiv(self):
        m1 = Matrix([[0.0, 2.0], [1.0, 4.0], [4.0, 10.0]])
        self.assertRaises(TypeError, m1.__rtruediv__, 2)

    def test_matrix_mul_ok(self):
        m1 = Matrix([[0.0, 2.0], [1.0, 4.0], [4.0, 10.0]])
        m2 = m1 * 2
        m3 = Matrix([[0.0, 2.0, 2.0], [1.0, 4.0, 3.0]])
        m4 = m1 * m3
        m5 = Matrix([[0.0, 1.0, 2.0], [0.0, 2.0, 4.0]])
        v1 = Vector([[1], [2], [3]])
        m6 = m5 * v1
        self.assertEqual(m2.data, [[0.0, 4.0], [2.0, 8.0], [8.0, 20.0]])
        self.assertEqual(m4.data, [[2.0, 8.0, 6.0], [4.0, 18.0, 14.0], [10.0, 48.0, 38.0]])
        self.assertEqual(m6.data, ([[8], [16]]))
    def test_matrix_mul_fail(self):
        m1 = Matrix([[0.0, 2.0], [1.0, 4.0], [4.0, 10.0]])
        m2 = Matrix([[0.0, 1.0, 0.0], [2.0, 3.0, 0.0], [2.0, 3.0, 0.0]])
        self.assertRaises(ValueError, m1.__mul__, m2)
        self.assertRaises(ValueError, m1.__mul__, 'a')

    def test_matrix_transpose(self):
        m1 = Matrix([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
        m2 = m1.T()
        m3 = Matrix([[0., 2., 4.], [1., 3., 5.]])
        m4 = m3.T()
        self.assertEqual(m2.data, [[0., 2., 4.], [1., 3., 5.]])
        self.assertEqual(m2.shape, (2, 3))
        self.assertEqual(m3.shape, (2, 3))
        self.assertEqual(m4.data, m1.data)


class TestVector(unittest.TestCase):

    def test_vector_init_ok(self):
        v1 = Vector([[1, 2, 3]])
        v2 = Vector([[1], [2], [3]])
        self.assertEqual(v1.data, [[1, 2, 3]])
        self.assertEqual(v1.shape, (1, 3))
        self.assertEqual(v2.data, [[1], [2], [3]])
        self.assertEqual(v2.shape, (3, 1))

    def test_vector_init_fail(self):
        self.assertRaises(ValueError, Vector, [[1, 2], [3, 4]])
        self.assertRaises(ValueError, Vector, [[1, 2, 3], [4, 5, 6]])

    def test_vector_add_ok(self):
        v1 = Vector([[1, 2, 3]])
        v2 = Vector([[4, 5, 6]])
        v3 = v1 + v2
        self.assertEqual(v3.data, [[5, 7, 9]])

    def test_vector_add_fail(self):
        v1 = Vector([[1, 2, 3]])
        v2 = Vector([[4, 5]])
        self.assertRaises(ValueError, v1.__add__, v2)

    def test_vector_sub_ok(self):
        v1 = Vector([[1, 2, 3]])
        v2 = Vector([[4, 5, 6]])
        v3 = v1 - v2
        self.assertEqual(v3.data, [[-3, -3, -3]])

    def test_vector_sub_fail(self):
        v1 = Vector([[1, 2, 3]])
        v2 = Vector([[4, 5]])
        self.assertRaises(ValueError, v1.__sub__, v2)

    def test_vector_mul_ok(self):
        v1 = Vector([[1, 2, 3]])
        v2 = v1 * 2
        self.assertEqual(v2.data, [[2, 4, 6]])

    def test_vector_mul_fail(self):
        v1 = Vector([[1, 2, 3]])
        self.assertRaises(TypeError, v1.__mul__, 'a')

    def test_vector_div_ok(self):
        v1 = Vector([[2, 4, 6]])
        v2 = v1 / 2
        self.assertEqual(v2.data, [[1, 2, 3]])

    def test_vector_div_fail(self):
        v1 = Vector([[2, 4, 6]])
        self.assertRaises(TypeError, v1.__truediv__, 'a')

    def test_vector_dot_product_ok(self):
        v1 = Vector([[1, 2, 3]])
        v2 = Vector([[4, 5, 6]])
        result = v1.dot(v2)
        self.assertEqual(result, 32)

    def test_vector_dot_product_fail(self):
        v1 = Vector([[1, 2, 3]])
        v2 = Vector([[4, 5]])
        self.assertRaises(ValueError, v1.dot, v2)

    def test_vector_rmul(self):
        v1 = Vector([[1, 2, 3]])
        v2 = 2 * v1
        self.assertEqual(v2.data, [[2, 4, 6]])

    def test_vector_rdiv(self):
        v1 = Vector([[2, 4, 6]])
        self.assertRaises(TypeError, v1.__rtruediv__, 2)


