using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Morning.TLNN
{
    public class Matrix
    {
        public uint RowCount => _row;
        public uint ColumnCount => _column;
        public double[][] Data => _data;

        private double[][] _data;
        private uint _row;
        private uint _column;

        public static Matrix operator +(Matrix left, Matrix right) => left.Add(right);
        public static Matrix operator -(Matrix left, Matrix right) => left.Add(-1 * right);
        public static Matrix operator *(Matrix left, Matrix right) => left.Multiply(right);
        public static Matrix operator *(Matrix m, double scalar) => m.Multiply(scalar);
        public static Matrix operator *(double scalar, Matrix m) => m.Multiply(scalar);
        public static Matrix operator ^(Matrix left, Matrix right) => left.AdamarMultiply(right);
        public double[] this[int index] => _data[index];
        public Matrix(uint row, uint column)
        {
            Init(row, column);
        }
        public Matrix(double[][] data) {
            _data = data;
        }
        public static Matrix One(uint row, uint column)
        {
            var result = new Matrix(row, column);

            for(var n = 0; n < result._row; n++)
            {
                for(var m = 0; m < result._column; m++)
                {
                    if (n == m) result[n][m] = 1;
                    else result[n][m] = 0;
                }
            }

            return result;
        }
        public Matrix Add(Matrix other)
        {
            var result = new Matrix(_row, _column);
            for (var n = 0; n < _row; n++)
            {
                for (var m = 0; m < _column; m++)
                {
                    result[n][m] = _data[n][m] + other[n][m];
                }
            }
            return result;
        }
        public Matrix Multiply(Matrix other)
        {
            var result = new Matrix(_row, other._column);

            for (var i = 0; i < _row; i++)
            {
                for (var j = 0; j < other._column; j++)
                {
                    for (var k = 0; k < _column; k++)
                    {
                        result[i][j] += _data[i][k] * other[k][j];
                    }
                }
            }

            return result;
        }
        public Matrix Multiply(double scalar)
        {
            var result = new Matrix(_row, _column);

            for (var n = 0; n < _row; n++)
            {
                for (var m = 0; m < _column; m++)
                {
                    result[n][m] = _data[n][m] * scalar;
                }
            }

            return result;
        }
        public Matrix AdamarMultiply(Matrix other)
        {
            var result = new Matrix(1, _column);

            for(var i =0;i< _column; i++)
            {
                result[0][i] = _data[0][i] * other._data[0][i];
            }

            return result;
        } 
        public Matrix Transponse()
        {
            var result = new Matrix(_column, _row);
            for (int n = 0; n < _row; n++)
            {
                for (int m = 0; m < _column; m++)
                {
                    result[m][n] = _data[n][m];
                }
            }
            return result;
        }
        public double Det()
        {
            if(_row != _column ) return double.NaN;
            return Det(this);
        }
        public double AD(uint i, uint j)
        {
            return ((i + j) % 2 == 0 ? 1 : -1) * Det(DecreaseMatrix(i, j));
        }
        public Matrix Inverse()
        {
            var det = Det();
            var result = new Matrix(_row,_column);

            for (uint n = 0; n < result._row; n++)
            {
                for (uint m = 0; m < result._column; m++)
                {
                    result[(int)n][m] = AD(n, m);
                }
            }

            return result.Transponse().Multiply(1 / det);
        }
        public Matrix Map(Func<double, double> selector)
        {
            for (var i = 0; i < _row; i++)
            {
                for (var j = 0; j < _column; j++)
                {
                    _data[i][j] = selector(_data[i][j]);
                }
            }
            return this;
        }
        public double Sum()
        {
            var sum = 0.0;

            for(var n = 0; n < _row; n++)
            {
                for(var  m = 0; m < _column; m++)
                {
                    sum+= _data[n][m];
                }
            }

            return sum;
        }
        private void Init(uint n, uint m)
        {
            _row = n;
            _column = m;
            _data = new double[n][];
            for (var i = 0; i < n; i++)
            {
                _data[i] = new double[m];
            }
        }
        private double Det(Matrix m)
        {
            var order = m._row;
            if (order > 2)
            {
                double result = 0.0;
                for (int i = 0; i < order; i++)
                {
                    result += ((0 + i) % 2 == 0 ? 1 : -1) * m._data[0][i] * Det(m.DecreaseMatrix(0, (uint)i));
                }
                return result;
            }
            else if (order == 2)
            {
                return m[0][0] * m[1][1] - m[1][0] * m[0][1];
            }
            else
            {
                return m[0][0];
            }
        }
        private double AD(Matrix m, uint i, uint j)
        {
            var order = m._row;

            if (order > 2)
            {
                double result = 0.0;
                for (int n = 0; n < order; n++)
                {
                    result += ((i + j) % 2 == 0 ? 1 : -1) * Det(m.DecreaseMatrix(i, j));
                }
                return result;
            }
            else if (order == 2)
            {
                return m[0][0] * m[1][1] - m[1][0] * m[0][1];
            }
            else
            {
                return m[0][0];
            }
        }
        private Matrix DecreaseMatrix(uint i, uint j)
        {
            var result = new Matrix(_row -1, _column - 1);

            int x = 0;
            for (int n = 0; n < _row; n++)
            {
                if (n == i) continue;
                int y = 0;
                for (int m = 0; m < _column; m++)
                {
                    if (m == j) continue;
                    result[x][y] = _data[n][m];
                    y++;
                }
                x++;
            }

            return result;
        }
    }
}
