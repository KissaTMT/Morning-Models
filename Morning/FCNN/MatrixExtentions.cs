using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Morning.FCNN
{
    public static class MatrixExtentions
    {
        public static Matrix<double> PointwiseOperation(this Matrix<double> source, Func<double, double> selector)
        {
            var result = CreateMatrix.DenseOfMatrix(source);

            for (var i = 0; i < result.RowCount; i++)
            {
                for(var j=0;j < result.ColumnCount; j++)
                {
                    result[i,j] = selector(source[i,j]);
                }
            }

            return result;
        }
    }
}
