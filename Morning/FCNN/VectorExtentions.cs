using MathNet.Numerics.LinearAlgebra;
namespace Morning.FCNN
{
    public static class VectorExtentions
    {
        public static Vector<double> PointwiseOperation(this Vector<double> source, Func<double, double> selector)
        {
            var result = CreateVector.DenseOfVector(source);

            for(var i = 0; i < result.Count; i++)
            {
                result[i] = selector(source[i]);
            }

            return result;
        }
        public static Vector<double> Softmax(this Vector<double> vector)
        {
            var result = CreateVector.Dense<double>(vector.Count);
            result = vector.PointwiseOperation(Math.Exp);
            var sum = result.Sum();
            return result.PointwiseOperation(i => i / sum);
        }
    }
}
