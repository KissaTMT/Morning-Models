using Morning.ActivationFunctions;
using MathNet.Numerics.LinearAlgebra;

namespace Morning.TLNN
{
    public class TLNNLA
    {
        public Dictionary<int,List<Vector<double>>> LogW = new Dictionary<int, List<Vector<double>>>();
        public Dictionary<int,List<Vector<double>>> LogU = new Dictionary<int, List<Vector<double>>>();
        public Dictionary<int,List<Vector<double>>> LogV = new Dictionary<int, List<Vector<double>>>();
        public Dictionary<int,List<Vector<double>>> Logh = new Dictionary<int, List<Vector<double>>>();
        public Dictionary<int,List<Vector<double>>> Logz = new Dictionary<int, List<Vector<double>>>();
        public Dictionary<int, List<double>> Logn = new Dictionary<int, List<double>>();

        
        public Dictionary<int,List<Vector<double>>> LogDV = new Dictionary<int, List<Vector<double>>>();
        public Dictionary<int,List<Vector<double>>> LogDh = new Dictionary<int, List<Vector<double>>>();
        public Dictionary<int,List<Vector<double>>> LogDz = new Dictionary<int, List<Vector<double>>>();
        public Dictionary<int,List<Vector<double>>> LogDW = new Dictionary<int, List<Vector<double>>>();
        public Dictionary<int,List<Vector<double>>> LogDU = new Dictionary<int, List<Vector<double>>>();
        public Dictionary<int, List<double>> LogDn = new Dictionary<int, List<double>>();

        private Vector<double> h;
        private Vector<double> z;
        private Vector<double> W;
        private Vector<double> U;
        private Vector<double> V;
        private double n;

        private IActivationFunction _f = new Tanh();

        public TLNNLA(int hiddenCount)
        {
            W = CreateVector.Dense<double>(hiddenCount);
            U = CreateVector.Dense<double>(hiddenCount);
            V = CreateVector.Dense<double>(hiddenCount);

            InitWeights();


        }
        public double Forward(double t)
        {
            z = t * W + U;
            h = z.Map(_f.Activate);
            n = h.PointwiseMultiply(U).Sum();
            return n;
        }

        public List<double> Train(double[] T, double[] Y, double errorThresholdValue, int epochs)
        {
            var errors = new List<double>();

            var previousError = 1000000.0;
            var error = previousError;
            var alpha = 0.1;

            for(var k=0;k<epochs;k++)
            {
                error = MSE(T, Y);
                errors.Add(error);

                //if (error < previousError) alpha /= 10;
                //else alpha *= 10;
                //previousError = error;

                LogW[k] = new List<Vector<double>>();
                LogU[k] = new List<Vector<double>>();
                LogV[k] = new List<Vector<double>>();
                Logz[k] = new List<Vector<double>>();
                Logh[k] = new List<Vector<double>>();
                Logn[k] = new List<double>();

                LogDW[k] = new List<Vector<double>>();
                LogDU[k] = new List<Vector<double>>();
                LogDV[k] = new List<Vector<double>>();
                LogDz[k] = new List<Vector<double>>();
                LogDh[k] = new List<Vector<double>>();
                LogDn[k] = new List<double>();

                for (var i = 0; i < T.Length; i++)
                {
                    Forward(T[i]);

                    var de_dn = 2*(n - Y[i]);
                    var de_dV = de_dn * h;
                    var de_dh = de_dn * V;
                    var de_dz = de_dh.PointwiseMultiply(z.Map(_f.DerivativeActivateFunction));
                    var de_dU = de_dz;
                    var de_dW = de_dz * T[i];

                    V -= alpha * de_dV;
                    U -= alpha * de_dU;
                    W -= alpha * de_dW;

                    LogW[k].Add(W);
                    LogU[k].Add(U);
                    LogV[k].Add(V);
                    Logz[k].Add(z);
                    Logh[k].Add(h);
                    Logn[k].Add(n);

                    LogDW[k].Add(de_dW);
                    LogDU[k].Add(de_dU);
                    LogDV[k].Add(de_dV);
                    LogDz[k].Add(de_dz);
                    LogDh[k].Add(de_dh);
                    LogDn[k].Add(de_dn);

                    //LM algorithm
                    //var de_dV2 = de_dV.ToColumnMatrix() * de_dV.ToRowMatrix();
                    //var de_dU2 = de_dU.ToColumnMatrix() * de_dU.ToRowMatrix();
                    //var de_dW2 = de_dW.ToColumnMatrix() * de_dW.ToRowMatrix();
                    //V -= (de_dV2 + alpha * CreateMatrix.DenseIdentity<double>(de_dV2.RowCount)).Inverse() * de_dV;
                    //U -= (de_dU2 + alpha * CreateMatrix.DenseIdentity<double>(de_dU2.RowCount)).Inverse() * de_dU;
                    //W -= (de_dW2 + alpha * CreateMatrix.DenseIdentity<double>(de_dW2.RowCount)).Inverse() * de_dW;
                }
                if (error < errorThresholdValue) break;
            }
            return errors;
        }
        private double MSE(double[] x, double[] y)
        {
            var n = x.Length;
            var sum = 0.0;
            for (var i = 0; i < n; i++)
            {
                sum += Math.Pow(y[i] - Forward(x[i]),2);
            }
            return sum/n;
        }
        private void InitWeights()
        {
            var rnd = new Random();
            var b = 0.7 * W.Count;
            for (var i = 0; i < U.Count; i++)
            {
                U[i] = -b + rnd.NextDouble() * b;
            }
            var sum = 0.0;
            for (var i = 0; i < W.Count; i++)
            {
                W[i] = -0.5 + rnd.NextDouble();
                sum += Math.Pow(W[i], 2);
            }
            var s = Math.Sqrt(sum);
            for (var i = 0; i < W.Count; i++)
            {
                W[i] = b * W[i] / s;
            }
            for (var i = 0; i < V.Count; i++)
            {
                V[i] = -0.5 + rnd.NextDouble();
            }
        }
    }
}