using MathNet.Numerics.LinearAlgebra;
using Morning.ActivationFunctions;

namespace Morning.FCNN
{
    public class FCNN
    {
        public Matrix<double>[] Weights;
        public Vector<double>[] Biases;

        public IActivationFunction ActivationFunction = new Sigmoid();

        private Vector<double>[] _inputs;
        private Vector<double>[] _outputs;
        private Vector<double>[] _activatedOutputs;
        private Vector<double>[] _deltas;

        private int _layerCount;

        public FCNN(params int[] neuronsOnLayers)
        {
            _layerCount = neuronsOnLayers.Length-1;

            Weights = new Matrix<double>[_layerCount];
            Biases = new Vector<double>[_layerCount];

            _inputs = new Vector<double>[_layerCount];
            _outputs = new Vector<double>[_layerCount];
            _activatedOutputs = new Vector<double>[_layerCount];
            _deltas = new Vector<double>[_layerCount];

            for (var i = 1; i < neuronsOnLayers.Length; i++)
            {
                Weights[i - 1] = CreateMatrix.Random<double>(neuronsOnLayers[i - 1], neuronsOnLayers[i]);
                Biases[i - 1] = CreateVector.Dense<double>(neuronsOnLayers[i]);

                _inputs[i - 1] = CreateVector.Dense<double>(neuronsOnLayers[i - 1]);
                _outputs[i - 1] = CreateVector.Dense<double>(neuronsOnLayers[i]);
                _activatedOutputs[i - 1] = CreateVector.Dense<double>(neuronsOnLayers[i]);
                _deltas[i - 1] = CreateVector.Dense<double>(neuronsOnLayers[i]);
            }
        }
        public FCNN(double[][,] weights, double[][] biases)
        {
            _layerCount = weights.Length;

            Weights = new Matrix<double>[_layerCount];
            Biases = new Vector<double>[_layerCount];

            _inputs = new Vector<double>[_layerCount];
            _outputs = new Vector<double>[_layerCount];
            _activatedOutputs = new Vector<double>[_layerCount];
            _deltas = new Vector<double>[_layerCount];

            for (var i = 0; i < weights.Length; i++)
            {
                Weights[i] = CreateMatrix.DenseOfArray(weights[i]);
                Biases[i] = CreateVector.DenseOfArray(biases[i]);

                _inputs[i] = CreateVector.Dense<double>(Weights[i].RowCount);
                _outputs[i] = CreateVector.Dense<double>(Weights[i].ColumnCount);
                _activatedOutputs[i] = CreateVector.Dense<double>(Weights[i].ColumnCount);
                _deltas[i] = CreateVector.Dense<double>(Weights[i].ColumnCount);
            }
        }
        public double[] Predict(double[] inputs)
        {
            return Predict(CreateVector.Dense(inputs)).AsArray();
        }

        public double[] Train(double[,] outputs, double[,] inputs, double errorThresholdValue, double learningRate, int epochNumber)
        {
            var inputMatrix = CreateMatrix.DenseOfArray(inputs);
            var outputMatrix = CreateMatrix.DenseOfArray(outputs);

            var errors = new LinkedList<double>();

            for (var i = 0; i < epochNumber; i++)
            {
                var error = 0.0;
                for (var k = 0; k < inputMatrix.RowCount; k++)
                {
                    var input = inputMatrix.Row(k);
                    var output = outputMatrix.Row(k);
                    Predict(input);
                    error += Backpropagation(output, learningRate);
                }
                errors.AddLast(error / inputMatrix.RowCount);
                if (errors.Last() <= errorThresholdValue) break;
            }
            return errors.ToArray();
        }
        
        public double[] Train(double[,] outputs, double[,] inputs, double learningRate, int epoch)
        {
            var errors = new double[epoch];
            var inputMatrix = CreateMatrix.DenseOfArray(inputs);
            var outputMatrix = CreateMatrix.DenseOfArray(outputs);
            for(var i = 0; i < epoch; i++)
            {
                var error = 0.0;
                for (var k = 0; k < inputMatrix.RowCount; k++)
                {
                    var input = inputMatrix.Row(k);
                    var output = outputMatrix.Row(k);
                    Predict(input);
                    error += Backpropagation(output, learningRate);
                }
                errors[i] = error / inputMatrix.RowCount;
            }
            return errors;
        }
        private Vector<double> Predict(Vector<double> inputs)
        {
            _inputs[0] = inputs.Clone();
            _outputs[0] = _inputs[0] * Weights[0] + Biases[0];
            _activatedOutputs[0] = _outputs[0].PointwiseOperation(ActivationFunction.Activate);


            for (var layerIndex = 1; layerIndex < _layerCount - 1; layerIndex++)
            {
                _inputs[layerIndex] = _activatedOutputs[layerIndex].Clone();
                _outputs[layerIndex] = _inputs[layerIndex] * Weights[layerIndex] + Biases[layerIndex];
                _activatedOutputs[layerIndex] = _outputs[layerIndex].PointwiseOperation(ActivationFunction.Activate);
            }

            _inputs[_layerCount - 1] = _activatedOutputs[_layerCount - 2].Clone();
            _outputs[_layerCount - 1] = _inputs[_layerCount - 1] * Weights[_layerCount - 1] + Biases[_layerCount - 1];
            _activatedOutputs[_layerCount - 1] = _outputs[_layerCount - 1].PointwiseOperation(ActivationFunction.Activate);

            return _activatedOutputs[_layerCount - 1];
        }
        private double Backpropagation(Vector<double> expected, double alpha)
        {
            var lastlayerIndex = _layerCount - 1;

            var difference = _activatedOutputs[lastlayerIndex] - expected;

            var delta = difference;
            var deltaWeight = alpha * _inputs[lastlayerIndex].ToColumnMatrix() * difference.ToRowMatrix();
            var deltaBias = alpha * delta;

            UpdateWeights(lastlayerIndex, delta, deltaBias, deltaWeight);

            for (var layerIndex = lastlayerIndex - 1; layerIndex >= 0; layerIndex--)
            {
                delta = (_deltas[layerIndex + 1] * Weights[layerIndex + 1].Transpose()).PointwiseMultiply((_outputs[layerIndex].PointwiseOperation(ActivationFunction.DerivativeActivateFunction)));
                deltaWeight = alpha * _inputs[layerIndex].ToColumnMatrix() * delta.ToRowMatrix();
                deltaBias = alpha * delta;

                UpdateWeights(layerIndex, delta, deltaBias, deltaWeight);
            }
            var error = difference.Sum() / difference.Count;
            
            return (error * error) / 2;
        }
        private void UpdateWeights(int layerIndex,Vector<double> delta, Vector<double> deltaBias, Matrix<double> deltaWeight)
        {
            _deltas[layerIndex] = delta;
            Weights[layerIndex] -= deltaWeight;
            Biases[layerIndex] -= deltaBias;
        }
    }
}