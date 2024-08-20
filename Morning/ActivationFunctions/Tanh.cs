namespace Morning.ActivationFunctions
{
    public class Tanh : IActivationFunction
    {
        public double Activate(double x) => Math.Tanh(x);

        public double DerivativeActivateFunction(double x) => (1 / Math.Pow(Math.Cosh(x), 2));
    }
}
