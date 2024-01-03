using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Morning.ActivationFunctions
{
    public class Sigmoid : IActivationFunction
    {
        public double Activate(double x) => 1.0 / (1.0 + Math.Exp(-x));
        public double DerivativeActivateFunction(double x) => Activate(x) * (1 - Activate(x));
    }
}
