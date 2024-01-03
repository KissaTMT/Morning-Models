using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Morning.ActivationFunctions
{
    public class Relu : IActivationFunction
    {
        public double Activate(double x) => Math.Max(0, x);
        public double DerivativeActivateFunction(double x) => x >= 0 ? 1 : 0;
    }
}
