using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Morning.ActivationFunctions
{
    public interface IActivationFunction
    {
        public double Activate(double x);
        public double DerivativeActivateFunction(double x);
    }
}
