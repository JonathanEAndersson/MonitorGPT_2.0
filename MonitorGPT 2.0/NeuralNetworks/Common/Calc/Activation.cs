using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MonitorGPT_2._0.NeuralNetworks.Common.Calc
{
    public static class Activation
    {
        public static float ActivationPReLUFunction(float x)
        {
            return MathF.Max(0.01F * x, x);
        }
        public static float ActivationPReLUFunctionDerivative(float x)
        {
            return x > 0 ? 1 : 0.01F;
        }
    }
}
