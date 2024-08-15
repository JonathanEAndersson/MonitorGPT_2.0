using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MonitorGPT_2._0.NeuralNetworks.Common.Calc
{
    public static class Activation
    {
        public static float ActivationSigmoidFunction(float x)
        {
            return 1 / (1 + MathF.Exp(-x));
        }
        public static float ActivationSigmoidFunctionDerivative(float x)
        {
            return ActivationSigmoidFunction(x) * (1 - ActivationSigmoidFunction(x));
        }
        public static float ActivationReLUFunction(float x)
        {
            return MathF.Max(0, x);
        }
        public static float ActivationReLUFunctionDerivative(float x)
        {
            return x;
        }
        public static float ActivationLeakyReLUFunction(float x)
        {
            return x > 0 ? x : 0.01F * x;
        }
        public static float ActivationLeakyReLUFunctionDerivative(float x)
        {
            return x > 0 ? 1 : 0.01F;
        }
    }
}