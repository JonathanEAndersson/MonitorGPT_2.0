using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MonitorGPT_2._0.NeuralNetworks.Common.Calc
{
    public static class Initialization
    {
        static Random random = new Random();
        public static async Task<float> InitializationXavier(int vIn, float vOut)
        {
            double stdDev = Math.Sqrt(2.0 / (vIn + vOut));
            return (float)(random.NextDouble() * 2 * stdDev - stdDev);
        }
        public static async Task<float> InitializationXavier(int previousLayerSize)
        {
            return Convert.ToSingle(random.NextDouble() * Math.Sqrt(1.0 / previousLayerSize));
        }
        public static async Task<float> InitializationKaiming(int previousLayerSize)
        {
            float std = (float)Math.Sqrt(2.0 / previousLayerSize);
            float u1 = random.NextSingle();
            float u2 = random.NextSingle();
            float z1 = (float)(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2));
            return z1 * std;
        }
    }
}
