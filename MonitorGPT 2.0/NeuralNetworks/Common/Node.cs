using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MonitorGPT_2._0.NeuralNetworks.Common
{
    public class Node
    {
        public Layer parentLayer {  get; set; }
        public Weight[] weights {  get; set; }
        public Bias bias { get; set; }
        public Synapse synapse { get; set; }
        public float activation { get; set; }
        public float gradient { get; set; }
        public float[] weightVelocities { get; set; }
        public float biasVelocity { get; set; }
        public Node(Layer parentLayer, Weight[] weights, Bias bias, Synapse synapse)
        {
            this.parentLayer = parentLayer;
            this.weights = weights;
            this.bias = bias;
            this.activation = 0.0f;
            this.gradient = 0.0f;
            this.synapse = synapse;
        }
    }
}
