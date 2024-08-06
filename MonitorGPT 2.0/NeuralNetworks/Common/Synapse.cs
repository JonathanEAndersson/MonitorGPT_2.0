using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MonitorGPT_2._0.NeuralNetworks.Common
{
    public class Synapse
    {
        public Node[] inputNodes { get; set; }
        public Node[] outputNodes { get; set; }
        public Synapse(Node[] inputNodes, Node[] outputNodes)
        {
            this.inputNodes = inputNodes;
            this.outputNodes = outputNodes;
        }
    }
}
