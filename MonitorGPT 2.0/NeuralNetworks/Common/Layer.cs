using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MonitorGPT_2._0.NeuralNetworks.Common
{
    public abstract class Layer
    {
        public int column {  get; set; }
        public Node[] nodes { get; set; }
    }
    public class InputLayer : Layer
    {

    }

    public class HiddenLayer : Layer 
    {

    }
    public class OutputLayer : Layer
    {

    }
}