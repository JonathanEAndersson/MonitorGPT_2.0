using MonitorGPT_2._0.NeuralNetworks.Common;
using MonitorGPT_2._0.NeuralNetworks.Common.Calc;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace MonitorGPT_2._0.NeuralNetworks.Network
{
    internal class NeuralNetwork
    {
        public Layer[] _layers { get; set; }
        private float momentum = 0.99F;
        private float learningRate = 0.001F;
        private float errorThreshold = 0.00000001F;
        public NeuralNetwork(float[] layerMap)
        {
            _layers = NeuralNetworkBuilder(layerMap).GetAwaiter().GetResult();
            InitializeVelocities();
        }
        private static async Task<Layer[]> NeuralNetworkBuilder(float[] layerMap)
        {
            Layer[] layers = await SpawnLayers(layerMap);
            layers = await AssignNeuralNetwork(layers);
            return layers;
        }
        private static async Task<Layer[]> SpawnLayers(float[] layerMap)
        {
            int layerAmount = layerMap.Length;
            Layer[] layers = new Layer[layerAmount];
            Random random = new Random();
            for (int i = 0; i < layerAmount; i++)
            {
                int nodeAmount = (int)layerMap[i];
                Node[] nodes = new Node[nodeAmount];
                int weightAmount = (0 < i) ? (int)layerMap[i - 1] : 0;

                for (int j = 0; j < nodeAmount; j++)
                {
                    Weight[] weights = new Weight[weightAmount];
                    for (int k = 0; k < weightAmount; k++)
                    {
                        weights[k] = new Weight { value = await Initialization.InitializationKaiming(layers[i - 1].nodes.Length) };
                    }
                    nodes[j] = new Node(layers[i], weights, new Bias { value = (i == 0) ? 0 : await Initialization.InitializationXavier(layers[i - 1].nodes.Length) }, new Synapse(new Node[0], new Node[0]));
                }
                if (i == 0)
                {
                    layers[i] = new InputLayer { nodes = nodes };
                }
                else if (i == layerAmount - 1)
                {
                    layers[i] = new OutputLayer { nodes = nodes };
                }
                else
                {
                    layers[i] = new HiddenLayer { nodes = nodes };
                }
            }
            return layers;
        }
        private static async Task<Layer[]> AssignNeuralNetwork(Layer[] layers)
        {
            int layerAmount = layers.Length;
            for (int i = 0; i < layerAmount; i++)
            {
                layers[i].column = i + 1;
                for (int j = 0; j < layers[i].nodes.Length; j++)
                {
                    layers[i].nodes[j].parentLayer = layers[i];
                    if (i != 0 && i != layerAmount - 1)
                    {
                        layers[i].nodes[j].synapse.inputNodes = layers[i - 1].nodes;
                        layers[i].nodes[j].synapse.outputNodes = layers[i + 1].nodes;
                    }
                    else if (i == 0)
                    {
                        layers[i].nodes[j].synapse.outputNodes = layers[i + 1].nodes;
                    }
                    else
                    {
                        layers[i].nodes[j].synapse.inputNodes = layers[i - 1].nodes;
                    }
                }
            }
            return layers;
        }
        private void InitializeVelocities()
        {
            foreach (var layer in _layers)
            {
                foreach (var node in layer.nodes)
                {
                    node.weightVelocities = new float[node.weights.Length];
                    node.biasVelocity = 0f;
                }
            }
        }
        private float[] Ask()
        {
            Console.WriteLine("Enter 14 values separated by spaces (e.g., 1 2 3):");
            string input = Console.ReadLine();
            string[] inputValues = input.Split(' ');
            float[] floatValues = new float[_layers[0].nodes.Length];
            for (int i = 0; i < floatValues.Length; i++)
            {
                floatValues[i] = float.Parse(inputValues[i].Trim());
            }
            return floatValues;
        }
        public async Task<float[]> Start()
        {
            float[] results = new float[_layers[0].nodes.Length];
            float[] values = Ask();
            Console.WriteLine("Enter the 14 desired values separated by spaces (e.g., 1 2 3):");
            string input = Console.ReadLine();
            string[] inputValues = input.Split(' ');
            float[] targetValues = new float[_layers[0].nodes.Length];
            for (int i = 0; i < targetValues.Length; i++)
            {
                targetValues[i] = float.Parse(inputValues[i].Trim());
            }
            for (int i = 0; i < values.Length; i++)
            {
                _layers[0].nodes[i].activation = values[i];
            }
            var stopwatch = new System.Diagnostics.Stopwatch();
            stopwatch.Start();
            for (int k = 0; ; k++)
            {
                await Forwardpropagate();
                for (int i = 0; i < _layers[_layers.Length - 1].nodes.Length; i++)
                {
                    results[i] = _layers[_layers.Length - 1].nodes[i].activation;
                }
                foreach (var result in results)
                {
                    Console.Write(MathF.Round(result, 6).ToString() + "    ");
                }
                Console.Write("\t Error: " + CalculateError(results, targetValues));
                Console.Write("\n");
                await Backpropagate(targetValues);
                float error = CalculateError(results, targetValues);
                if (error < errorThreshold)
                {
                    Console.WriteLine($"Training complete. Final error: {error}, Iterations: {k}");
                    break;
                }
            }
            stopwatch.Stop();
            Console.WriteLine($"Time taken: {stopwatch.ElapsedMilliseconds} ms");

            return results;
        }
        private async Task Forwardpropagate()
        {
            for (int i = 1; i < _layers.Length; i++)
            {
                for (int j = 0; j < _layers[i].nodes.Length; j++)
                {
                    float[] z = new float[_layers[i-1].nodes.Length];
                    for (int k = 0; k < _layers[i - 1].nodes.Length; k++)
                    {
                        z[k] = _layers[i].nodes[j].synapse.inputNodes[k].activation * _layers[i].nodes[j].weights[k].value;
                    }
                    _layers[i].nodes[j].activation = Activation.ActivationLeakyReLUFunction(z.Sum() + _layers[i].nodes[j].bias.value);
                }
            }
        }
        private async Task Backpropagate(float[] targetValues)
        {
            var outputLayer = _layers[_layers.Length - 1];
            for (int i = 0; i < outputLayer.nodes.Length; i++) // Output Layer
            {
                var outputNode = outputLayer.nodes[i];
                outputNode.gradient = (outputNode.activation - targetValues[i]) * Activation.ActivationLeakyReLUFunctionDerivative(outputNode.activation);
            }
            for (int i = _layers.Length - 2; i >= 1; i--) // Hidden Layers
            {
                var hiddenLayer = _layers[i];
                var nextLayer = _layers[i + 1];
                for (int j = 0; j < _layers[i].nodes.Length; j++)
                {
                    var hiddenNode = hiddenLayer.nodes[j];
                    float deltaSum = 0;
                    for (int k = 0; k < nextLayer.nodes.Length; k++)
                    {
                        var outputNode = nextLayer.nodes[k];
                        deltaSum += outputNode.gradient * outputNode.weights[j].value;
                    }
                    hiddenNode.gradient = deltaSum * Activation.ActivationLeakyReLUFunctionDerivative(hiddenNode.activation);
                }
            }
            for (int i = 1; i < _layers.Length; i++) // Update Weights, Biases, and Momentum
            {
                var layer = _layers[i];
                for (int j = 0; j < layer.nodes.Length; j++)
                {
                    var node = layer.nodes[j];
                    for (int k = 0; k < node.synapse.inputNodes.Length; k++)
                    {
                        var inputNode = node.synapse.inputNodes[k];
                        float weightGradient = node.gradient * inputNode.activation;
                        node.weightVelocities[k] = momentum * node.weightVelocities[k] - learningRate * weightGradient;
                        node.weights[k].value += node.weightVelocities[k];
                    }
                    node.biasVelocity = momentum * node.biasVelocity - learningRate * node.gradient;
                    node.bias.value += node.biasVelocity;
                }
            }
        }
        private float CalculateError(float[] output, float[] target)
        {
            float sumSquaredError = 0f;
            for (int i = 0; i < output.Length; i++)
            {
                float error = output[i] - target[i];
                sumSquaredError += error * error;
            }
            return sumSquaredError / output.Length;
        }
    }
}
