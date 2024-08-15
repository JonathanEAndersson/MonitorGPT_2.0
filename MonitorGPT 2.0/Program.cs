using MonitorGPT_2._0.NeuralNetworks.Network;

namespace MonitorGPT_2._0
{
    internal class Program
    {
        static async Task Main(string[] args)
        {
            float[] layersConfig = { 14, 28, 28, 28, 14 };
            NeuralNetwork network = new NeuralNetwork(layersConfig);
            await network.Start();
            Console.WriteLine("Done");
            Console.ReadLine();
        }
    }
}
