using MonitorGPT_2._0.NeuralNetworks.Network;

namespace MonitorGPT_2._0
{
    internal class Program
    {
        static async Task Main(string[] args)
        {
            float[] layersConfig = { 3, 100, 100, 3 };
            NeuralNetwork network = new NeuralNetwork(layersConfig);
            await network.Start();
            Console.WriteLine("Done");
            Console.ReadLine();
        }
    }
}
