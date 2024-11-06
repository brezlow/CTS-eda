using System;
using KSplittingNamespace;

namespace edaContest
{
    internal class Program
    {
        static void Main(string[] args)
        {
            if (args.Length < 6)
            {
                Console.WriteLine("Usage: <program> -def_file <circuit file> -constraint <constraint file> -output <output file>");
                return;
            }

            string circuitFilePath = args[1];
            string constraintFilePath = args[3];
            string outputFilePath = args[5];

            // 创建 CircuitData 实例
            FileReader reader = new FileReader();
            CircuitData circuitData = reader.ReadCircuitData(circuitFilePath, constraintFilePath);

            // 获取触发器集合
            List<Node> triggers = new List<Node>();
            foreach (var ff in circuitData.FFInstances)
            {
                triggers.Add(new Node(ff.Position.X, ff.Position.Y, 0, triggers.Count, circuitData.FFSize.Width, circuitData.FFSize.Height));
            }

            // 创建 KSplittingClustering 实例
            double alpha = 1.0; // 根据需要设置 alpha 值
            int maxFanout = circuitData.MaxFanout;
            int maxNetRC = (int)circuitData.MaxNetRC;
            KSplittingClustering kSplitting = new KSplittingClustering(triggers, circuitData.FloorplanSize.Width, circuitData.FloorplanSize.Height, 0, alpha, maxFanout, maxNetRC);



            // 执行聚类算法
            List<List<Node>> clusters = kSplitting.ExecuteClustering();


            // 输出聚类结果
            Console.WriteLine($"聚类团数目: {clusters.Count}");
            for (int i = 0; i < clusters.Count; i++)
            {
                Console.WriteLine($"聚类团 {i + 1}:");
                foreach (var node in clusters[i])
                {
                    Console.WriteLine($"  节点 {node.Id}: ({node.X}, {node.Y})");
                }
            }


            // FileWriter writer = new FileWriter();
            // writer.WriteOutput(outputFilePath, circuitData, nets);

        }

        static void PrintCircuitData(CircuitData data)
        {
            Console.WriteLine($"Units: {data.Units}");
            Console.WriteLine($"Floorplan Size: {data.FloorplanSize.Width} x {data.FloorplanSize.Height}");
            Console.WriteLine($"FF Size: {data.FFSize.Width} x {data.FFSize.Height}");
            Console.WriteLine($"Buffer Size: {data.BufferSize.Width} x {data.BufferSize.Height}");
            Console.WriteLine($"Clock Root Position: ({data.ClockRootPosition.X}, {data.ClockRootPosition.Y})");

            Console.WriteLine("FF Instances:");
            foreach (var ff in data.FFInstances)
            {
                Console.WriteLine($"  Name: {ff.Name}, Position: ({ff.Position.X}, {ff.Position.Y})");
            }

            Console.WriteLine("Buffer Instances:");
            foreach (var buffer in data.BufferInstances)
            {
                Console.WriteLine($"  Name: {buffer.Name}, Position: ({buffer.Position.X}, {buffer.Position.Y})");
            }
        }
    }
}