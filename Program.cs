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
            Console.WriteLine($"触发器数目:{circuitData.FFInstances.Count}");
            List<Node> triggers = new List<Node>();
            foreach (var ff in circuitData.FFInstances)
            {
                triggers.Add(new Node(ff.Position.X, ff.Position.Y, triggers.Count, ff.Name, circuitData.FFSize.Width, circuitData.FFSize.Height));
            }

            // 建立所有元件结构
            List<CircuitComponent> CircuitComponents = new List<CircuitComponent>();

            // 将FF实例添加到CircuitComponents中
            foreach (var ff in circuitData.FFInstances)
            {
                CircuitComponents.Add(new CircuitComponent(ff.Position.X, ff.Position.Y, ff.Name, circuitData.FFSize.Width, circuitData.FFSize.Height, circuitData.FFSize.Width * circuitData.FFSize.Height));
            }

            // 建立全局Buffer元件结构
            List<BufferInstance> TotalBuffer = new List<BufferInstance>();

            // 开始第一层聚类
            // 创建 KSplittingClustering 实例
            double alpha = 7.5; // 根据需要设置 alpha 值
            int maxFanout = circuitData.MaxFanout;
            // 计算障碍物面积
            double obstacleArea = CircuitComponents.Sum(component => component.Area);
            Console.WriteLine($"障碍物面积: {obstacleArea}");

            int maxNetRC = (int)circuitData.MaxNetRC;
            KSplittingClustering kSplitting = new KSplittingClustering(triggers, circuitData.FloorplanSize.Width, circuitData.FloorplanSize.Height, circuitData.BufferSize.Height, circuitData.BufferSize.Width, obstacleArea, alpha, circuitData.NetUnitR, circuitData.NetUnitC, maxFanout, maxNetRC, maxFanout, CircuitComponents, TotalBuffer);

            Console.WriteLine("开始执行聚类算法...");

            // 执行聚类算法
            List<Node> bottomBuffer = kSplitting.ExecuteClustering();
            Console.WriteLine("聚类算法执行完毕");
            Console.WriteLine($"BottomBuffer数目:{bottomBuffer.Count}");
            // 将第一次聚类的缓冲器添加到CircuitComponents中
            foreach (var buffer in bottomBuffer)
            {
                CircuitComponents.Add(new CircuitComponent(buffer.X, buffer.Y, buffer.Name, circuitData.BufferSize.Width, circuitData.BufferSize.Height, circuitData.BufferSize.Width * circuitData.BufferSize.Height));
            }

            // 循环进行聚类，直到得到最上一层聚类及 buffer 只有一个时结束聚类
            while (bottomBuffer.Count > 1)
            {
                // 更新 triggers 为上一次聚类得到的 buffer
                triggers = bottomBuffer;

                // 创建新的 KSplittingClustering 实例
                kSplitting = new KSplittingClustering(triggers, circuitData.FloorplanSize.Width, circuitData.FloorplanSize.Height, circuitData.BufferSize.Height, circuitData.BufferSize.Width, obstacleArea, alpha, circuitData.NetUnitR, circuitData.NetUnitC, maxFanout, maxNetRC, maxFanout, CircuitComponents, TotalBuffer, false);

                Console.WriteLine("开始执行聚类算法...");

                // 执行聚类算法
                bottomBuffer = kSplitting.ExecuteClustering();
                Console.WriteLine("聚类算法执行完毕");
                Console.WriteLine($"Buffer数目:{bottomBuffer.Count}");
                // 将本次聚类的缓冲器添加到 CircuitComponents 中
                foreach (var buffer in bottomBuffer)
                {
                    CircuitComponents.Add(new CircuitComponent(buffer.X, buffer.Y, buffer.Name, circuitData.BufferSize.Width, circuitData.BufferSize.Height, circuitData.BufferSize.Width * circuitData.BufferSize.Height));
                }
            }

            circuitData.BufferInstances = TotalBuffer;



            FileWriter writer = new FileWriter();
            writer.WriteOutput(outputFilePath, circuitData);
        }

    }
}