using System;

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

            // 打印 CircuitData 数据
            // PrintCircuitData(circuitData);

            //List<Net> nets = new List<Net>
            //{
            //new Net("net_clk", "CLK", new List<string> { "BUF1" }),
            //new Net("net_buf1", "BUF1", new List<string> { "BUF2", "BUF3", "BUF4", "BUF5" }),
            //// 添加其他 net 数据
            //};

            Console.WriteLine("数据加载成功。");

            FileWriter writer = new FileWriter();
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