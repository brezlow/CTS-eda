using System;
using System.Collections.Generic;
using System.IO;

class FileWriter
{
    public void WriteOutput(string filePath, CircuitData data, List<Net> nets)
    {
        using (StreamWriter sw = new StreamWriter(filePath))
        {
            // 写入单位声明
            sw.WriteLine($"UNITS DISTANCE MICRONS {data.Units} ;");

            // 写入 Floorplan 大小
            sw.WriteLine($"DIEAREA ( 0 0 ) ( 0 {data.FloorplanSize.Height} ) ( {data.FloorplanSize.Width} {data.FloorplanSize.Height} ) ( {data.FloorplanSize.Width} 0 ) ;");

            // 写入 FF 和 BUF 尺寸
            sw.WriteLine($"FF ( {data.FFSize.Width} {data.FFSize.Height} ) ;");
            sw.WriteLine($"BUF ( {data.BufferSize.Width} {data.BufferSize.Height} ) ;");

            // 写入时钟源位置
            sw.WriteLine($"CLK ( {data.ClockRootPosition.X} {data.ClockRootPosition.Y} ) ;");

            // 写入组件部分
            sw.WriteLine($"COMPONENTS {data.FFInstances.Count + data.BufferInstances.Count} ;");
            foreach (var ff in data.FFInstances)
            {
                sw.WriteLine($"- {ff.Name} FF ( {ff.Position.X} {ff.Position.Y} ) ;");
            }
            foreach (var buf in data.BufferInstances)
            {
                sw.WriteLine($"- {buf.Name} BUF ( {buf.Position.X} {buf.Position.Y} ) ;");
            }
            sw.WriteLine("END COMPONENTS");

            // 写入连接关系 NETS 部分
            sw.WriteLine($"NETS {nets.Count} ;");
            foreach (var net in nets)
            {
                sw.Write($"- {net.Name} ( {net.Source} ) (");
                for (int i = 0; i < net.Sinks.Count; i++)
                {
                    sw.Write($" {net.Sinks[i]}");
                    if (i < net.Sinks.Count - 1)
                        sw.Write(" ");
                }
                sw.WriteLine(" ) ;");
            }
            sw.WriteLine("END NETS");
        }
    }
}
