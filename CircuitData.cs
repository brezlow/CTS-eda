using System.Collections.Generic;

// 电路数据类型, 用于存储电路布局信息
public class CircuitData
{
    public int Units { get; set; }
    public (int Width, int Height) FloorplanSize { get; set; }
    public (int Width, int Height) FFSize { get; set; }
    public (int Width, int Height) BufferSize { get; set; }
    public (int X, int Y) ClockRootPosition { get; set; }
    public List<FFInstance> FFInstances { get; set; } = new List<FFInstance>();
    public List<BufferInstance> BufferInstances { get; set; } = new List<BufferInstance>();

    // 约束属性，如最大扇出、rc值等
    public double NetUnitR { get; set; }
    public double NetUnitC { get; set; }
    public double MaxNetRC { get; set; }
    public int MaxFanout { get; set; }
    public double BufferDelay { get; set; }
}

public class FFInstance
{
    public required string Name { get; set; }
    public (int X, int Y) Position { get; set; }
}

public class BufferInstance
{
    public required string Name { get; set; }
    public (int X, int Y) Position { get; set; }
    public List<Node> ContainedNodes { get; set; } = new List<Node>(); // 添加 ContainedNodes 属性
}

public class Node
{
    public Node(int x, int y, double delay, int id, int width, int height)
    {
        X = x;
        Y = y;
        Delay = delay;
        Id = id;
        Width = width;
        Height = height;
    }

    public int Id { get; }
    public int X { get; }
    public int Y { get; }
    public double Delay { get; set; } // 平均延迟，用于中层聚类的补偿计算
    public int Width { get; }  // 长方形的宽度
    public int Height { get; } // 长方形的高度
    public string BufferName { get; set; } // 添加 BufferName 属性
}

class Net
{
    public string Name { get; set; }
    public string Source { get; set; }
    public List<string> Sinks { get; set; }

    public Net(string name, string source, List<string> sinks)
    {
        Name = name;
        Source = source;
        Sinks = sinks;
    }
}