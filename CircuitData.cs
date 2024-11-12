using System.Collections.Generic;
using System.Reflection.Metadata.Ecma335;

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
    public List<string> ContainedNodeNames { get; set; } = new List<string>(); // 修改为包含子元件名称的字符串列表
    public double AverageManhattanDistance { get; set; } // 添加表示该缓冲器聚类内部平均集线曼哈顿距离的属性
}

public class Node
{
    public Node(int x, int y, int id, string name, int width, int height)
    {
        X = x;
        Y = y;
        Id = id;
        Name = name;
        Width = width;
        Height = height;
    }

    public int Id { get; set; }
    public int X { get; set; }
    public int Y { get; set; }
    public string Name { get; set; }
    public double Delay { get; set; } // 平均延迟，用于中层聚类的补偿计算
    public int Width { get; }  // 长方形的宽度
    public int Height { get; } // 长方形的高度
}

public class CircuitComponent
{
    public CircuitComponent(int x, int y, string name, int width, int height, double area)
    {
        X = x;
        Y = y;
        Name = name;
        Width = width;
        Height = height;
        Area = area;
    }
    public string Name { get; set; }
    public double Area { get; set; }

    public int Width { get; }
    public int Height { get; }
    public int X { get; set; }
    public int Y { get; set; }
}