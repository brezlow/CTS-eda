using System.Collections.Generic;

class CircuitData
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

