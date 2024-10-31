class Constraint
{
    public int MaxFanout { get; set; }
    public double MaxRC { get; set; }
    public double BufferDelay { get; set; }
    public (double Resistance, double Capacitance) UnitRC { get; set; }
}
