using System.Collections.Generic;

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
