using System;
using System.Collections.Generic;
using System.IO;

class FileReader
{
    public CircuitData ReadCircuitData(string circuitFilePath, string constraintFilePath)
    {
        CircuitData data = new CircuitData();
        ReadCircuitFile(circuitFilePath, data);
        ReadConstraintFile(constraintFilePath, data);
        return data;
    }

    private void ReadCircuitFile(string filePath, CircuitData data)
    {
        try
        {
            using (StreamReader sr = new StreamReader(filePath))
            {
                string line;
                while ((line = sr.ReadLine()!) != null)
                {
                    if (line.StartsWith("UNITS DISTANCE MICRONS"))
                    {
                        data.Units = int.Parse(line.Split(' ')[3]);
                    }
                    else if (line.StartsWith("DIEAREA"))
                    {
                        ParseDieArea(line, data);
                    }
                    else if (line.StartsWith("FF"))
                    {
                        data.FFSize = ParseSize(line);
                    }
                    else if (line.StartsWith("BUF"))
                    {
                        data.BufferSize = ParseSize(line);
                    }
                    else if (line.StartsWith("CLK"))
                    {
                        data.ClockRootPosition = ParsePosition(line);
                    }
                    else if (line.StartsWith("COMPONENTS"))
                    {
                        ParseComponents(sr, data.FFInstances);
                    }
                }
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine("读取电路文件时出错：" + ex.Message);
        }
    }

    private void ReadConstraintFile(string filePath, CircuitData data)
    {
        try
        {
            using (StreamReader sr = new StreamReader(filePath))
            {
                string line;
                while ((line = sr.ReadLine()!) != null)
                {
                    // 解析约束数据
                    var parts = line.Split('=');
                    if (parts.Length == 2)
                    {
                        string key = parts[0].Trim();
                        string value = parts[1].Trim();

                        switch (key)
                        {
                            case "net_unit_r":
                                data.NetUnitR = double.Parse(value);
                                break;
                            case "net_unit_c":
                                data.NetUnitC = double.Parse(value);
                                break;
                            case "max_net_rc":
                                data.MaxNetRC = double.Parse(value);
                                break;
                            case "max_fanout":
                                data.MaxFanout = int.Parse(value);
                                break;
                            case "buffer_delay":
                                data.BufferDelay = double.Parse(value);
                                break;
                        }
                    }
                }
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine("读取约束文件时出错：" + ex.Message);
        }
    }

    private void ParseDieArea(string line, CircuitData data)
    {
        var parts = line.Split(new char[] { ' ', '(', ')' }, StringSplitOptions.RemoveEmptyEntries);
        int x = int.Parse(parts[5]);
        int y = int.Parse(parts[6]);
        data.FloorplanSize = (x, y);
    }

    private (int width, int height) ParseSize(string line)
    {
        var parts = line.Split(new char[] { ' ', '(', ')' }, StringSplitOptions.RemoveEmptyEntries);
        int width = int.Parse(parts[1]);
        int height = int.Parse(parts[2]);
        return (width, height);
    }

    private (int x, int y) ParsePosition(string line)
    {
        var parts = line.Split(new char[] { ' ', '(', ')' }, StringSplitOptions.RemoveEmptyEntries);
        int x = int.Parse(parts[1]);
        int y = int.Parse(parts[2]);
        return (x, y);
    }

    private void ParseComponents(StreamReader sr, List<FFInstance> ffInstances)
    {
        string line;
        while ((line = sr.ReadLine()!) != null && !line.StartsWith("END COMPONENTS"))
        {
            if (line.StartsWith("-"))
            {
                var parts = line.Split(new char[] { ' ', '(', ')' }, StringSplitOptions.RemoveEmptyEntries);
                if (parts[2] == "FF")
                {
                    FFInstance ff = new FFInstance
                    {
                        Name = parts[1],
                        Position = (int.Parse(parts[3]), int.Parse(parts[4]))
                    };
                    ffInstances.Add(ff);
                }
            }
        }
    }
}
