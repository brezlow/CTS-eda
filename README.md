`FileReader`类包含了解析输入文件的具体逻辑。`CircuitData`则是一个数据容器，存储解析结果。`FFInstance`和`BufferInstance`分别表示不同组件的实例



* **主程序（Program.cs）**：程序入口，调用`FileReader`类读取输入数据。
* **FileReader.cs**：负责读取电路和约束文件，解析特定参数。
* **CircuitData.cs**：储存解析后的电路数据（例如FF尺寸、缓冲区大小、时钟根位置等）。
* **Component.cs**：定义不同组件类（如`FFInstance`、`BufferInstance`），包含其特有属性
  
  

---

1. **主程序结构**：`Main`方法负责解析输入参数并读取电路和约束数据文件，然后将数据写入输出文件。程序目前实现了文件读写和数据结构的创建部分，数据读取成功后打印确认信息。

2. **数据读取（`FileReader`类）**：
   
   * `ReadCircuitData`方法从电路文件和约束文件中加载数据。
   * `ReadCircuitFile`方法解析电路文件，包括单位、芯片面积、FF和缓冲器尺寸、时钟根位置等。
   * `ReadConstraintFile`方法解析约束文件，包括电阻电容单位值、最大网络RC、最大扇出等约束条件。
   * `ParseDieArea`、`ParseSize`、`ParsePosition`等方法用于解析电路文件的具体信息。
   * `ParseComponents`方法负责从文件中提取FF实例，可以扩展以支持缓冲器实例的解析。

3. **数据结构**：`CircuitData`、`Constraint`、`FFInstance`、`BufferInstance`和`Net`类构成了CTS设计所需的基本数据结构。
   
   * `CircuitData`类中包含了单位、芯片面积、时钟根位置、FF和缓冲器尺寸及实例列表，还包括约束条件的字段。
   * `Constraint`类存储了约束条件的详细信息，如最大扇出、最大RC值、缓冲器延迟等。
   * `Net`类描述了电路的网络连接结构。

4. **输出文件写入（`FileWriter`类）**：`WriteOutput`方法将电路数据和网络连接结构写入到指定的输出文件，保持了文件格式的一致性。

### 后续的开发建议

为了实现核心的时钟树综合功能，可以考虑以下几个步骤：

1. **定义时钟树结构**：为每个节点（FF或缓冲器）和连接关系设计合适的数据结构。

2. **算法实现**：CTS核心算法可以包含以下步骤：
   
   * **时钟树生成**：递归地在时钟根和FF之间插入缓冲器，构建扇出受控的树结构。
   * **路径延迟计算**：基于每条路径的RC值和缓冲器延迟，计算从时钟根到每个FF的延迟。
   * **扇出控制**：依据最大扇出限制，动态添加缓冲器以分担负载。

3. **更新程序结构**：
   
   * 在`Main`方法中，将时钟树生成逻辑放在加载数据之后，写入输出文件之前。
   * 可在`CircuitData`中添加用于存储生成时钟树的字段。
     
     

---

# K-分裂程序的函数

ExecuteClustering

```csharp
public List<List<Node>> ExecuteClustering()
```

功能：执行K分裂聚类算法。 返回值：返回一个包含多个聚类的列表，每个聚类是一个节点列表。



----



输入文件格式：

执行方式建议：
{运行程序文件} -def_file {电路文件全路径} -constraint {约束文件全路径} -output solution.def 

示例： 
文件路径：
/home/public/case1/problem.def 
/home/public/case1/constraints.txt 


