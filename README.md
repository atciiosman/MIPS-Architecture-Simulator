# MIPS Simulator (with GUI)

This project is a MIPS simulator that provides a graphical user interface (GUI) to simulate MIPS assembly instructions. Users can write, execute, and debug MIPS assembly programs interactively.

## Features

### Assembly Code Execution
- Write MIPS assembly instructions through the editor.
- Execute instructions step by step or run the entire program.

### Register and Memory Visualization
- View register values in real-time.
- Monitor the state of instruction and data memory.

### Machine Code Conversion
- Convert MIPS assembly instructions into binary machine code.

### GUI Interaction
- Interactive GUI with separate panels:
  - Assembly input.
  - Register and memory visualization.
  - Execution trace.
  - Machine code output.

## Requirements
- **Python 3.8+**
- **tkinter library** (Included in most Python installations)

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/eminosmanatci/MIPS-Architecture-Simulator
   ```
2. Navigate to the project directory:
   ```bash
   cd mips-simulator-gui
   ```
3. Run the simulator:
   ```bash
   python mips_simulator.py
   ```

## How to Use
1. **Write your assembly code**: Enter your MIPS assembly code in the Assembly editor.
2. **Use the buttons**:
   - **Load Program**: Loads the written program into the simulator.
   - **Step**: Executes one instruction at a time.
   - **Run**: Executes the entire program.
   - **Convert to MC**: Converts assembly instructions to machine code.
   - **Reset**: Resets the simulator to its initial state.
3. Monitor output in the respective sections:
   - **Registers**: Displays the state of registers.
   - **Instruction Memory**: Shows the loaded program.
   - **Data Memory**: Displays memory values used in the program.
   - **Execution Trace**: Logs each executed instruction and state changes.

## Components
### Backend (MIPS_Simulator)
- **Instruction Parsing**: Validates and parses MIPS assembly instructions.
- **Execution Logic**: Executes MIPS instructions and updates registers and memory.
- **Machine Code Conversion**: Converts assembly instructions to binary machine code.

### Frontend (MIPS_GUI)
- **tkinter GUI**: Provides an interactive interface for writing and executing programs.
- **Real-Time Updates**: Dynamically updates register and memory states during execution.

## Design Decisions
1. **Educational Simplicity**: Designed for learning and understanding MIPS architecture.
2. **Real-Time Feedback**: Visualization of state changes aids debugging.
3. **Scalability**: Modular design allows for easy extension of supported instructions.

## Known Limitations
1. **Instruction Set**: Limited to basic MIPS instructions (e.g., `add`, `sub`, `lw`, `sw`, `beq`, etc.).
2. **Error Handling**: Error messages for invalid instructions could be more descriptive.
3. **Performance**: Large programs may experience performance issues.

