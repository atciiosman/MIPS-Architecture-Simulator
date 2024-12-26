import tkinter as tk
from tkinter import scrolledtext, messagebox
from collections import defaultdict
from datetime import datetime


class MIPS_Simulator:
    def __init__(self):
        self.registers = {i: 0 for i in range(32)}
        self.registers[0] = 0
        self.registers[31] = 0
        self.instruction_memory = defaultdict(int)
        self.data_memory = defaultdict(int)
        self.pc = 0
        self.instructions = []
        self.labels = {}
        self.execution_trace = []
        self.output_log = []
        self.instruction_memory_size = 512
        self.data_memory_size = 512

        self.register_map = {
            "$zero": 0, "$at": 1, "$v0": 2, "$v1": 3,
            "$a0": 4, "$a1": 5, "$a2": 6, "$a3": 7,
            "$t0": 8, "$t1": 9, "$t2": 10, "$t3": 11,
            "$t4": 12, "$t5": 13, "$t6": 14, "$t7": 15,
            "$s0": 16, "$s1": 17, "$s2": 18, "$s3": 19,
            "$s4": 20, "$s5": 21, "$s6": 22, "$s7": 23,
            "$t8": 24, "$t9": 25, "$k0": 26, "$k1": 27,
            "$gp": 28, "$sp": 29, "$fp": 30, "$ra": 31
        }
        self.reverse_register_map = {v: k for k, v in self.register_map.items()}

    def _get_register_number(self, reg):
        if reg in self.register_map:
            return self.register_map[reg]
        return int(reg[2:])

    def _get_machine_code_register_number(self, reg):
        if reg in self.register_map:
            return self.register_map[reg]
        return int(reg[2:])

    def load_program(self, instructions):
        self.instructions = instructions.splitlines()
        self.pc = 0
        self.execution_trace.clear()
        self.labels.clear()
        self.output_log.clear()
        self.instruction_memory.clear()  # Clear before loading new program

        cleaned_instructions = []
        for instruction in self.instructions:
            instruction = instruction.split("#")[0].strip()
            if instruction:
                cleaned_instructions.append(instruction)

        self.instructions = cleaned_instructions

        # Check for minimum instruction memory size
        if len(self.instructions) * 4 > self.instruction_memory_size:
            raise MemoryError("Program size exceeds instruction memory limit")

        for index, instruction in enumerate(self.instructions):
            parts = instruction.split(":")
            if len(parts) > 1:
                label = parts[0].strip()
                self.labels[label] = index
                self.instructions[index] = parts[1].strip()

        for index, instruction in enumerate(self.instructions):
            self.instruction_memory[index] = instruction

    def execute_instruction(self, instruction):
        parts = instruction.split()
        if not parts:
            return

        op = parts[0]

        try:
            if op == "add":
                if len(parts) != 4:
                    raise Exception(f"Invalid format for {op} instruction: {instruction}")
                rd, rs, rt = self._get_register_number(parts[1].strip(",")), self._get_register_number(
                    parts[2].strip(",")), self._get_register_number(parts[3])
                old_value = self.registers[rd]
                self.registers[rd] = self.registers[rs] + self.registers[rt]
                self.output_log.append(
                    f"  {self.get_register_name(rd)} changed from {old_value} to {self.registers[rd]}")
            elif op == "addi":
                if len(parts) != 4:
                    raise Exception(f"Invalid format for {op} instruction: {instruction}")
                rd, rs, imm = self._get_register_number(parts[1].strip(",")), self._get_register_number(
                    parts[2].strip(",")), int(parts[3])
                old_value = self.registers[rd]
                self.registers[rd] = self.registers[rs] + imm
                self.output_log.append(
                    f"  {self.get_register_name(rd)} changed from {old_value} to {self.registers[rd]}")
            elif op == "sub":
                if len(parts) != 4:
                    raise Exception(f"Invalid format for {op} instruction: {instruction}")
                rd, rs, rt = self._get_register_number(parts[1].strip(",")), self._get_register_number(
                    parts[2].strip(",")), self._get_register_number(parts[3])
                old_value = self.registers[rd]
                self.registers[rd] = self.registers[rs] - self.registers[rt]
                self.output_log.append(
                    f"  {self.get_register_name(rd)} changed from {old_value} to {self.registers[rd]}")
            elif op == "and":
                if len(parts) != 4:
                    raise Exception(f"Invalid format for {op} instruction: {instruction}")
                rd, rs, rt = self._get_register_number(parts[1].strip(",")), self._get_register_number(
                    parts[2].strip(",")), self._get_register_number(parts[3])
                old_value = self.registers[rd]
                self.registers[rd] = self.registers[rs] & self.registers[rt]
                self.output_log.append(
                    f"  {self.get_register_name(rd)} changed from {old_value} to {self.registers[rd]}")
            elif op == "or":
                if len(parts) != 4:
                    raise Exception(f"Invalid format for {op} instruction: {instruction}")
                rd, rs, rt = self._get_register_number(parts[1].strip(",")), self._get_register_number(
                    parts[2].strip(",")), self._get_register_number(parts[3])
                old_value = self.registers[rd]
                self.registers[rd] = self.registers[rs] | self.registers[rt]
                self.output_log.append(
                    f"  {self.get_register_name(rd)} changed from {old_value} to {self.registers[rd]}")
            elif op == "xor":
                if len(parts) != 4:
                    raise Exception(f"Invalid format for {op} instruction: {instruction}")
                rd, rs, rt = self._get_register_number(parts[1].strip(",")), self._get_register_number(
                    parts[2].strip(",")), self._get_register_number(parts[3])
                old_value = self.registers[rd]
                self.registers[rd] = self.registers[rs] ^ self.registers[rt]
                self.output_log.append(
                    f"  {self.get_register_name(rd)} changed from {old_value} to {self.registers[rd]}")
            elif op == "slt":
                if len(parts) != 4:
                    raise Exception(f"Invalid format for {op} instruction: {instruction}")
                rd, rs, rt = self._get_register_number(parts[1].strip(",")), self._get_register_number(
                    parts[2].strip(",")), self._get_register_number(parts[3])
                old_value = self.registers[rd]
                self.registers[rd] = 1 if self.registers[rs] < self.registers[rt] else 0
                self.output_log.append(
                    f"  {self.get_register_name(rd)} changed from {old_value} to {self.registers[rd]}")
            elif op == "slti":
                if len(parts) != 4:
                    raise Exception(f"Invalid format for {op} instruction: {instruction}")
                rd, rs, imm = self._get_register_number(parts[1].strip(",")), self._get_register_number(
                    parts[2].strip(",")), int(parts[3])
                old_value = self.registers[rd]
                self.registers[rd] = 1 if self.registers[rs] < imm else 0
                self.output_log.append(
                    f"  {self.get_register_name(rd)} changed from {old_value} to {self.registers[rd]}")
            elif op == "andi":
                if len(parts) != 4:
                    raise Exception(f"Invalid format for {op} instruction: {instruction}")
                rd, rs, imm = self._get_register_number(parts[1].strip(",")), self._get_register_number(
                    parts[2].strip(",")), int(parts[3])
                old_value = self.registers[rd]
                self.registers[rd] = self.registers[rs] & imm
                self.output_log.append(
                    f"  {self.get_register_name(rd)} changed from {old_value} to {self.registers[rd]}")
            elif op == "ori":
                if len(parts) != 4:
                    raise Exception(f"Invalid format for {op} instruction: {instruction}")
                rd, rs, imm = self._get_register_number(parts[1].strip(",")), self._get_register_number(
                    parts[2].strip(",")), int(parts[3])
                old_value = self.registers[rd]
                self.registers[rd] = self.registers[rs] | imm
                self.output_log.append(
                    f"  {self.get_register_name(rd)} changed from {old_value} to {self.registers[rd]}")
            elif op == "sll":
                if len(parts) != 4:
                    raise Exception(f"Invalid format for {op} instruction: {instruction}")
                rd, rt, shamt = self._get_register_number(parts[1].strip(",")), self._get_register_number(
                    parts[2].strip(",")), int(parts[3])
                old_value = self.registers[rd]
                self.registers[rd] = self.registers[rt] << shamt
                self.output_log.append(
                    f"  {self.get_register_name(rd)} changed from {old_value} to {self.registers[rd]}")
            elif op == "srl":
                if len(parts) != 4:
                    raise Exception(f"Invalid format for {op} instruction: {instruction}")
                rd, rt, shamt = self._get_register_number(parts[1].strip(",")), self._get_register_number(
                    parts[2].strip(",")), int(parts[3])
                old_value = self.registers[rd]
                self.registers[rd] = self.registers[rt] >> shamt
                self.output_log.append(
                    f"  {self.get_register_name(rd)} changed from {old_value} to {self.registers[rd]}")
            elif op == "lw":
                if len(parts) != 3:
                    raise Exception(f"Invalid format for {op} instruction: {instruction}")
                rd, offset_rs = self._get_register_number(parts[1].strip(",")), parts[2]
                offset_parts = offset_rs.split("(")
                if len(offset_parts) != 2:
                    raise Exception(f"Invalid format for {op} instruction: {instruction}")
                try:
                    offset, rs = int(offset_parts[0]), self._get_register_number(offset_parts[1].strip(")"))
                except ValueError:
                    raise Exception(f"Invalid offset value: {offset_parts[0]}")
                # Check for data memory bounds before reading
                if (self.registers[rs] + offset) < 0 or (self.registers[rs] + offset) >= self.data_memory_size:
                    raise MemoryError(f"Memory access violation at address {self.registers[rs] + offset}")

                old_value = self.registers[rd]
                self.registers[rd] = self.data_memory[self.registers[rs] + offset]
                self.output_log.append(
                    f"  {self.get_register_name(rd)} changed from {old_value} to {self.registers[rd]}")
            elif op == "sw":
                if len(parts) != 3:
                    raise Exception(f"Invalid format for {op} instruction: {instruction}")
                rs, offset_rd = self._get_register_number(parts[1].strip(",")), parts[2]
                offset_parts = offset_rd.split("(")
                if len(offset_parts) != 2:
                    raise Exception(f"Invalid format for {op} instruction: {instruction}")
                try:
                    offset, rd = int(offset_parts[0]), self._get_register_number(offset_parts[1].strip(")"))
                except ValueError:
                    raise Exception(f"Invalid offset value: {offset_parts[0]}")
                # Check for data memory bounds before writing
                if (self.registers[rd] + offset) < 0 or (self.registers[rd] + offset) >= self.data_memory_size:
                    raise MemoryError(f"Memory access violation at address {self.registers[rd] + offset}")

                self.data_memory[self.registers[rd] + offset] = self.registers[rs]
                self.output_log.append(f"  Memory at {self.registers[rd] + offset} changed to {self.registers[rs]}")
            elif op == "beq":
                if len(parts) != 4:
                    raise Exception(f"Invalid format for {op} instruction: {instruction}")
                rs, rt, label = self._get_register_number(parts[1].strip(",")), self._get_register_number(
                    parts[2].strip(",")), parts[3]
                if self.registers[rs] == self.registers[rt]:
                    if label not in self.labels:
                        raise Exception(f"Label {label} not found")
                    self.pc = self.labels[label] - 1
                    self.output_log.append(f"  Branch taken to label: {label} (PC = {self.pc * 4})")

            elif op == "bne":
                if len(parts) != 4:
                    raise Exception(f"Invalid format for {op} instruction: {instruction}")
                rs, rt, label = self._get_register_number(parts[1].strip(",")), self._get_register_number(
                    parts[2].strip(",")), parts[3]
                if self.registers[rs] != self.registers[rt]:
                    if label not in self.labels:
                        raise Exception(f"Label {label} not found")
                    self.pc = self.labels[label] - 1
                    self.output_log.append(f"  Branch taken to label: {label} (PC = {self.pc * 4})")
            elif op == "j":
                if len(parts) != 2:
                    raise Exception(f"Invalid format for {op} instruction: {instruction}")
                label = parts[1]
                if label not in self.labels:
                    raise Exception(f"Label {label} not found")
                self.pc = self.labels[label] - 1
                self.output_log.append(f"  Jumped to label: {label} (PC = {self.pc * 4})")
            elif op == "jal":
                if len(parts) != 2:
                    raise Exception(f"Invalid format for {op} instruction: {instruction}")
                label = parts[1]
                if label not in self.labels:
                    raise Exception(f"Label {label} not found")
                self.registers[31] = (self.pc + 1) * 4
                self.output_log.append(f"  $ra set to {self.registers[31]}")
                self.pc = self.labels[label] - 1
                self.output_log.append(f"  Jumped to label: {label} (PC = {self.pc * 4})")
            elif op == "jr":
                if len(parts) != 2:
                    raise Exception(f"Invalid format for {op} instruction: {instruction}")
                rs = self._get_register_number(parts[1])
                self.pc = (self.registers[rs] // 4) - 1
                self.output_log.append(
                    f"  Jumped to address in register {self.get_register_name(rs)} (PC = {self.pc * 4})")
            else:
                raise Exception(f"Unsupported instruction: {op}")
        except Exception as e:
            self.output_log.append(f"Error executing {instruction}: {e}")
        except MemoryError as e:
            self.output_log.append(f"Error executing {instruction}: {e}")

    def step(self):
        if self.pc < len(self.instructions):
            instruction = self.instructions[self.pc]
            self.execution_trace.append(f"Executing: {instruction} (PC=0x{self.pc * 4:08X})")
            try:
                self.execute_instruction(instruction)
            except Exception as e:
                self.execution_trace.append(f"Error: {e}")

            self.pc += 1
        else:
            raise Exception("No more instructions to execute.")

    def reset(self):
        self.registers = {i: 0 for i in range(32)}
        self.registers[0] = 0
        self.registers[31] = 0
        self.instruction_memory.clear()
        self.data_memory.clear()
        self.pc = 0
        self.instructions.clear()
        self.labels.clear()
        self.execution_trace.clear()
        self.output_log.clear()

    def get_execution_trace(self):
        return "\n".join(self.execution_trace)

    def get_output_log(self):
        return "\n".join(self.output_log)

    def get_register_name(self, register_number):
        if register_number in self.reverse_register_map:
            return self.reverse_register_map[register_number]
        return f"$r{register_number}"

    def convert_to_machine_code(self):
        machine_codes = []
        for instruction in self.instructions:
            parts = instruction.split()
            if not parts:
                continue

            op = parts[0]
            try:
                if op == "add":
                    rd, rs, rt = parts[1].strip(","), parts[2].strip(","), parts[3]
                    machine_codes.append(
                        f"000000 {self._get_machine_code_register_number(rs):05b} {self._get_machine_code_register_number(rt):05b} {self._get_machine_code_register_number(rd):05b} 00000 100000")
                elif op == "addi":
                    rd, rs, imm = parts[1].strip(","), parts[2].strip(","), int(parts[3])
                    machine_codes.append(
                        f"001000 {self._get_machine_code_register_number(rs):05b} {self._get_machine_code_register_number(rd):05b} {imm:016b}")
                elif op == "sub":
                    rd, rs, rt = parts[1].strip(","), parts[2].strip(","), parts[3]
                    machine_codes.append(
                        f"000000 {self._get_machine_code_register_number(rs):05b} {self._get_machine_code_register_number(rt):05b} {self._get_machine_code_register_number(rd):05b} 00000 100010")
                elif op == "and":
                    rd, rs, rt = parts[1].strip(","), parts[2].strip(","), parts[3]
                    machine_codes.append(
                        f"000000 {self._get_machine_code_register_number(rs):05b} {self._get_machine_code_register_number(rt):05b} {self._get_machine_code_register_number(rd):05b} 00000 100100")
                elif op == "or":
                    rd, rs, rt = parts[1].strip(","), parts[2].strip(","), parts[3]
                    machine_codes.append(
                        f"000000 {self._get_machine_code_register_number(rs):05b} {self._get_machine_code_register_number(rt):05b} {self._get_machine_code_register_number(rd):05b} 00000 100101")
                elif op == "xor":
                    rd, rs, rt = parts[1].strip(","), parts[2].strip(","), parts[3]
                    machine_codes.append(
                        f"000000 {self._get_machine_code_register_number(rs):05b} {self._get_machine_code_register_number(rt):05b} {self._get_machine_code_register_number(rd):05b} 00000 100110")
                elif op == "slt":
                    rd, rs, rt = parts[1].strip(","), parts[2].strip(","), parts[3]
                    machine_codes.append(
                        f"000000 {self._get_machine_code_register_number(rs):05b} {self._get_machine_code_register_number(rt):05b} {self._get_machine_code_register_number(rd):05b} 00000 101010")
                elif op == "slti":
                    rd, rs, imm = parts[1].strip(","), parts[2].strip(","), int(parts[3])
                    machine_codes.append(
                        f"001010 {self._get_machine_code_register_number(rs):05b} {self._get_machine_code_register_number(rd):05b} {imm:016b}")
                elif op == "andi":
                    rd, rs, imm = parts[1].strip(","), parts[2].strip(","), int(parts[3])
                    machine_codes.append(
                        f"001100 {self._get_machine_code_register_number(rs):05b} {self._get_machine_code_register_number(rd):05b} {imm:016b}")
                elif op == "ori":
                    rd, rs, imm = parts[1].strip(","), parts[2].strip(","), int(parts[3])
                    machine_codes.append(
                        f"001101 {self._get_machine_code_register_number(rs):05b} {self._get_machine_code_register_number(rd):05b} {imm:016b}")
                elif op == "sll":
                    rd, rt, shamt = parts[1].strip(","), parts[2].strip(","), int(parts[3])
                    machine_codes.append(
                        f"000000 00000 {self._get_machine_code_register_number(rt):05b} {self._get_machine_code_register_number(rd):05b} {shamt:05b} 000000")
                elif op == "srl":
                    rd, rt, shamt = parts[1].strip(","), parts[2].strip(","), int(parts[3])
                    machine_codes.append(
                        f"000000 00000 {self._get_machine_code_register_number(rt):05b} {self._get_machine_code_register_number(rd):05b} {shamt:05b} 000010")
                elif op == "lw":
                    rd, offset_rs = parts[1].strip(","), parts[2]
                    offset_parts = offset_rs.split("(")
                    offset, rs = int(offset_parts[0]), offset_parts[1].strip(")")
                    machine_codes.append(
                        f"100011 {self._get_machine_code_register_number(rs):05b} {self._get_machine_code_register_number(rd):05b} {offset:016b}")
                elif op == "sw":
                    rs, offset_rd = parts[1].strip(","), parts[2]
                    offset_parts = offset_rd.split("(")
                    offset, rd = int(offset_parts[0]), offset_parts[1].strip(")")
                    machine_codes.append(
                        f"101011 {self._get_machine_code_register_number(rd):05b} {self._get_machine_code_register_number(rs):05b} {offset:016b}")
                elif op == "beq":
                    rs, rt, label = parts[1].strip(","), parts[2].strip(","), parts[3]
                    if label not in self.labels:
                        raise Exception(f"Label {label} not found")
                    address = self.labels[label] * 4
                    machine_codes.append(
                        f"000100 {self._get_machine_code_register_number(rs):05b} {self._get_machine_code_register_number(rt):05b} {address:016b}")
                elif op == "bne":
                    rs, rt, label = parts[1].strip(","), parts[2].strip(","), parts[3]
                    if label not in self.labels:
                        raise Exception(f"Label {label} not found")
                    address = self.labels[label] * 4
                    machine_codes.append(
                        f"000101 {self._get_machine_code_register_number(rs):05b} {self._get_machine_code_register_number(rt):05b} {address:016b}")
                elif op == "j":
                    label = parts[1]
                    if label not in self.labels:
                        raise Exception(f"Label {label} not found")
                    address = self.labels[label] * 4
                    machine_codes.append(f"000010 {address:026b}")
                elif op == "jal":
                    label = parts[1]
                    if label not in self.labels:
                        raise Exception(f"Label {label} not found")
                    address = self.labels[label] * 4
                    machine_codes.append(f"000011 {address:026b}")
                elif op == "jr":
                    rs = parts[1]
                    machine_codes.append(
                        f"000000 {self._get_machine_code_register_number(rs):05b} 00000 00000 00000 001000")
                else:
                    machine_codes.append(f"Unsupported instruction: {instruction}")
            except Exception as e:
                machine_codes.append(f"Error converting {instruction}: {e}")
        return "\n".join(machine_codes)


class MIPS_GUI:
    def __init__(self, root):
        self.simulator = MIPS_Simulator()

        root.title("MIPS Simulator")
        root.geometry("1200x800")
        root.configure(bg='#2C3E50')

        main_frame = tk.Frame(root, bg='#2C3E50')
        main_frame.pack(expand=True, fill="both", padx=10, pady=10)

        left_top_frame = tk.Frame(main_frame, bg='#2C3E50')
        left_top_frame.pack(side="left", fill="both", expand=True)

        assembly_frame = tk.LabelFrame(left_top_frame, text="ASSEMBLY",
                                       bg='#34495E', fg='white',
                                       font=('Arial', 10, 'bold'))
        assembly_frame.pack(side="left", fill="both", expand=True, padx=5, pady=3)  # reduced pady
        self.program_input = scrolledtext.ScrolledText(assembly_frame, height=10, width=35,  # reduced height
                                                       bg='#ECF0F1', fg='#2C3E50',
                                                       font=('Consolas', 11))
        self.program_input.pack(padx=5, pady=5, fill="both", expand=True)

        convert_frame = tk.LabelFrame(left_top_frame, text="Convert Machine Code",
                                      bg='#34495E', fg='white',
                                      font=('Arial', 10, 'bold'))
        convert_frame.pack(side="left", fill="both", expand=True, padx=5, pady=3)  # reduced pady
        self.machine_code_display = scrolledtext.ScrolledText(convert_frame, height=10, width=35,  # reduced height
                                                              bg='#ECF0F1', fg='#2C3E50',
                                                              font=('Consolas', 11))
        self.machine_code_display.pack(padx=5, pady=5, fill="both", expand=True)

        right_top_frame = tk.LabelFrame(main_frame, text="Register",
                                        bg='#34495E', fg='white',
                                        font=('Arial', 10, 'bold'))
        right_top_frame.pack(side="right", fill="both", expand=True, padx=5, pady=3)  # reduced pady
        self.registers_display = scrolledtext.ScrolledText(right_top_frame, height=10, width=30,  # reduced height
                                                           bg='#ECF0F1', fg='#2C3E50',
                                                           font=('Consolas', 11))
        self.registers_display.pack(padx=5, pady=5, fill="both", expand=True)

        bottom_frame = tk.Frame(root, bg='#2C3E50')
        bottom_frame.pack(side="bottom", fill="both", expand=True, padx=10, pady=5)

        control_frame = tk.Frame(bottom_frame, bg='#2C3E50')
        control_frame.pack(pady=5)

        button_style = {
            'font': ('Arial', 10, 'bold'),
            'width': 15,
            'height': 2,
            'border': 0,
            'borderwidth': 0,
            'cursor': 'hand2'
        }

        self.load_button = tk.Button(control_frame, text="Load Program",
                                     bg='#3498DB', fg='white',
                                     activebackground='#2980B9',
                                     command=self.load_program,
                                     **button_style)
        self.load_button.pack(side="left", padx=5)

        self.step_button = tk.Button(control_frame, text="Step",
                                     bg='#2ECC71', fg='white',
                                     activebackground='#27AE60',
                                     command=self.step,
                                     **button_style)
        self.step_button.pack(side="left", padx=5)

        self.run_button = tk.Button(control_frame, text="Run",
                                    bg='#E74C3C', fg='white',
                                    activebackground='#C0392B',
                                    command=self.run_program,
                                    **button_style)
        self.run_button.pack(side="left", padx=5)

        self.convert_button = tk.Button(control_frame, text="Convert to MC",
                                        bg='#9B59B6', fg='white',
                                        activebackground='#8E44AD',
                                        command=self.convert_to_machine_code,
                                        **button_style)
        self.convert_button.pack(side="left", padx=5)

        self.reset_button = tk.Button(control_frame, text="Reset",
                                      bg='#95A5A6', fg='white',
                                      activebackground='#7F8C8D',
                                      command=self.reset_simulator,
                                      **button_style)
        self.reset_button.pack(side="left", padx=5)

        output_container = tk.Frame(bottom_frame, bg='#2C3E50')
        output_container.pack(fill="both", expand=True, pady=5)

        output_frame = tk.LabelFrame(output_container, text="OUTPUT",
                                     bg='#34495E', fg='white',
                                     font=('Arial', 10, 'bold'))
        output_frame.pack(side="left", fill="both", expand=True, padx=5, pady=3)  # reduced pady
        self.output_area = scrolledtext.ScrolledText(output_frame, height=5, width=70,  # reduced height
                                                     bg='#ECF0F1', fg='#2C3E50',
                                                     font=('Consolas', 11))
        self.output_area.pack(padx=5, pady=5, fill="both", expand=True)

        trace_frame = tk.LabelFrame(output_container, text="EXECUTION TRACE",
                                    bg='#34495E', fg='white',
                                    font=('Arial', 10, 'bold'))
        trace_frame.pack(side="right", fill="both", expand=True, padx=5, pady=3)  # reduced pady
        self.trace_area = scrolledtext.ScrolledText(trace_frame, height=5, width=70,  # reduced height
                                                    bg='#ECF0F1', fg='#2C3E50',
                                                    font=('Consolas', 11))
        self.trace_area.pack(padx=5, pady=5, fill="both", expand=True)

        instruction_frame = tk.LabelFrame(bottom_frame, text="INSTRUCTION MEMORY",
                                          bg='#34495E', fg='white',
                                          font=('Arial', 10, 'bold'))
        instruction_frame.pack(fill="both", expand=True, pady=5, padx=5)
        self.instruction_memory_display = scrolledtext.ScrolledText(instruction_frame, height=5, width=140,
                                                                    # reduced height
                                                                    bg='#ECF0F1', fg='#2C3E50',
                                                                    font=('Consolas', 11))

        self.instruction_memory_display.pack(padx=5, pady=5, fill="both", expand=True)

        data_frame = tk.LabelFrame(bottom_frame, text="DATA MEMORY",
                                   bg='#34495E', fg='white',
                                   font=('Arial', 10, 'bold'))
        data_frame.pack(fill="both", expand=True, pady=5, padx=5)
        self.data_memory_display = scrolledtext.ScrolledText(data_frame, height=5, width=140,  # reduced height
                                                             bg='#ECF0F1', fg='#2C3E50',
                                                             font=('Consolas', 11))
        self.data_memory_display.pack(padx=5, pady=5, fill="both", expand=True)

        for button in [self.load_button, self.step_button, self.run_button,
                       self.convert_button, self.reset_button]:
            button.bind("<Enter>", lambda e, btn=button: btn.configure(relief=tk.RAISED))
            button.bind("<Leave>", lambda e, btn=button: btn.configure(relief=tk.FLAT))

    def load_program(self):
        program_text = self.program_input.get("1.0", tk.END).strip()
        try:
            self.simulator.load_program(program_text)
            self.display_output("Program loaded successfully.")
            self.update_registers()
            self.update_memory()
        except MemoryError as e:
            self.display_output(str(e))
        except Exception as e:
            self.display_output(f"Error loading program: {e}")

    def step(self):
        try:
            self.simulator.step()
            self.display_output(self.simulator.get_output_log())
            self.display_trace(self.simulator.get_execution_trace())
            self.update_registers()
            self.update_memory()
            self.simulator.output_log.clear()
        except Exception as e:
            self.display_output(str(e))

    def run_program(self):
        while True:
            try:
                self.simulator.step()
            except Exception as e:
                self.display_output(str(e))
                break
        self.display_output(self.simulator.get_output_log())
        self.display_trace(self.simulator.get_execution_trace())
        self.update_registers()
        self.update_memory()
        self.simulator.output_log.clear()

    def convert_to_machine_code(self):
        machine_code = self.simulator.convert_to_machine_code()
        self.machine_code_display.config(state="normal")
        self.machine_code_display.delete("1.0", tk.END)
        self.machine_code_display.insert(tk.END, machine_code)
        self.machine_code_display.config(state="disabled")

    def reset_simulator(self):
        self.simulator.reset()

        self.program_input.delete("1.0", tk.END)

        self.machine_code_display.config(state="normal")
        self.machine_code_display.delete("1.0", tk.END)
        self.machine_code_display.config(state="disabled")

        self.registers_display.config(state="normal")
        self.registers_display.delete("1.0", tk.END)
        self.registers_display.config(state="disabled")

        self.output_area.config(state="normal")
        self.output_area.delete("1.0", tk.END)
        self.output_area.config(state="disabled")

        self.trace_area.config(state="normal")
        self.trace_area.delete("1.0", tk.END)
        self.trace_area.config(state="disabled")

        self.instruction_memory_display.config(state="normal")
        self.instruction_memory_display.delete("1.0", tk.END)
        self.instruction_memory_display.config(state="disabled")

        self.data_memory_display.config(state="normal")
        self.data_memory_display.delete("1.0", tk.END)
        self.data_memory_display.config(state="disabled")

        self.display_output("Simulator has been reset.")

    def update_registers(self):
        self.registers_display.config(state="normal")
        self.registers_display.delete("1.0", tk.END)

        for reg_number, value in self.simulator.registers.items():
            reg_name = self.simulator.get_register_name(reg_number)
            self.registers_display.insert(tk.END, f"{reg_name}: {value}\n")

        self.registers_display.config(state="disabled")

    def update_memory(self):
        self.instruction_memory_display.config(state="normal")
        self.instruction_memory_display.delete("1.0", tk.END)
        for address, instruction in self.simulator.instruction_memory.items():
            self.instruction_memory_display.insert(tk.END, f"0x{address * 4:08X}: {instruction}\n")
        self.instruction_memory_display.config(state="disabled")

        self.data_memory_display.config(state="normal")
        self.data_memory_display.delete("1.0", tk.END)
        for address, value in self.simulator.data_memory.items():
            self.data_memory_display.insert(tk.END, f"0x{address * 4:08X}: {value}\n")
        self.data_memory_display.config(state="disabled")

    def display_output(self, output):
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        self.output_area.config(state="normal")
        self.output_area.delete("1.0", tk.END)
        self.output_area.insert(tk.END, f"[{current_time}] {output}\n")
        self.output_area.see(tk.END)
        self.output_area.config(state="disabled")

    def display_trace(self, output):

        self.trace_area.config(state="normal")
        self.trace_area.delete("1.0", tk.END)

        lines = output.split('\n')
        formatted_output = ""
        for line in lines:
            if line.strip():
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                formatted_output += f"[{timestamp}] {line}\n"

        self.trace_area.insert(tk.END, formatted_output)
        self.trace_area.see(tk.END)
        self.trace_area.config(state="disabled")


if __name__ == "__main__":
    root = tk.Tk()
    app = MIPS_GUI(root)
    root.mainloop()
