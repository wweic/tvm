/*!
 *  Copyright (c) 2018 by Contributors
 * \file tvm/relay/vm/vm.h
 * \brief Abstract device memory management API
 */
#ifndef TVM_RELAY_VM_VM_H_
#define TVM_RELAY_VM_VM_H_

#include <tvm/relay/expr_functor.h>
#include <tvm/relay/logging.h>
#include <tvm/runtime/memory_manager.h>
#include <tvm/runtime/object.h>
#include <string>
#include <unordered_map>
#include <vector>
#include <memory>

namespace tvm {
namespace relay {
namespace vm {

using namespace tvm::runtime;

using RegName = size_t;

enum struct Opcode {
  Move,
  Ret,
  Invoke,
  InvokeClosure,
  InvokePacked,
  AllocTensor,
  AllocDatatype,
  AllocClosure,
  GetField,
  If,
  Select,
  LoadConst,
  Goto
};

struct Instruction {
  struct TensorInfo {
      RegName shape_register;
      size_t ndim;
      DLDataType dtype;
  };

  Opcode op;

  // Destination register that the opcode writes to
  RegName dst;

  union {
    TensorInfo tensor_info;

    // For InvokeClosure
    struct {
      RegName closure;
      size_t closure_args_num;
      RegName* closure_args;
    };
    // For Ret
    struct {
      RegName result;
    };
    // For Move
    struct {
      RegName from;
    };
    struct {
      size_t packed_index;
      size_t arity;
      size_t output_size;
      RegName* packed_args;
    };
    // For Select node
    struct {
      RegName select_cond;
      RegName select_op1;
      RegName select_op2;
    };
    // For If node
    struct {
      RegName if_cond;
      size_t true_offset;
      size_t false_offset;
    };
    // For Invoke
    struct {
      size_t func_index;
      size_t num_args;
      RegName* invoke_args_registers;
    };
    struct {
      size_t const_index;
    };
    struct {
      size_t pc_offset;
    };
    // For GetField
    struct {
      RegName object;
      size_t field_index;
    };
    // For AllocDatatype
    struct {
      size_t constructor_tag;
      size_t num_fields;
      RegName* datatype_fields;
    };
    // For AllocClosure
    struct {
      size_t clo_index;
      size_t num_freevar;
      RegName* free_vars;
    };
  };

  Instruction();
  Instruction(const Instruction& instr);
  ~Instruction();

  friend std::ostream& operator<<(std::ostream& os, const Instruction&);
};

// Helpers to build instructions.
Instruction Select(RegName cond, RegName op1, RegName op2, RegName dst);
Instruction Ret(RegName result);
Instruction InvokePacked(size_t packed_index, size_t arity, size_t output_size,
                         const std::vector<RegName>& args);
Instruction AllocTensor(RegName shape_register, const std::vector<int64_t>& shape,
                        DLDataType dtype, RegName dst);
Instruction AllocDatatype(size_t tag, size_t num_fields, const std::vector<RegName>& fields,
                          RegName dst);
Instruction AllocClosure(size_t func_index, size_t num_freevar,
                         const std::vector<RegName>& free_vars, RegName dst);
Instruction GetField(RegName object, size_t field_index, RegName dst);
Instruction If(RegName cond, size_t true_branch, size_t false_branch);
Instruction Goto(size_t pc_offset);
Instruction Invoke(size_t func_index, const std::vector<RegName>& args, RegName dst);
Instruction InvokeClosure(RegName closure, const std::vector<RegName>& args, RegName dst);
Instruction LoadConst(size_t const_index, RegName dst);
Instruction Move(RegName src, RegName dst);

struct VMFunction {
  std::string name;
  size_t params;
  std::vector<Instruction> instructions;
  size_t register_file_size;

  VMFunction(std::string name, size_t params, std::vector<Instruction> instructions,
             size_t register_file_size)
    : name(name), params(params), instructions(instructions), register_file_size(register_file_size)
      {}

  VMFunction() {}

  friend std::ostream& operator<<(std::ostream& os, const VMFunction&);
};

void VMFunctionPrint(const VMFunction& vm_func);

/*! \brief A representation of a stack frame.
 *
 * We store the current frame's information on the call stack (frames)
 * when we finish execution we restore the virtual machine state.
 */
struct VMFrame {
    size_t pc;
    size_t func_index;
    size_t args;
    const Instruction* code;

    std::vector<Object> register_file;

    RegName caller_return_register;

    VMFrame(size_t pc, size_t func_index, size_t args, const Instruction* code,
            size_t register_file_size)
      : pc(pc), func_index(func_index), args(args), code(code), register_file(register_file_size),
        caller_return_register(0) {}
};

struct VirtualMachine {
    // TODO(@jroesch):
    std::vector<PackedFunc> packed_funcs;
    std::vector<VMFunction> functions;
    std::vector<VMFrame> frames;
    std::vector<Object> constants;

    // Frame State
    size_t func_index;
    const Instruction* code;
    size_t pc;
    // Special register to save function call return value
    Object return_register;

    std::vector<TVMContext> ctxs;

    // Interface debugging.
    std::unordered_map<GlobalVar, size_t, NodeHash, NodeEqual> global_map;
    std::unordered_map<size_t, Constructor> tag_index_map;

    void PushFrame(size_t arg_count, size_t ret_pc, const VMFunction& vm_func);
    size_t PopFrame();
    void InvokeGlobal(const VMFunction& func, const std::vector<Object>& args);
    void Run();

    inline void WriteRegister(RegName r, Object v);
    inline Object ReadRegister(RegName r);

    Object Invoke(const VMFunction& func, const std::vector<Object>& args);
    Object Invoke(const GlobalVar& global, const std::vector<Object>& args);

    VirtualMachine() :
      functions(), frames(),
      func_index(0), code(nullptr), pc(0) {}

    void Init(const std::vector<TVMContext>& ctxs);

    static VirtualMachine FromModule(const Module& module,
                                     const std::vector<TVMContext>& ctxs);
};

bool IsClosure(const Function& func);
Module LambdaLift(const Module& module);
Module InlinePrimitives(const Module& module);

VirtualMachine CompileModule(const Module& mod);

}  // namespace vm
}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_VM_VM_H_
