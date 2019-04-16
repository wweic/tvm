/*!
 *  Copyright (c) 2018 by Contributors
 * \file tvm/relay/vm/vm.h
 * \brief A virtual machine for executing Relay programs.
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

/*! \brief A register name. */
using RegName = size_t;

/*! \brief A enumeration of Relay's opcodes.
 *
 * The opcode is used to implement instruction
 * as a tagged union.
*/
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
  Opcode op;

  // Destination register that the opcode writes to
  RegName dst;

  union {
    struct /* AllocTensorOperands */ {
      RegName shape_register;
      size_t ndim;
      DLDataType dtype;
    };
    struct /* InvokeClosureOperands */ {
      RegName closure;
      size_t closure_args_num;
      RegName* closure_args;
    };
    struct /* ReturnOperands */ {
      RegName result;
    };
    struct /* MoveOperands */ {
      RegName from;
    };
    struct /* PackedOperands */ {
      size_t packed_index;
      size_t arity;
      size_t output_size;
      RegName* packed_args;
    };
    struct /* SelectOperands */ {
      RegName select_cond;
      RegName select_op1;
      RegName select_op2;
    };
    struct /* IfOperands */ {
      RegName if_cond;
      size_t true_offset;
      size_t false_offset;
    };
    struct /* InvokeOperands */ {
      size_t func_index;
      size_t num_args;
      RegName* invoke_args_registers;
    };
    struct /* ConstOperands */ {
      size_t const_index;
    };
    struct /* JumpOperands */ {
      size_t pc_offset;
    };
    struct /* ProjOperands */ {
      RegName object;
      size_t field_index;
    };
    struct /* AllocDatatypeOperands */ {
      size_t constructor_tag;
      size_t num_fields;
      RegName* datatype_fields;
    };
    struct /* AllocClosureOperands */ {
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

/*! \brief */
Instruction Select(RegName cond, RegName op1, RegName op2, RegName dst);
/*! \brief */
Instruction Ret(RegName result);
/*! \brief */
Instruction InvokePacked(size_t packed_index, size_t arity, size_t output_size,
                         const std::vector<RegName>& args);
/*! \brief */
Instruction AllocTensor(RegName shape_register, const std::vector<int64_t>& shape,
                        DLDataType dtype, RegName dst);
/*! \brief */
Instruction AllocDatatype(size_t tag, size_t num_fields, const std::vector<RegName>& fields,
                          RegName dst);
/*! \brief */
Instruction AllocClosure(size_t func_index, size_t num_freevar,
                         const std::vector<RegName>& free_vars, RegName dst);
/*! \brief */
Instruction GetField(RegName object, size_t field_index, RegName dst);
/*! \brief */
Instruction If(RegName cond, size_t true_branch, size_t false_branch);
/*! \brief */
Instruction Goto(size_t pc_offset);
/*! \brief */
Instruction Invoke(size_t func_index, const std::vector<RegName>& args, RegName dst);
/*! \brief */
Instruction InvokeClosure(RegName closure, const std::vector<RegName>& args, RegName dst);
/*! \brief */
Instruction LoadConst(size_t const_index, RegName dst);
/*! \brief */
Instruction Move(RegName src, RegName dst);

/*! \brief A representation of a Relay function in the VM.
 *
 * Contains metadata about the compiled function, as
 * we as the compiled VM instructions.
 */
struct VMFunction {
  /*! \brief The function's name. */
  std::string name;
  /*! \brief The number of function parameters. */
  size_t params;
  /*! \brief The instructions representing the function. */
  std::vector<Instruction> instructions;
  /*! \brief The size of the frame for this function */
  size_t register_file_size;

  VMFunction(std::string name,
             size_t params,
             std::vector<Instruction> instructions,
             size_t register_file_size)
    : name(name),
      params(params),
      instructions(instructions),
      register_file_size(register_file_size)
      {}

  VMFunction() {}

  friend std::ostream& operator<<(std::ostream& os, const VMFunction&);
};

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

/*! \brief The virtual machine.
 *
 * The virtual machine contains all the current execution state,
 * as well as the global view of functions, the global constant
 * table, the compiled operators.
 */
struct VirtualMachine {
    /*! \brief The virtual machine's packed function table. */
    std::vector<PackedFunc> packed_funcs;
    /*! \brief The virtual machine's function table. */
    std::vector<VMFunction> functions;
    /*! \brief The frame stack. */
    std::vector<VMFrame> frames;
    /*! \brief The constant pool. */
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

/*! \brief Compile a module into a virtual machine which can be executed. */
VirtualMachine CompileModule(const Module& mod);

}  // namespace vm
}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_VM_VM_H_
