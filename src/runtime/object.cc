/*!
 *  Copyright (c) 2017 by Contributors
 * \file object.cc
 * \brief TVM runtime object used by VM.
 */

#include <tvm/logging.h>
#include <tvm/runtime/object.h>
#include <iostream>

namespace tvm {
namespace runtime {

std::ostream& operator<<(std::ostream& os, const ObjectTag& tag) {
  switch (tag) {
    case ObjectTag::kClosure:
      os << "Closure";
      break;
    case ObjectTag::kDatatype:
      os << "Datatype";
      break;
    case ObjectTag::kTensor:
      os << "Tensor";
      break;
    case ObjectTag::kExternalFunc:
      os << "ExternalFunction";
      break;
    default:
      LOG(FATAL) << "Invalid object tag: found " << static_cast<int>(tag);
  }
  return os;
}

Object Object::Tensor(const NDArray& data) {
  ObjectPtr<ObjectCell> ptr = MakeObject<TensorCell>(data);
  return Object(ptr);
}

Object Object::Datatype(size_t tag, const std::vector<Object>& fields) {
  ObjectPtr<ObjectCell> ptr = MakeObject<DatatypeCell>(tag, fields);
  return Object(ptr);
}

Object Object::Tuple(const std::vector<Object>& fields) {
  return Object::Datatype(0, fields);
}

Object Object::Closure(size_t func_index, const std::vector<Object>& free_vars) {
  ObjectPtr<ObjectCell> ptr = MakeObject<ClosureCell>(func_index, free_vars);
  return Object(ptr);
}

ObjectPtr<TensorCell> Object::AsTensor() const {
  CHECK(ptr.get());
  CHECK(ptr.get()->tag == ObjectTag::kTensor);
  return ptr.As<TensorCell>();
}

ObjectPtr<DatatypeCell> Object::AsDatatype() const {
  CHECK(ptr.get());
  CHECK(ptr.get()->tag == ObjectTag::kDatatype);
  return ptr.As<DatatypeCell>();
}

ObjectPtr<ClosureCell> Object::AsClosure() const {
  CHECK(ptr.get());
  CHECK(ptr.get()->tag == ObjectTag::kClosure);
  return ptr.As<ClosureCell>();
}

NDArray ToNDArray(const Object& obj) {
  auto tensor = obj.AsTensor();
  return tensor->data;
}

}  // namespace runtime
}  // namespace tvm
