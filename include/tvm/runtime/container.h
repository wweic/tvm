/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file tvm/runtime/container.h
 * \brief Common POD(plain old data) container types.
 */
#ifndef TVM_RUNTIME_CONTAINER_H_
#define TVM_RUNTIME_CONTAINER_H_
#include <tvm/runtime/object.h>
#include <initializer_list>
#include <vector>

namespace tvm {
namespace runtime {

/*! \brief An object representing a structure or enumeration. */
class ADTObj : public Object {
 public:
  /*! \brief The tag representing the constructor used. */
  uint32_t tag_;
  /*! \brief Number of fields in the ADT object. */
  uint32_t size_;
  // The fields of the structure follows directly in memory.

  /*!
   * \brief Access element at index
   * \param idx The index of the element.
   * \return The ObjectRef at the index.
   */
  ObjectRef operator[](size_t idx) const;

  void* AddressOf(int i) const;

  virtual ~ADTObj();

  static constexpr const uint32_t _type_index = TypeIndex::kVMADT;
  static constexpr const char* _type_key = "vm.ADT";
  TVM_DECLARE_FINAL_OBJECT_INFO(ADTObj, Object);
};

/*! \brief reference to algebraic data type objects. */
class ADT : public ObjectRef {
 public:
  ADT(uint32_t tag, std::vector<ObjectRef> fields);

  template<typename Iterator>
  ADT(uint32_t tag, Iterator begin, Iterator end);

  ADT(uint32_t tag, std::initializer_list<ObjectRef> init);

  void Dump();
  
  /*!
   * \brief construct a tuple object.
   * \param fields The fields of the tuple.
   * \return The constructed tuple type.
   */
  static ADT Tuple(std::vector<ObjectRef> fields);

  TVM_DEFINE_OBJECT_REF_METHODS(ADT, ObjectRef, ADTObj);
};

}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_CONTAINER_H_
