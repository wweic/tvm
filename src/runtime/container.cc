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
/*
 * \file src/runtime/container.cc
 * \brief POD container type implementations.
 */
#include <dmlc/logging.h>
#include <tvm/runtime/container.h>
#include <cstdint>
#include "object_internal.h"
#include "runtime_base.h"

namespace tvm {
namespace runtime {

ObjectRef ADTObj::operator[](size_t idx) const {
  if (idx > this->size_) {
    LOG(FATAL) << "Index out of bound at " << idx << " bound is " << this->size_ << "\n";
  }
  ObjectRef* field_p = reinterpret_cast<ObjectRef*>(AddressOf(idx));
  std::cout << (void*)field_p << " count " << field_p->get()->use_count() << "\n";
  return *field_p;
}

void* ADTObj::AddressOf(int i) const {
  ADTObj* self = const_cast<ADTObj*>(this);
  char* fields = reinterpret_cast<char*>(self) + sizeof(ADTObj);
  ObjectRef* field_p = reinterpret_cast<ObjectRef*>(fields + i * sizeof(ObjectRef));
  return field_p;
}

ADTObj::~ADTObj() {
  for (size_t i = 0; i < size_; ++i) {
    ObjectRef* fp = reinterpret_cast<ObjectRef*>(AddressOf(i));
    fp->ObjectRef::~ObjectRef();
  }
}

ADT::ADT(uint32_t tag, std::vector<ObjectRef> fields) {
  ADT(tag, fields.begin(), fields.end());
  return;

  size_t num_elems = fields.size();
  auto ptr = make_array<ADTObj, ObjectRef>(num_elems);
  ptr->tag_ = tag;
  ptr->size_ = num_elems;
  for (size_t i = 0; i < num_elems; ++i) {
    void* field_p = ptr->AddressOf(i);
    new (field_p) ObjectRef(fields[i]);
    ObjectRef* op = reinterpret_cast<ADTObj*>(field_p);
    std::cout << "Vector Setting to field " << (void*)op << " count: " << op->get()->use_count() << "\n";
  }
  data_ = std::move(ptr);
}

ADT::ADT(uint32_t tag, std::initializer_list<ObjectRef> init) {
  ADT(tag, init.begin(), init.end());
}

template<typename Iterator>
ADT::ADT(uint32_t tag, Iterator begin, Iterator end) {
  size_t num_elems = std::distance(begin, end);
  auto ptr = make_array<ADTObj, ObjectRef>(num_elems);
  ptr->tag_ = tag;
  ptr->size_ = num_elems;
  auto it = begin;
  for (size_t i = 0; i < num_elems; ++i) {
    void* field_p = ptr->AddressOf(i);
    new (field_p) ObjectRef(*it);
    ObjectRef* op = reinterpret_cast<ObjectRef*>(field_p);
    std::cout << "Setting to field " << (void*)op << " count: " << op->get()->use_count() << "\n";
    ++it;
  }
  data_ = std::move(ptr);
}

ADT ADT::Tuple(std::vector<ObjectRef> fields) {
  return ADT(0, fields);
}

}  // namespace runtime
}  // namespace tvm
