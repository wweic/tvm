# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=no-else-return, unidiomatic-typecheck, invalid-name
"""A prelude containing useful global functions and ADT definitions."""
import os
from .ty import GlobalTypeVar, TensorType, Any, TypeVar, FuncType, TupleType, scalar_type
from .expr import Var, Function, GlobalVar, Let, If, Tuple, TupleGetItem, const
from .op.tensor import add, subtract, equal
from .adt import Constructor, TypeData, Clause, Match
from .adt import PatternConstructor, PatternVar, PatternWildcard
from . import op
from .parser import fromtext
__PRELUDE_PATH__ = os.path.dirname(os.path.realpath(__file__))
from .module import Module

class Prelude:
    """Contains standard definitions."""

    def define_tensor_adt(self):
        """dynamic tensor
        """
        self.tensor_t = GlobalTypeVar("tensor_t")
        tensor0_type = TensorType([])        
        tensor1_type = TensorType([Any()])
        tensor2_type = TensorType([Any(), Any()])        
        tensor3_type = TensorType([Any(), Any(), Any()])                
        self.tensor_nil = Constructor("tensor_nil", [], self.tensor_t)
        self.tensor0 = Constructor("tensor0", [tensor0_type], self.tensor_t)
        self.tensor1 = Constructor("tensor1", [tensor1_type], self.tensor_t)
        self.tensor2 = Constructor("tensor2", [tensor2_type], self.tensor_t)
        self.tensor3 = Constructor("tensor3", [tensor3_type], self.tensor_t)
        self.mod[self.tensor_t] = TypeData(self.tensor_t, [], [self.tensor_nil, self.tensor0, self.tensor1, self.tensor2])

    def define_tensor_add_one(self):
        self.tensor_add_one = GlobalVar("tensor_add_one")
        x = Var("x", self.tensor_t())
        t0 = Var("t0")
        t1 = Var("t1")
        t2 = Var("t2")                
        tensor0_case = Clause(PatternConstructor(self.tensor0, [PatternVar(t0)]), self.tensor1(op.expand_dims(t0, 0, 1)))        
        tensor1_case = Clause(PatternConstructor(self.tensor1, [PatternVar(t1)]), self.tensor2(op.expand_dims(t1, 0, 1)))
        tensor2_case = Clause(PatternConstructor(self.tensor2, [PatternVar(t2)]), self.tensor3(op.expand_dims(t2, 0, 1)))
        self.mod[self.tensor_add_one] = Function([x], Match(x, [tensor0_case, tensor1_case, tensor2_case]))

    def define_tensor_concat(self):
        self.tensor_concatenate = GlobalVar("tensor_concatenate")
        x = Var("x", self.tensor_t())
        y = Var("y", self.tensor_t())        

        t11 = Var("t11")
        t12 = Var("t12")
        t21 = Var("t21")
        t22 = Var("t22")
        tensor1_case = Clause(PatternConstructor(self.tensor1, [PatternVar(t11)]),
                              Match(y, [Clause(PatternConstructor(self.tensor1, [PatternVar(t12)]), 
                                        self.tensor1(op.concatenate([t11, t12], axis=0)) )]
                                    ))
        tensor2_case = Clause(PatternConstructor(self.tensor2, [PatternVar(t21)]), 
                              Match(y, [Clause(PatternConstructor(self.tensor2, [PatternVar(t22)]), 
                                        self.tensor2(op.concatenate([t21, t22], axis=0)) )]
                                    ))
        self.mod[self.tensor_concatenate] = Function([x, y], Match(x, [tensor1_case, tensor2_case]))        

    def define_list_adt(self):
        """Defines a LISP-style list ADT. An empty list is
        represented by nil(). A member x can be appended to the
        front of a list l via the constructor cons(x, l)."""
        self.l = GlobalTypeVar("list")
        a = TypeVar("a")
        self.nil = Constructor("nil", [], self.l)
        self.cons = Constructor("cons", [a, self.l(a)], self.l)
        self.mod[self.l] = TypeData(self.l, [a], [self.nil, self.cons])

    def define_tensor_array(self):
        self.tensor_array = GlobalVar("tensor_array")
        n = Var("x", scalar_type('int32'))        
        body = If(equal(n, const(0)),
                  self.nil(),
                  self.cons(self.tensor_nil(), self.tensor_array(subtract(n, const(1)))))
        self.mod[self.tensor_array] = Function([n], body, self.l(self.tensor_t()), [])

    def define_tensor_array_read(self):
        self.tensor_array_read = GlobalVar("tensor_array_read")
        tensor_array = Var("tensor_array", self.l(self.tensor_t()))
        n = Var("x", scalar_type('int32'))
        self.mod[self.tensor_array_read] = Function([tensor_array, n], self.nth(tensor_array, n), self.tensor_t(), [])

    def define_tensor_array_size(self):
        self.tensor_array_size = GlobalVar("tensor_array_size")
        tensor_array = Var("tensor_array", self.l(self.tensor_t()))
        self.mod[self.tensor_array_size] = Function([tensor_array], self.length(tensor_array), scalar_type('int32'), [])

    def define_tensor_array_write(self):
        self.tensor_array_write = GlobalVar("tensor_array_write")
        tensor_array = Var("tensor_array", self.l(self.tensor_t()))
        n = Var("x", scalar_type('int32'))
        v = Var("v", self.tensor_t())
        self.mod[self.tensor_array_write] = Function([tensor_array, n, v], self.update(tensor_array, n, v), self.l(self.tensor_t()), [])

    def define_tensor_array_stack(self):
        self.tensor_array_stack = GlobalVar("tensor_array_stack")
        tensor_array = Var("tensor_array", self.l(self.tensor_t()))
        tensor_array_add_one = self.map(self.tensor_add_one, tensor_array)
        tensors = self.foldl(self.tensor_concatenate, self.hd(tensor_array_add_one), self.tl(tensor_array_add_one))
        self.mod[self.tensor_array_stack] = Function([tensor_array], tensors, self.tensor_t(), [])

    def define_tensor_array_unstack_tensor1(self):
        self.tensor_array_unstack_tensor1_helper = GlobalVar("tensor_array_unstack_tensor1_helper")        
        tensor = Var("t", TensorType([Any()]))        
        up = Var("up", scalar_type('int32'))
        i = Var("i", scalar_type('int32'))        

        helper_body = If(equal(i, up), self.nil(), self.cons(self.tensor0(op.take(tensor, i)), 
            self.tensor_array_unstack_tensor1_helper(add(i, const(1)), up, tensor))
                         )
        self.mod[self.tensor_array_unstack_tensor1_helper] = Function([i, up, tensor], helper_body, self.l(self.tensor_t()), [])

        self.tensor_array_unstack_tensor1 = GlobalVar("tensor_array_unstack_tensor1")
        tensor1 = Var("tensor", TensorType([Any()]))
        shape = op.shape_of(tensor1)
        ndim = op.take(shape, const(0))
        self.mod[self.tensor_array_unstack_tensor1] = Function([tensor1], self.tensor_array_unstack_tensor1_helper(const(0), ndim, tensor1), 
            self.l(self.tensor_t()), [])

    def define_tensor_array_unstack_tensor2(self):
        self.tensor_array_unstack_tensor2_helper = GlobalVar("tensor_array_unstack_tensor2_helper")
        tensor = Var("t", TensorType([Any(), Any()]))
        up = Var("up", scalar_type('int32'))
        i = Var("i", scalar_type('int32'))

        helper_body = If(equal(i, up),
                         self.nil(),
                         self.cons(self.tensor1(op.take(tensor, i, axis=0)),
                                   self.tensor_array_unstack_tensor2_helper(
                                       add(i, const(1)), up, tensor)))
        self.mod[self.tensor_array_unstack_tensor2_helper] = Function([i, up, tensor], helper_body, self.l(self.tensor_t()), [])

        self.tensor_array_unstack_tensor2 = GlobalVar("tensor_array_unstack_tensor2")
        tensor2 = Var("tensor", TensorType([Any(), Any()]))
        shape = op.shape_of(tensor2)
        ndim = op.take(shape, const(0))
        self.mod[self.tensor_array_unstack_tensor2] = Function([tensor2], self.tensor_array_unstack_tensor2_helper(const(0), ndim, tensor2),
            self.l(self.tensor_t()), [])
        
        print(self.mod[self.tensor_array_unstack_tensor2].astext())
        print(self.mod[self.tensor_array_unstack_tensor2_helper].astext())

    def define_tensor_array_scatter(self):
        self.tensor_array_scatter_helper = GlobalVar("tensor_array_scatter_helper")
        ta = Var("ta", self.l(self.tensor_t()))
        current = Var("current", scalar_type('int32'))
        limit = Var("limit", scalar_type('int32'))
        indices_ = Var('indices_', TensorType([Any()], 'int32'))
        values_ = Var('values_', self.l(self.tensor_t()))

        helper_body = If(equal(current, limit),
                         ta,
                         self.tensor_array_scatter_helper(
                             self.tensor_array_write(ta, op.take(indices_, current), self.tensor_array_read(values_, current)),
                             add(current, const(1)),
                             limit, indices_, values_))

        self.mod[self.tensor_array_scatter_helper] = Function([ta, current, limit, indices_, values_], helper_body, self.l(self.tensor_t()), [])

        self.tensor_array_scatter = GlobalVar("tensor_array_scatter")
        tensor_array = Var("tensor_array", self.l(self.tensor_t()))
        indices = Var('indices', TensorType([Any()], 'int32'))
        values = Var('values', self.l(self.tensor_t()))

        indices_shape = op.shape_of(indices)
        limit = op.take(indices_shape, const(0))
        body = self.tensor_array_scatter_helper(tensor_array, const(0), limit, indices, values)
        self.mod[self.tensor_array_scatter] = Function([tensor_array, indices, values], body, self.l(self.tensor_t()), [])

    def define_tensor_array_gather(self):
        self.tensor_array_gather_helper = GlobalVar("tensor_array_gather_helper")
        ta = Var("ta", self.l(self.tensor_t()))
        accu = Var("accu", self.l(self.tensor_t()))        
        current = Var("current", scalar_type('int32'))
        limit = Var("limit", scalar_type('int32'))
        indices_ = Var('indices_', TensorType([Any()], 'int32'))

        helper_body = If(equal(current, const(0)),
                         self.tensor_array_stack(accu),
                         self.tensor_array_gather_helper(
                             ta,
                             self.cons(self.tensor_array_read(ta, op.take(indices_, subtract(current, const(1)))), accu),
                             subtract(current, const(1)),
                             limit, indices_))

        self.mod[self.tensor_array_gather_helper] = \
            Function([ta, accu, current, limit, indices_], helper_body, self.tensor_t(), [])

        self.tensor_array_gather = GlobalVar("tensor_array_gather")
        tensor_array = Var("tensor_array", self.l(self.tensor_t()))
        indices = Var('indices', TensorType([Any()], 'int32'))

        indices_shape = op.shape_of(indices)
        limit = op.take(indices_shape, const(0))
        body = self.tensor_array_gather_helper(tensor_array, self.nil(), limit, limit, indices)
        self.mod[self.tensor_array_gather] = Function([tensor_array, indices], body, self.tensor_t(), [])

    def define_tensor_take(self):
        self.tensor_take = GlobalVar('tensor_take')
        t = Var('tensor', self.tensor_t())
        lower = Var('lower', scalar_type('int32'))
        upper = Var('upper', scalar_type('int32'))

        t1 = Var('t1')
        t2 = Var('t2')
        t3 = Var('t3')
        tensor1_case = Clause(PatternConstructor(self.tensor1, [PatternVar(t1)]), self.tensor1(op.take(t1, op.arange(lower, upper, dtype='int32')))
                              )
        tensor2_case = Clause(PatternConstructor(self.tensor2, [PatternVar(t2)]), self.tensor2(op.take(t2, op.arange(lower, upper, dtype='int32'), axis=0))
                              )
        tensor3_case = Clause(PatternConstructor(self.tensor3, [PatternVar(t3)]), self.tensor3(op.take(t3, op.arange(lower, upper, dtype='int32'), axis=0))
                              )
        self.mod[self.tensor_take] = Function([t, lower, upper], Match(t, [tensor1_case, tensor2_case, tensor3_case]), self.tensor_t(), [])

    def define_tensor_array_split(self):
        self.tensor_array_split_helper = GlobalVar('ta_split_helper')
        ta1 = Var("tensor_array", self.l(self.tensor_t()))
        value1 = Var('value1', self.tensor_t())
        offset1 = Var('offset1', scalar_type('int32'))
        current1 = Var('current1', scalar_type('int32'))
        limit1 = Var('limit1', scalar_type('int32'))
        lengths1 = Var('lengths', TensorType([Any()], 'int32'))

        helper1_body = If(equal(current1, limit1),
                          ta1,
                          self.tensor_array_write(
                              self.tensor_array_split_helper(
                                  ta1,
                                  value1,
                                  add(offset1, op.take(lengths1, current1)),
                                  add(current1, const(1)),
                                  limit1,
                                  lengths1
                              ),
                              current1,
                              self.tensor_take(value1,
                                               offset1,
                                               add(
                                                 op.take(lengths1, current1),
                                                 offset1)
                          )))
        self.mod[self.tensor_array_split_helper] = \
            Function([ta1, value1, offset1, current1, limit1, lengths1], helper1_body, self.l(self.tensor_t()), [])


        self.tensor_array_split = GlobalVar("tensor_array_split")
        tensor_array = Var("tensor_array", self.l(self.tensor_t()))
        value = Var('value', self.tensor_t())
        lengths = Var('lengths', TensorType([Any()], 'int32'))

        lengths_shape = op.shape_of(lengths)
        lengths_limit = op.take(lengths_shape, const(0))
        body = self.tensor_array_split_helper(
            tensor_array,
            value,
            const(0),
            const(0),
            lengths_limit,
            lengths)
        self.mod[self.tensor_array_split] = Function([tensor_array, value, lengths], body, self.l(self.tensor_t()), [])

    def define_tensor_array_concat(self):
        self.tensor_array_concat = GlobalVar('tensor_array_concat')
        tensor_array = Var("tensor_array", self.l(self.tensor_t()))
        hd = Var("hd")
        tl = Var("tl")

        nil_case = Clause(PatternConstructor(self.nil), self.tensor_nil())
        cons_case = Clause(PatternConstructor(self.cons, [PatternVar(hd), PatternVar(tl)]),
                           Match(tl, [
                               Clause(PatternConstructor(self.nil), hd),
                               Clause(PatternWildcard(), self.tensor_concatenate(hd, self.tensor_array_concat(tl)))
                           ], False)
                           )

        self.mod[self.tensor_array_concat] = Function([tensor_array], Match(tensor_array, [nil_case, cons_case], False), self.tensor_t(), [])

    def define_list_hd(self):
        """Defines a function to get the head of a list. Assume the list has at least one
        element.

        hd(l) : list[a] -> a
        """
        self.hd = GlobalVar("hd")
        a = TypeVar("a")
        x = Var("x", self.l(a))
        y = Var("y")
        z = Var("z")
        cons_case = Clause(PatternConstructor(self.cons, [PatternVar(y), PatternVar(z)]), y)
        self.mod[self.hd] = Function([x], Match(x, [cons_case], False), a, [a])

    def define_list_tl(self):
        """Defines a function to get the tail of a list.

        tl(l) : list[a] -> list[a]
        """
        self.tl = GlobalVar("tl")
        a = TypeVar("a")
        x = Var("x", self.l(a))
        y = Var("y")
        z = Var("z")
        cons_case = Clause(PatternConstructor(self.cons, [PatternVar(y), PatternVar(z)]), z)
        self.mod[self.tl] = Function([x], Match(x, [cons_case], False), self.l(a), [a])


    def define_list_nth(self):
        """Defines a function to get the nth element of a list.

        nth(l) : list[a] -> Tensor[(), int32] -> a
        """
        self.nth = GlobalVar("nth")
        a = TypeVar("a")
        x = Var("x", self.l(a))
        n = Var("n", scalar_type('int32'))

        body = If(equal(n, const(0)),
                  self.hd(x),
                  self.nth(self.tl(x), subtract(n, const(1))))

        self.mod[self.nth] = Function([x, n], body, a, [a])


    def define_list_update(self):
        """Defines a function to update the nth element of a list and return the updated list.

        update(l, i, v) : list[a] -> Tensor[(), int32] -> a -> list[a]
        """
        self.update = GlobalVar("update")
        a = TypeVar("a")
        l = Var("l", self.l(a))
        n = Var("n", scalar_type('int32'))
        v = Var("v", a)

        body = If(equal(n, const(0)),
                  self.cons(v, self.tl(l)),
                  self.cons(self.hd(l),
                            self.update(self.tl(l),
                                        subtract(n, const(1)),
                                        v)))

        self.mod[self.update] = Function([l, n, v], body, self.l(a), [a])


    def define_list_map(self):
        """Defines a function for mapping a function over a list's
        elements. That is, map(f, l) returns a new list where
        the ith member is f applied to the ith member of l.

        map(f, l) : fn<a, b>(fn(a) -> b, list[a]) -> list[b]
        """
        self.map = GlobalVar("map")
        a = TypeVar("a")
        b = TypeVar("b")
        f = Var("f", FuncType([a], b))
        x = Var("x", self.l(a))
        y = Var("y")
        z = Var("z")
        nil_case = Clause(PatternConstructor(self.nil), self.nil())
        cons_case = Clause(PatternConstructor(self.cons, [PatternVar(y), PatternVar(z)]),
                           self.cons(f(y), self.map(f, z)))
        self.mod[self.map] = Function([f, x], Match(x, [nil_case, cons_case]), self.l(b), [a, b])


    def define_list_foldl(self):
        """Defines a left-way fold over a list.

        foldl(f, z, l) : fn<a, b>(fn(a, b) -> a, a, list[b]) -> a

        foldl(f, z, cons(a1, cons(a2, cons(a3, cons(..., nil)))))
        evaluates to f(...f(f(f(z, a1), a2), a3)...)
        """
        self.foldl = GlobalVar("foldl")
        a = TypeVar("a")
        b = TypeVar("b")
        f = Var("f", FuncType([a, b], a))
        av = Var("av", a)
        bv = Var("bv", self.l(b))
        y = Var("y")
        z = Var("z")
        nil_case = Clause(PatternConstructor(self.nil), av)
        cons_case = Clause(PatternConstructor(self.cons, [PatternVar(y), PatternVar(z)]),
                           self.foldl(f, f(av, y), z))
        self.mod[self.foldl] = Function([f, av, bv],
                                        Match(bv, [nil_case, cons_case]), a, [a, b])


    def define_list_foldr(self):
        """Defines a right-way fold over a list.

        foldr(f, l, z) : fn<a, b>(fn(a, b) -> b, list[a], b) -> b

        foldr(f, cons(a1, cons(a2, cons(..., cons(an, nil)))), z)
        evalutes to f(a1, f(a2, f(..., f(an, z)))...)
        """
        self.foldr = GlobalVar("foldr")
        a = TypeVar("a")
        b = TypeVar("b")
        f = Var("f", FuncType([a, b], b))
        av = Var("av", self.l(a))
        bv = Var("bv", b)
        y = Var("y")
        z = Var("z")
        nil_case = Clause(PatternConstructor(self.nil), bv)
        cons_case = Clause(PatternConstructor(self.cons, [PatternVar(y), PatternVar(z)]),
                           f(y, self.foldr(f, bv, z)))
        self.mod[self.foldr] = Function([f, bv, av],
                                        Match(av, [nil_case, cons_case]), b, [a, b])


    def define_list_foldr1(self):
        """Defines a right-way fold over a nonempty list.

        foldr1(f, l) : fn<a>(fn(a, a) -> a, list[a]) -> a

        foldr1(f, cons(a1, cons(a2, cons(..., cons(an, nil)))))
        evalutes to f(a1, f(a2, f(..., f(an-1, an)))...)
        """
        self.foldr1 = GlobalVar("foldr1")
        a = TypeVar("a")
        f = Var("f", FuncType([a, a], a))
        av = Var("av", self.l(a))
        x = Var("x")
        y = Var("y")
        z = Var("z")
        one_case = Clause(PatternConstructor(self.cons,
                                             [PatternVar(x), PatternConstructor(self.nil)]), x)
        cons_case = Clause(PatternConstructor(self.cons, [PatternVar(y), PatternVar(z)]),
                           f(y, self.foldr1(f, z)))
        self.mod[self.foldr1] = Function([f, av],
                                         Match(av, [one_case, cons_case], False), a, [a])


    def define_list_concat(self):
        """Defines a function that concatenates two lists.

        concat(l1, l2) : fn<a>(list[a], list[a]) -> list[a]"""
        self.concat = GlobalVar("concat")
        a = TypeVar("a")
        l1 = Var("l1", self.l(a))
        l2 = Var("l2", self.l(a))
        h = Var("h")
        t = Var("t")
        updater = Function([h, t], self.cons(h, t))
        self.mod[self.concat] = Function([l1, l2],
                                         self.foldr(updater, l2, l1),
                                         self.l(a), [a])


    def define_list_filter(self):
        """Defines a function that filters a list.

        filter(f, l) : fn<a>(fn(a) -> Tensor[(), bool], list[a]) -> list[a]

        It returns the sublist of l consisting of the elements for which f returns true.
        """
        self.filter = GlobalVar("filter")
        a = TypeVar("a")
        f = Var("f", FuncType([a], scalar_type("bool")))
        l = Var("l", self.l(a))
        h = Var("h")
        t = Var("t")
        nil_case = Clause(PatternConstructor(self.nil), self.nil())
        cons_case = Clause(PatternConstructor(self.cons, [PatternVar(h), PatternVar(t)]),
                           If(f(h), self.cons(h, self.filter(f, t)), self.filter(f, t)))
        self.mod[self.filter] = Function([f, l], Match(l, [nil_case, cons_case]), self.l(a), [a])


    def define_list_zip(self):
        """Defines a function that combines two lists into a list of tuples of their elements.

        zip(l, m) : fn<a, b>(list[a], list[b]) -> list[(a, b)]

        The zipped list will be the length of the shorter list.
        """
        self.zip = GlobalVar("zip")
        a = TypeVar("a")
        b = TypeVar("b")
        nil_case = Clause(PatternConstructor(self.nil), self.nil())
        l1 = Var("l1")
        l2 = Var("l2")
        h1 = Var("h1")
        h2 = Var("h2")
        t1 = Var("t1")
        t2 = Var("t2")
        inner_cons_case = Clause(PatternConstructor(self.cons, [PatternVar(h2), PatternVar(t2)]),
                                 self.cons(Tuple([h1, h2]), self.zip(t1, t2)))
        outer_cons_case = Clause(PatternConstructor(self.cons, [PatternVar(h1), PatternVar(t1)]),
                                 Match(l2, [nil_case, inner_cons_case]))
        self.mod[self.zip] = Function([l1, l2], Match(l1, [nil_case, outer_cons_case]),
                                      self.l(TupleType([a, b])), [a, b])


    def define_list_rev(self):
        """Defines a function that reverses a list.

        rev(l) : fn<a>(list[a]) -> list[a]
        """
        self.rev = GlobalVar("rev")
        a = TypeVar("a")
        l = Var("l", self.l(a))
        x = Var("x")
        y = Var("y")
        updater = Function([y, x], self.cons(x, y))
        self.mod[self.rev] = Function([l],
                                      self.foldl(updater, self.nil(), l),
                                      self.l(a), [a])


    def define_list_map_accumr(self):
        """Defines an accumulative map, which is a fold that simulataneously updates
        an accumulator value and a list of results.

        map_accumr(f, s, l) : fn<a, b, c>(fn(a, b) -> (a, c), a, list[b]) -> (a, list[c])

        This map proceeds through l from right to left.
        """
        self.map_accumr = GlobalVar("map_accumr")
        a = TypeVar("a")
        b = TypeVar("b")
        c = TypeVar("c")
        f = Var("f", FuncType([a, b], TupleType([a, c])))
        acc = Var("acc", a)
        l = Var("l", self.l(b))
        v = Var("v", b)
        p = Var("p", TupleType([a, self.l(c)]))
        f_out = Var("f_out", TupleType([a, c]))
        updater = Function([v, p],
                           Let(f_out, f(TupleGetItem(p, 0), v),
                               Tuple([TupleGetItem(f_out, 0),
                                      self.cons(TupleGetItem(f_out, 1),
                                                TupleGetItem(p, 1))])),
                           TupleType([a, self.l(c)]))
        self.mod[self.map_accumr] = Function([f, acc, l],
                                             self.foldr(updater, Tuple([acc, self.nil()]), l),
                                             TupleType([a, self.l(c)]),
                                             [a, b, c])


    def define_list_map_accuml(self):
        """Defines an accumulative map, which is a fold that simulataneously updates
        an accumulator value and a list of results.

        map_accuml(f, s, l) : fn<a, b, c>(fn(a, b) -> (a, c), a, list[b]) -> (a, list[c])

        This map proceeds through l from left to right.
        """
        self.map_accuml = GlobalVar("map_accuml")
        a = TypeVar("a")
        b = TypeVar("b")
        c = TypeVar("c")
        f = Var("f", FuncType([a, b], TupleType([a, c])))
        acc = Var("acc", a)
        l = Var("l", self.l(b))
        v = Var("v", b)
        p = Var("p", TupleType([a, self.l(c)]))
        f_out = Var("f_out", TupleType([a, c]))
        updater = Function([p, v],
                           Let(f_out, f(TupleGetItem(p, 0), v),
                               Tuple([TupleGetItem(f_out, 0),
                                      self.cons(TupleGetItem(f_out, 1),
                                                TupleGetItem(p, 1))])),
                           TupleType([a, self.l(c)]))
        self.mod[self.map_accuml] = Function([f, acc, l],
                                             self.foldl(updater, Tuple([acc, self.nil()]), l),
                                             TupleType([a, self.l(c)]),
                                             [a, b, c])


    def define_optional_adt(self):
        """Defines an optional ADT, which can either contain some other
        type or nothing at all."""
        self.optional = GlobalTypeVar("optional")
        a = TypeVar("a")
        self.some = Constructor("some", [a], self.optional)
        self.none = Constructor("none", [], self.optional)
        self.mod[self.optional] = TypeData(self.optional, [a], [self.some, self.none])


    def define_list_unfoldr(self):
        """Defines a function that builds up a list starting from a seed value.

        unfoldr(f, s) : fn<a, b>(fn(a) -> Optional[(a, b)], a) -> list[b]

        f returns an option containing a new seed and an output value. f will
        continue to be called on the new seeds until it returns None. All the
        output values will be combined into a list, right to left.
        """
        self.unfoldr = GlobalVar("unfoldr")
        a = TypeVar("a")
        b = TypeVar("b")
        f = Var("f", FuncType([a], self.optional(TupleType([a, b]))))
        s = Var("s", a)
        p = Var("p", TupleType([a, b]))
        none_case = Clause(PatternConstructor(self.none), self.nil())
        some_case = Clause(PatternConstructor(self.some, [PatternVar(p)]),
                           self.cons(TupleGetItem(p, 1),
                                     self.unfoldr(f, TupleGetItem(p, 0))))
        self.mod[self.unfoldr] = Function([f, s], Match(f(s), [none_case, some_case]),
                                          self.l(b), [a, b])


    def define_list_unfoldl(self):
        """Defines a function that builds up a list starting from a seed value.

        unfoldl(f, s) : fn<a, b>(fn(a) -> Optional[(a, b)], a) -> list[b]

        f returns an option containing a new seed and an output value. f will
        continue to be called on the new seeds until it returns None. All the
        output values will be combined into a list, left to right.
        """
        self.unfoldl = GlobalVar("unfoldl")
        a = TypeVar("a")
        b = TypeVar("b")
        f = Var("f", FuncType([a], self.optional(TupleType([a, b]))))
        s = Var("s", a)
        # easiest way to implement is to do a right unfold and reverse
        self.mod[self.unfoldl] = Function([f, s],
                                          self.rev(self.unfoldr(f, s)),
                                          self.l(b), [a, b])


    def define_list_sum(self):
        """Defines a function that computes the sum of a list of integer scalars."""
        self.sum = GlobalVar("sum")
        a = Var("a", self.l(scalar_type('int32')))
        x = Var('x')
        y = Var('y')
        addf = Function([x, y], add(x, y))
        self.mod[self.sum] = Function([a], self.foldl(addf, const(0), a))


    def define_list_length(self):
        """Defines a function that returns the length of a list"""
        self.length = GlobalVar("length")
        a = TypeVar("a")
        x = Var("x", self.l(a))
        y = Var("y")
        nil_case = Clause(PatternConstructor(self.nil), const(0))
        cons_case = Clause(PatternConstructor(self.cons, [PatternWildcard(), PatternVar(y)]),
                           add(const(1), self.length(y)))
        self.mod[self.length] = Function([x],
                                         Match(x, [nil_case, cons_case]), scalar_type('int32'), [a])


    def define_tree_adt(self):
        """Defines a tree ADT. A tree can contain any type.
        It has only one constructor, rose(x, l), where x is the content
        of that point of the tree and l is a list of more trees of the
        same type. A leaf is thus rose(x, nil()).
        """
        self.tree = GlobalTypeVar("tree")
        a = TypeVar("a")
        self.rose = Constructor("rose", [a, self.l(self.tree(a))], self.tree)
        self.mod[self.tree] = TypeData(self.tree, [a], [self.rose])


    def define_tree_map(self):
        """Defines a function that maps over a tree. The function
        is applied to each subtree's contents.

        Signature: fn<a, b>(f : fn(a) -> b, t : tree[a]) -> tree[b]
        """
        self.tmap = GlobalVar("tmap")
        a = TypeVar("a")
        b = TypeVar("b")
        t = Var("t", self.tree(a))
        f = Var("f", FuncType([a], b))
        x = Var("x", self.tree(a))
        y = Var("y")
        z = Var("z")
        rose_case = Clause(PatternConstructor(self.rose, [PatternVar(y), PatternVar(z)]),
                           self.rose(f(y), self.map(Function([x], self.tmap(f, x)), z)))
        self.mod[self.tmap] = Function([f, t],
                                       Match(t, [rose_case]), self.tree(b), [a, b])


    def define_tree_size(self):
        """Defines a function that computes the size of a tree.

        Signature: fn<a>(t : tree[a]) -> Tensor[(), int32]
        """
        self.size = GlobalVar("size")
        a = TypeVar("a")
        t = Var("t", self.tree(a))
        z = Var("z")
        rose_case = Clause(PatternConstructor(self.rose, [PatternWildcard(), PatternVar(z)]),
                           add(const(1), self.sum(self.map(self.size, z))))
        self.mod[self.size] = Function([t],
                                       Match(t, [rose_case]), scalar_type('int32'), [a])


    def define_iterate(self):
        """Defines a function that take a number n and a function f;
        returns a closure that takes an argument and applies f
        n times to its argument.

        Signature: fn<a>(f : fn(a) -> a, n : Tensor[(), int32]) -> fn(a) -> a
        """
        self.iterate = GlobalVar("iterate")
        a = TypeVar("a")
        f = Var("f", FuncType([a], a))
        x = Var("x", scalar_type('int32'))
        body = If(equal(x, const(0)),
                  self.id,
                  self.compose(f,
                               self.iterate(f, subtract(x, const(1)))))
        self.mod[self.iterate] = Function([f, x],
                                          body,
                                          FuncType([a], a),
                                          [a])

    def load_prelude(self):
        """
        Parses the portions of the Prelude written in Relay's text format and adds
        them to the module.
        """
        prelude_file = os.path.join(__PRELUDE_PATH__, "prelude.rly")
        with open(prelude_file) as prelude:
            prelude = fromtext(prelude.read())
            self.mod.update(prelude)
            self.id = self.mod.get_global_var("id")
            self.compose = self.mod.get_global_var("compose")


    def __init__(self, mod=None):
        if mod is None:
            mod = Module()
        self.mod = mod
        self.load_prelude()
        self.define_list_adt()
        self.define_list_hd()
        self.define_list_tl()
        self.define_list_map()
        self.define_list_foldl()
        self.define_list_foldr()
        self.define_list_foldr1()
        self.define_list_concat()
        self.define_list_filter()
        self.define_list_zip()
        self.define_list_rev()
        self.define_list_map_accumr()
        self.define_list_map_accuml()

        self.define_optional_adt()
        self.define_list_unfoldr()
        self.define_list_unfoldl()

        self.define_list_length()
        self.define_list_nth()
        self.define_list_update()
        self.define_list_sum()

        self.define_tree_adt()
        self.define_tree_map()
        self.define_tree_size()

        self.define_iterate()

        self.define_tensor_adt()
        self.define_tensor_take()
        self.define_tensor_add_one()
        self.define_tensor_concat()
        self.define_tensor_array()
        self.define_tensor_array_read()
        self.define_tensor_array_size()
        self.define_tensor_array_write()        
        self.define_tensor_array_stack()
        self.define_tensor_array_unstack_tensor1()
        self.define_tensor_array_unstack_tensor2()
        self.define_tensor_array_scatter()
        self.define_tensor_array_gather()
        self.define_tensor_array_split()
        self.define_tensor_array_concat()
