import os
import mxnet as mx
from mxnet import gluon

import tvm
import numpy as np
from tvm import relay
from tvm.relay.vm import eval_vm, eta_expand
from tvm.relay.scope_builder import ScopeBuilder

def test_id():
    x = relay.var('x', shape=(10, 10))
    f = relay.Function([x], x)
    x_data = np.random.rand(10, 10).astype('float64')
    res = eval_vm(f, tvm.cpu(), x_data)
    tvm.testing.assert_allclose(res.asnumpy(), x_data)

def test_op():
    x = relay.var('x', shape=(10, 10))
    f = relay.Function([x], x + x)
    x_data = np.random.rand(10, 10).astype('float32')
    res = eval_vm(f, tvm.cpu(), x_data)
    tvm.testing.assert_allclose(res.asnumpy(), x_data + x_data)

def any(x):
    x = relay.op.nn.batch_flatten(x)
    return relay.op.min(x, axis=[0, 1])

def test_cond():
    x = relay.var('x', shape=(10, 10))
    y = relay.var('x', shape=(10, 10))
    # f = relay.Function([x, y], relay.op.equal(x, y))
    f = relay.Function([x, y], any(relay.op.equal(x, y)))
    x_data = np.random.rand(10, 10).astype('float32')
    y_data = np.random.rand(10, 10).astype('float32')

    # same
    res = eval_vm(f, tvm.cpu(), x_data, x_data)
    tvm.testing.assert_allclose(res.asnumpy(), True)

    # diff
    res = eval_vm(f, tvm.cpu(), x_data, y_data)
    tvm.testing.assert_allclose(res.asnumpy(), False)


def test_simple_if():
    x = relay.var('x', shape=(10, 10))
    y = relay.var('x', shape=(10, 10))
    f = relay.Function([x, y],
        relay.If(any(relay.op.equal(x, y)), x, y))
    x_data = np.random.rand(10, 10).astype('float32')
    y_data = np.random.rand(10, 10).astype('float32')

    # same
    res = eval_vm(f, tvm.cpu(), x_data, x_data)
    tvm.testing.assert_allclose(res.asnumpy(), x_data)

    # diff
    res = eval_vm(f, tvm.cpu(), x_data, y_data)
    tvm.testing.assert_allclose(res.asnumpy(), y_data)

def test_simple_call():
    mod = relay.module.Module({})
    sum_up = relay.GlobalVar('sum_up')
    i = relay.var('i', shape=[], dtype='int32')
    sb = ScopeBuilder()
    sb.ret(i)
    func = relay.Function([i], sb.get(), ret_type=relay.TensorType([], 'int32'))
    mod[sum_up] = func
    i_data = np.array(0, dtype='int32')
    # Refactor this bit
    mod[mod.entry_func] = relay.Function([], sum_up)
    result = eval_vm(mod, tvm.cpu(), i_data)
    tvm.testing.assert_allclose(result.asnumpy(), i_data)

def test_count_loop():
    mod = relay.module.Module({})
    sum_up = relay.GlobalVar('sum_up')
    i = relay.var('i', shape=[], dtype='int32')
    sb = ScopeBuilder()
    with sb.if_scope(relay.equal(i, relay.const(0, dtype='int32'))):
        sb.ret(i)
    with sb.else_scope():
        one_less = relay.subtract(i, relay.const(1, dtype='int32'))
        rec_call = relay.Call(sum_up, [one_less])
        sb.ret(relay.add(rec_call, i))
    func = relay.Function([i], sb.get(), ret_type=relay.TensorType([], 'int32'))
    mod[sum_up] = func
    i_data = np.array(0, dtype='int32')
    mod[mod.entry_func] = relay.Function([], sum_up)
    result = eval_vm(mod, tvm.cpu(), i_data)
    tvm.testing.assert_allclose(result.asnumpy(), i_data)

def test_sum_loop():
    mod = relay.module.Module({})
    sum_up = relay.GlobalVar('sum_up')
    i = relay.var('i', shape=[], dtype='int32')
    accum = relay.var('accum', shape=[], dtype='int32')
    sb = ScopeBuilder()
    with sb.if_scope(relay.equal(i, relay.const(0, 'int32'))):
        sb.ret(accum)
    with sb.else_scope():
        one_less = relay.subtract(i, relay.const(1, 'int32'))
        new_accum = relay.add(accum, i)
        sb.ret(relay.Call(sum_up, [one_less, new_accum]))
    func = relay.Function([i, accum], sb.get())
    mod[sum_up] = func
    i_data = np.array(10, dtype='int32')
    accum_data = np.array(0, dtype='int32')
    mod[mod.entry_func] = relay.Function([], sum_up)
    result = eval_vm(mod, tvm.cpu(), i_data, accum_data)
    tvm.testing.assert_allclose(result.asnumpy(), sum(range(1, 11)))

def test_tuple_fst():
    ttype = relay.TupleType([relay.TensorType((1,)), relay.TensorType((10,))])
    tup = relay.var('tup', type_annotation=ttype)
    f = relay.Function([tup], relay.TupleGetItem(tup, 0))
    i_data = np.random.rand(41).astype('float32')
    j_data = np.random.rand(10).astype('float32')
    result = eval_vm(f, tvm.cpu(), (i_data, j_data))
    tvm.testing.assert_allclose(result.asnumpy(), i_data)

def test_tuple_second():
    ttype = relay.TupleType([relay.TensorType((1,)), relay.TensorType((10,))])
    tup = relay.var('tup', type_annotation=ttype)
    f = relay.Function([tup], relay.TupleGetItem(tup, 1))
    i_data = np.random.rand(41).astype('float32')
    j_data = np.random.rand(10).astype('float32')
    result = eval_vm(f, tvm.cpu(), (i_data, j_data))
    tvm.testing.assert_allclose(result.asnumpy(), j_data)

def test_let_tensor():
    sb = relay.ScopeBuilder()
    shape = (1,)
    x = relay.var('x', shape=shape, dtype='float32')
    x1 = relay.var('x1', shape=shape, dtype='float32')

    x1 = sb.let(x1, x)
    xplusone = x1 + relay.const(42.0, 'float32')
    sb.ret(xplusone)
    body = sb.get()

    f = relay.Function([x], body)

    x_data = np.random.rand(*shape).astype('float32')
    result = eval_vm(f, tvm.cpu(), x_data)
    tvm.testing.assert_allclose(result.asnumpy(), x_data + 42.0)

def test_let_scalar():
    sb = relay.ScopeBuilder()

    x = relay.var('x', 'float32')
    x1 = relay.var('x1', 'float32')

    x1 = sb.let(x1, x)
    xplusone = x1 + relay.const(42.0, 'float32')
    sb.ret(xplusone)
    body = sb.get()

    f = relay.Function([x], body)

    x_data = np.array(np.random.rand()).astype('float32')
    result = eval_vm(f, tvm.cpu(), x_data)
    tvm.testing.assert_allclose(result.asnumpy(), x_data + 42.0)

def import_mxnet_model(cell_type, input_size, hidden_size, fname, batch=1, seq_len=100):
    ctx = mx.context.cpu()
    dtype = 'float32'
    if cell_type == 'gru' or cell_type == 'rnn':
        num_states = 1
    elif cell_type == 'lstm':
        num_states = 2
    else:
        raise RuntimeError("Unsupported RNN cell type: %s" % cell_type)
    data_names = ['data0']
    inputs = [mx.nd.random.uniform(shape=(seq_len, batch, input_size), dtype=dtype, ctx=ctx)]
    for i in range(num_states):
        data_names.append('data%s' % (i+1))
        inputs.append(mx.nd.zeros((batch, hidden_size), dtype=dtype, ctx=ctx))

    model_data_dir = os.path.dirname(os.path.realpath(__file__))
    net = gluon.nn.SymbolBlock.imports("%s/model_zoo_data/%s-symbol.json.data" % (model_data_dir, fname), data_names,
                                       "%s/model_zoo_data/%s-0001.params.data" % (model_data_dir, fname), ctx=ctx)
    net.hybridize()
    inputs = []
    inputs.append(mx.sym.Variable("data"))
    inputs.append(mx.sym.Variable("state"))
    return relay.frontend.from_mxnet(net, {}, input_symbols=inputs)

def test_rnn():
    net = import_mxnet_model('rnn', 128, 128, "rnn_i128_h128")
#    execute_mxnet_model('gru', 128, 128, "gru_i128_h128")

if __name__ == "__main__":
    test_id()
    # test_op()
    # test_cond()
    # test_simple_if()
    # test_simple_call()
    # test_count_loop()
    # test_sum_loop()
    # test_tuple_fst()
    # test_tuple_second()
    # test_let_scalar()
    # test_let_tensor()
    # test_rnn()
