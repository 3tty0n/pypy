import py
from rpython.rtyper.lltypesystem import rffi, lltype
from pypy.module.cpyext.test.test_api import BaseApiTest, raises_w
from pypy.module.cpyext.api import Py_ssize_tP, PyObjectP, PyTypeObjectPtr
from pypy.module.cpyext.pyobject import make_ref, from_ref
from pypy.interpreter.error import OperationError
from pypy.module.cpyext.test.test_cpyext import AppTestCpythonExtensionBase
from pypy.module.cpyext.dictproxyobject import *
from pypy.module.cpyext.dictobject import *
from pypy.module.cpyext.pyobject import decref

class TestDictObject(BaseApiTest):
    def test_dict(self, space):
        d = PyDict_New(space)
        assert space.eq_w(d, space.newdict())

        assert space.eq_w(PyDict_GetItem(space, space.wrap({"a": 72}),
                                             space.wrap("a")),
                          space.wrap(72))

        PyDict_SetItem(space, d, space.wrap("c"), space.wrap(42))
        assert space.eq_w(space.getitem(d, space.wrap("c")),
                          space.wrap(42))

        space.setitem(d, space.wrap("name"), space.wrap(3))
        assert space.eq_w(PyDict_GetItem(space, d, space.wrap("name")),
                          space.wrap(3))

        space.delitem(d, space.wrap("name"))
        assert not PyDict_GetItem(space, d, space.wrap("name"))

        buf = rffi.str2charp("name")
        assert not PyDict_GetItemString(space, d, buf)
        rffi.free_charp(buf)

        assert PyDict_Contains(space, d, space.wrap("c"))
        assert not PyDict_Contains(space, d, space.wrap("z"))

        PyDict_DelItem(space, d, space.wrap("c"))
        with raises_w(space, KeyError):
            PyDict_DelItem(space, d, space.wrap("name"))
        assert PyDict_Size(space, d) == 0

        space.setitem(d, space.wrap("some_key"), space.wrap(3))
        buf = rffi.str2charp("some_key")
        PyDict_DelItemString(space, d, buf)
        assert PyDict_Size(space, d) == 0
        with raises_w(space, KeyError):
            PyDict_DelItemString(space, d, buf)
        rffi.free_charp(buf)

        d = space.wrap({'a': 'b'})
        PyDict_Clear(space, d)
        assert PyDict_Size(space, d) == 0

    def test_check(self, space):
        d = PyDict_New(space, )
        assert PyDict_Check(space, d)
        assert PyDict_CheckExact(space, d)
        sub = space.appexec([], """():
            class D(dict):
                pass
            return D""")
        d = space.call_function(sub)
        assert PyDict_Check(space, d)
        assert not PyDict_CheckExact(space, d)
        i = space.wrap(2)
        assert not PyDict_Check(space, i)
        assert not PyDict_CheckExact(space, i)

    def test_keys(self, space):
        w_d = space.newdict()
        space.setitem(w_d, space.wrap("a"), space.wrap("b"))

        assert space.eq_w(PyDict_Keys(space, w_d), space.wrap(["a"]))
        assert space.eq_w(PyDict_Values(space, w_d), space.wrap(["b"]))
        assert space.eq_w(PyDict_Items(space, w_d), space.wrap([("a", "b")]))

    def test_merge(self, space):
        w_d = space.newdict()
        space.setitem(w_d, space.wrap("a"), space.wrap("b"))

        w_d2 = space.newdict()
        space.setitem(w_d2, space.wrap("a"), space.wrap("c"))
        space.setitem(w_d2, space.wrap("c"), space.wrap("d"))
        space.setitem(w_d2, space.wrap("e"), space.wrap("f"))

        PyDict_Merge(space, w_d, w_d2, 0)
        assert space.unwrap(w_d) == dict(a='b', c='d', e='f')
        PyDict_Merge(space, w_d, w_d2, 1)
        assert space.unwrap(w_d) == dict(a='c', c='d', e='f')

    def test_update(self, space):
        w_d = space.newdict()
        space.setitem(w_d, space.wrap("a"), space.wrap("b"))

        w_d2 = PyDict_Copy(space, w_d)
        assert not space.is_w(w_d2, w_d)
        space.setitem(w_d, space.wrap("c"), space.wrap("d"))
        space.setitem(w_d2, space.wrap("e"), space.wrap("f"))

        PyDict_Update(space, w_d, w_d2)
        assert space.unwrap(w_d) == dict(a='b', c='d', e='f')

    def test_update_doesnt_accept_list_of_tuples(self, space):
        w_d = space.newdict()
        space.setitem(w_d, space.wrap("a"), space.wrap("b"))

        w_d2 = space.wrap([("c", "d"), ("e", "f")])

        with raises_w(space, AttributeError):
            PyDict_Update(space, w_d, w_d2)
        assert space.unwrap(w_d) == dict(a='b') # unchanged

    def test_iter(self, space):
        w_dict = space.sys.getdict(space)
        py_dict = make_ref(space, w_dict)

        ppos = lltype.malloc(Py_ssize_tP.TO, 1, flavor='raw')
        ppos[0] = 0
        pkey = lltype.malloc(PyObjectP.TO, 1, flavor='raw')
        pvalue = lltype.malloc(PyObjectP.TO, 1, flavor='raw')

        try:
            w_copy = space.newdict()
            while PyDict_Next(space, w_dict, ppos, pkey, pvalue):
                w_key = from_ref(space, pkey[0])
                w_value = from_ref(space, pvalue[0])
                space.setitem(w_copy, w_key, w_value)
        finally:
            lltype.free(ppos, flavor='raw')
            lltype.free(pkey, flavor='raw')
            lltype.free(pvalue, flavor='raw')

        decref(space, py_dict) # release borrowed references

        assert space.eq_w(space.len(w_copy), space.len(w_dict))
        assert space.eq_w(w_copy, w_dict)

    def test_iterkeys(self, space):
        w_dict = space.sys.getdict(space)
        py_dict = make_ref(space, w_dict)

        ppos = lltype.malloc(Py_ssize_tP.TO, 1, flavor='raw')
        pkey = lltype.malloc(PyObjectP.TO, 1, flavor='raw')
        pvalue = lltype.malloc(PyObjectP.TO, 1, flavor='raw')

        keys_w = []
        values_w = []
        try:
            ppos[0] = 0
            while PyDict_Next(space, w_dict, ppos, pkey, None):
                w_key = from_ref(space, pkey[0])
                keys_w.append(w_key)
            ppos[0] = 0
            while PyDict_Next(space, w_dict, ppos, None, pvalue):
                w_value = from_ref(space, pvalue[0])
                values_w.append(w_value)
        finally:
            lltype.free(ppos, flavor='raw')
            lltype.free(pkey, flavor='raw')
            lltype.free(pvalue, flavor='raw')

        decref(space, py_dict) # release borrowed references

        assert space.eq_w(space.newlist(keys_w),
                          space.call_function(
                             space.w_list,
                             space.call_method(w_dict, "keys")))
        assert space.eq_w(space.newlist(values_w),
                          space.call_function(
                             space.w_list,
                             space.call_method(w_dict, "values")))

    def test_dictproxy(self, space):
        w_dict = space.sys.get('modules')
        w_proxy = PyDictProxy_New(space, w_dict)
        assert space.contains_w(w_proxy, space.wrap('sys'))
        raises(OperationError, space.setitem,
               w_proxy, space.wrap('sys'), space.w_None)
        raises(OperationError, space.delitem,
               w_proxy, space.wrap('sys'))
        raises(OperationError, space.call_method, w_proxy, 'clear')
        assert PyDictProxy_Check(space, w_proxy)

    def test_typedict1(self, space):
        py_type = make_ref(space, space.w_int)
        py_dict = rffi.cast(PyTypeObjectPtr, py_type).c_tp_dict
        ppos = lltype.malloc(Py_ssize_tP.TO, 1, flavor='raw')

        ppos[0] = 0
        pkey = lltype.malloc(PyObjectP.TO, 1, flavor='raw')
        pvalue = lltype.malloc(PyObjectP.TO, 1, flavor='raw')
        try:
            w_copy = space.newdict()
            while PyDict_Next(space, py_dict, ppos, pkey, pvalue):
                w_key = from_ref(space, pkey[0])
                w_value = from_ref(space, pvalue[0])
                space.setitem(w_copy, w_key, w_value)
        finally:
            lltype.free(ppos, flavor='raw')
            lltype.free(pkey, flavor='raw')
            lltype.free(pvalue, flavor='raw')
        decref(space, py_type) # release borrowed references
        # do something with w_copy ?

class AppTestDictObject(AppTestCpythonExtensionBase):
    def test_dictproxytype(self):
        module = self.import_extension('foo', [
            ("dict_proxy", "METH_VARARGS",
             """
                 PyObject * dict;
                 PyObject * proxydict;
                 int i;
                 if (!PyArg_ParseTuple(args, "O", &dict))
                     return NULL;
                 proxydict = PyDictProxy_New(dict);
#ifdef PYPY_VERSION  // PyDictProxy_Check[Exact] are PyPy-specific.
                 if (!PyDictProxy_Check(proxydict)) {
                    Py_DECREF(proxydict);
                    PyErr_SetNone(PyExc_ValueError);
                    return NULL;
                 }
                 if (!PyDictProxy_CheckExact(proxydict)) {
                    Py_DECREF(proxydict);
                    PyErr_SetNone(PyExc_ValueError);
                    return NULL;
                 }
#endif  // PYPY_VERSION
                 i = PyObject_Size(proxydict);
                 Py_DECREF(proxydict);
                 return PyLong_FromLong(i);
             """),
            ])
        assert module.dict_proxy({'a': 1, 'b': 2}) == 2

    def test_update(self):
        module = self.import_extension('foo', [
            ("update", "METH_VARARGS",
             '''
             if (PyDict_Update(PyTuple_GetItem(args, 0), PyTuple_GetItem(args, 1)))
                return NULL;
             Py_RETURN_NONE;
             ''')])
        d = {"a": 1}
        module.update(d, {"c": 2})
        assert d == dict(a=1, c=2)
        d = {"a": 1}
        raises(AttributeError, module.update, d, [("c", 2)])

    def test_typedict2(self):
        module = self.import_extension('foo', [
            ("get_type_dict", "METH_O",
             '''
                PyObject* value = args->ob_type->tp_dict;
                if (value == NULL) value = Py_None;
                Py_INCREF(value);
                return value;
             '''),
            ])
        d = module.get_type_dict(1)
        assert d['real'].__get__(1, 1) == 1
    def test_advanced(self):
        module = self.import_extension('foo', [
            ("dict_len", "METH_O",
            '''
                int ret = args->ob_type->tp_as_mapping->mp_length(args);
                return PyLong_FromLong(ret);
            '''),
            ("dict_setitem", "METH_VARARGS",
            '''
                int ret;
                PyObject * dict = PyTuple_GetItem(args, 0);
                if (PyTuple_Size(args) < 3 || !dict || 
                        !dict->ob_type->tp_as_mapping ||
                        !dict->ob_type->tp_as_mapping->mp_ass_subscript)
                    return PyLong_FromLong(-1);
                ret = dict->ob_type->tp_as_mapping->mp_ass_subscript(
                        dict, PyTuple_GetItem(args, 1),
                        PyTuple_GetItem(args, 2));
                return PyLong_FromLong(ret);
            '''),
            ("dict_delitem", "METH_VARARGS",
            '''
                int ret;
                PyObject * dict = PyTuple_GetItem(args, 0);
                if (PyTuple_Size(args) < 2 || !dict || 
                        !dict->ob_type->tp_as_mapping ||
                        !dict->ob_type->tp_as_mapping->mp_ass_subscript)
                    return PyLong_FromLong(-1);
                ret = dict->ob_type->tp_as_mapping->mp_ass_subscript(
                        dict, PyTuple_GetItem(args, 1), NULL);
                return PyLong_FromLong(ret);
            '''),
            ("dict_next", "METH_VARARGS",
            '''
                PyObject *key, *value;
                PyObject *arg = NULL;
                Py_ssize_t pos = 0;
                int ret = 0;
                if ((PyArg_ParseTuple(args, "|O", &arg))) {
                    if (arg && PyDict_Check(arg)) {
                        while (PyDict_Next(arg, &pos, &key, &value))
                            ret ++;
                        /* test no crash if pos is not reset to 0*/
                        while (PyDict_Next(arg, &pos, &key, &value))
                            ret ++;
                    }
                }
                return PyLong_FromLong(ret);
            '''),
            ])
        d = {'a': 1, 'b':2}
        assert module.dict_len(d) == 2
        assert module.dict_setitem(d, 'a', 'c') == 0
        assert d['a'] == 'c'
        assert module.dict_delitem(d, 'a') == 0
        r = module.dict_next({'a': 1, 'b': 2})
        assert r == 2
