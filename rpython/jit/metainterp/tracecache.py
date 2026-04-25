"""
Persistent trace cache (Idea 3): reuse optimized traces across PyPy processes.

Key   = sha256(greenkey_fingerprint || type_profile_snapshot || JIT_VERSION)
Value = binary-encoded ResOperation stream + descr-table + assumptions.

The backend code (assembly) is *not* cached -- we redo that every run to
absorb ISA and address-space differences. What we cache is the result of the
costly trace-recording + optimization passes: the final pre-backend IR.

Activation:
  PYPY_TRACE_CACHE_DIR=path   root directory. Missing dir or empty var
                               disables the cache.
  PYPY_TRACE_CACHE_VERBOSE=1   log hits/misses.

Design notes:
  * Wire format is a hand-rolled tagged binary with a hand-rolled binary
    header (no JSON, no pickle). Designed to be translation-friendly: only
    bytes, struct.pack/unpack, and plain ints/floats.
  * Descr objects can't be serialized directly (they wrap GC/C-level state).
    Instead:
      - descrs that live in `metainterp_sd.opcode_descrs` are stored by
        their index in that list. They roundtrip cleanly across runs.
      - any other descr (TargetToken, JitCellToken, FailDescr, ...) is
        flagged `dynamic`; an entry that references any dynamic descr
        gets `has_dynamic_descrs=1` in meta, and the replay helper
        refuses to hand the entry to the backend. This is the conservative
        way to keep a prototype correct without a full descr registry.
  * record_known_result / quasi_immutable assumptions are stored as a
    packed list of (kind_tag, payload_string) pairs. Re-validation on
    load is done via a caller-supplied hook.
"""

import errno
import hashlib
import os
import struct
import sys

CACHE_DIR = os.environ.get('PYPY_TRACE_CACHE_DIR', '').strip()
VERBOSE = os.environ.get('PYPY_TRACE_CACHE_VERBOSE', '') == '1'

MAGIC = b'PYPYTC03'
_JIT_VERSION_BYTES = 16


def _compute_jit_version():
    """Hash a handful of JIT core files so edits to them invalidate the
    cache. Cheap, catches the common "I changed the optimizer" footgun.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    files = [
        os.path.join(here, 'pyjitpl.py'),
        os.path.join(here, 'compile.py'),
        os.path.join(here, 'resoperation.py'),
        os.path.join(here, 'heapcache.py'),
        os.path.join(here, '..', 'codewriter', 'jitcode.py'),
    ]
    h = hashlib.sha256()
    h.update(sys.version.encode('utf-8', 'replace'))
    for p in files:
        try:
            with open(p, 'rb') as f:
                h.update(f.read())
        except (OSError, IOError):
            h.update(('<missing:' + p + '>').encode('utf-8', 'replace'))
    return h.hexdigest()[:_JIT_VERSION_BYTES]


JIT_VERSION = _compute_jit_version()


def is_enabled():
    return bool(CACHE_DIR) and os.path.isdir(CACHE_DIR)


# --- fingerprinting --------------------------------------------------------

def fingerprint_greenkey(greenkey):
    h = hashlib.sha256()
    if greenkey is None:
        h.update(b'<none>')
        return h.hexdigest()
    for g in greenkey:
        tag = getattr(g, 'type', '?')
        h.update(('%s:' % tag).encode('ascii', 'replace'))
        val = None
        for attr in ('value', '_value', 'getint', 'getref_base'):
            a = getattr(g, attr, None)
            if callable(a):
                try:
                    val = a()
                    break
                except Exception:
                    continue
            elif a is not None:
                val = a
                break
        if val is None:
            val = repr(g)
        h.update(repr(val).encode('utf-8', 'replace'))
        h.update(b'|')
    return h.hexdigest()


def fingerprint_type_profile(type_tokens):
    if type_tokens is None:
        return hashlib.sha256(b'<none>').hexdigest()
    if isinstance(type_tokens, dict):
        items = sorted(type_tokens.items())
    else:
        items = sorted(list(type_tokens))
    h = hashlib.sha256()
    for k, v in items:
        h.update(repr(k).encode('utf-8', 'replace'))
        h.update(b'=')
        h.update(repr(v).encode('utf-8', 'replace'))
        h.update(b'|')
    return h.hexdigest()


def make_key(greenkey, type_profile, extra=b''):
    h = hashlib.sha256()
    h.update(JIT_VERSION.encode('ascii'))
    h.update(b'|')
    h.update(fingerprint_greenkey(greenkey).encode('ascii'))
    h.update(b'|')
    h.update(fingerprint_type_profile(type_profile).encode('ascii'))
    if extra:
        if isinstance(extra, bytes):
            h.update(b'|'); h.update(extra)
        else:
            h.update(b'|'); h.update(extra.encode('utf-8', 'replace'))
    return h.hexdigest()


# --- wire format -----------------------------------------------------------

# Arg tags. Compact and ASCII-visible for dump readability.
TAG_BOX_REF    = 0x01  # back-reference to a previously-declared box by id
TAG_CONST_INT  = 0x02  # s64 integer literal
TAG_CONST_FLT  = 0x03  # f64 float literal
TAG_CONST_PTR  = 0x04  # u32 opcode_descrs index (0 = null)
TAG_CONST_REPR = 0x05  # fallback: length-prefixed repr() string

# Input arg kind tags (distinct from arg tags so we can encode the type).
IN_KIND_INT   = 0x01
IN_KIND_REF   = 0x02
IN_KIND_FLOAT = 0x03

# Result type tags (kept numeric so decoders can tell void from i/r/f cheaply).
RES_VOID  = ord('v')
RES_INT   = ord('i')
RES_REF   = ord('r')
RES_FLOAT = ord('f')

# Descr kind tags: how a descr was registered at store time.
DESCR_KIND_STATIC  = 0x01  # `metainterp_sd.opcode_descrs[index]` lookup
DESCR_KIND_DYNAMIC = 0x02  # dynamic descr; store repr only (no backend replay)

# Assumption kind tags.
ASSUMP_QUASI_IMMUT    = 0x01
ASSUMP_KNOWN_RESULT   = 0x02
ASSUMP_OTHER          = 0x7F

# Meta value tags.
META_U64 = 0x01
META_F64 = 0x02
META_STR = 0x03


def _pack_u8(v):   return struct.pack('>B', v & 0xFF)
def _pack_u16(v):  return struct.pack('>H', v & 0xFFFF)
def _pack_u32(v):  return struct.pack('>I', v & 0xFFFFFFFF)
def _pack_s64(v):  return struct.pack('>q', v)
def _pack_u64(v):  return struct.pack('>Q', v & 0xFFFFFFFFFFFFFFFF)
def _pack_f64(v):  return struct.pack('>d', v)


def _pack_lenstr(s):
    if isinstance(s, bytes):
        b = s
    else:
        b = s.encode('utf-8', 'replace')
    if len(b) > 0xFFFF:
        b = b[:0xFFFF]
    return _pack_u16(len(b)) + b


class _Reader(object):
    def __init__(self, data):
        self.data = data
        self.pos = 0

    def _take(self, n):
        if self.pos + n > len(self.data):
            raise ValueError('truncated cache entry')
        v = self.data[self.pos:self.pos + n]
        self.pos += n
        return v

    def u8(self):  return struct.unpack('>B', self._take(1))[0]
    def u16(self): return struct.unpack('>H', self._take(2))[0]
    def u32(self): return struct.unpack('>I', self._take(4))[0]
    def s64(self): return struct.unpack('>q', self._take(8))[0]
    def u64(self): return struct.unpack('>Q', self._take(8))[0]
    def f64(self): return struct.unpack('>d', self._take(8))[0]

    def lenstr(self):
        n = self.u16()
        return self._take(n)


# --- entry model ----------------------------------------------------------

class SerializedArg(object):
    """Single op argument. The tag dictates how `value` is interpreted."""
    __slots__ = ('tag', 'value')

    def __init__(self, tag, value):
        self.tag = tag
        self.value = value

    def __repr__(self):
        return '<Arg tag=%d value=%r>' % (self.tag, self.value)


class SerializedInput(object):
    """Input arg record: (kind_tag, box_id)."""
    __slots__ = ('kind', 'box_id')

    def __init__(self, kind, box_id):
        self.kind = kind
        self.box_id = box_id

    def __repr__(self):
        return '<Input kind=%d box=%d>' % (self.kind, self.box_id)


class SerializedOp(object):
    __slots__ = ('opnum', 'args', 'descr_kind', 'descr_ref',
                 'result_type', 'result_box_id')

    def __init__(self, opnum, args, descr_kind, descr_ref,
                 result_type, result_box_id):
        self.opnum = opnum
        self.args = args
        self.descr_kind = descr_kind
        self.descr_ref = descr_ref  # int (opcode_descrs index) for STATIC,
                                     # string repr for DYNAMIC, 0/None for missing
        self.result_type = result_type
        self.result_box_id = result_box_id

    def __repr__(self):
        return ('<Op num=%d args=%d descr=%d/%r res=%r/%d>' %
                (self.opnum, len(self.args), self.descr_kind,
                 self.descr_ref, chr(self.result_type), self.result_box_id))


class Assumption(object):
    __slots__ = ('kind', 'payload')

    def __init__(self, kind, payload):
        self.kind = kind
        self.payload = payload

    def __repr__(self):
        return '<Assump k=%d p=%r>' % (self.kind, self.payload)


class TraceCacheEntry(object):
    __slots__ = ('inputargs', 'operations', 'assumptions', 'meta',
                 'has_dynamic_descrs')

    def __init__(self, inputargs, operations, assumptions=None, meta=None,
                 has_dynamic_descrs=False):
        self.inputargs = inputargs
        self.operations = operations
        self.assumptions = assumptions or []
        self.meta = meta or {}
        self.has_dynamic_descrs = has_dynamic_descrs

    def __repr__(self):
        return ('<TraceCacheEntry ops=%d inputs=%d assumps=%d dyn=%d>' %
                (len(self.operations), len(self.inputargs),
                 len(self.assumptions), int(self.has_dynamic_descrs)))


# --- binary header encode/decode ------------------------------------------

def _encode_assumptions(assumptions):
    out = [_pack_u16(len(assumptions))]
    for a in assumptions:
        if isinstance(a, dict):
            kind = a.get('kind', 'other')
            payload = a.get('id') or a.get('payload') or repr(a)
            if kind == 'quasi_immutable':
                kind_tag = ASSUMP_QUASI_IMMUT
            elif kind == 'known_result':
                kind_tag = ASSUMP_KNOWN_RESULT
            else:
                kind_tag = ASSUMP_OTHER
            out.append(_pack_u8(kind_tag))
            out.append(_pack_lenstr(str(payload)))
        elif isinstance(a, Assumption):
            out.append(_pack_u8(a.kind))
            out.append(_pack_lenstr(a.payload))
        else:
            out.append(_pack_u8(ASSUMP_OTHER))
            out.append(_pack_lenstr(repr(a)))
    return b''.join(out)


def _decode_assumptions(r):
    n = r.u16()
    out = []
    for _ in range(n):
        k = r.u8()
        p = r.lenstr()
        try:
            p = p.decode('utf-8', 'replace')
        except AttributeError:
            pass
        out.append(Assumption(k, p))
    return out


def _encode_meta(meta):
    out = [_pack_u16(len(meta))]
    for k in sorted(meta.keys()):
        v = meta[k]
        out.append(_pack_lenstr(k))
        if isinstance(v, bool):
            out.append(_pack_u8(META_U64))
            out.append(_pack_u64(1 if v else 0))
        elif isinstance(v, int) or (sys.version_info[0] < 3 and isinstance(v, long)):  # noqa
            out.append(_pack_u8(META_U64))
            out.append(_pack_u64(int(v)))
        elif isinstance(v, float):
            out.append(_pack_u8(META_F64))
            out.append(_pack_f64(v))
        else:
            out.append(_pack_u8(META_STR))
            out.append(_pack_lenstr(str(v)))
    return b''.join(out)


def _decode_meta(r):
    n = r.u16()
    meta = {}
    for _ in range(n):
        k = r.lenstr().decode('utf-8', 'replace')
        t = r.u8()
        if t == META_U64:
            meta[k] = r.u64()
        elif t == META_F64:
            meta[k] = r.f64()
        elif t == META_STR:
            meta[k] = r.lenstr().decode('utf-8', 'replace')
        else:
            raise ValueError('unknown meta tag %d' % t)
    return meta


# --- body encode/decode ---------------------------------------------------

def _encode_arg(arg):
    if arg.tag == TAG_BOX_REF:
        return _pack_u8(arg.tag) + _pack_u32(arg.value)
    if arg.tag == TAG_CONST_INT:
        return _pack_u8(arg.tag) + _pack_s64(arg.value)
    if arg.tag == TAG_CONST_FLT:
        return _pack_u8(arg.tag) + _pack_f64(float(arg.value))
    if arg.tag == TAG_CONST_PTR:
        return _pack_u8(arg.tag) + _pack_u32(arg.value)
    if arg.tag == TAG_CONST_REPR:
        return _pack_u8(arg.tag) + _pack_lenstr(arg.value)
    raise ValueError('unknown arg tag %d' % arg.tag)


def _decode_arg(r):
    tag = r.u8()
    if tag == TAG_BOX_REF:   return SerializedArg(tag, r.u32())
    if tag == TAG_CONST_INT: return SerializedArg(tag, r.s64())
    if tag == TAG_CONST_FLT: return SerializedArg(tag, r.f64())
    if tag == TAG_CONST_PTR: return SerializedArg(tag, r.u32())
    if tag == TAG_CONST_REPR:
        s = r.lenstr()
        try:
            s = s.decode('utf-8', 'replace')
        except AttributeError:
            pass
        return SerializedArg(tag, s)
    raise ValueError('unknown arg tag %d' % tag)


def encode_entry(entry):
    parts = []
    parts.append(_pack_u16(len(entry.inputargs)))
    for inp in entry.inputargs:
        parts.append(_pack_u8(inp.kind))
        parts.append(_pack_u32(inp.box_id))
    parts.append(_pack_u32(len(entry.operations)))
    for op in entry.operations:
        parts.append(_pack_u16(op.opnum))
        parts.append(_pack_u8(len(op.args)))
        for a in op.args:
            parts.append(_encode_arg(a))
        parts.append(_pack_u8(op.descr_kind))
        if op.descr_kind == DESCR_KIND_STATIC:
            parts.append(_pack_u32(int(op.descr_ref)))
        elif op.descr_kind == DESCR_KIND_DYNAMIC:
            parts.append(_pack_lenstr(op.descr_ref or ''))
        else:
            # 0 = no descr
            pass
        parts.append(_pack_u8(op.result_type))
        if op.result_type != RES_VOID:
            parts.append(_pack_u32(op.result_box_id))
    return b''.join(parts)


def decode_entry(data):
    r = _Reader(data)
    n_in = r.u16()
    inputargs = []
    for _ in range(n_in):
        kind = r.u8()
        box_id = r.u32()
        inputargs.append(SerializedInput(kind, box_id))
    n_ops = r.u32()
    ops = []
    for _ in range(n_ops):
        opnum = r.u16()
        n_args = r.u8()
        args = [_decode_arg(r) for _ in range(n_args)]
        dk = r.u8()
        if dk == DESCR_KIND_STATIC:
            dref = r.u32()
        elif dk == DESCR_KIND_DYNAMIC:
            s = r.lenstr()
            try:
                s = s.decode('utf-8', 'replace')
            except AttributeError:
                pass
            dref = s
        else:
            dref = 0
        res_type = r.u8()
        if res_type != RES_VOID:
            res_box = r.u32()
        else:
            res_box = 0
        ops.append(SerializedOp(opnum, args, dk, dref, res_type, res_box))
    return TraceCacheEntry(inputargs, ops)


# --- on-disk layout --------------------------------------------------------

def _path_for(key):
    assert CACHE_DIR
    return os.path.join(CACHE_DIR, key + '.tc')


def _ensure_dir():
    if not CACHE_DIR:
        return False
    try:
        os.makedirs(CACHE_DIR)
    except OSError as e:
        if e.errno != errno.EEXIST:
            if VERBOSE:
                sys.stderr.write('[tracecache] mkdir failed: %s\n' % e)
            return False
    return True


def store(key, entry, assumption_records=None):
    if not is_enabled():
        return False
    if not _ensure_dir():
        return False
    assumptions = assumption_records if assumption_records is not None \
        else entry.assumptions
    try:
        header = (_encode_assumptions(assumptions) +
                  _encode_meta(entry.meta) +
                  _pack_u8(1 if entry.has_dynamic_descrs else 0))
        body = encode_entry(entry)
        tmp = _path_for(key) + '.tmp'
        with open(tmp, 'wb') as f:
            f.write(MAGIC)
            f.write(JIT_VERSION.encode('ascii').ljust(_JIT_VERSION_BYTES, b' '))
            f.write(_pack_u32(len(header)))
            f.write(header)
            f.write(_pack_u32(len(body)))
            f.write(body)
        os.rename(tmp, _path_for(key))
        if VERBOSE:
            sys.stderr.write('[tracecache] stored %s (%d ops, dyn=%d)\n' %
                             (key[:12], len(entry.operations),
                              int(entry.has_dynamic_descrs)))
        Stats.stores += 1
        return True
    except Exception as e:
        if VERBOSE:
            sys.stderr.write('[tracecache] store failed: %s\n' % e)
        return False


def load(key, validate=None):
    """Load a cache entry. `validate(assumptions) -> bool` returns True to
    accept the entry, False to reject (miss). Returns the entry or None.
    """
    if not is_enabled():
        return None
    path = _path_for(key)
    if not os.path.exists(path):
        Stats.misses += 1
        return None
    try:
        with open(path, 'rb') as f:
            magic = f.read(len(MAGIC))
            if magic != MAGIC:
                return None
            version = f.read(_JIT_VERSION_BYTES).rstrip(b' ').decode('ascii')
            if version != JIT_VERSION:
                if VERBOSE:
                    sys.stderr.write('[tracecache] version mismatch, discard %s\n'
                                     % key[:12])
                try:
                    os.remove(path)
                except OSError:
                    pass
                return None
            (hl,) = struct.unpack('>I', f.read(4))
            header_bytes = f.read(hl)
            hr = _Reader(header_bytes)
            assumptions = _decode_assumptions(hr)
            meta = _decode_meta(hr)
            has_dynamic = hr.u8() != 0
            if validate is not None:
                if not validate(assumptions):
                    if VERBOSE:
                        sys.stderr.write('[tracecache] assumptions invalid for %s\n'
                                         % key[:12])
                    return None
            (bl,) = struct.unpack('>I', f.read(4))
            body = f.read(bl)
            entry = decode_entry(body)
            entry.assumptions = assumptions
            entry.meta = meta
            entry.has_dynamic_descrs = has_dynamic
    except Exception as e:
        if VERBOSE:
            sys.stderr.write('[tracecache] load failed: %s\n' % e)
        Stats.load_failures += 1
        return None
    if VERBOSE:
        sys.stderr.write('[tracecache] hit %s\n' % key[:12])
    Stats.hits += 1
    return entry


def invalidate(key):
    if not CACHE_DIR:
        return
    path = _path_for(key)
    try:
        os.remove(path)
    except OSError:
        pass


# --- stats -----------------------------------------------------------------

class Stats(object):
    hits = 0
    misses = 0
    stores = 0
    load_failures = 0


def reset_stats():
    Stats.hits = 0
    Stats.misses = 0
    Stats.stores = 0
    Stats.load_failures = 0


# --- LazyDescr placeholder (structural tests) -----------------------------

class LazyDescr(object):
    __slots__ = ('descr_id', 'repr_str')

    def __init__(self, descr_id, repr_str):
        self.descr_id = descr_id
        self.repr_str = repr_str

    def __repr__(self):
        return '<LazyDescr id=%d repr=%r>' % (self.descr_id, self.repr_str)


# --- structural replay (no backend) ---------------------------------------

class _InputArgPlaceholder(object):
    __slots__ = ('box_id', 'kind')

    def __init__(self, box_id, kind):
        self.box_id = box_id
        self.kind = kind

    def __repr__(self):
        return '<InputArg#%d kind=%d>' % (self.box_id, self.kind)


class _ReplayedOp(object):
    __slots__ = ('opnum', 'args', 'descr', 'result_type', 'result_box_id')

    def __init__(self, opnum, args, descr, result_type, result_box_id):
        self.opnum = opnum
        self.args = args
        self.descr = descr
        self.result_type = result_type
        self.result_box_id = result_box_id

    def getopnum(self):    return self.opnum
    def getarglist(self):  return list(self.args)
    def getdescr(self):    return self.descr

    def __repr__(self):
        return ('<_ReplayedOp num=%d args=%d descr=%r res=%s>' %
                (self.opnum, len(self.args), self.descr, chr(self.result_type)))


def _default_const_factory_int(v):
    class _CI(object):
        __slots__ = ('value',)
        def __init__(self, v): self.value = v
    c = _CI(v); c.value = v; return c


def _default_const_factory_float(v):
    class _CF(object):
        __slots__ = ('value',)
        def __init__(self, v): self.value = v
    c = _CF(v); c.value = v; return c


def _default_const_factory_ptr(v):
    class _CP(object):
        __slots__ = ('ptr_id',)
        def __init__(self, v): self.ptr_id = v
    c = _CP(v); c.ptr_id = v; return c


def _default_const_factory_repr(v):
    class _CR(object):
        __slots__ = ('repr_str',)
        def __init__(self, v): self.repr_str = v
    c = _CR(v); c.repr_str = v; return c


_default_const_factory = {
    TAG_CONST_INT:  _default_const_factory_int,
    TAG_CONST_FLT:  _default_const_factory_float,
    TAG_CONST_PTR:  _default_const_factory_ptr,
    TAG_CONST_REPR: _default_const_factory_repr,
}


def replay_to_operations(entry, descr_resolver=None, const_factory=None):
    """Reconstruct a structural stand-in sequence of operations. Used by
    tests that verify the wire format without touching the JIT backend.
    """
    if const_factory is None:
        const_factory = _default_const_factory
    boxes_by_id = {}
    inputargs = []
    for inp in entry.inputargs:
        b = _InputArgPlaceholder(inp.box_id, inp.kind)
        boxes_by_id[inp.box_id] = b
        inputargs.append(b)
    descr_cache = {}

    def _resolve_descr(op):
        if op.descr_kind == 0:
            return None
        key = (op.descr_kind, op.descr_ref)
        if key in descr_cache:
            return descr_cache[key]
        if descr_resolver is not None:
            d = descr_resolver(op.descr_kind, op.descr_ref)
        else:
            d = LazyDescr(0, repr(op.descr_ref))
        descr_cache[key] = d
        return d

    operations = []
    for op in entry.operations:
        args = []
        for a in op.args:
            if a.tag == TAG_BOX_REF:
                b = boxes_by_id.get(a.value)
                if b is None:
                    b = _InputArgPlaceholder(a.value, IN_KIND_INT)
                    boxes_by_id[a.value] = b
                args.append(b)
            else:
                factory = const_factory.get(a.tag)
                if factory is None:
                    raise ValueError('no const factory for tag %d' % a.tag)
                args.append(factory(a.value))
        descr = _resolve_descr(op)
        stub = _ReplayedOp(op.opnum, args, descr,
                           op.result_type, op.result_box_id)
        if op.result_type != RES_VOID:
            boxes_by_id[op.result_box_id] = stub
        operations.append(stub)
    return inputargs, operations


# --- real replay (into the live JIT) --------------------------------------

class CacheReplayUnsupported(Exception):
    """Raised when a cached entry can't be safely handed to the backend
    (dynamic descrs, missing opcode_descrs index, etc.)."""


def _descr_resolver_from_metainterp_sd(metainterp_sd):
    """Return a resolver callable that maps (descr_kind, descr_ref) back to
    a live descr by looking up `metainterp_sd.opcode_descrs` for static
    descrs and refusing to resolve dynamic ones.
    """
    opcode_descrs = getattr(metainterp_sd, 'opcode_descrs', None)

    def resolve(kind, ref):
        if kind == DESCR_KIND_STATIC:
            if opcode_descrs is None:
                raise CacheReplayUnsupported('metainterp_sd.opcode_descrs missing')
            if not (0 <= ref < len(opcode_descrs)):
                raise CacheReplayUnsupported(
                    'opcode_descrs index %d out of range' % ref)
            return opcode_descrs[ref]
        # Dynamic descr: we cannot rebuild it.
        raise CacheReplayUnsupported('dynamic descr cannot be replayed')
    return resolve


def replay_to_real_operations(entry, metainterp_sd, descr_resolver=None):
    """Reconstruct real `InputArg*` and `ResOperation` instances from a
    cache entry. Box identity is preserved across args and results.

    Raises CacheReplayUnsupported when the entry references a descr that
    cannot be rebuilt (dynamic descrs, out-of-range indices, ...). Callers
    should treat that as a miss.
    """
    if entry.has_dynamic_descrs:
        raise CacheReplayUnsupported('entry has dynamic descrs')

    from rpython.jit.metainterp.resoperation import (
        ResOperation, InputArgInt, InputArgRef, InputArgFloat)
    from rpython.jit.metainterp.history import ConstInt, ConstFloat, ConstPtr
    from rpython.jit.codewriter import longlong
    from rpython.rtyper.lltypesystem import lltype, llmemory

    if descr_resolver is None:
        descr_resolver = _descr_resolver_from_metainterp_sd(metainterp_sd)

    boxes_by_id = {}
    inputargs = []
    for inp in entry.inputargs:
        if inp.kind == IN_KIND_INT:
            b = InputArgInt(0)
        elif inp.kind == IN_KIND_FLOAT:
            b = InputArgFloat(longlong.ZEROF)
        elif inp.kind == IN_KIND_REF:
            b = InputArgRef(lltype.nullptr(llmemory.GCREF.TO))
        else:
            raise CacheReplayUnsupported(
                'unknown input kind %d' % inp.kind)
        boxes_by_id[inp.box_id] = b
        inputargs.append(b)

    def _make_arg(a):
        if a.tag == TAG_BOX_REF:
            b = boxes_by_id.get(a.value)
            if b is None:
                raise CacheReplayUnsupported(
                    'forward box ref %d' % a.value)
            return b
        if a.tag == TAG_CONST_INT:
            return ConstInt(a.value)
        if a.tag == TAG_CONST_FLT:
            return ConstFloat(longlong.getfloatstorage(float(a.value)))
        if a.tag == TAG_CONST_PTR:
            # For now only null pointers are roundtrip-safe.
            if a.value == 0:
                return ConstPtr(lltype.nullptr(llmemory.GCREF.TO))
            raise CacheReplayUnsupported(
                'non-null const_ptr %d not yet supported' % a.value)
        if a.tag == TAG_CONST_REPR:
            raise CacheReplayUnsupported(
                'const_repr (%r) not rebuildable' % a.value)
        raise CacheReplayUnsupported('unknown arg tag %d' % a.tag)

    operations = []
    for op in entry.operations:
        args = [_make_arg(a) for a in op.args]
        if op.descr_kind == 0:
            descr = None
        else:
            descr = descr_resolver(op.descr_kind, op.descr_ref)
        real_op = ResOperation(op.opnum, args, descr=descr)
        if op.result_type != RES_VOID:
            boxes_by_id[op.result_box_id] = real_op
        operations.append(real_op)
    return inputargs, operations


# --- builder: turn a live loop into a TraceCacheEntry ---------------------

class _Builder(object):
    """Walks live ResOperation objects and produces a TraceCacheEntry.

    Box identity is preserved via id()->box_id. Descr handling:
      * if the descr is in `opcode_descrs_set` (a set-of-ids supplied by
        the caller), store DESCR_KIND_STATIC + its index.
      * otherwise, record DESCR_KIND_DYNAMIC + repr() and flag the entry
        as having dynamic descrs (so the real-replay path refuses it).
    """

    def __init__(self, opcode_descrs=None):
        self._box_ids = {}
        self._next_box_id = 1
        if opcode_descrs is not None:
            # id(descr) -> index in opcode_descrs
            self._static_descr_map = {id(d): i
                                      for i, d in enumerate(opcode_descrs)}
        else:
            self._static_descr_map = {}
        self._has_dynamic = False

    def _box_id(self, box):
        bid = self._box_ids.get(id(box))
        if bid is None:
            bid = self._next_box_id
            self._next_box_id += 1
            self._box_ids[id(box)] = bid
        return bid

    def _input_kind(self, box):
        t = getattr(box, 'type', '?')
        if t == 'i': return IN_KIND_INT
        if t == 'r': return IN_KIND_REF
        if t == 'f': return IN_KIND_FLOAT
        return IN_KIND_INT

    def _encode_const(self, box):
        # Dispatch on the box's declared type first so that a const-float
        # with both getint() and getfloatstorage() available (like the test
        # fakes) doesn't silently downcast.
        t = getattr(box, 'type', '?')
        if t == 'f':
            getfloatstorage = getattr(box, 'getfloatstorage', None)
            if getfloatstorage is not None:
                try:
                    return SerializedArg(TAG_CONST_FLT,
                                         float(getfloatstorage()))
                except Exception:
                    pass
        if t == 'r':
            getref_base = getattr(box, 'getref_base', None)
            if getref_base is not None:
                try:
                    r = getref_base()
                    return SerializedArg(TAG_CONST_PTR, 0 if not r else 1)
                except Exception:
                    pass
        if t == 'i':
            getint = getattr(box, 'getint', None)
            if getint is not None:
                try:
                    return SerializedArg(TAG_CONST_INT, int(getint()))
                except Exception:
                    pass
        # Untyped fallback: try accessors in order.
        getint = getattr(box, 'getint', None)
        if getint is not None:
            try:
                return SerializedArg(TAG_CONST_INT, int(getint()))
            except Exception:
                pass
        getfloatstorage = getattr(box, 'getfloatstorage', None)
        if getfloatstorage is not None:
            try:
                return SerializedArg(TAG_CONST_FLT, float(getfloatstorage()))
            except Exception:
                pass
        return SerializedArg(TAG_CONST_REPR, repr(box))

    def _encode_arg(self, box):
        bid = self._box_ids.get(id(box))
        if bid is not None:
            return SerializedArg(TAG_BOX_REF, bid)
        is_const = getattr(box, 'is_constant', None)
        if callable(is_const) and is_const():
            return self._encode_const(box)
        return SerializedArg(TAG_BOX_REF, self._box_id(box))

    def _encode_descr(self, descr):
        if descr is None:
            return 0, 0
        idx = self._static_descr_map.get(id(descr), -1)
        if idx >= 0:
            return DESCR_KIND_STATIC, idx
        self._has_dynamic = True
        return DESCR_KIND_DYNAMIC, repr(descr)

    def _encode_input(self, box):
        bid = self._box_id(box)
        return SerializedInput(self._input_kind(box), bid)

    def _encode_op(self, op):
        opnum = op.getopnum()
        args = [self._encode_arg(a) for a in op.getarglist()]
        dk, dref = self._encode_descr(op.getdescr())
        rtype = getattr(op, 'type', 'v')
        if rtype not in ('i', 'r', 'f'):
            rtype = 'v'
        res_type = ord(rtype)
        res_box_id = 0 if res_type == RES_VOID else self._box_id(op)
        return SerializedOp(opnum, args, dk, dref, res_type, res_box_id)

    def build(self, inputargs, operations, assumptions=None, meta=None):
        in_args = [self._encode_input(b) for b in inputargs]
        ops = [self._encode_op(o) for o in operations]
        return TraceCacheEntry(
            inputargs=in_args,
            operations=ops,
            assumptions=assumptions or [],
            meta=meta or {},
            has_dynamic_descrs=self._has_dynamic)


def build_entry_from_loop(loop, opcode_descrs=None, assumptions=None,
                           meta=None):
    return _Builder(opcode_descrs=opcode_descrs).build(
        loop.inputargs, loop.operations,
        assumptions=assumptions, meta=meta)
