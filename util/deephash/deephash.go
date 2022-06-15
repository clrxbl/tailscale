// Copyright (c) 2020 Tailscale Inc & AUTHORS All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package deephash hashes a Go value recursively, in a predictable order,
// without looping. The hash is only valid within the lifetime of a program.
// Users should not store the hash on disk or send it over the network.
// The hash is sufficiently strong and unique such that
// Hash(x) == Hash(y) is an appropriate replacement for x == y.
//
// The definition of equality is identical to reflect.DeepEqual except:
//	* Floating-point values are compared based on the raw bits,
//	  which means that NaNs (with the same bit pattern) are treated as equal.
//	* Types which implement interface { AppendTo([]byte) []byte } use
//	  the AppendTo method to produce a textual representation of the value.
//	  Thus, two values are equal if AppendTo produces the same bytes.
//
// WARNING: This package, like most of the tailscale.com Go module,
// should be considered Tailscale-internal; we make no API promises.
package deephash

import (
	"bufio"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"hash"
	"log"
	"math"
	"reflect"
	"sync"
	"time"
	"unsafe"
)

// There is much overlap between the theory of serialization and hashing.
// A hash (useful for determining equality) can be produced by printing a value
// and hashing the output. The format must:
//	* be deterministic such that the same value hashes to the same output, and
//	* be parsable such that the same value can be reproduced by the output.
//
// The logic below hashes a value by printing it to a hash.Hash.
// To be parsable, it assumes that we know the Go type of each value:
//	* scalar types (e.g., bool or int32) are printed as fixed-width fields.
//	* list types (e.g., strings, slices, and AppendTo buffers) are prefixed
//	  by a fixed-width length field, followed by the contents of the list.
//	* slices, arrays, and structs print each element/field consecutively.
//	* interfaces print with a 1-byte prefix indicating whether it is nil.
//	  If non-nil, it is followed by a fixed-width field of the type index,
//	  followed by the format of the underlying value.
//	* pointers print with a 1-byte prefix indicating whether the pointer is
//	  1) nil, 2) previously seen, or 3) newly seen. Previously seen pointers are
//	  followed by a fixed-width field with the index of the previous pointer.
//	  Newly seen pointers are followed by the format of the underlying value.
//	* maps print with a 1-byte prefix indicating whether the map pointer is
//	  1) nil, 2) previously seen, or 3) newly seen. Previously seen pointers
//	  are followed by a fixed-width field of the index of the previous pointer.
//	  Newly seen maps are printed as a fixed-width field with the XOR of the
//	  hash of every map entry. With a sufficiently strong hash, this value is
//	  theoretically "parsable" by looking up the hash in a magical map that
//	  returns the set of entries for that given hash.

const scratchSize = 128

// hasher is reusable state for hashing a value.
// Get one via hasherPool.
type hasher struct {
	h          hash.Hash
	bw         *bufio.Writer
	scratch    [scratchSize]byte
	visitStack visitStack
}

func (h *hasher) reset() {
	if h.h == nil {
		h.h = sha256.New()
	}
	if h.bw == nil {
		h.bw = bufio.NewWriterSize(h.h, h.h.BlockSize())
	}
	h.bw.Flush()
	h.h.Reset()
}

// Sum is an opaque checksum type that is comparable.
type Sum struct {
	sum [sha256.Size]byte
}

func (s1 *Sum) xor(s2 Sum) {
	for i := 0; i < sha256.Size; i++ {
		s1.sum[i] ^= s2.sum[i]
	}
}

func (s Sum) String() string {
	return hex.EncodeToString(s.sum[:])
}

var (
	once sync.Once
	seed uint64
)

func (h *hasher) sum() (s Sum) {
	h.bw.Flush()
	// Sum into scratch & copy out, as hash.Hash is an interface
	// so the slice necessarily escapes, and there's no sha256
	// concrete type exported and we don't want the 'hash' result
	// parameter to escape to the heap:
	copy(s.sum[:], h.h.Sum(h.scratch[:0]))
	return s
}

var hasherPool = &sync.Pool{
	New: func() any { return new(hasher) },
}

// Hash returns the hash of v.
func Hash(v any) (s Sum) {
	h := hasherPool.Get().(*hasher)
	defer hasherPool.Put(h)
	h.reset()
	once.Do(func() {
		seed = uint64(time.Now().UnixNano())
	})
	h.hashUint64(seed)
	h.hashValue(reflect.ValueOf(v))
	return h.sum()
}

// Update sets last to the hash of v and reports whether its value changed.
func Update(last *Sum, v ...any) (changed bool) {
	sum := Hash(v)
	if sum == *last {
		// unchanged.
		return false
	}
	*last = sum
	return true
}

var appenderToType = reflect.TypeOf((*appenderTo)(nil)).Elem()

type appenderTo interface {
	AppendTo([]byte) []byte
}

func (h *hasher) hashUint8(i uint8) {
	h.bw.WriteByte(i)
}
func (h *hasher) hashUint16(i uint16) {
	binary.LittleEndian.PutUint16(h.scratch[:2], i)
	h.bw.Write(h.scratch[:2])
}
func (h *hasher) hashUint32(i uint32) {
	binary.LittleEndian.PutUint32(h.scratch[:4], i)
	h.bw.Write(h.scratch[:4])
}
func (h *hasher) hashUint64(i uint64) {
	binary.LittleEndian.PutUint64(h.scratch[:8], i)
	h.bw.Write(h.scratch[:8])
}
func (h *hasher) hashVarint(x uint64) {
	i := 0
	for x >= 0x80 {
		h.scratch[i] = byte(x) | 0x80
		x >>= 7
		i++
	}
	h.scratch[i] = byte(x)
	h.bw.Write(h.scratch[:i+1])
}

var (
	uint8Type    = reflect.TypeOf(byte(0))
	timeTimeType = reflect.TypeOf(time.Time{})
)

// returns ok if it was handled; else slow path runs
type typeHasherFunc func(h *hasher, v reflect.Value) (ok bool)

// A non-nil typeHasherFuture is populated into the typeHasher map
// when its type is first requested, before its func is created.
// Its func field fn is only populated once the type has been created.
// This is used for recursive types.
type typeHasherFuture struct {
	fn typeHasherFunc // nil until created
}

var typeHasher sync.Map // map[reflect.Type]*typeHasherFuture

var genTypeHasherMu sync.Mutex // held while populating the map

func getTypeHasher(t reflect.Type) typeHasherFunc {
	if f, ok := typeHasher.Load(t); ok {
		return f.(*typeHasherFuture).fn
	}
	genTypeHasherMu.Lock()
	defer genTypeHasherMu.Unlock()

	fu := getTypeHasherFutureLocked(t)
	return fu.fn
}

func getTypeHasherFutureLocked(t reflect.Type) *typeHasherFuture {
	if f, ok := typeHasher.Load(t); ok {
		return f.(*typeHasherFuture)
	}
	fu := new(typeHasherFuture)
	typeHasher.Store(t, fu)
	fu.fn = genTypeHasherLocked(t)
	return fu
}

func hashUint(h *hasher, v reflect.Value) bool {
	h.hashVarint(v.Uint())
	return true
}

func hashInt(h *hasher, v reflect.Value) bool {
	i64 := v.Int()
	ux := uint64(i64) << 1
	if i64 < 0 {
		ux = ^ux
	}
	h.hashVarint(ux)
	return true
}

func hashStructAppenderTo(h *hasher, v reflect.Value) bool {
	if !v.CanInterface() {
		return false // slow path
	}
	var a appenderTo
	if v.CanAddr() {
		a = v.Addr().Interface().(appenderTo)
	} else {
		a = v.Interface().(appenderTo)
	}
	size := h.scratch[:8]
	record := a.AppendTo(size)
	binary.LittleEndian.PutUint64(record, uint64(len(record)-len(size)))
	h.bw.Write(record)
	return true
}

func hashStructFields(t reflect.Type) typeHasherFunc {
	type indexAndFuture struct {
		index int
		fu    *typeHasherFuture
	}
	fields := make([]indexAndFuture, 0, t.NumField())
	for i, n := 0, t.NumField(); i < n; i++ {
		sf := t.Field(i)
		if sf.Type.Size() == 0 {
			continue
		}
		fields = append(fields, indexAndFuture{i, getTypeHasherFutureLocked(sf.Type)})
	}
	return func(h *hasher, v reflect.Value) bool {
		for _, f := range fields {
			if !f.fu.fn(h, v.Field(f.index)) {
				return false
			}
		}
		return true
	}
}

func hashPtrToMemoryRange(eleType reflect.Type) typeHasherFunc {
	size := eleType.Size()
	return func(h *hasher, v reflect.Value) bool {
		if v.IsNil() {
			h.hashUint8(0) // indicates nil
		} else {
			h.hashUint8(1) // indicates visiting a pointer
			h.bw.Write(unsafe.Slice((*byte)(v.UnsafePointer()), size))
		}
		return true
	}
}

const debug = false

func genTypeHasherLocked(t reflect.Type) typeHasherFunc {
	if debug {
		log.Printf("generating func for %v", t)
	}

	switch t.Kind() {
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return hashInt
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		return hashUint
	case reflect.Float32:
		return (*hasher).hashFloat32
	case reflect.Float64:
		return (*hasher).hashFloat64
	case reflect.Complex64:
		return (*hasher).hashComplex64
	case reflect.Complex128:
		return (*hasher).hashComplex128
	case reflect.String:
		return (*hasher).hashString
	case reflect.Slice:
		et := t.Elem()
		if canMemHash(et) {
			return (*hasher).hashSliceMem
		}
		efu := getTypeHasherFutureLocked(et)
		return hashSliceElements(efu)
	case reflect.Struct:
		if t == timeTimeType {
			return (*hasher).hashTime
		}
		if t.Implements(appenderToType) {
			return hashStructAppenderTo
		}
		return hashStructFields(t)
	case reflect.Pointer:
		et := t.Elem()
		if canMemHash(et) {
			return hashPtrToMemoryRange(et)
		}
		if t.Implements(appenderToType) {
			return func(h *hasher, v reflect.Value) bool {
				if !v.CanInterface() {
					return false // slow path
				}
				if v.IsNil() {
					h.hashUint8(0) // indicates nil
					return true
				}
				h.hashUint8(1) // indicates visiting a pointer
				a := v.Interface().(appenderTo)
				size := h.scratch[:8]
				record := a.AppendTo(size)
				binary.LittleEndian.PutUint64(record, uint64(len(record)-len(size)))
				h.bw.Write(record)
				return true
			}
		}
		if !typeIsRecursive(t) {
			efu := getTypeHasherFutureLocked(et)
			return func(h *hasher, v reflect.Value) bool {
				if v.IsNil() {
					h.hashUint8(0) // indicates nil
					return true
				}
				h.hashUint8(1) // indicates visiting a pointer
				return efu.fn(h, v.Elem())
			}
		}
	}

	return func(h *hasher, v reflect.Value) bool {
		if debug {
			log.Printf("unhandled type %v", v.Type())
		}
		return false
	}
}

// hashString hashes v, of kind String.
func (h *hasher) hashString(v reflect.Value) bool {
	s := v.String()
	h.hashUint64(uint64(len(s)))
	h.bw.WriteString(s)
	return true
}

func (h *hasher) hashFloat32(v reflect.Value) bool {
	h.hashUint32(math.Float32bits(float32(v.Float())))
	return true
}

func (h *hasher) hashFloat64(v reflect.Value) bool {
	h.hashUint64(math.Float64bits(v.Float()))
	return true
}

func (h *hasher) hashComplex64(v reflect.Value) bool {
	c := complex64(v.Complex())
	h.hashUint32(math.Float32bits(real(c)))
	h.hashUint32(math.Float32bits(imag(c)))
	return true
}

func (h *hasher) hashComplex128(v reflect.Value) bool {
	c := v.Complex()
	h.hashUint64(math.Float64bits(real(c)))
	h.hashUint64(math.Float64bits(imag(c)))
	return true
}

// hashString hashes v, of kind time.Time.
func (h *hasher) hashTime(v reflect.Value) bool {
	var t time.Time
	if v.CanAddr() {
		t = *(v.Addr().Interface().(*time.Time))
	} else {
		t = v.Interface().(time.Time) // allocs
	}
	b := t.AppendFormat(h.scratch[:0], time.RFC3339Nano)
	h.bw.Write(b)
	h.hashVarint(uint64(len(b)))
	return true
}

// hashSliceMem hashes v, of kind Slice, with a memhash-able element type.
func (h *hasher) hashSliceMem(v reflect.Value) bool {
	vLen := v.Len()
	h.hashUint64(uint64(vLen))
	if vLen == 0 {
		return true
	}
	h.bw.Write(unsafe.Slice((*byte)(v.UnsafePointer()), v.Type().Elem().Size()*uintptr(vLen)))
	return true
}

func hashSliceElements(efu *typeHasherFuture) typeHasherFunc {
	return func(h *hasher, v reflect.Value) bool {
		vLen := v.Len()
		h.hashUint64(uint64(vLen))
		for i := 0; i < vLen; i++ {
			if !efu.fn(h, v.Index(i)) {
				return false
			}
		}
		return true
	}
}

// canMemHash reports whether a slice of t can be hashed by looking at its
// contiguous bytes in memory alone. (e.g. structs with gaps aren't memhashable)
func canMemHash(t reflect.Type) bool {
	switch t.Kind() {
	case reflect.Bool, reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uintptr, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64,
		reflect.Float64, reflect.Float32, reflect.Complex128, reflect.Complex64:
		return true
	case reflect.Array:
		return canMemHash(t.Elem())
	case reflect.Struct:
		var sumFieldSize uintptr
		for i, numField := 0, t.NumField(); i < numField; i++ {
			sf := t.Field(i)
			if !canMemHash(sf.Type) {
				// Special case for 0-width fields that aren't at the end.
				if sf.Type.Size() == 0 && i < numField-1 {
					continue
				}
				return false
			}
			sumFieldSize += sf.Type.Size()
		}
		return sumFieldSize == t.Size() // else there are gaps
	}
	return false // TODO: more
}

// typeIsRecursive reports whether t has a path back to itself.
func typeIsRecursive(t reflect.Type) bool {
	inStack := map[reflect.Type]bool{}

	var visitType func(t reflect.Type) (isRecursiveSoFar bool)
	visitType = func(t reflect.Type) (isRecursiveSoFar bool) {
		if canMemHash(t) {
			// Anything mem-hashable can't have pointers
			return false
		}
		switch t.Kind() {
		case reflect.String:
			return false
		}
		if t.Size() == 0 {
			return false
		}

		if inStack[t] {
			return true
		}
		inStack[t] = true
		defer func() { delete(inStack, t) }()

		switch t.Kind() {
		default:
			panic("unhandled kind " + t.Kind().String())
		case reflect.Interface:
			// Assume the worst for now. TODO(bradfitz): in some cases
			// we should be able to prove that it's not recursive. Not worth
			// it for now.
			return true
		case reflect.Array, reflect.Chan, reflect.Pointer, reflect.Slice:
			return visitType(t.Elem())
		case reflect.Map:
			if visitType(t.Key()) {
				return true
			}
			if visitType(t.Elem()) {
				return true
			}
		case reflect.Struct:
			for i, numField := 0, t.NumField(); i < numField; i++ {
				if visitType(t.Field(i).Type) {
					return true
				}
			}
			return false
		}
		return false
	}
	return visitType(t)
}

const useTypeHashers = true // set to false for comparisons with old way

func (h *hasher) hashValue(v reflect.Value) {
	if !v.IsValid() {
		return
	}
	if useTypeHashers {
		fn := getTypeHasher(v.Type())
		if fn(h, v) {
			return
		}
	}
	w := h.bw
	if debug {
		log.Printf("doing slow path for %v", v.Type())
	}

	// TODO(dsnet): Avoid cycle detection for types that cannot have cycles.

	// Generic handling.
	switch v.Kind() {
	default:
		panic(fmt.Sprintf("unhandled kind %v for type %v", v.Kind(), v.Type()))
	case reflect.Ptr:
		if v.IsNil() {
			h.hashUint8(0) // indicates nil
			return
		}

		// Check for cycle.
		ptr := pointerOf(v)
		if idx, ok := h.visitStack.seen(ptr); ok {
			h.hashUint8(2) // indicates cycle
			h.hashUint64(uint64(idx))
			return
		}
		h.visitStack.push(ptr)
		defer h.visitStack.pop(ptr)

		h.hashUint8(1) // indicates visiting a pointer
		h.hashValue(v.Elem())
	case reflect.Struct:
		for i, n := 0, v.NumField(); i < n; i++ {
			h.hashValue(v.Field(i))
		}
	case reflect.Slice, reflect.Array:
		vLen := v.Len()
		if v.Kind() == reflect.Slice {
			h.hashUint64(uint64(vLen))
		}
		if v.Type().Elem() == uint8Type && v.CanInterface() {
			if vLen > 0 && vLen <= scratchSize {
				// If it fits in scratch, avoid the Interface allocation.
				// It seems tempting to do this for all sizes, doing
				// scratchSize bytes at a time, but reflect.Slice seems
				// to allocate, so it's not a win.
				n := reflect.Copy(reflect.ValueOf(&h.scratch).Elem(), v)
				w.Write(h.scratch[:n])
				return
			}
			fmt.Fprintf(w, "%s", v.Interface())
			return
		}
		for i := 0; i < vLen; i++ {
			// TODO(dsnet): Perform cycle detection for slices,
			// which is functionally a list of pointers.
			// See https://github.com/google/go-cmp/blob/402949e8139bb890c71a707b6faf6dd05c92f4e5/cmp/compare.go#L438-L450
			h.hashValue(v.Index(i))
		}
	case reflect.Interface:
		if v.IsNil() {
			h.hashUint8(0) // indicates nil
			return
		}
		v = v.Elem()

		h.hashUint8(1) // indicates visiting interface value
		h.hashType(v.Type())
		h.hashValue(v)
	case reflect.Map:
		// Check for cycle.
		ptr := pointerOf(v)
		if idx, ok := h.visitStack.seen(ptr); ok {
			h.hashUint8(2) // indicates cycle
			h.hashUint64(uint64(idx))
			return
		}
		h.visitStack.push(ptr)
		defer h.visitStack.pop(ptr)

		h.hashUint8(1) // indicates visiting a map
		h.hashMap(v)
	case reflect.String:
		s := v.String()
		h.hashUint64(uint64(len(s)))
		w.WriteString(s)
	case reflect.Bool:
		if v.Bool() {
			h.hashUint8(1)
		} else {
			h.hashUint8(0)
		}
	case reflect.Int8:
		h.hashUint8(uint8(v.Int()))
	case reflect.Int16:
		h.hashUint16(uint16(v.Int()))
	case reflect.Int32:
		h.hashUint32(uint32(v.Int()))
	case reflect.Int64, reflect.Int:
		h.hashUint64(uint64(v.Int()))
	case reflect.Uint8:
		h.hashUint8(uint8(v.Uint()))
	case reflect.Uint16:
		h.hashUint16(uint16(v.Uint()))
	case reflect.Uint32:
		h.hashUint32(uint32(v.Uint()))
	case reflect.Uint64, reflect.Uint, reflect.Uintptr:
		h.hashUint64(uint64(v.Uint()))
	case reflect.Float32:
		h.hashUint32(math.Float32bits(float32(v.Float())))
	case reflect.Float64:
		h.hashUint64(math.Float64bits(float64(v.Float())))
	case reflect.Complex64:
		h.hashUint32(math.Float32bits(real(complex64(v.Complex()))))
		h.hashUint32(math.Float32bits(imag(complex64(v.Complex()))))
	case reflect.Complex128:
		h.hashUint64(math.Float64bits(real(complex128(v.Complex()))))
		h.hashUint64(math.Float64bits(imag(complex128(v.Complex()))))
	}
}

type mapHasher struct {
	h    hasher
	val  valueCache      // re-usable values for map iteration
	iter reflect.MapIter // re-usable map iterator
}

var mapHasherPool = &sync.Pool{
	New: func() any { return new(mapHasher) },
}

type valueCache map[reflect.Type]reflect.Value

func (c *valueCache) get(t reflect.Type) reflect.Value {
	v, ok := (*c)[t]
	if !ok {
		v = reflect.New(t).Elem()
		if *c == nil {
			*c = make(valueCache)
		}
		(*c)[t] = v
	}
	return v
}

// hashMap hashes a map in a sort-free manner.
// It relies on a map being a functionally an unordered set of KV entries.
// So long as we hash each KV entry together, we can XOR all
// of the individual hashes to produce a unique hash for the entire map.
func (h *hasher) hashMap(v reflect.Value) {
	mh := mapHasherPool.Get().(*mapHasher)
	defer mapHasherPool.Put(mh)

	iter := &mh.iter
	iter.Reset(v)
	defer iter.Reset(reflect.Value{}) // avoid pinning v from mh.iter when we return

	var sum Sum
	k := mh.val.get(v.Type().Key())
	e := mh.val.get(v.Type().Elem())
	mh.h.visitStack = h.visitStack // always use the parent's visit stack to avoid cycles
	for iter.Next() {
		k.SetIterKey(iter)
		e.SetIterValue(iter)
		mh.h.reset()
		mh.h.hashValue(k)
		mh.h.hashValue(v)
		sum.xor(mh.h.sum())
	}
	h.bw.Write(append(h.scratch[:0], sum.sum[:]...)) // append into scratch to avoid heap allocation
}

// visitStack is a stack of pointers visited.
// Pointers are pushed onto the stack when visited, and popped when leaving.
// The integer value is the depth at which the pointer was visited.
// The length of this stack should be zero after every hashing operation.
type visitStack map[pointer]int

func (v visitStack) seen(p pointer) (int, bool) {
	idx, ok := v[p]
	return idx, ok
}

func (v *visitStack) push(p pointer) {
	if *v == nil {
		*v = make(map[pointer]int)
	}
	(*v)[p] = len(*v)
}

func (v visitStack) pop(p pointer) {
	delete(v, p)
}

// pointer is a thin wrapper over unsafe.Pointer.
// We only rely on comparability of pointers; we cannot rely on uintptr since
// that would break if Go ever switched to a moving GC.
type pointer struct{ p unsafe.Pointer }

func pointerOf(v reflect.Value) pointer {
	return pointer{unsafe.Pointer(v.Pointer())}
}

// hashType hashes a reflect.Type.
// The hash is only consistent within the lifetime of a program.
func (h *hasher) hashType(t reflect.Type) {
	// This approach relies on reflect.Type always being backed by a unique
	// *reflect.rtype pointer. A safer approach is to use a global sync.Map
	// that maps reflect.Type to some arbitrary and unique index.
	// While safer, it requires global state with memory that can never be GC'd.
	rtypeAddr := reflect.ValueOf(t).Pointer() // address of *reflect.rtype
	h.hashUint64(uint64(rtypeAddr))
}
