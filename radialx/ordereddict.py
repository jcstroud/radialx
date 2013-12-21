#! /usr/bin/env python

"""
    pyTableLib : a table data structure library for python
    Copyright (C) 2008, James C. Stroud

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

# To Do
# * insort

from itertools import izip

class OrderedDictError(Exception): pass
class InitError(OrderedDictError): pass
class InsertError(OrderedDictError): pass

class OrderedDict(dict):
  def __init__(self, items=()):

    def _fail(msg=None):
      raise InitError, msg

    if hasattr(items, 'keys'):
      self._keys = list(items.keys())
    else:
      if hasattr(items, 'next'):
        items = list(items)
      try:
        items = [tuple(i) for i in items]
        keys = [k for (k, v) in items]
      except TypeError, ValueError:
        _fail('items must be dict-like or key-value pairs')
      if len(set(keys)) != len(keys):
        _fail('multiple occurances of the same key is not permitted')
      else:
        self._keys = keys

    dict.__init__(self, items)

  def __getstate__(self):
    return {'_keys' : self._keys[:]}

  def __setstate__(self, adict):
    if '_keys' in adict:
      self.__dict__ = {'_keys' : adict['_keys'][:]}
    else:
      self.__dict__ = {'_keys' : adict.keys()}

  def __delitem__(self, k):
    dict.__delitem__(self, k)
    self._keys.remove(k)
  def __getitem__(self, k):
    if all(hasattr(k, a) for a in ['start', 'stop', 'step']):
      keys = self._keys[k]
      return OrderedDict((k, self[k]) for k in keys)
    else:
      return dict.__getitem__(self, k)
  def __iter__(self):
    return iter(self._keys)
  def __str__(self):
    """
    Gives ipython type output.
    """
    rstr = ["{\n"]
    empty = True
    for (k, v) in self.iteritems():
      empty = False
      if str(k) == k:
        k = "'%s'" % k
      if str(v) == v:
        v = "'%s'" % v
      rstr.append(" %s: %s,\n" % (k, v))
    if not empty:
      rstr[-1] = rstr[-1][:-2]
    rstr.append("}")
    return "".join(rstr)
  def __repr__(self):
    return '%s(%s)' % (self.__class__.__name__, str(self))
  def __reversed__(self):
    return reversed(self._keys)
  def __setitem__(self, k, v):
    if all(hasattr(k, a) for a in ('start', 'stop', 'step')):
      keys = self._keys[k]
      print keys, v
      for key, val in izip(keys, v):
        dict.__setitem__(self, key, val)
    else:
      new_key = (k not in self)
      dict.__setitem__(self, k, v)
      if new_key:
        try:
          self._keys.append(k)
        except AttributeError:
          self._keys = [k]
  def clear(self):
    self._keys = []
    return dict.clear(self)
  def copy(self):
    return self.__class__(self)
  def index(self, k):
    return self._keys.index(k)
  def insert(self, index, item):
    k, v = item
    if k in self:
      msg = 'Could not insert with duplicate key: %s' % key
      raise InsertError, msg
    dict.__setitem__(self, k, v)
    self._keys.insert(index, k)
  def insert_items(self, index, items):
    new_keys = []
    for k, v in items:
      if k in self:
        msg = 'Could not insert with duplicate key: %s' % k
        raise InsertError, msg
      else:
        new_keys.append(k)
    for k, v in items:
      dict.__setitem__(self, k, v)
    self._keys[index:index] = new_keys
  def item(self, i):
    k = self._keys[i]
    return (k, self[k])
  def items(self):
    return [(k, self[k]) for k in self._keys]
  def iteritems(self):
    def _f():
      for k in self._keys:
        yield (k, self[k])
    return _f()
  def iterkeys(self):
    return iter(self._keys)
  def itervalues(self):
    def _f():
      for k in self._keys:
        yield self[k]
    return _f()
  def key(self, i):
    return self._keys[i]
  def keys(self):
    return self._keys[:]
  def pop(self, k):
    v  = dict.pop(self, k)
    self._keys.remove(k)
    return v
  def popitem(self, i=-1):
    k = self._keys.pop(i)
    v = dict.pop(self, k)
    return (k, v)
  def swap(self, key1, key2):
    _keys = self._keys
    i1 = _keys.index(key1)
    i2 = _keys.index(key2)
    _keys[i1], _keys[i2] = _keys[i2], _keys[i1]
  def move(self, key, dist):
    _keys = self._keys
    i = _keys.index(key)
    j = i + dist
    self._move_from_to(i, j)
  def move_to(self, key, j):
    _keys = self._keys
    numkeys = len(_keys)
    last = numkeys - 1
    if not -numkeys <= j < numkeys:
      raise IndexError, 'new index, %d, out of range' % j
    i = _keys.index(key)
    if j < 0:
      j = numkeys + j
    self._move_from_to(i, j)
  def _move_from_to(self, i, j):
    _keys = self._keys
    key = _keys.pop(i)
    _keys[j:j] = key
  
  def reorder(self, keys):
    for k in keys:
      if not k in self:
        raise KeyError, 'key not found: %s' % k
    if len(keys) != len(self):
      raise ValueError, 'too few keys specified'
    self._keys = list(keys)
  def setdefault(self, k, d=None, lazy=None):
    if not ((d is None) or (lazy is None)):
      raise ValueError, 'either d or lazy or both must be None'
    if k in self:
      d = self[k]
    else:
      if lazy is not None:
        d = lazy()
      self[k] = d
    return d
  def update(self, other, index=None, **kwargs):
    """
    Values of kwargs take precedence over values in other, but
    ordering in other takes precendence over kwargs.
    Reasoning: documentation of dict.update suggests that::

      update(other, **kwargs)

    is equivalent to::

      update(other)
      update(kwargs)

    So the results of OrderedDict.update reflects this implicit
    ordering of operations.
    """
    other.update(kwargs)
    new_keys = [k for k in other if k not in self]
    if index is None:
      self._keys.extend(new_keys)
    else:
      self._keys[index:index] = new_keys
    dict.update(self, other)
  def reverse(self):
    self._keys.reverse()
  def reversed(self):
    od = self.copy()
    od.reverse()
    return od
  def sort(self, *args, **kwargs):
    self._keys.sort(*args, **kwargs)
  def sorted(self, *args, **kwargs):
    od = self.copy()
    od.sort(*args, **kwargs)
    return od
  def value(self, i):
    k = self._keys[i]
    return self[k]
  def values(self):
    return [self[k] for k in self._keys]
  def every(self, n):
    new = [self.__class__() for i in xrange(n)]
    ii = self.iteritems()
    try:
      while True:
        for s in new:
          k, v = ii.next()
          s[k] = v
    except StopIteration:
      pass
    return new

def test():
  d = OrderedDict((('Last', 'Williams'), ('First', 'Mary'), ('Middle', 'Francis')))
  print d
  print d.keys()
  d['FavColor'] = 'chartreuse'
  print d
  del(d['First'])
  print d
  d['First'] = 'Alice'
  print d
  d.reorder(['First', 'Middle', 'Last', 'FavColor'])
  print d
  d.sort()
  print d
  d.reorder(['First', 'Middle', 'Last', 'FavColor'])
  print d
  print d.sorted()
  print d.index('Middle')
  d.insert(3, ('FavFlavor', 'chocolate'))
  # d.insert(3, ('Pet', 'Cat'))
  print d
  d.insert_items(3, (('Sport', 'Hockey'), ('Pet', 'Dog')))
  print d
  # d.insert_items(3, (('Sport', 'Soccer'), ('Town', 'Boulder')))
  print d.item(3)
  print d.items()
  itit = d.iteritems()
  print itit
  for it in itit:
    print it
  itkey = d.iterkeys()
  print itkey
  for k in itkey:
    print k
  itval = d.itervalues()
  for v in itval:
    print v
  print d.key(4)
  print d.keys()
  print d.pop('Sport')
  print d
  # d.pop('Sport')
  d.insert(3, ('Sport', 'Soccer'))
  print d
  print d.popitem(3)
  print d
  d.insert(3, ('Sport', 'Soccer'))
  print d
  print d.popitem()
  print d
  d['FavColor'] = 'chartreuse'
  print d
  print d.setdefault('FavColor', 'fuscia')
  print d
  print d.setdefault('Hair', 'brown')
  print d
  print d.get('Bicycle', 'schwinn')
  print d.get('Hair', 'orange')
  d.update(OrderedDict((('Eyes', 'blue'), ('Spouse', 'Pat'))),
           Car='jag', Music='rocknroll', Spouse='Chris')
  print d
  print d.values()
  d[:5] = ('Bob', 'Carol', 'Ted', 'Alice', 'Fred')
  d[5:5] = ('x')
  print d
  x = d[5:8]
  print x.keys()
  print x
  for y in reversed(x):
    print y,
  print
  print repr(x)

if __name__ == "__main__":
  test()
