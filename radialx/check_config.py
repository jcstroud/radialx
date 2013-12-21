#! /usr/bin/env python

import re

import configobj
from validate import *

class CheckError(Exception):
  pass
class ConfigFileError(CheckError):
  pass

def unrepr(s):
  """
  Allows for unquotable strings surrounded by "`"s.
  """
  if s[0] == "`" and s[-1] == "`":
    v = s[1:-1]
  elif not s:
    v = s
  else:
    v = configobj._builder.build(configobj.getObj(s))
  return v

# Overriding configobj unrepr
# configobj.unrepr = unrepr


def unquote(s):
  if s[0] == "`" and s[-1] == "`":
    s = s[1:-1]
  return s

def typed_list(value, length=None, min=None, max=None,
                      empty=None, force=True, type_=None):
    """
    This will force unrepr if None is given as type_.
    If unrepr is not desired, just use validate.tuple.
    If length is given, min and max will be ignored.
    If empty is set to True, then None can be given
    as a value.
    """
    line_endingRE = re.compile(",\s*?\n", re.DOTALL)
    quoted_listRE = re.compile("^\s*\[\s*(.*?)\]\s*$", re.DOTALL)
    if str(value) == value:
      if (empty is not None):
        if value.lower() == empty.lower():
          value = []
      else:
        # configobj should have already listified single-line lists
        ql_match = quoted_listRE.match(value)
        if line_endingRE.search(value) is not None:
          if ql_match is not None:
            value = ql_match.group(1)
          value = ", ".join(line_endingRE.split(value)).strip()
          # abuse configobj to spite the author
          value = configobj.ConfigObj(["value = " + value])['value']

    if length is None:
      try:
        min = int(min)
        max = int(max)
      except TypeError:
        pass
    else:
      try:
          min = int(length)
          max = int(length)
      except ValueError:
          raise VdtParamError('length', length)

    if force:
      if str(value) == value:
        value = [value]

    if not isinstance(value, (list, tuple)):
      raise VdtTypeError(value)

    emptied = ((empty is not None) and (value == []))
    if not emptied:
      if min is not None:
        if len(value) < min:
          raise VdtValueTooShortError(value)
      if max is not None:
        if len(value) > max:
          raise VdtValueTooLongError(value)

    out = []
    for entry in value:
      if type_ is None:
        try:
          entry = configobj.unrepr(entry)
        except (configobj.UnknownType, SyntaxError):
          # this allows explicit strings to go un-unrepred'
          entry = unquote(entry)
      else:
        try:
          entry = type_(entry)
        except TypeError:
            e = VdtTypeError(value)
            e.msg = "%s not a valid type for conversion." % repr(value)
            raise e
        except ValueError:
            e = VdtValueError(value)
            e.msg = "Can not convert '%s' to a %s." % (entry, type_)
            raise e
      out.append(entry)
    return out

def typed_tuple(value, length=None, min=None, max=None,
                       empty=None, force=True, type_=None):
  """
  This will force unrepr if None is given as type_.
  If unrepr is not desired, just use validate.tuple.
  If length is given, min and max will be ignored.
  """
  return tuple(typed_list(value, length, min, max,
                                 empty, force, type_))

def int_list(value, length=None, min=None, max=None,
                                 empty=None, force=True):
  return typed_list(value, length, min, max, empty, force, int)
def integer_list(value, length=None, min=None, max=None,
                                     empty=None, force=True):
  return typed_list(value, length, min, max, empty, force, int)
def float_list(value, length=None, min=None, max=None,
                                   empty=None, force=True):
  return typed_list(value, length, min, max, empty, force, float)
def str_list(value, length=None, min=None, max=None,
                                 empty=None, force=True):
  return typed_list(value, length, min, max, empty, force, str)
def string_list(value, length=None, min=None, max=None,
                                    empty=None, force=True):
  return typed_list(value, length, min, max, empty, force, str)

def int_tuple(value, length=None, min=None, max=None,
                                  empty=None, force=True):
  return typed_tuple(value, length, min, max, empty, force, int)
def integer_tuple(value, length=None, min=None, max=None,
                                      empty=None, force=True):
  return typed_tuple(value, length, min, max, empty, force, int)
def float_tuple(value, length=None, min=None, max=None,
                                    empty=None, force=True):
  return typed_tuple(value, length, min, max, empty, force, float)
def str_tuple(value, length=None, min=None, max=None,
                                  empty=None, force=True):
  return typed_tuple(value, length, min, max, empty, force, str)
def string_tuple(value, length=None, min=None, max=None,
                                     empty=None, force=True):
  return typed_tuple(value, length, min, max, empty, force, str)

def number_format(value):
  numRE = re.compile(r'(\d+\.?)(\d*)([diouxXeEfFgG])$')
  e = None
  try:
    m = numRE.search(value)
  except TypeError, e:
    e = VdtTypeError(value)
  if m is None:
    e = VdtValueError(value)
  else:
    flgs = value[:m.start()]
    if not all(((f in "#0- +") and
                (flgs.count(f) <= 1)) for f in flgs):
      e = VdtValueError(value)
  if len(m.group(2)) > 1:
    e = VdtValueError(value)
    e.msg = "Precision of %s is too large." % repr(value)
  if e is not None:
    if not hasattr(e, "msg"):
      e.msg = "%s is not a valid formatting string." % repr(value)
    raise e
  return value

def update_validator(validator):
  globals_ = globals()
  checks_ = [
              "typed_list", "typed_tuple",
              "int_list", "integer_list",
              "float_list", "str_list", "string_list",
              "int_tuple", "integer_tuple", "float_tuple",
              "str_tuple", "string_tuple",
              "number_format"
            ]
  for c in checks_:
    validator.functions[c] = globals_[c]


def make_errors(c, result):
  flat = configobj.flatten_errors(c, result)
  rstr = []
  for (section_list, key, _) in flat:
    sec = ", ".join(section_list)
    if key is not None:
      params = (key, sec)
      if sec == "":
        tmplt = "- The '%s' setting of the top level section failed."
        msg = tmplt % key
      else:
        tmplt = "- The key '%s' of section '%s' failed."
        msg = tmplt % (key, sec)
      rstr.append("   " + msg)
    else:
      rstr.append("   - The section '%s' is missing." % sec)
  return "\n".join(rstr)

def load(config_name, validation_name):
  validator = Validator()
  update_validator(validator)

  c = configobj.ConfigObj(config_name, configspec=validation_name)

  result = c.validate(validator)

  if result != True:
    error_message = make_errors(c, result)
    tmplt = "The configuration file named '%s' is not valid."
    e = ConfigFileError(tmplt % c.filename)
    e.context = { 'result' : result,
                  'error_message' : error_message }
                  
    print e.context['error_message']
    raise e

  return c
