import os
import pint
import numpy as np
import typing
import json
from src.UnitSystem import *


class Serializable:

    serialize_members = []

    def serialize(self, chkpt_dir: str, obj_dict: dict = None, chkpt_subdir: str = "") -> None:
        if not chkpt_dir.endswith("/"):
            chkpt_dir += '/'
        var_dict = Serializable.__setup_serialization(self, chkpt_dir + chkpt_subdir)

        if obj_dict is None:
            obj_dict = {}

        to_serialize = self.get_serialize_members()

        for m in dir(self):
            member = getattr(self, m)
            if m in to_serialize or isinstance(member, Serializable):
                Serializable.__save_member(m, member, var_dict['members'], chkpt_dir, obj_dict, chkpt_subdir)

        Serializable.save(chkpt_dir + chkpt_subdir, var_dict)

    @classmethod
    def get_serialize_members(cls):
        members = set()  # if class not setting serialize_members or set something already set in the parent class
        # mro = method resolution order -> iterate over class precedence list and collect serialize_members from all
        for base in reversed(cls.__mro__):
            if hasattr(base, 'serialize_members'):
                for m in base.serialize_members:
                    if m not in members:
                        members.add(m)
        return list(members)

    @classmethod
    def deserialize(cls, chkpt_dir: str, obj_dict: dict = None, chkpt_subdir: str = "") -> typing.Any:
        from src.Factory import Fabricable
        if chkpt_dir[-1:] != '/':
            chkpt_dir += '/'

        type, id, chkpt = Serializable.load_checkpoint(chkpt_dir + chkpt_subdir)
        if issubclass(cls, Fabricable):
            obj = cls.factory.create_blank(type)
        else:
            # This assumes an empty constructor which is in most cases not given (however, for now it suffices for the serializable classes)
            # A cleaner future solution would be to create an interface that guarantees the existence of a create_blank function or similar
            obj = cls()
        obj.load_members(chkpt_dir, chkpt, id, obj_dict, chkpt_subdir)
        return obj

    @staticmethod
    def load_checkpoint(chkpt_dir: str) -> (str, str, dict[str, dict]):
        with open(chkpt_dir + "sim_vars.json", 'r') as jfile:
            var_dict = json.load(fp=jfile)

        return var_dict['type'], var_dict['id'], var_dict['members']

    def load_members(self, chkpt_dir: str, chkpt: dict, id: int, obj_dict = None, chkpt_subdir: str = "") -> None:
        if obj_dict is None:
            obj_dict = {}
        obj_dict[id] = self

        for member in chkpt:
            setattr(self, member, Serializable.__load_member(chkpt_dir, chkpt[member], obj_dict, chkpt_subdir))


    @staticmethod
    def __load_member(chkpt_dir: str, member: typing.Any, obj_dict: dict = None, chkpt_subdir: str = "") -> typing.Any:
        if isinstance(member, dict) and 'type' in member:
            if member['type'] == "np.ndarray":
                arr = np.load(chkpt_dir + chkpt_subdir + member['dump_file'])
                if 'unit' in member:
                    return ureg.Quantity(arr, member['unit'])
                else:
                    return arr
            elif 'unit' in member:
                return ureg.Quantity(member['value'], member['unit'])
            elif member['type'] == "list":
                tmp = []
                for m in member['elems']:
                    tmp.append(Serializable.__load_member(chkpt_dir, member['elems'][m], obj_dict, chkpt_subdir))
                return tmp
            elif member['type'] == "tuple":
                tmp = []
                for m in member['elems']:
                    tmp.append(Serializable.__load_member(chkpt_dir, member['elems'][m], obj_dict, chkpt_subdir))
                return tuple(tmp)
            elif member['type'] == "dict":
                tmp = {}
                for m in member['dict']:
                    tmp[m] = Serializable.__load_member(chkpt_dir, member['dict'][m], obj_dict, chkpt_subdir)
                return tmp
            elif 'id' in member and member['id'] in obj_dict:
                return obj_dict[member['id']]
            elif 'dir' in member:
                return Serializable.__deserialize_member(chkpt_dir, member, obj_dict, chkpt_subdir)
        else:
            return member

    @staticmethod
    def __deserialize_member(chkpt_dir: str, member_dict: dict, obj_dict: dict, chkpt_subdir: str = "") -> typing.Any:
        type = member_dict['type']

        module_name, class_name = type.rsplit('.', 1)
        module = __import__(module_name, fromlist=[class_name])
        cls = getattr(module, class_name)
        return cls.deserialize(chkpt_dir, obj_dict, member_dict['dir'])

    @staticmethod
    def __setup_serialization(obj: typing.Any, chkpt_dir: str) -> (str, dict[str, dict]):
        if not os.path.exists(chkpt_dir):
            os.makedirs(chkpt_dir)

        var_dict = {}
        var_dict['members'] = {}
        var_dict['type'] = obj.__class__.__name__
        var_dict['id'] = str(id(obj))

        return var_dict

    @staticmethod
    def __save_member(name: str, member: typing.Any, var_dict: dict, chkpt_dir: str, obj_dict: dict[int, typing.Any],
                      chkpt_subdir: str = "") -> None:
        if isinstance(member, Serializable):
            var_dict[name] = {}
            var_dict[name]['type'] = member.__class__.__module__ + "." + member.__class__.__name__
            var_dict[name]['id'] = str(id(member))
            if str(id(member)) in obj_dict:
                var_dict[name]['dir'] = obj_dict[str(id(member))]
            else:
                chkpt_subdir += name + "/"
                var_dict[name]['dir'] = chkpt_subdir
                obj_dict[str(id(member))] = var_dict[name]['dir']
                member.serialize(chkpt_dir, obj_dict, chkpt_subdir)
        elif isinstance(member, pint.Quantity):
            Serializable.__save_member_pint(name, member, var_dict, chkpt_dir + chkpt_subdir)
        elif isinstance(member, np.ndarray):
            np_dump_file = name + ".npy"
            np.save(chkpt_dir + chkpt_subdir + np_dump_file, member)
            var_dict[name] = {}
            var_dict[name]['type'] = "np.ndarray"
            var_dict[name]['dump_file'] = np_dump_file
        # is not instance of a class
        elif not hasattr(member, '__dict__'):
            chkpt_subdir += name + "/"
            if isinstance(member, dict):
                var_dict[name] = {}
                var_dict[name]['type'] = "dict"
                var_dict[name]['dict'] = {}
                for key in member:
                    Serializable.__save_member(key, member[key], var_dict[name]['dict'], chkpt_dir, obj_dict, chkpt_subdir)
            elif isinstance(member, list):
                var_dict[name] = {}
                var_dict[name]['type'] = "list"
                var_dict[name]['elems'] = {}
                for idx in range(len(member)):
                    Serializable.__save_member(str(idx), member[idx], var_dict[name]['elems'], chkpt_dir, obj_dict, chkpt_subdir)
            elif isinstance(member, tuple):
                var_dict[name] = {}
                var_dict[name]['type'] = "tuple"
                var_dict[name]['elems'] = {}
                for idx in range(len(member)):
                    Serializable.__save_member(str(idx), member[idx], var_dict[name]['elems'], chkpt_dir, obj_dict, chkpt_subdir)
            else:
                var_dict[name] = member

    @staticmethod
    def __save_member_pint(name: str, member: pint.Quantity, var_dict: dict, chkpt_dir: str) -> None:
        var_dict[name] = {}
        member_dict = var_dict[name]
        member_dict['unit'] = str(member.units)

        dequantified = member.magnitude
        if not isinstance(dequantified, np.ndarray):
            member_dict['value'] = dequantified
            member_dict['type'] = "scalar"
        else:
            np_dump_file = name + ".npy"
            np.save(chkpt_dir + np_dump_file, dequantified)
            member_dict['type'] = "np.ndarray"
            member_dict['dump_file'] = np_dump_file

    @staticmethod
    def save(chkpt_dir: str, var_dict: dict) -> None:
        with open(chkpt_dir + "sim_vars.json", 'w') as jfile:
            json.dump(fp=jfile, obj=var_dict, indent=2)
