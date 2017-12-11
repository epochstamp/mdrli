import hashlib
import sys
from configobj import ConfigObj
from validate import Validator
from joblib import load,dump
import importlib
import os
import inspect
import shutil

def capitalizeFirstLetter(s):
    return s[0].upper() + s[1:]

def md5_file(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def list_classes_module_with_parent(modulefile,parent):
        cls = []
        module = importlib.import_module(modulefile.replace("/",".").replace(".py",""))
        def isparent(c):
                try:
                        return c != parent and issubclass(c, parent)
                except:
                        return False
        for name, obj in inspect.getmembers(module, isparent):
            cls.append(name)
        return cls

def get_mod_object(folder,module,modtype,*args,**kwargs):
    mod = __import__(folder + "." + module + "." + modtype, fromlist=[capitalizeFirstLetter(module)])
    try:
        return getattr(mod, capitalizeFirstLetter(module))(*args)
    except:
        return getattr(mod, capitalizeFirstLetter(module))(**kwargs)



def get_mod_class(folder,module,modtype):
    mod = __import__(folder + "." + module + "." + modtype, fromlist=[capitalizeFirstLetter(module)])
    return getattr(mod, capitalizeFirstLetter(module))


def load_dump(dumptype,module,name):
    return load("dumps/" + dumptype + "/" + module + "/" + name + ".dump")

def dump_dump(dumptype,module,name,value):
    try:
        os.makedirs("dumps/" + dumptype + "/" + module)
    except:
        pass
    dump(value,"dumps/" + dumptype + "/" + module + "/" + name + ".dump")

    
def parse_conf(conf_file):
    directory = os.path.dirname(conf_file)
    validator = Validator()
    try:
        params = ConfigObj(conf_file, directory + "/validator")
        params.validate(validator, copy=True)
    except:
        params = ConfigObj(conf_file)
    return params

def copy_rename(old_file_name, new_file_name):
        src_dir= os.curdir
        dst_dir= os.path.join(os.curdir , "")
        src_file = os.path.join(src_dir, old_file_name)
        shutil.copy(src_file,dst_dir)
        
        dst_file = os.path.join(dst_dir, old_file_name)
        new_dst_file_name = os.path.join(dst_dir, new_file_name)
        os.rename(dst_file, new_dst_file_name)
