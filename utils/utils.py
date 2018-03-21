import hashlib
import sys
from configobj import ConfigObj
from configparser import ConfigParser
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

def erase_dict_from_keyword_list(dict_to_erase, strkeyvalue_list, warning=False):
    dict_res = dict(dict_to_erase)
    lst = list(filter(lambda x : len(x) == 2, map(lambda y : y.split("="), strkeyvalue_list)))
    for [k,v] in lst:
        if k not in dict_to_erase and warning:
            print("Warning : keyword " + str(k) + " is not a key from the dict to erase") 
        dict_res[k] = v
    return dict_res

def revalidate_dict_from_conf_module(dict_args, modtype,module):
    write_conf(dict_args, "cfgs/"+modtype+"/" + module + "/temp")
    dict_res = parse_conf("cfgs/"+modtype+"/" + module + "/temp")
    os.remove("cfgs/"+modtype+"/" + module + "/temp")   
    return dict_res       

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

def get_mod_object(folder,module,modtype,args,kwargs,mode=0):
    mod = __import__(folder + "." + module + "." + modtype, fromlist=[capitalizeFirstLetter(module)])

    if mode == 0:
        return getattr(mod, capitalizeFirstLetter(module))(*args,**kwargs)
    elif mode == 1:
        return getattr(mod, capitalizeFirstLetter(module))(**kwargs)  
    else:
        return getattr(mod, capitalizeFirstLetter(module))(*args)



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

def flatten(l, ltypes=(list, tuple)):
    ltype = type(l)
    l = list(l)
    i = 0
    while i < len(l):
        while isinstance(l[i], ltypes):
            if not l[i]:
                l.pop(i)
                i -= 1
                break
            else:
                l[i:i + 1] = l[i]
        i += 1
    return ltype(l)
    
def parse_conf(conf_file, get_sections=False):
    directory = os.path.dirname(conf_file)
    validator = Validator()
    try:
        params = ConfigObj(conf_file, configspec=directory + "/validator")
        params.validate(validator, copy=True)
    except Exception as e:
        params = ConfigObj(conf_file)
    if get_sections:
        cfgparser = ConfigParser()
        cfgparser.read_file(open(conf_file))
        return params, cfgparser.sections()
    return params

def write_conf(conf_dict,conf_file):
    config = ConfigObj()
    config.filename = conf_file
    for k,v in conf_dict.items():
       config[k] = v
    config.write()
    

def copy_rename(old_file_name, new_file_name):
        src_dir= os.curdir
        dst_dir= os.path.join(os.curdir , "")
        src_file = os.path.join(src_dir, old_file_name)
        shutil.copy(src_file,dst_dir)
        
        dst_file = os.path.join(dst_dir, old_file_name)
        new_dst_file_name = os.path.join(dst_dir, new_file_name)
        os.rename(dst_file, new_dst_file_name)
