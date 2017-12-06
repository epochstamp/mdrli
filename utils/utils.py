import hashlib
import sys
from configobj import ConfigObj
from joblib import load,dump
import os

def capitalizeFirstLetter(s):
    return s[0].upper() + s[1:]

def md5_file(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()



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
    params = ConfigObj(conf_file)
    return params
