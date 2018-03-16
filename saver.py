
def save(net,name):
    model_json = net.to_json()
    with open(name+".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    net.save_weights(name+".h5")
def load(name)