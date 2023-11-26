import paths
import pickle
import torch
from torch.nn import Module
from model.ModelParameters import ModelParameters

def save_model(model : Module, name):
    torch.save(model.state_dict(), paths.PARAM_PATH + "_" + name)

def load_model(model : Module, name):
    model.load_state_dict(torch.load(paths.PARAM_PATH + "_" + name))
    model.eval()

def save_model_and_hp(model : Module, hyperparam : ModelParameters, batch_size, name):
    dict = {}
    dict["param"] = model.state_dict()
    dict["hyperparam"] = hyperparam
    dict["batch_size"] = batch_size
    with open(paths.PARAM_PATH + "_" + name, 'wb') as handle:
        pickle.dump(dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_model_from_file(file: str | None = None):
    if file is None:
        filename = paths.select_file(choose_file=True)
    else:
        filename = file
    with open(filename, 'rb') as handle:
        dict = pickle.load(handle)
        hyperparam : ModelParameters = dict["hyperparam"]
        param = dict["param"]
        batch_size = dict["batch_size"]

        model = hyperparam.instantiate_new_model()
        model.load_state_dict(param)
        model = model.to(device=paths.device)
        return model, batch_size, dict["advisor"]

def get_save_name(advisor, modelName, algName, run_index, extension):
    return "A" + str(advisor) + "_" + "R" + str(run_index) + "_" + modelName + "_" + algName  + extension
