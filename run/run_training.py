from model import NormalVAE as vae
from data import GenreateTopologies as Top
import torch
import torch.optim as optim
import paths
import train_util

if __name__ == "__main__":

    topology_sets = Top.Topologies()
    indices = list(range(topology_sets.length))
    indices.append(indices)

    data_in_tensor = torch.from_numpy(topology_sets.get_topology_tensor(indices, mask=True)).float().to(paths.device)
    data_out_tensor = torch.from_numpy(topology_sets.get_topology_tensor(indices, mask=False)).float().to(paths.device)

    # expand dims of tensor in channel 1
    data_in_tensor = data_in_tensor.unsqueeze(1)
    data_out_tensor = data_out_tensor.unsqueeze(1)

    num_epochs = 20
    batch_size = 64

    model_parameters = vae.SimpleVAEParameters(input_channels=1,
                                               hidden_size=64,
                                               num_layers=3,
                                               latent_dim=10,
                                               kernel_size=3,
                                               stride=2)

    model = model_parameters.instantiate_new_model().to(paths.device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, num_epochs + 1):
        vae.train(model=model, optimizer=optimizer, batch_size=batch_size, epoch=epoch, data_in_tensor=data_in_tensor, data_out_tensor=data_out_tensor)

    save_name = paths.PARAM_PATH + model_parameters.getName() + ".pickle"
    train_util.save_model_and_hp(model, model_parameters, batch_size=batch_size, name=save_name)