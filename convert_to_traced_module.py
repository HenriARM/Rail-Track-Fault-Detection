import torch

device = "cpu"
model_filename = "./exp_eff_b7_lr_e3/best_model.pt"
model = torch.load(model_filename, map_location="cpu")
model = model.eval()
model = model.to(device)

# Save model as a traced script:
example_input = torch.rand(1, 3, 224, 224)
traced_script_module = torch.jit.trace(model, example_input)
traced_script_module.save("model.pt")
