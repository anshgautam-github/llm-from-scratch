# The recommended way is to save a model's so-called state_dict, a dictionary mapping each layer to its parameters, using the torch.save function as follows:
model = GPTModel(GPT_CONFIG_124M)
torch.save(model.state_dict(), "model.pth")

# Then, after saving the model weights via the state_dict, we can load the model weights into a new GPTModel model instance as follows:
model = GPTModel(GPT_CONFIG_124M)
model.load_state_dict(torch.load("model.pth"))
model.eval()

# Using torch.save, we can save both the model and optimizer state_dict contents as follows:
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)

torch.save({
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    }, 
    "model_and_optimizer.pth"
)

# Then, we can restore the model and optimizer states as follows by first loading the saved data via torch.load and then using the load_state_dict method:
checkpoint = torch.load("model_and_optimizer.pth")
model = GPTModel(GPT_CONFIG_124M)
model.load_state_dict(checkpoint["model_state_dict"])
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.1)
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
model.train();