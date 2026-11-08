Put your trained model here (.pt / .pth / .pkl).

Saved with torch.save(model, "name.pt")  -> loaded as it is.
Saved with torch.save(model.state_dict(), ...) -> rebuilt with the "Model settings" in the sidebar.
