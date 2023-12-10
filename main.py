import magic_sdk as magic
import torch.nn as nn
import torch

# Example usage of the functions in the magic_sdk module

# Scraping Google's homepage
content = magic.scrape_google()
print("Scraped content from Google's homepage:")
print(content[:200])  # Displaying the first 200 characters

# Scraping a specific URL
url_content = magic.scrape(url="http://example.com")
print("\nScraped content from example.com:")
print(url_content[:200])  # Displaying the first 200 characters

# For demonstration purposes, we'll create a dummy model and save it
class DummyModel(nn.Module):
    def forward(self, x):
        return x * 2

# Saving a dummy model to a file
dummy_model = DummyModel()
torch.save(dummy_model, "model.pt")


# Loading the dummy model using magic_sdk
loaded_model = torch.load("model.pt")

# Running inference with the loaded model (using a dummy tensor)
dummy_input = magic.torch_tensor([1, 2, 3])
result = magic.run_inference(model=loaded_model, input_tensor=dummy_input)
sm = magic.torch_softmax(dummy_input)
print(sm)
print("\nInference result with dummy model:")
print(result)

# Displaying result in dictionary format
magic.display_result_dict({"result": result.tolist()})

magic.print_cow_ascii()

