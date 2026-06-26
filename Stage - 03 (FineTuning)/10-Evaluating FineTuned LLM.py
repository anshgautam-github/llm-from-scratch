# The following code uses the generate method in the same manner as before; however, we now iterate over the entire test_set.
# Also, instead of printing the model responses, we add them to the test_set dictionary:
from tqdm import tqdm

for i, entry in tqdm(enumerate(test_data), total=len(test_data)):

    input_text = format_input(entry)

    token_ids = generate(
        model=model,
        idx=text_to_token_ids(input_text, tokenizer).to(device),
        max_new_tokens=256,
        context_size=BASE_CONFIG["context_length"],
        eos_id=50256
    )
    generated_text = token_ids_to_text(token_ids, tokenizer)
    response_text = generated_text[len(input_text):].replace("### Response:", "").strip()

    test_data[i]["model_response"] = response_text


with open("instruction-data-with-response.json", "w") as file:
    json.dump(test_data, file, indent=4)  # "indent" for pretty-printing


# Let's verify that the responses have been correctly added to the test_set dictionary by examining one of the entries:
print(test_data[0])

# Finally, we save the model as gpt2-medium355M-sft.pth file to be able to reuse it in future projects:
import re
file_name = f"{re.sub(r'[ ()]', '', CHOOSE_MODEL) }-sft.pth"
torch.save(model.state_dict(), file_name)
print(f"Model saved as {file_name}")

# Load model via
# model.load_state_dict(torch.load("gpt2-medium355M-sft.pth"))

# EVALUATING THE FINE-TUNED LLM

# The following code verifies that the Ollama session is running properly before we use Ollama to evaluate the test set responses generated:
import psutil

def check_if_running(process_name):
    running = False
    for proc in psutil.process_iter(["name"]):
        if process_name in proc.info["name"]:
            running = True
            break
    return running

ollama_running = check_if_running("ollama")

if not ollama_running:
    raise RuntimeError("Ollama not running. Launch ollama before proceeding.")
print("Ollama running:", check_if_running("ollama"))


# An alternative to the ollama run command for interacting with the model is through its REST API using Python.

# The following query_model function demonstrates how to use the API:
# Step 1: Create the data payload as a dictionary
# Step 2: Convert the dictionary to a JSON formatted string and encode it to bytes
# Step 3: Create a request object, setting the method to POST and adding necessary headers
# Step 4: Send the request and capture the response
import urllib.request

def query_model(
    prompt,
    model="llama3",
    url="http://localhost:11434/api/chat"
):
    # Create the data payload as a dictionary
    data = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "options": {     # Settings below are required for deterministic responses
            "seed": 123,
            "temperature": 0,
            "num_ctx": 2048
        }
    }


    # Convert the dictionary to a JSON formatted string and encode it to bytes
    payload = json.dumps(data).encode("utf-8")

    # Create a request object, setting the method to POST and adding necessary headers
    request = urllib.request.Request(
        url,
        data=payload,
        method="POST"
    )
    request.add_header("Content-Type", "application/json")

    # Send the request and capture the response
    response_data = ""
    with urllib.request.urlopen(request) as response:
        # Read and decode the response
        while True:
            line = response.readline().decode("utf-8")
            if not line:
                break
            response_json = json.loads(line)
            response_data += response_json["message"]["content"]

    return response_data


# Here's an example of how to use the query_llama function we just implemented:
model = "llama3"
result = query_model("What do Llamas eat?", model)
print(result)

# First, we apply this approach to the first three examples from the test set that we examined:
for entry in test_data[:3]:
    prompt = (
        f"Given the input `{format_input(entry)}` "
        f"and correct output `{entry['output']}`, "
        f"score the model response `{entry['model_response']}`"
        f" on a scale from 0 to 100, where 100 is the best score. "
    )
    print("\nDataset response:")
    print(">>", entry['output'])
    print("\nModel response:")
    print(">>", entry["model_response"])
    print("\nScore:")
    print(">>", query_model(prompt))
    print("\n-------------------------")



# The following generate_model_scores function uses a modified the prompt telling the model to "Respond with the integer number only.":
def generate_model_scores(json_data, json_key, model="llama3"):
    scores = []
    for entry in tqdm(json_data, desc="Scoring entries"):
        prompt = (
            f"Given the input `{format_input(entry)}` "
            f"and correct output `{entry['output']}`, "
            f"score the model response `{entry[json_key]}`"
            f" on a scale from 0 to 100, where 100 is the best score. "
            f"Respond with the integer number only."
        )
        score = query_model(prompt, model)
        try:
            scores.append(int(score))
        except ValueError:
            print(f"Could not convert score: {score}")
            continue

    return scores


# To further improve our model's performance, we can explore various strategies, such as:

# (1) Adjusting the hyperparameters during finetuning, such as the learning rate, batch size, or number of epochs.

# (2) Increasing the size of the training dataset or diversifying the examples to cover a broader range of topics and styles.

# (3) Experimenting with different prompts or instruction formats to guide the model's responses more effectively.

# (4) Considering the use of a larger pretrained model, which may have greater capacity to capture complex patterns and generate more accurate responses.

# (5) We can also use parameter efficient fine-tuning techniques like LoRA.