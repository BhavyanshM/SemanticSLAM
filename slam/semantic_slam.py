import torch
from datasets import load_dataset
from transformers import BertTokenizer, BertModel, pipeline

def dataset_main():
    wikitext_dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")

    unmasker = pipeline('fill-mask', model=model, tokenizer=tokenizer)

    # Get the 11th example from the training set (index 10)
    input = wikitext_dataset["train"][10]
    print(input['text'])


    # Tokenize the input
    encoded_input = tokenizer(input["text"], return_tensors="pt")

    # Mask the 6th token (index 5)
    masked_index = 5
    encoded_input["input_ids"][0, masked_index] = tokenizer.mask_token_id

    # Print the masked input
    print(tokenizer.decode(encoded_input["input_ids"][0]))

    # Run the model to unmask the token
    output = model(**encoded_input)

    # Get the logits for the masked token
    print(output.last_hidden_state.shape, output.pooler_output.shape)

    predictions = output.last_hidden_state

    print(predictions[0, masked_index].shape)

    probs = torch.nn.functional.softmax(predictions[0, masked_index], dim=-1)

    top_k_weights, top_k_indices = torch.topk(probs, 5, sorted=True)

    for i, pred_idx in enumerate(top_k_indices):
        predicted_token = tokenizer.convert_ids_to_tokens([pred_idx])[0]
        token_weight = top_k_weights[i]
        print("[MASK]: '%s'"%predicted_token, " | weights:", float(token_weight))


