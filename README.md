<img src="assets/banner.png" alt="MIKLIUM LM" width="100%">

# MIKLIUM LM
## miklium-lm-mini

MIKLIUM LM is an advanced large language model developed by OpenAGI for the MIKLIUM ecosystem. We are introducing **miklium-lm-mini**, the most compact and accessible model within the MIKLIUM LM family, designed for rapid deployment and experimentation, this model supports chain of thought reasoning.

## About the Model

**Total Parameters:** 1.73M

The model architecture and training pipeline are built entirely using Python. 

As a **foundation model**, `miklium-lm-mini` is currently pre-trained on a small, curated dataset. While it demonstrates capable foundational reasoning, unlocking its full potential requires further re-training and  fine-tuning, feeding it expanded training datasets, and training across additional epochs. 

This repository includes not only the model itself but also a hands-on, beautifully designed web interface to chat directly with the model and experience its capabilities firsthand.

## Live Demo

Experience `miklium-lm-mini` in action on our official web interface:
[Test the Model Here](https://aaa-g.github.io/miklium-lm-mini/website/index)

## Sample Interaction

Here is an example demonstrating the prompt format and the reasoning capabilities of the model:

```text
Prompt: <user> What is the capital of France?
Response: 
<ai> <think> The user is asking for the capital of France. The capital of France is Paris. <eos>
```

## License

This model is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.