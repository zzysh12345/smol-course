# Generating Preference Datasets

Within [the chapter on preference alignment](../2_preference_alignment/README.md) we learned about preference alignment. In this section we will explore how to generate preference datasets for preference alignment. We will built on top of the methods that have been introduced in [generating instruction datasets](./instruction_datasets.md). On top of that, we will show how to add extra completions to the dataset using basic prompting or by using EvolQuality to improve the quality of responses. Lastly, we will show how UltraFeedback can be used to generate scores and critiques.

## Prompting for a Second Opinion

Preference data is a dataset with multiple `completions`. We can add more `completions` to a dataset by prompting a model to generate them. When doing this we need to ensure that the second completion is not too similar to the first completion in terms overall quality and phrasing. This is important because the models needs to be optimised for a clear preference.

 Each `completion` is a response to a `prompt`, where we know which completion is preferred over the other, normally denoted as `chosen` and `rejected`. The score is a measure of how much one response is preferred over the other. In general these scores can be absolute, subjective, or relative. For this course we will focus on the first two, because they are most valuable for creating preference datasets.

### Model pooling

You can use models from different model-families to generate a second completion, which is called model pooling. To further improve the quality of the second completion, you can use using different generation arguments, like tweaking the `temperature`. Lastly, you can use different prompt templates or system prompts to generate a second completion with ensure diversity based on specific characteristics defined in the template. In theory, we could take two models of varying quality and use the better one as the `chosen` completion.

Let's start with model pooling by loading the [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) and [HuggingFaceTB/SmolLM2-1.7B-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct) models using the `transformers` integration of the `distilabel` library. Using this model, we will create two synthetic `responses` for a given `prompt`. We will create another pipeline with `LoadDataFromDicts`, `TextGeneration`, and `GroupColumns`. We wil first load data, then use two generation steps, and then group the results. We connect the steps and flow the data through the pipeline using the `>>` operator and `[]`, which means that we want to use the output of the previous step as the input for both steps within the list.

```python
from distilabel.llms import TransformersLLM
from distilabel.pipeline import Pipeline
from distilabel.steps import GroupColumns, LoadDataFromDicts
from distilabel.steps.tasks import TextGeneration

with Pipeline() as pipeline:
    data = LoadDataFromDicts(data=[{"instruction": "What is synthetic data?"}])
    llm_a = TransformersLLM(model="HuggingFaceTB/SmolLM2-1.7B-Instruct")
    gen_a = TextGeneration(llm=llm_a)
    llm_b = TransformersLLM(model="Qwen/Qwen2.5-1.5B-Instruct")
    gen_b = TextGeneration(llm=llm_b)
    group = GroupColumns(columns=["generation"])
    data >> [gen_a, gen_b] >> group

if __name__ == "__main__":
    distiset = pipeline.run()
    print(distiset["default"]["train"]["grouped_generation"][0])
# {[
#   'Synthetic data is artificially generated data that mimics real-world usage.',
#   'Synthetic data refers to data that has been generated artificially.'
# ]}
```

As you can see, we have two synthetic `completions` for the given `prompt`. We could have boosted diversity by initialising the `TextGeneration` steps with specific `system_prompt` or by passing generation arguments to the `TransformersLLM`. Since we are familiar with the concept of a pipeline, we will now park this

### EvolQuality

EvolQuality is similar to the [EvolInstruct](./instruction_datasets.md#evolinstruct)  a prompting technique but it evolves `completions` instead of the input `prompt`. The task takes both a `prompt` and `completion` and evolves the `completion` into a version that is better at responding to the `prompt` based on a set of criteria. This better version is defined according to a set of criteria by improving helpfulness, relevance, deepening, creativity, or details. Because this automatically generates a second completion, we can use it to add more `completions` to a dataset. In theory, we could even assume the evolution is better than the original completion and use it as the `chosen` completion out of the box.

 The prompt is [implemented in distilabel](https://github.com/argilla-io/distilabel/tree/main/src/distilabel/steps/tasks/evol_quality) and a simplified version is shown below:

```bash
I want you act as a Response Rewriter.
Given prompt a and a response, rewrite the response into a better version.
Complicate the prompt based on the following criteria:
{{ criteria }}

# Prompt
{{ input }}

# Response
{{ output }}

# Improved Response
```

To use it, we need to pass the `llm` to the [EvolQuality class](https://distilabel.argilla.io/dev/components-gallery/tasks/evolquality/). Let's use the synthetic `prompt` and `completion` from [the Model Pooling section](#model-pooling) as input and evolve it into a better version. For this example, we will only evolve for one generation.

```python
from distilabel.llms import TransformersLLM
from distilabel.steps.tasks import EvolQuality

llm = TransformersLLM(model="HuggingFaceTB/SmolLM2-1.7B-Instruct")
evol_quality = EvolQuality(llm=llm, num_evolutions=1)
evol_quality.load()

instruction = "What is synthetic data?"
completion = "Synthetic data is artificially generated data that mimics real-world usage."

next(evol_quality.process([{
    "instruction": instruction,
    "response": completion
}]))
# The process of generating synthetic data through manual prompting involves creating artificial data sets that mimic real-world usage patterns.
```

The `response` is now more complex and specific to the `instruction`. This is a good start but as we have seen with EvolInstruct, evolved generations are not always better. Hence it is important to use additional evaluation techniques to ensure the quality of the dataset. We will explore this in the next section.

## Creating Scores

### UltraFeedback


