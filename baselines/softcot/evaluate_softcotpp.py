
import re
import argparse

from tqdm import tqdm
import torch

from transformers import AutoTokenizer, GenerationConfig
from fastNLP import logger

from llm_model import ScalingEfficientSoftCoTFromSmallModel
from data_loader import GSM8KLoader, StrategyQALoader, AugASDivLoader, AQuALoader, DULoader
from utils import pre_process_gsm8k, pre_process_strategy_qa, pre_process_aqua, pre_process_du


args = argparse.ArgumentParser()
args.add_argument('--base_model_id', type=str, default='Qwen/Qwen3-8B')
args.add_argument('--assistant_model_id', type=str, default='Qwen/Qwen3-0.6B')
args.add_argument('--params_file_name', type=str, default=None)
args.add_argument('--base_model_ckpt', type=str, default=None)
args.add_argument('--assistant_model_ckpt', type=str, default=None)
args.add_argument('--num_thought_tokens', type=int, default=2)
args.add_argument('--num_scaling_times', type=int, default=10)
args.add_argument('--num_return_sequences', type=int, default=1)
args.add_argument('--task_name', type=str, choices=[
    'gsm8k', 'strategyqa', 'asdiv-aug', 'aqua', 'du'
])
args.add_argument('--model_name', type=str, default='soft', choices=[
    'scaling'
])
args.add_argument('--print_input', action='store_true', default=False)
args.add_argument('--print_response', action='store_true', default=False)
args.add_argument('--test_k', type=int, default=0)
args.add_argument('--seed', type=int, default=42)
args.add_argument('--tune_base_model', action='store_true', default=False)
args.add_argument('--tune_assistant_model', action='store_true', default=False)
arg = args.parse_args()
logger.info(f'Args: {arg.__dict__}')

base_model_id = arg.base_model_id
assistant_model_id = arg.assistant_model_id
params_file_name = arg.params_file_name
base_model_ckpt = arg.base_model_ckpt
assistant_model_ckpt = arg.assistant_model_ckpt
num_thought_tokens = arg.num_thought_tokens
num_scaling_times = arg.num_scaling_times
num_return_sequences = arg.num_return_sequences
task_name = arg.task_name
model_name = arg.model_name
print_input = arg.print_input
print_response = arg.print_response
test_k = arg.test_k
seed = arg.seed
tune_base_model = arg.tune_base_model
tune_assistant_model = arg.tune_assistant_model

large_model_name = base_model_id.split('/')[-1]
small_model_name = assistant_model_id.split('/')[-1]

if base_model_ckpt in ['None']:
    base_model_ckpt = None
if assistant_model_ckpt in ['None']:
    assistant_model_ckpt = None

model_dtype = torch.bfloat16
param_dtype = str(model_dtype)

base_tokenizer = AutoTokenizer.from_pretrained(base_model_id, token='your-huggingface-token')
assistant_tokenizer = AutoTokenizer.from_pretrained(assistant_model_id, token='your-huggingface-token')

if 'Llama' in base_model_id:
    base_special_token = ['<|end_of_text|>', '<|reserved_special_token_0|>', '<|reserved_special_token_1|>']
    base_backbone = 'llama'
elif 'Qwen2.5' in base_model_id:
    base_special_token = ['<|endoftext|>', '<|box_start|>', '<|box_end|>']
    base_backbone = 'qwen'
elif 'Qwen3' in base_model_id:
    base_special_token = ['<|endoftext|>', '<think>', '</think>']
    base_backbone = 'qwen3'
else:
    raise NotImplementedError
if 'Llama' in assistant_model_id:
    assistant_special_token = ['<|end_of_text|>', '<|reserved_special_token_0|>', '<|reserved_special_token_1|>']
    assistant_backbone = 'llama'
elif 'Qwen2.5' in assistant_model_id:
    assistant_special_token = ['<|endoftext|>', '<|box_start|>', '<|box_end|>']
    assistant_backbone = 'qwen'
elif 'Qwen3' in assistant_model_id:
    assistant_special_token = ['<|endoftext|>', '<think>', '</think>']
    assistant_backbone = 'qwen3'
else:
    raise NotImplementedError

if model_name in ['scaling']:
    model = ScalingEfficientSoftCoTFromSmallModel(
        assistant_model_id,
        base_model_id,
        num_thought_tokens,
        tune_base_model=tune_base_model,
        tune_assistant_model=tune_assistant_model,
        path_to_projection_module=params_file_name,
        path_to_small_language_model=assistant_model_ckpt,
        num_scaling_times=num_scaling_times,
    )
else:
    raise NotImplementedError
logger.info(f'Successfully Init Model `{model.__class__.__name__}`')
model.eval()
model.assistant_model.eval()
model.base_model.eval()

if task_name in ['gsm8k']:
    db = GSM8KLoader().load()
    preprocess_method = pre_process_gsm8k
elif task_name in ['strategyqa']:
    db = StrategyQALoader().load()
    preprocess_method = pre_process_strategy_qa
elif task_name in ['asdiv-aug']:
    db = AugASDivLoader().load()
    preprocess_method = pre_process_gsm8k
elif task_name in ['aqua']:
    db = AQuALoader().load()
    preprocess_method = pre_process_aqua
elif task_name in ['du']:
    db = DULoader().load()
    preprocess_method = pre_process_du
else:
    raise NotImplementedError

ds = db.get_dataset('test')

if test_k > 0:
    ds = ds[: test_k]

generation_config = GenerationConfig.from_pretrained(base_model_id)
if base_backbone in ['llama']:
    generation_config.pad_token_id = 128009
elif base_backbone in ['qwen', 'qwen3']:
    generation_config.pad_token_id = 151643
else:
    raise NotImplementedError
generation_config.top_p = 1.0
generation_config.temperature = 1.0

correct_count = 0
avg_token = 0
for idx, ins in enumerate(tqdm(ds)):

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    if task_name in ['gsm8k', 'asdiv-aug', 'aqua']:
        answer = ins['answer'].split('\n')[-1]
        assert answer.startswith('####')
        answer = answer.replace(',', '')
        if task_name in ['gsm8k', 'asdiv-aug']:
            if '.' in answer:
                answer = float(answer[4:])
            else:
                answer = int(answer[4:])
        else:
            answer = answer[4:].strip()
    elif task_name in ['strategyqa', 'du']:
        answer = ins['answer']
    else:
        raise NotImplementedError

    logger.info(f'Ground Truth Answer: {answer}')

    inputs = preprocess_method(
        ins, base_tokenizer, assistant_tokenizer, num_thought_tokens,
        add_bot_eot=(num_thought_tokens > 0), split='test',
        base_special_token=base_special_token,
        assistant_special_token=assistant_special_token,
        base_backbone=base_backbone,
        assistant_backbone=assistant_backbone,
        device=model.device,
    )
    if print_input:
        logger.info(f'Raw Inputs for Base Model: {base_tokenizer.decode(inputs["input_ids"][0])}')
        # logger.info(f'Raw Inputs for Assistant Model: {assistant_tokenizer.decode(inputs["assistant_input_ids"][0])}')

    terminators = [
        base_tokenizer.eos_token_id,
    ]
    if base_backbone in ['llama']:
        terminators.append(base_tokenizer.convert_tokens_to_ids("<|eot_id|>"))

    model_answer_list = []
    model_answer_count = {}

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    original_inputs_embeds = model.base_model.get_input_embeddings()(inputs['input_ids'])

    torch.manual_seed(41)
    torch.cuda.manual_seed(41)

    inputs_embeds_list = model.get_inputs_embeds_for_base_model(
        inputs['assistant_input_ids'],
        inputs['assistant_attention_mask'],
        inputs['input_ids'],
        original_inputs_embeds,
        inputs['thought_index'],
        print_input,
    )

    output_list = []

    max_new_tokens = 1024
    if base_backbone in ['qwen3']:
        max_new_tokens = 4096
        if task_name in ['du', 'strategyqa']:
            max_new_tokens /= 4

    if num_return_sequences > 1:
        for inputs_embeds in inputs_embeds_list:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            outputs = model.base_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=inputs['attention_mask'],
                max_new_tokens=max_new_tokens,
                eos_token_id=terminators,
                do_sample=True,
                generation_config=generation_config,
                num_return_sequences=num_return_sequences,
            )
            output_list.append(outputs)
    else:
        all_inputs_embeds = torch.cat(inputs_embeds_list, dim=0)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        outputs = model.base_model.generate(
            inputs_embeds=all_inputs_embeds,
            attention_mask=inputs['attention_mask'],
            max_new_tokens=max_new_tokens,
            eos_token_id=terminators,
            do_sample=True,
            generation_config=generation_config,
            num_return_sequences=num_return_sequences,
        )
        output_list.append(outputs)


    for i, outputs in enumerate(output_list):
        for j in range(outputs.shape[0]):
            response = outputs[j]
            avg_token += response.shape[0] / outputs.shape[0] / len(output_list)
            raw_model_answer = base_tokenizer.decode(response, skip_special_tokens=True)

            if print_response:
                logger.info(f'Answer ({idx + 1}-{i + 1}-{j + 1}/{len(ds)}): {base_tokenizer.decode(response)}<|end-of-response|>')

            if task_name in ['gsm8k', 'asdiv-aug']:
                cleaned_model_answer = raw_model_answer.replace(',', '')
                cleaned_model_answer = cleaned_model_answer.replace('%', '')
                cleaned_model_answer = cleaned_model_answer.replace('$', '')
            else:
                cleaned_model_answer = raw_model_answer

            match = re.findall(r'\s*([\d,]+(?:\.\d+)?)\s*', cleaned_model_answer)

            if task_name in ['gsm8k', 'asdiv-aug']:
                try:
                    if match:
                        last_match = match[-1]
                        cleaned_match = last_match.replace(',', '')
                        cleaned_match = cleaned_match.replace('%', '')
                        cleaned_match = cleaned_match.replace('$', '')
                        if '.' in cleaned_match:
                            model_answer = round(float(cleaned_match), 2)
                        else:
                            model_answer = int(cleaned_match)
                    else:
                        model_answer = None
                    if model_answer is None and not print_response:
                        logger.info(f'None Model Answer ({idx + 1}-{i + 1}-{j + 1}/{len(ds)}): {base_tokenizer.decode(response)}')
                except Exception as e:
                    model_answer = None
                    logger.error(f'Error: {e}')
            elif task_name in ['strategyqa']:
                last_yes = re.search(r'\bsey\b', raw_model_answer.lower()[::-1])
                if last_yes is not None:
                    last_yes = last_yes.start()
                else:
                    last_yes = len(raw_model_answer)
                last_no = re.search(r'\bon\b', raw_model_answer.lower()[::-1])
                if last_no is not None:
                    last_no = last_no.start()
                else:
                    last_no = len(raw_model_answer)
                if last_yes == last_no == len(raw_model_answer):
                    model_answer = None
                else:
                    model_answer = last_yes < last_no
            elif task_name in ['aqua', 'du']:
                m_answer = re.search(r'\b[a-f]\b', raw_model_answer.lower()[::-1])
                if m_answer is not None:
                    model_answer = m_answer.group(0).upper()
                else:
                    model_answer = None
            else:
                raise NotImplementedError

            model_answer_list.append(model_answer)
            if model_answer in model_answer_count and model_answer is not None:
                model_answer_count[model_answer] += 1
            else:
                model_answer_count[model_answer] = 1

    max_model_count = 0
    final_model_answer = None

    for k, v in model_answer_count.items():
        if v > max_model_count:
            final_model_answer = k
            max_model_count = v

    logger.info(f'Ground Truth Answer: {answer}')
    logger.info(f'Model Answer: {model_answer_list}')
    logger.info(f'Model Answer Count: {model_answer_count}')
    logger.info(f'Final Model Answer: {final_model_answer}')
    is_correct = (final_model_answer == answer)
    logger.info(f'Is Correct: {is_correct}')
    if is_correct:
        correct_count += 1
    logger.info(f'Average generated tokens ({idx + 1} instances): {round(avg_token / (idx + 1), 2)}')
    logger.info(f'Correct Count: {correct_count}/{idx + 1}')
    logger.info(f'{"-" * 20}')
