from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
import copy
import os
import json
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from datasets import Dataset, load_dataset
from peft import AutoPeftModelForCausalLM, LoraConfig, get_peft_model, PeftConfig, PeftModel

from transformers import (
    AutoTokenizer, 
    HfArgumentParser, 
    TrainingArguments, 
    BitsAndBytesConfig, 
    AutoModelForCausalLM,
    AutoModelForSequenceClassification, 
    pipeline, 
)

from trl import DPOTrainer, PPOTrainer, PPOConfig
from trl.core import LengthSampler
from accelerate import Accelerator

from custom_reward import RewardConfig, MultiRewardTrainer

@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """

    # data parameters
    beta: Optional[float] = field(default=0.1, metadata={"help": "the beta parameter for DPO loss"})

    # training parameters
    model_name_or_path: Optional[str] = field(
        default="meta-llama/Llama-2-7b-chat-hf",
        metadata={"help": "the location of the SFT model name or path"},
    )
    dataset_name: Optional[str] = field(default="Anthropic/hh-rlhf", metadata={"help": "the dataset name for training"})
    eval_datasets: Optional[str] = field(default="hellaswag", metadata={"help": "the dataset name for eval"})
    dataset_text_field: Optional[str] = field(default="text", metadata={"help": "the text field of the dataset"})
    
    learning_rate: Optional[float] = field(default=5e-4, metadata={"help": "optimizer learning rate"})
    lr_scheduler_type: Optional[str] = field(default="cosine", metadata={"help": "the lr scheduler type"})
    warmup_steps: Optional[int] = field(default=100, metadata={"help": "the number of warmup steps"})
    weight_decay: Optional[float] = field(default=0.05, metadata={"help": "the weight decay"})
    optimizer_type: Optional[str] = field(default="paged_adamw_32bit", metadata={"help": "the optimizer type"})

    per_device_train_batch_size: Optional[int] = field(default=4, metadata={"help": "train batch size per device"})
    per_device_eval_batch_size: Optional[int] = field(default=1, metadata={"help": "eval batch size per device"})
    gradient_accumulation_steps: Optional[int] = field(
        default=4, metadata={"help": "the number of gradient accumulation steps"}
    )
    gradient_checkpointing: Optional[bool] = field(
        default=False, metadata={"help": "whether to use gradient checkpointing"}
    )

    load_in_8bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 8 bits precision"})
    load_in_4bit: Optional[bool] = field(default=True, metadata={"help": "load the model in 4 bits precision"})
    use_peft: Optional[bool] = field(default=True, metadata={"help": "Wether to use PEFT or not to train adapters"})
    lora_alpha: Optional[float] = field(default=16, metadata={"help": "the lora alpha parameter"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "the lora dropout parameter"})
    lora_r: Optional[int] = field(default=128, metadata={"help": "the lora r parameter"})
    
    max_prompt_length: Optional[int] = field(default=512, metadata={"help": "the maximum prompt length"})
    max_seq_length: Optional[int] = field(default=1024, metadata={"help": "the maximum sequence length"})
    num_rewards: Optional[int] = field(default=1, metadata={"help": "the number of reward heads"})

    num_train_epochs: Optional[int] = field(
        default=1, metadata={"help": "Number of training epochs"}
    )
    max_steps: Optional[int] = field(default=-1, metadata={"help": "max number of training steps (override epochs)"})
    logging_steps: Optional[int] = field(default=10, metadata={"help": "the logging frequency"})
    save_steps: Optional[int] = field(default=100, metadata={"help": "the saving frequency"})
    eval_steps: Optional[int] = field(default=100, metadata={"help": "the evaluation frequency"})

    output_dir: Optional[str] = field(default="results", metadata={"help": "the output directory"})
    log_freq: Optional[int] = field(default=1, metadata={"help": "the logging frequency"})
    trust_remote_code: Optional[bool] = field(default=True, metadata={"help": "Enable `trust_remote_code`"})

    # instrumentation
    sanity_check: Optional[bool] = field(default=False, metadata={"help": "only train on 10 samples"})
    report_to: Optional[str] = field(
        default="none",
        metadata={
            "help": 'The list of integrations to report the results and logs to. Supported platforms are `"azure_ml"`,'
            '`"comet_ml"`, `"mlflow"`, `"neptune"`, `"tensorboard"`,`"clearml"` and `"wandb"`. '
            'Use `"all"` to report to all integrations installed, `"none"` for no integrations.'
        },
    )
    
    # debug argument for distributed training
    ignore_bias_buffers: Optional[bool] = field(
        default=False,
        metadata={
            "help": "fix for DDP issues with LM bias/mask buffers - invalid scalar type,`inplace operation. See"
            "https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992"
        },
    )

    # AL parameters
    rm_freeze_env: Optional[bool] = field(default=False, metadata={"help": "wheather to freeze encoder in reward model or not"})
    init_samples: Optional[int] = field(default=1000, metadata={"help": "number of inital samples for warming up reward model"})
    bo_iters: Optional[int] = field(default=20, metadata={"help": "number of BO iterations"})
    topk_acqf: Optional[int] = field(default=1000, metadata={"help": "number of acquistion samples in each bo iteration"})
    algo: Optional[str] = field(default="max_rw", metadata={"help": "Acquistion function"})
    
parser = HfArgumentParser(ScriptArguments)
script_args, _ = parser.parse_args_into_dataclasses(return_remaining_strings=True)

######################################################
################## CONFIGURATIONS ###################
######################################################

if script_args.load_in_8bit and script_args.load_in_4bit:
    raise ValueError("You can't load the model in 8 bits and 4 bits at the same time")
elif script_args.load_in_8bit or script_args.load_in_4bit:
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=script_args.load_in_8bit,
        load_in_4bit=script_args.load_in_4bit,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=False,
    )
    # Copy the model to each device
    device_map = {"": Accelerator().local_process_index}
else:
    device_map = None
    quantization_config = None
print(quantization_config)

if script_args.use_peft:
    peft_config = LoraConfig(
        r=script_args.lora_r,
        lora_alpha=script_args.lora_alpha,
        lora_dropout=script_args.lora_dropout,
        target_modules=[
            "q_proj",
            "v_proj",
            "k_proj",
            "out_proj",
            "fc_in",
            "fc_out",
            "wte",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )
    peft_config_rw = LoraConfig(
        r=script_args.lora_r,
        lora_alpha=script_args.lora_alpha,
        lora_dropout=script_args.lora_dropout,
        bias="none",
        task_type="SEQ_CLS",
        modules_to_save=["scores"]
    )
else:
    peft_config = None
    peft_config_rw = None
    
######################################################
################## GENERATOR MODEL ###################
######################################################

def get_generator(
    model_name_or_path,
    quantization_config,
    device_map,
    script_args
):
    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        low_cpu_mem_usage=True,
        quantization_config=quantization_config,
        device_map=device_map,
        trust_remote_code=script_args.trust_remote_code
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    model.config.pad_token_id = model.config.eos_token_id

    if script_args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]
    return model

######################################################
################## REWARD MODEL ######################
######################################################

def get_rw_model(
    model_name_or_path,
    quantization_config,
    device_map,
    script_args
): 
    rw_model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        low_cpu_mem_usage=True,
        quantization_config=quantization_config,
        device_map=device_map,
        trust_remote_code=script_args.trust_remote_code,
        num_labels=script_args.num_rewards
    )
    rw_model.config.use_cache = False
    rw_model.config.pretraining_tp = 2
    rw_model.config.pad_token_id = rw_model.config.eos_token_id


    # JUST FOR GPT2
    # rw_model.config.pad_token_id = model.config.eos_token_id

    if script_args.rm_freeze_env:
        # freeze_layers = ['transformer'] # JUST FOR GPT2
        freeze_layers = ['model']
        for name, param in rw_model.named_parameters():
            freeze = sum([l in name for l in freeze_layers])
            if freeze > 0:
                param.requires_grad = False
            
    return rw_model


def get_rw_model_with_adapter(
    base_model_name_or_path,
    peft_adapter_path,
    quantization_config,
    device_map,
    script_args
): 
    rw_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name_or_path,
        low_cpu_mem_usage=True,
        quantization_config=quantization_config,
        device_map=device_map,
        trust_remote_code=script_args.trust_remote_code,
        num_labels=script_args.num_rewards
    )

    rw_model.config.use_cache = False
    rw_model.config.pretraining_tp = 2
    rw_model.config.pad_token_id = rw_model.config.eos_token_id

    merge_rw_model = PeftModel.from_pretrained(rw_model, peft_adapter_path)

    # JUST FOR GPT2
    # rw_model.config.pad_token_id = model.config.eos_token_id

    for name, param in merge_rw_model.named_parameters():
        if 'score' in name:  #  'score' is the name of the last layer
            param.requires_grad = True
        else:
            param.requires_grad = False
            
    return merge_rw_model

######################################################
#################### TOKENIZER #######################
######################################################

tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path)
tokenizer.pad_token = tokenizer.eos_token

######################################################

######################################################
################# DATASET FUNCTIONS ##################
######################################################

def preprocess_function_reward(examples):
    new_examples = {
        "input_ids_chosen": [],
        "attention_mask_chosen": [],
        "input_ids_rejected": [],
        "attention_mask_rejected": [],
    }
    for chosen, rejected in zip(examples["chosen"], examples["rejected"]):
        tokenized_chosen = tokenizer(chosen, truncation=True)
        tokenized_rejected = tokenizer(rejected, truncation=True)

        new_examples["input_ids_chosen"].append(tokenized_chosen["input_ids"])
        new_examples["attention_mask_chosen"].append(tokenized_chosen["attention_mask"])
        new_examples["input_ids_rejected"].append(tokenized_rejected["input_ids"])
        new_examples["attention_mask_rejected"].append(tokenized_rejected["attention_mask"])

    return new_examples

def get_dataset_for_reward(
    train_dataset,
    eval_dataset,
    max_seq_length: int,
    sanity_check: bool = False,
    num_proc=24
):
    if sanity_check:
        train_dataset = train_dataset.select(range(min(len(train_dataset), 10)))
        eval_dataset = eval_dataset.select(range(min(len(eval_dataset), 10)))
        
    # Preprocess the dataset and filter out examples that are longer than script_args.max_length
    train_dataset_processed = train_dataset.map(
        preprocess_function_reward,
        batched=True,
        num_proc=num_proc,
    )
    train_dataset_processed = train_dataset_processed.filter(
        lambda x: len(x["input_ids_chosen"]) <= max_seq_length
        and len(x["input_ids_rejected"]) <= max_seq_length
    )

    eval_dataset_processed = eval_dataset.map(
        preprocess_function_reward,
        batched=True,
        num_proc=num_proc,
    )
    eval_dataset_processed = eval_dataset_processed.filter(
        lambda x: len(x["input_ids_chosen"]) <= max_seq_length
        and len(x["input_ids_rejected"]) <= max_seq_length
    )
    
    return train_dataset_processed, eval_dataset_processed

def preprocess_function_finetuning(samples) -> Dict[str, str]:
    return {
        "prompt": [sample.split("Assistant: ")[0] + "Assistant: " for sample in samples["chosen"]],
        "chosen": [sample.split("Assistant: ")[-1] for sample in samples["chosen"]],
        "rejected": [sample.split("Assistant: ")[-1] for sample in samples["rejected"]],
    }
    
def get_dataset_for_finetuning(
    train_dataset,
    eval_dataset,
    max_seq_length: int,
    sanity_check: bool = False,
    num_proc=24,
) -> Dataset:
    """Load the stack-exchange-paired dataset from Hugging Face and convert it to the necessary format.

    The dataset is converted to a dictionary with the following structure:
    {
        'prompt': List[str],
        'chosen': List[str],
        'rejected': List[str],
    }

    Prompts are structured as follows:
      "Question: " + <prompt> + "\n\nAnswer: "
    """
    original_columns = train_dataset.column_names

    if sanity_check:
        train_dataset = train_dataset.select(range(min(len(train_dataset), 10)))
        eval_dataset = eval_dataset.select(range(min(len(eval_dataset), 10)))

    train_dataset_processed = train_dataset.map(
        preprocess_function_finetuning,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns,
    )

    train_dataset_processed = train_dataset_processed.filter(
        lambda x: len(x["prompt"]) + len(x["chosen"]) <= max_seq_length
        and len(x["prompt"]) + len(x["rejected"]) <= max_seq_length
    )
    
    eval_dataset_processed = eval_dataset.map(
        preprocess_function_finetuning,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns,
    )

    eval_dataset_processed = eval_dataset_processed.filter(
        lambda x: len(x["prompt"]) + len(x["chosen"]) <= max_seq_length
        and len(x["prompt"]) + len(x["rejected"]) <= max_seq_length
    )
    
    return train_dataset_processed, eval_dataset_processed

######################################################
##################### RUNNERS ########################
######################################################
def run_reward_training(
    model, 
    tokenizer,
    train_dataset,
    eval_dataset,
    peft_config,
    script_args
):
    output_dir = os.path.join(script_args.output_dir, "reward_model")
    
    training_args = RewardConfig(
        output_dir=output_dir,
        max_length=script_args.max_seq_length,
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        num_train_epochs=script_args.num_train_epochs,
        max_steps=script_args.max_steps,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        gradient_checkpointing=script_args.gradient_checkpointing,
        learning_rate=script_args.learning_rate,
        report_to=script_args.report_to,
        remove_unused_columns=False,
        optim="adamw_torch",
        logging_steps=script_args.logging_steps,
        # evaluation_strategy="steps" if script_args.eval_split != "none" else "no",
        # eval_steps=None if script_args.eval_split == "none" else 30000,
    )

    rw_trainer = MultiRewardTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        # peft_config=peft_config,
    )
    
    rw_trainer.train()
    rw_trainer.model.save_pretrained(output_dir)

def run_dpo_finetuning(
    model, 
    model_ref, 
    tokenizer,
    train_dataset,
    eval_dataset,
    peft_config,
    script_args
):
    output_dir = os.path.join(script_args.output_dir, "generator_model")
    
    training_args = TrainingArguments(
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        per_device_eval_batch_size=script_args.per_device_eval_batch_size,
        num_train_epochs=script_args.num_train_epochs,
        max_steps=script_args.max_steps,
        logging_steps=script_args.logging_steps,
        # save_steps=script_args.save_steps,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        gradient_checkpointing=script_args.gradient_checkpointing,
        learning_rate=script_args.learning_rate,
        evaluation_strategy="steps",
        eval_steps=script_args.eval_steps,
        output_dir=output_dir,
        lr_scheduler_type=script_args.lr_scheduler_type,
        warmup_steps=script_args.warmup_steps,
        optim=script_args.optimizer_type,
        bf16=False,
        remove_unused_columns=False,
        run_name="dpo_llama2",
        report_to=script_args.report_to
    )
    
    if peft_config:
        # ValueError: You passed both a ref_model and a peft_config. 
        # For training PEFT adapters with DPO there is no need to pass a reference model. 
        # Please pass `ref_model=None` in case you want to train PEFT adapters, or pass a ref_model with `force_use_ref_model=True` in DPOTrainer's init. if you want to use a different ref_model.
        model_ref = None
    
    # model = get_peft_model(model, peft_config)
    # model.config.use_cache = False
        
    dpo_trainer = DPOTrainer(
        model,
        model_ref, # None if peft_config
        args=training_args,
        beta=script_args.beta,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
        max_prompt_length=script_args.max_prompt_length,
        max_length=script_args.max_seq_length
    )
    
    dpo_trainer.train()
    dpo_trainer.model.save_pretrained(output_dir)

if __name__ == '__main__':
    # LOAD DATASETS
    full_dataset = load_dataset(
        script_args.dataset_name,
        split="train"
    )

    train_dataset = copy.deepcopy(full_dataset[:script_args.init_samples])
    train_dataset = Dataset.from_dict(train_dataset)

    if script_args.sanity_check:
        dataset_endidx = script_args.init_samples + 100
    else:
        dataset_endidx = None
    unobserved_dataset = copy.deepcopy(full_dataset[script_args.init_samples:dataset_endidx])
    unobserved_dataset = Dataset.from_dict(unobserved_dataset)

    del full_dataset
    
    eval_dataset = load_dataset(
        script_args.dataset_name,
        split="test"
    )

    # WARM-UP LLM (GENERATOR)
    print("="*10, " WARMING UP GENERATOR ", "="*10)
    train_dataset_lm, eval_dataset_lm = get_dataset_for_finetuning(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        max_seq_length=script_args.max_seq_length,
        sanity_check=script_args.sanity_check
    )
    
    model = get_generator(
        script_args.model_name_or_path,
        quantization_config,
        device_map,
        script_args
    )
    model_ref = copy.deepcopy(model) if peft_config else None

    run_dpo_finetuning(
        model = model,
        model_ref = model_ref,
        tokenizer = tokenizer,
        train_dataset = train_dataset_lm,
        eval_dataset = eval_dataset_lm,
        peft_config = peft_config,
        script_args = script_args
    )

    del model
    del model_ref
        
    # SAVING TOKENIZER
    tokenizer.save_pretrained(os.path.join(script_args.output_dir, "generator_model"))

    # Define the directory path
    eval_dir = os.path.join(script_args.output_dir, script_args.eval_datasets)

    # Check if the directory exists
    if not os.path.exists(eval_dir):
        # Create the directory if it doesn't exist
        os.makedirs(eval_dir)

    eval_metrics = []

    for iter in range(script_args.bo_iters):

        ################################################################
        # PREPARE REWARD MODELS
        ################################################################
        if iter == 0:
            rw_model_path = os.path.join(script_args.output_dir, "generator_model")
            if not os.path.exists(rw_model_path):
                rw_model_path = script_args.model_name_or_path

            # rw_model = get_rw_model(
            #     script_args.model_name_or_path,
            #     quantization_config,
            #     device_map,
            #     script_args
            # )
                
            rw_model = get_rw_model_with_adapter(
                script_args.model_name_or_path,
                os.path.join(script_args.output_dir, "generator_model"), # Load new adapter model
                quantization_config,
                device_map,
                script_args
            )
            

            train_dataset_rw, eval_dataset_rw = get_dataset_for_reward(
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                max_seq_length=script_args.max_seq_length,
                sanity_check=script_args.sanity_check
            )
        else:
            ################################################################
            # TRAINING REWARD MODELS
            ################################################################
            print("="*10, " TRAINING REWARD MODEL ", "="*10)
            train_dataset_rw, eval_dataset_rw = get_dataset_for_reward(
                train_dataset=train_dataset, # ACTIVE QUERIES
                eval_dataset=eval_dataset,
                max_seq_length=script_args.max_seq_length,
                sanity_check=script_args.sanity_check
            )
        
            rw_model = get_rw_model_with_adapter(
                script_args.model_name_or_path,
                os.path.join(script_args.output_dir, "generator_model"), # Load new adapter model
                quantization_config,
                device_map,
                script_args
            )
            
            run_reward_training(
                model = rw_model,
                tokenizer = tokenizer,
                train_dataset = train_dataset_rw,
                eval_dataset = eval_dataset_rw,
                peft_config = peft_config_rw,
                script_args = script_args
            )
    
        if script_args.algo == "max_rw":
            ################################################################
            # INFERENCE AND SELECTING SAMPLES BY USING REWARD MODEL
            ################################################################
            print("="*10, " INFERENCE AND SELECTING SAMPLES BY USING REWARD MODEL ", "="*10)

            list_rw_value = []
            for sample in tqdm(unobserved_dataset):
                tokenized_sample = tokenizer(
                    [sample['chosen'], sample['rejected']], 
                    truncation=True, padding=True, 
                    return_tensors="pt"
                )
                rw_value = rw_model(**tokenized_sample).logits.mean(-1).sum()
                list_rw_value.append(rw_value.item())
            del rw_model
                
            list_rw_value = np.array(list_rw_value)
            selected_idxs = np.argpartition(list_rw_value, -script_args.topk_acqf)[-script_args.topk_acqf:]
            selected_samples = unobserved_dataset[selected_idxs]
            
        elif script_args.algo == "random":
            selected_idxs = np.random.choice(
                np.arange(len(unobserved_dataset)), 
                size=script_args.topk_acqf, 
                replace=False
            )
            selected_samples = unobserved_dataset[selected_idxs]
        else:
            raise NotImplementedError
            
        # Update train_dataset
        train_dataset = Dataset.from_dict(selected_samples) # ACTIVE QUERIES
        unobserved_dataset = unobserved_dataset.select(
            (
                i for i in range(len(unobserved_dataset)) 
                if i not in selected_idxs
            )
        )
        
        ################################################################
        # FINE_TUNING LLM (GENERATOR)
        ################################################################
        print("="*10, " FINETUNING GENERATOR ", "="*10)
        train_dataset_lm, eval_dataset_lm = get_dataset_for_finetuning(
            train_dataset=train_dataset, # ACTIVE QUERIES
            eval_dataset=eval_dataset,
            max_seq_length=script_args.max_seq_length,
            sanity_check=script_args.sanity_check
        )
    
        model = get_generator(
            os.path.join(script_args.output_dir, "generator_model"),
            quantization_config,
            device_map,
            script_args
        )

        model_ref = copy.deepcopy(model) if peft_config else None

        run_dpo_finetuning(
            model = model,
            model_ref = model_ref,
            tokenizer = tokenizer,
            train_dataset = train_dataset_lm,
            eval_dataset = eval_dataset_lm,
            peft_config = peft_config,
            script_args = script_args
        )
        
        del model
        del model_ref
        
        ################################################################
        # EVALUATING LLM (GENERATOR)
        ################################################################
        print("="*10, " EVALUATING GENERATOR ", "="*10)

        # Construct the full path including the file name
        eval_json_file = os.path.join(eval_dir, f'eval_results_{iter}.json')

        # Define the CLI command as a string
        cli_command = f'lm_eval --model hf\
                        --model_args pretrained={script_args.model_name_or_path},peft={os.path.join(script_args.output_dir, "generator_model")},trust_remote_code=True,load_in_4bit=True\
                        --tasks {script_args.eval_datasets} \
                        --device cuda:0 \
                        --batch_size 8 \
                        --output_path {eval_json_file} \
                        {"--limit 100" if script_args.sanity_check else ""}'
        
        exit_code = os.system(cli_command)
        # Check the exit code for success or failure
        if exit_code == 0:
            print("Command executed successfully")
        else:
            print("Command failed with exit code:", exit_code)

        with open(eval_json_file) as json_file:
            eval_results = json.load(json_file)

        eval_metrics.append({
            "iteration": iter,
            "metric": eval_results['results'][script_args.eval_datasets]
        })
        

    # Store final result
    all_result_path = os.path.join(script_args.output_dir, "all_result.json")
    try:
        with open(all_result_path, 'w') as json_file:
            json.dump(eval_metrics, json_file, indent=4)
        # Print message if successful
        print(f"Result has been stored at: {all_result_path}")
    except Exception as e:
        # Print error message if writing fails
        print(f"Error occurred while writing result: {e}")

# python run_pipeline.py --sanity_check True --init_samples 10 --bo_iters 10 --topk_acqf 10 --output_dir /lfs/local/0/sttruong/lhf
    