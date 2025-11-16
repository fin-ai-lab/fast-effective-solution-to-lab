#!/bin/bash
export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"

models=(
    # "Llama-3.2-1B-Instruct"
    # "Llama-3.2-3B-Instruct"
    "Llama-3.1-8B-Instruct"
)

# Define trainers, their experiments, and their corresponding number of epochs
declare -A trainer_experiments
declare -A trainer_epochs

trainer_experiments["GradAscent"]="unlearn/tofu/default.yaml"
trainer_experiments["GradDiff"]="unlearn/tofu/default.yaml"
trainer_experiments["NPO"]="unlearn/tofu/default.yaml"
trainer_experiments["DPO"]="unlearn/tofu/idk.yaml"
trainer_experiments["RMU"]="unlearn/tofu/default.yaml"

trainer_epochs["GradAscent"]=10
trainer_epochs["GradDiff"]=10
trainer_epochs["NPO"]=10
trainer_epochs["DPO"]=10
trainer_epochs["RMU"]=10

splits=(
    # "forget01 holdout01 retain99"
    # "forget05 holdout05 retain95"
    "forget10 holdout10 retain90"
)

per_device_train_batch_size=16 # on two gpus would make effective batch size 32
gradient_accumulation_steps=1

########################################################################################################################
########################################### Unlearn TOFU models ########################################################
########################################################################################################################

for split in "${splits[@]}"; do
    forget_split=$(echo $split | cut -d' ' -f1)
    holdout_split=$(echo $split | cut -d' ' -f2)
    retain_split=$(echo $split | cut -d' ' -f3)
    
    for model in "${models[@]}"; do
        for trainer in "${!trainer_epochs[@]}"; do
            epochs=${trainer_epochs[$trainer]}
            experiment=${trainer_experiments[$trainer]}
           
            task_name=tofu_${model}_${forget_split}_${trainer}
            model_path=open-unlearning/tofu_${model}_full
            echo ${task_name}: Unlearning ${model_path} using ${trainer} for ${epochs} epochs
            
            # Unlearn
            CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file configs/accelerate/default_config.yaml --main_process_port $MASTER_PORT \
            src/train.py --config-name=unlearn.yaml \
            experiment=${experiment} \
            trainer=${trainer} \
            task_name=${task_name} \
            model=${model} \
            forget_split=${forget_split} \
            retain_split=${retain_split} \
            model.model_args.pretrained_model_name_or_path=${model_path} \
            retain_logs_path=saves/eval/tofu_${model}_${retain_split}/TOFU_EVAL.json \
            trainer.args.per_device_train_batch_size=$per_device_train_batch_size \
            trainer.args.gradient_accumulation_steps=$gradient_accumulation_steps \
            trainer.args.num_train_epochs=${epochs} \
            trainer.args.ddp_find_unused_parameters=true \
            trainer.args.gradient_checkpointing=true \
            trainer.args.save_strategy="epoch"
            
            # Eval - Run two jobs in parallel on separate GPUs
            checkpoints=(13 26 39 52 65 78 91 104 117 130)
            for ((i=0; i<${#checkpoints[@]}; i+=2)); do
                checkpoint1=${checkpoints[i]}
                checkpoint2=${checkpoints[i+1]}
                
                echo "Evaluating checkpoints ${checkpoint1} and ${checkpoint2} in parallel"
                
                # Start first eval job on GPU 0 in background
                CUDA_VISIBLE_DEVICES=0 python src/eval.py \
                experiment=eval/tofu/default.yaml \
                forget_split=${forget_split} \
                holdout_split=${holdout_split} \
                model=${model} \
                task_name=${task_name}/checkpoint-${checkpoint1} \
                model.model_args.pretrained_model_name_or_path=saves/unlearn/${task_name}/checkpoint-${checkpoint1} \
                paths.output_dir=saves/unlearn/${task_name}/checkpoint-${checkpoint1}/evals \
                retain_logs_path=saves/eval/tofu_${model}_${retain_split}/TOFU_EVAL.json &
                
                # Start second eval job on GPU 1 in background (if it exists)
                if [[ -n "$checkpoint2" ]]; then
                    CUDA_VISIBLE_DEVICES=1 python src/eval.py \
                    experiment=eval/tofu/default.yaml \
                    forget_split=${forget_split} \
                    holdout_split=${holdout_split} \
                    model=${model} \
                    task_name=${task_name}/checkpoint-${checkpoint2} \
                    model.model_args.pretrained_model_name_or_path=saves/unlearn/${task_name}/checkpoint-${checkpoint2} \
                    paths.output_dir=saves/unlearn/${task_name}/checkpoint-${checkpoint2}/evals \
                    retain_logs_path=saves/eval/tofu_${model}_${retain_split}/TOFU_EVAL.json &
                fi
                
                # Wait for both jobs to complete before starting next pair
                wait
                echo "Completed evaluation of checkpoints ${checkpoint1} and ${checkpoint2}"
            done
        done
    done
done