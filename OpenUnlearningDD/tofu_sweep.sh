CUDA_VISIBLE_DEVICES=0 python src/eval.py experiment=eval/tofu/default \
  +model.model_handler=DD \
  +model.model_dd_big=open-unlearning/tofu_Llama-3.1-8B-Instruct_full \
  +model.model_dd_retain=open-unlearning/tofu_Llama-3.2-1B-Instruct_retain90 \
  +model.model_dd_forget=open-unlearning/tofu_Llama-3.2-1B-Instruct_full \
  +model.model_dd_alpha=0 \
  task_name=target

CUDA_VISIBLE_DEVICES=0 python src/eval.py experiment=eval/tofu/default \
  +model.model_handler=DD \
  +model.model_dd_big=open-unlearning/tofu_Llama-3.1-8B-Instruct_retain90 \
  +model.model_dd_retain=open-unlearning/tofu_Llama-3.2-1B-Instruct_retain90 \
  +model.model_dd_forget=open-unlearning/tofu_Llama-3.2-1B-Instruct_full \
  +model.model_dd_alpha=0 \
  task_name=retrain

for model_size in '3.2-1B' '3.2-3B'; do
   for alpha in 0.5 1.0 1.5 2.0 2.5 2.6 2.7 2.8 2.9 3.0 3.1 3.2 3.3 3.4 3.5; do
      CUDA_VISIBLE_DEVICES=0 python src/eval.py experiment=eval/tofu/default \
        +model.model_handler=DD \
        +model.model_dd_big=open-unlearning/tofu_Llama-3.1-8B-Instruct_full \
        +model.model_dd_retain=open-unlearning/tofu_Llama-$model_size-Instruct_retain90 \
        +model.model_dd_forget=open-unlearning/tofu_Llama-$model_size-Instruct_full \
        +model.model_dd_use_ngram=No \
        +model.model_dd_alpha=$alpha \
        task_name=alpha-$alpha-$model_size
    done
   for topk in 1 5 20 50 100; do
      CUDA_VISIBLE_DEVICES=0 python src/eval.py experiment=eval/tofu/default \
        +model.model_handler=DD \
        +model.model_dd_big=open-unlearning/tofu_Llama-3.1-8B-Instruct_full \
        +model.model_dd_retain=open-unlearning/tofu_Llama-$model_size-Instruct_retain90 \
        +model.model_dd_forget=open-unlearning/tofu_Llama-$model_size-Instruct_full \
        +model.model_dd_topk=$topk \
        +model.model_dd_use_ngram=No \
        +model.topk_vocab=TOFU \
        +model.model_dd_monte_carlo=Yes \
        task_name=tofu_rank/topk-$topk-$model_size
    done
done