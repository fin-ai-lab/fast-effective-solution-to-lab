# # Target and Retrain 

CUDA_VISIBLE_DEVICES=1 python src/eval.py experiment=eval/muse/default.yaml \
  data_split=News \
  +model.model_handler=DD \
  +model.model_dd_use_ngram=No \
  +model.model_dd_big=muse-bench/MUSE-news_target \
  +model.model_dd_retain=models/model_1/ \
  +model.model_dd_forget=models/model_2/ \
  +model.model_dd_alpha=0 \
  task_name=muse_target

CUDA_VISIBLE_DEVICES=0 python src/eval.py experiment=eval/muse/default.yaml \
  data_split=News \
  +model.model_handler=DD \
  +model.model_dd_use_ngram=No \
  +model.model_dd_big=muse-bench/MUSE-news_retrain \
  +model.model_dd_retain=models/model_1/ \
  +model.model_dd_forget=models/model_2/ \
  +model.model_dd_alpha=0 \
  task_name=muse_retrain

# # DD experiments

for model in "1.3b" "2.7b"; do
    for alpha in 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5 2.0 2.5 3.0; do
        CUDA_VISIBLE_DEVICES=0 python src/eval.py experiment=eval/muse/default.yaml \
        data_split=News \
        +model.model_handler=DD \
        +model.model_dd_use_ngram=No \
        +model.model_dd_big=muse-bench/MUSE-news_target \
        +model.model_dd_retain=models/$model/model_1/ \
        +model.model_dd_forget=models/$model/model_2/ \
        +model.model_dd_alpha=$alpha \
        task_name=muse_main/muse-$model-alpha-$alpha
    done
    
    for topk in 1 5 20 50 100 200 500 1000; do
        CUDA_VISIBLE_DEVICES=0 python src/eval.py experiment=eval/muse/default.yaml \
        data_split=News \
        +model.model_handler=DD \
        +model.model_dd_use_ngram=No \
        +model.model_dd_big=muse-bench/MUSE-news_target \
        +model.model_dd_retain=models/$model/model_1/ \
        +model.model_dd_forget=models/$model/model_2/ \
        +model.model_dd_topk=$topk \
        +model.topk_vocab=MUSE \
        +model.model_dd_monte_carlo=Yes \
        task_name=muse_main/muse-$model-topk-$topk
    done
done

# # Trigram model experiments

for topk in 1 2 3 5 10; do
  CUDA_VISIBLE_DEVICES=0 python src/eval.py experiment=eval/muse/default.yaml \
  data_split=News \
    +model.model_handler=DD \
    +model.model_dd_big=muse-bench/MUSE-news_target \
    +model.model_dd_use_ngram=Yes \
    +model.model_dd_retain=[data/news/raw/retain1.txt] \
    +model.model_dd_forget=[data/news/raw/forget.txt] \
    +model.model_dd_topk=$topk \
    +model.topk_vocab=MUSE \
    task_name=muse-Trigram-topk-$topk
done

for alpha in 5 10 15 20 25 30; do
  CUDA_VISIBLE_DEVICES=0 python src/eval.py experiment=eval/muse/default.yaml \
    +model.model_handler=DD \
    +model.model_dd_big=muse-bench/MUSE-news_target \
    +model.model_dd_use_ngram=Yes \
    +model.model_dd_retain=[data/news/raw/retain1.txt] \
    +model.model_dd_forget=[data/news/raw/forget.txt] \
    +model.model_dd_alpha=$alpha \
    task_name=muse-Trigram-alpha-$alpha
done

# Scaling and Sustainability benchmarks
for number in "3" "4" "5" "6" "7" "8"; do
  CUDA_VISIBLE_DEVICES=0 python src/eval.py experiment=eval/muse/default.yaml \
    data_split=News \
    +model.model_handler=DD \
    +model.model_dd_use_ngram=No \
    +model.model_dd_big=muse-bench/MUSE-news_target \
    +model.model_dd_retain=models/1.3b/model_1/ \
    +model.model_dd_forget=models/1.3b/model_$number/ \
    +model.model_dd_alpha=0.8 \
    task_name=muse-scalsust-linear-1.3b-$number

  CUDA_VISIBLE_DEVICES=0 python src/eval.py experiment=eval/muse/default.yaml \
    data_split=News \
    +model.model_handler=DD \
    +model.model_dd_use_ngram=No \
    +model.model_dd_big=muse-bench/MUSE-news_target \
    +model.model_dd_retain=models/1.3b/model_1/ \
    +model.model_dd_forget=models/1.3b/model_$number/ \
    +model.model_dd_topk=1000 \
    +model.topk_vocab=MUSE \
    task_name=muse-scalsust-rank-1.3b-$number
done