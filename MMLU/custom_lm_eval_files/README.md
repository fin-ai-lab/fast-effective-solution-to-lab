cp custom_lm_eval_files/__init__.py lm-evaluation-harness/lm_eval/models/__init__.py
cp custom_lm_eval_files/count_mmlu_questions.py lm-evaluation-harness/lm_eval/models/
cp custom_lm_eval_files/two_model_gemma.py lm-evaluation-harness/lm_eval/models/
cp custom_lm_eval_files/selection.py lm-evaluation-harness/lm_eval/filters/selection.py
rm -rf /lm-evaluation-harness/lm_eval/tasks/mmlu/default/
cp -r custom_lm_eval_files/default/ lm-evaluation-harness/lm_eval/tasks/mmlu/default/