from lm_eval.api.model import LM
from lm_eval.api.registry import register_model

@register_model("count-mmlu-questions")
class TwoModelGemma(LM):
    def __init__(self, run_index) -> None:
        super().__init__()
        self.run_index = run_index
            
    @classmethod
    def create_from_arg_string(cls, arg_string, additional_config=None):
        _, run_index = arg_string.split("=")
        if not run_index.isdigit():
            raise ValueError(f"Invalid run index: {run_index}. Expected a digit.")

        return cls(run_index=int(run_index))

    def generate_until(self, requests, disable_tqdm: bool = False):
        res = []
        number_ran = 0
        number_questions_in_each = {}
        number_questions_done_each = {}

        for i, request in enumerate(requests):
            res.append("")
            if request.task_name not in number_questions_in_each:
                number_questions_in_each[request.task_name] = 0
                number_questions_done_each[request.task_name] = 0
            number_questions_in_each[request.task_name] += 1
            if i % self.run_index == 0: 
                print(request)
                number_questions_done_each[request.task_name] += 1

        print(number_questions_done_each)
        print(number_questions_in_each)
        return res

    def loglikelihood():
        pass

    def loglikelihood_rolling():
        pass