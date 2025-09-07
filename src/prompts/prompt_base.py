
class ConstrainPromptsCreatedMeta(type):
    """We want the n prompts created to be a factor of 10 for pipeline simplicity."""
    def __new__(cls, name, bases, class_dict):
        original_init = class_dict.get('__init__')
        
        def new_init(self, *args, **kwargs):
            if original_init:
                original_init(self, *args, **kwargs)
            if 'n_prompts_created_per_generation' in kwargs and \
                    kwargs['n_prompts_created_per_generation'] not in [1, 2, 5, 10, 20]:
                raise ValueError("n_prompts_created_per_generation must be a factor of 10")
            
        class_dict['__init__'] = new_init

        return type.__new__(cls, name, bases, class_dict)


class PromptBase(metaclass=ConstrainPromptsCreatedMeta):
    def __init__(self, entropy_file: str = 'cases/adding_entropy.txt') -> None:
        with open(entropy_file, 'r') as f:
            self.entropy_lines = f.readlines()

    def generative_prompt(self) -> str:
        raise NotImplementedError
    
    def generative_system_prompt(self) -> str:
        return ""
    
    def relevance_check_prompt(self, question: str) -> str:
        raise NotImplementedError
    
    def relevance_check_system_prompt(self) -> str:
        return ""
    
    def subject_evaluation_prompt(self, question: str, answer: str) -> str:
        raise NotImplementedError
    
    def subject_evaluation_system_prompt(self) -> str:
        return ""
    
    # def get_optional_subject_prompt_prepend(self) -> str:
    #     return ""
    
    def get_top_eval_score(self) -> int:
        raise NotImplementedError

    def subject_model_system_prompt(self) -> str:
        return ""
