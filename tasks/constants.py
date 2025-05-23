SharedTasks = ["math", "coding", "general_qa", "other"]
SharedTask2Id = {subtask: i for i, subtask in enumerate(SharedTasks)}
SharedTask2Category = {v: k for k, v in SharedTask2Id.items()}

BBHSubTasks = ["boolean_expressions", "causal_judgement", "date_understanding", "disambiguation_qa", "dyck_languages", "formal_fallacies", "geometric_shapes", "hyperbaton", "logical_deduction_five_objects", "logical_deduction_seven_objects", "logical_deduction_three_objects", "movie_recommendation", "multistep_arithmetic_two", "navigate",
               "object_counting", "penguins_in_a_table", "reasoning_about_colored_objects", "ruin_names", "salient_translation_error_detection", "snarks", "sports_understanding", "temporal_sequences", "tracking_shuffled_objects_five_objects", "tracking_shuffled_objects_seven_objects", "tracking_shuffled_objects_three_objects", "web_of_lies", "word_sorting"]
BBHSubTask2Id = {subtask: i for i, subtask in enumerate(BBHSubTasks)}
BBHId2SubTask = {v: k for k, v in BBHSubTask2Id.items()}

SmolTalkSources = ["smol-magpie-ultra", "smol-constraints", "smol-rewrite", "smol-summarize", "apigen-80k", "everyday-conversations",
                   "explore-instruct-rewriting", "longalign", "metamathqa-50k", "numina-cot-100k", "openhermes-100k", "self-oss-instruct", "systemchats-30k"]
SmolTalkSource2Id = {source: i for i, source in enumerate(SmolTalkSources)}
SmolTalkId2Source = {v: k for k, v in SmolTalkSource2Id.items()}

SmolTalkMagpieUltraCategories = ["advice-seeking", "brainstorming", "coding", "creative-writing",
                                 "data-analysis", "editing", "information-seeking", "math", "planning", "reasoning", "role-playing"]
SmolTalkMagpieUltraCategory2Id = {
    category: i for i, category in enumerate(SmolTalkMagpieUltraCategories)}
SmolTalkMagpieUltraId2Category = {v: k for k,
                                  v in SmolTalkMagpieUltraCategory2Id.items()}

MMLUCategories = ["abstract_algebra", "anatomy", "astronomy", "auxiliary_train", "business_ethics", "clinical_knowledge", "college_biology", "college_chemistry", "college_computer_science", "college_mathematics", "college_medicine", "college_physics", "computer_security", "conceptual_physics", "econometrics", "electrical_engineering", "elementary_mathematics", "formal_logic", "global_facts", "high_school_biology", "high_school_chemistry", "high_school_computer_science", "high_school_european_history", "high_school_geography", "high_school_government_and_politics", "high_school_macroeconomics", "high_school_mathematics",
                  "high_school_microeconomics", "high_school_physics", "high_school_psychology", "high_school_statistics", "high_school_us_history", "high_school_world_history", "human_aging", "human_sexuality", "international_law", "jurisprudence", "logical_fallacies", "machine_learning", "management", "marketing", "medical_genetics", "miscellaneous", "moral_disputes", "moral_scenarios", "nutrition", "philosophy", "prehistory", "professional_accounting", "professional_law", "professional_medicine", "professional_psychology", "public_relations", "security_studies", "sociology", "us_foreign_policy", "virology", "world_religions"]
MMLUCategory2Id = {source: i for i, source in enumerate(MMLUCategories)}
MMLUId2Category = {v: k for k, v in MMLUCategory2Id.items()}


MMLUProCategories = ["biology", "business", "chemistry", "computer science", "economics",
                     "engineering", "health", "history", "law", "math", "other", "philosophy", "physics", "psychology"]
MMLUProCategory2Id = {category: i for i,
                      category in enumerate(MMLUProCategories)}
MMLUProId2Category = {v: k for k, v in MMLUProCategory2Id.items()}


SmolTalkSources2Shared = {

    "smol-constraints": "other",

    "smol-rewrite": "other",

    "smol-summarize": "other",

    "apigen-80k": "other",

    "everyday-conversations": "other",

    "explore-instruct-rewriting": "other",

    "longalign": "other",

    "metamathqa-50k": "math",
    "numina-cot-100k": "math",
    "openhermes-100k": "general_qa",
    "self-oss-instruct": "coding",


    "systemchats-30k": "other"
}


SmolTalkMagpieUltraCategories2Shared = {

    "advice-seeking": "other",

    "brainstorming": "other",
    "coding": "coding",





    "data-analysis": "general_qa",

    "editing": "other",
    "information-seeking": "general_qa",
    "math": "math",

    "planning": "other",


    "reasoning": "general_qa",

    "role-playing": "other"
}


MMLUProCategories2Shared = {
    "biology": "general_qa",
    "business": "general_qa",
    "chemistry": "general_qa",
    "computer science": "general_qa",
    "economics": "general_qa",
    "engineering": "general_qa",
    "health": "general_qa",
    "history": "general_qa",
    "law": "general_qa",
    "math": "math",
    "other": "general_qa",
    "philosophy": "general_qa",
    "physics": "general_qa",
    "psychology": "general_qa"
}

MMLUCategories2Shared = {
    "abstract_algebra": "math",
    "anatomy": "general_qa",
    "astronomy": "general_qa",
    "business_ethics": "general_qa",
    "clinical_knowledge": "general_qa",
    "college_biology": "general_qa",
    "college_chemistry": "general_qa",
    "college_computer_science": "general_qa",
    "college_mathematics": "math",
    "college_medicine": "general_qa",
    "college_physics": "general_qa",
    "computer_security": "general_qa",
    "conceptual_physics": "general_qa",
    "econometrics": "general_qa",
    "electrical_engineering": "general_qa",
    "elementary_mathematics": "math",
    "formal_logic": "general_qa",
    "global_facts": "general_qa",
    "high_school_biology": "general_qa",
    "high_school_chemistry": "general_qa",
    "high_school_computer_science": "coding",
    "high_school_european_history": "general_qa",
    "high_school_geography": "general_qa",
    "high_school_government_and_politics": "general_qa",
    "high_school_macroeconomics": "general_qa",
    "high_school_mathematics": "math",
    "high_school_microeconomics": "general_qa",
    "high_school_physics": "general_qa",
    "high_school_psychology": "general_qa",
    "high_school_statistics": "math",
    "high_school_us_history": "general_qa",
    "high_school_world_history": "general_qa",
    "human_aging": "general_qa",
    "human_sexuality": "general_qa",
    "international_law": "general_qa",
    "jurisprudence": "general_qa",
    "logical_fallacies": "general_qa",
    "machine_learning": "general_qa",
    "management": "general_qa",
    "marketing": "general_qa",
    "medical_genetics": "general_qa",
    "miscellaneous": "general_qa",
    "moral_disputes": "general_qa",
    "moral_scenarios": "general_qa",
    "nutrition": "general_qa",
    "philosophy": "general_qa",
    "prehistory": "general_qa",
    "professional_accounting": "general_qa",
    "professional_law": "general_qa",
    "professional_medicine": "general_qa",
    "professional_psychology": "general_qa",
    "public_relations": "general_qa",
    "security_studies": "general_qa",
    "sociology": "general_qa",
    "us_foreign_policy": "general_qa",
    "virology": "general_qa",
    "world_religions": "general_qa"
}


def get_shared_category_dictionary(config, processor_filepath):
    dataset_name, dataset_subset, *_ = processor_filepath.split(',')
    guidance_categories = config.guidance_categories
    assert guidance_categories is not None
    category2id = {subtask: i for i, subtask in enumerate(guidance_categories)}
    if dataset_name == "HuggingFaceTB/smoltalk":
        if dataset_subset == 'smol-magpie-ultra':
            data_category2shared = {}
            for k, v in SmolTalkMagpieUltraCategories2Shared.items():
                if k not in config.exclude_train_data_category:
                    data_category2shared[k] = v
                    assert v in guidance_categories, (
                        f'Issue detected w/ subcategory {v})')
        else:
            category = SmolTalkSources2Shared[dataset_subset]
            assert category in guidance_categories, (
                f'Issue detected w/ subset {dataset_subset} ({category})')
            data_category2shared = {None: category}
    elif dataset_name.startswith("cais/mmlu"):
        data_category2shared = MMLUCategories2Shared
    elif dataset_name.startswith("TIGER-Lab/MMLU-Pro"):
        data_category2shared = MMLUProCategories2Shared
    elif dataset_name.startswith("openai/gsm8k"):
        category = 'math'
        assert category in guidance_categories
        data_category2shared = {None: category}
    elif dataset_name.startswith("lightevalMATH"):
        category = 'math'
        assert category in guidance_categories
        data_category2shared = {None: category}
    elif dataset_name.startswith("allenai/ai2_arc"):
        category = 'general_qa'
        assert category in guidance_categories
        data_category2shared = {None: category}
    elif dataset_name.startswith("ybisk/piqa"):
        category = 'general_qa'
        assert category in guidance_categories
        data_category2shared = {None: category}
    elif dataset_name.startswith("openai/openai_humaneval"):
        category = 'coding'
        assert category in guidance_categories
        data_category2shared = {None: category}
    elif dataset_name.startswith("codeparrot/instructhumaneval"):
        category = 'coding'
        assert category in guidance_categories
        data_category2shared = {None: category}
    elif dataset_name.startswith("google-research-datasets/mbpp"):
        category = 'coding'
        assert category in guidance_categories
        data_category2shared = {None: category}
    else:
        raise NotImplementedError

    data_category2id = {}
    for k, v in data_category2shared.items():
        data_category2id[k] = category2id[v]

    return category2id, data_category2id, data_category2shared


def get_total_num_categories(config,):
    for filepath in config.train_filepaths:
        dataset_name, dataset_subset, train_split = filepath.split(',')
        if dataset_name == "HuggingFaceTB/smoltalk":
            if config.use_data_split_guidance:
                guidance_modulation_num_classes = config.num_data_categories
                if guidance_modulation_num_classes is not None:
                    if len(guidance_modulation_num_classes) > 0:
                        raise NotImplementedError
                    else:
                        raise NotImplementedError
                else:
                    raise NotImplementedError
