import os
import openai
import json
import itertools
from utils.prompt_pool import PromptGenerator
import fastapi_poe
import asyncio
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
poe_api_key = os.getenv("POE_API_KEY")


def stringtolist(description):
    return [descriptor[2:] for descriptor in description.split('\n') if (descriptor != '') and (descriptor.startswith('- '))]


def get_next_number(args, folder_path):
    files = os.listdir(folder_path)
    numbers = []
    for file in files:
        if file.startswith(f"{args.in_dataset}_{args.llm_model}") and file.endswith(".json"):
            try:
                number = int(file.split('_')[-1].split('.')[0])
                numbers.append(number)
            except ValueError:
                pass
    
    return 0 if not numbers else max(numbers) + 1


def partition(lst, size):
    for i in range(0, len(lst), size):
        yield list(itertools.islice(lst, i, i + size))
        

async def poe_chat(args, prompt):
    message = fastapi_poe.ProtocolMessage(role="user", content=prompt)
    response = ""
    retry_count = 0 
    max_retries = 10
    while retry_count < max_retries:
        try:
            async for partial in fastapi_poe.get_bot_response(messages=[message], bot_name=args.llm_model, api_key=poe_api_key, temperature=0.0):
                response += partial.text
            break 
        except Exception as e:
            print(f"Encountered an error: {e}. Retrying in 1 second...")
            await asyncio.sleep(1)
            retry_count += 1

    if retry_count == max_retries:
        print("Reached maximum retry attempts. Exiting.")
    return response


async def poe_chat_multiple_times(args, context, prompt):
    user_message = fastapi_poe.ProtocolMessage(role="user", content=prompt)
    context.append(user_message)
    response = ""
    retry_count = 0 
    max_retries = 10
    while retry_count < max_retries:
        try:
            async for partial in fastapi_poe.get_bot_response(messages=context, bot_name=args.llm_model, api_key=poe_api_key, temperature=0.0):
                response += partial.text
            break 
        except Exception as e:
            print(f"Encountered an error: {e}. Retrying in 1 second...")
            await asyncio.sleep(1)
            retry_count += 1

    context.append(fastapi_poe.ProtocolMessage(role="bot", content=response))

    return response, context


def get_completion(args, prompt):
    """Fetches GPT responses for given prompts."""
    try:
        response = openai.ChatCompletion.create(model=args.llm_model,
                                                  messages=[{"role": "user", "content": prompt}],
                                                  temperature=0.0)
        return response.choices[0].message["content"]
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return []


def get_completion_from_messages(args, context, new_prompt):
    try:
        context.append({"role": "user", "content": new_prompt})
        response = openai.ChatCompletion.create(
            model=args.llm_model,
            messages=context,
            temperature=0.0,
        )
        
        generated_text = response.choices[0].message["content"]
        context.append({"role": "assistant", "content": generated_text})
        return generated_text, context

    except Exception as e:
        print(f"An error occurred: {e}")
        return []


async def obtain_gpt_class_and_save(args, file_path, class_list):
    descriptors = {}
    prompt_gen = PromptGenerator()
    if args.llm_model in ['gpt-3.5-turbo', 'gpt-3.5-turbo-16k', 'gpt-4', 'gpt-4-1106-preview', 'gpt-4-0125-preview']:
        if args.ood_task in ['far', 'fine_grained', 'general', 
                             'fine_grained_irrelevant', 'fine_grained_dissimilar',
                             'far_irrelevant', 'far_dissimilar',]:
            # due the max_length limit, we guide LLMs to generate 50 outlier classes, thus we need to envision more times for L > 50
            envision_nums = 50
            envision_times = max(int(args.L/envision_nums)-1, 0)
            prompts = prompt_gen.get_prompt(ood_task=args.ood_task, in_dataset=args.in_dataset, class_info=class_list, envision_nums=envision_nums)
            response_texts = get_completion(args, prompts)

            context = []
            context.append({"role": "user", "content": prompts})
            context.append({"role": "assistant", "content": response_texts})

            for i in range(envision_times):
                print(i)
                new_prompt = prompt_gen.get_prompt_again(ood_task=args.ood_task, in_dataset=args.in_dataset, class_info=class_list, envision_nums=envision_nums)
                generated_text, context = get_completion_from_messages(args, context, new_prompt)
                response_texts = response_texts + generated_text
            
            descriptors_list = stringtolist(response_texts)
            descriptors = {args.ood_task: descriptors_list}
            
        elif args.ood_task in ['near', 'near_irrelevant', 'near_dissimilar']:
            envision_nums = args.L
            print(prompt_gen.get_prompt(ood_task=args.ood_task, in_dataset=args.in_dataset, class_info='dog', envision_nums=envision_nums))

            prompts = [prompt_gen.get_prompt(ood_task=args.ood_task, in_dataset=args.in_dataset, class_info=category.replace('_', ' '), envision_nums=envision_nums)for category in class_list]
            response_texts = [get_completion(args, prompt) for prompt in prompts]
            descriptors_list = [stringtolist(response_text) for response_text in response_texts]
            descriptors = {cat: descr for cat, descr in zip(class_list, descriptors_list)}
    else:
        if args.ood_task in ['far', 'fine_grained', 'general', 
                             'fine_grained_irrelevant', 'fine_grained_dissimilar',
                             'far_irrelevant', 'far_dissimilar',]:
            # due the max_length limit, we guide LLMs to generate 50 outlier classes, thus we need to envision more times for L > 50
            envision_nums = 50
            envision_times = max(int(args.L/envision_nums)-1, 0)
            prompts = prompt_gen.get_prompt(ood_task=args.ood_task, in_dataset=args.in_dataset, class_info=class_list, envision_nums=envision_nums)
            # we don't use 'prefix' in GPT, since GPT can follow our JSON format without the 'prefix' below
            prefix = "Before my question, I will give you an example; please strictly follow my answer format(just use '-' before every answer) and just give me the answer (the answer starts with -), no other words!!!  Do not include my questions and examples in your answer.\n"
            prompts = prefix + prompts
            response_texts = await poe_chat(args, prompts)

            context = []
            context.append(fastapi_poe.ProtocolMessage(role="user", content=prompts))
            context.append(fastapi_poe.ProtocolMessage(role="bot", content=response_texts))

            for i in range(envision_times):
                print(i)
                new_prompt = prompt_gen.get_prompt_again(ood_task=args.ood_task, in_dataset=args.in_dataset, class_info=class_list)
                generated_text, context = await poe_chat_multiple_times(args, context, new_prompt)
                response_texts = response_texts + generated_text
            
            descriptors_list = stringtolist(response_texts)
            descriptors = {args.ood_task: descriptors_list}
            
        elif args.ood_task in ['near', 'near_irrelevant', 'near_dissimilar']:
            envision_nums = args.L
            prefix = "Before my question, I will give you three examples; please strictly follow my answer format(just use '-' before every answer) and just give me the answer (your answer should start with '-'), no other words!!!  Do not include my questions and examples in your answer.\n"
            prompts = [prefix + prompt_gen.get_prompt(ood_task=args.ood_task, in_dataset=args.in_dataset, class_info=category.replace('_', ' '), envision_nums=envision_nums) for category in class_list]

            response_texts = []
            for prompt in prompts:
                response = await poe_chat(args, prompt)
                response_texts.append(response)
            descriptors_list = [stringtolist(response_text) for response_text in response_texts]
            descriptors = {cat: descr for cat, descr in zip(class_list, descriptors_list)}

    print(response_texts)
    with open(file_path, 'w') as fp:
        json.dump(descriptors, fp)


def load_llm_classes(args, test_labels):
    if args.ood_task in ['far', 'fine_grained', 'general', 
                        'fine_grained_irrelevant', 'fine_grained_dissimilar',
                        'far_irrelevant', 'far_dissimilar',]:
        assert args.L in [100, 300, 500]
    elif args.ood_task in ['near', 'near_irrelevant', 'near_dissimilar']:
        assert args.L in [1, 3, 10]
    else:
        raise NotImplementedError

    folder_path = os.path.join("envisioned_classes", f'{args.ood_task}_{args.L}')
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    if args.generate_class:
        print('Envisioning Outlier Exposure...')
        json_number = get_next_number(args, folder_path)
        file_path = os.path.join(folder_path, f"{args.in_dataset}_{args.llm_model}_{json_number}.json")
        asyncio.run(obtain_gpt_class_and_save(args, file_path, test_labels))
    else:
        if args.in_dataset.startswith("ImageNet_C") or args.in_dataset in ['ImageNet_sketch']:
            file_path = os.path.join(folder_path, f"ImageNet_{args.llm_model}_{args.json_number}.json")
        else:
            file_path = os.path.join(folder_path, f"{args.in_dataset}_{args.llm_model}_{args.json_number}.json")
    print('=== load json: ', file_path)
    gpt_class_dict = load_json(file_path)
    print('Get Envisioned Candidate Class Names.')

    gpt_class = []
    for key, value in gpt_class_dict.items():
        gpt_class.extend(value)
    
    # remove repeat
    gpt_class = [item.lower() for item in gpt_class]
    gpt_class = list(set(gpt_class))
    print('After set: ', len(gpt_class))
    
    test_labels = [item.lower() for item in test_labels]
    gpt_class = [item for item in gpt_class if item.lower() not in test_labels]
    print('After test set: ', len(gpt_class))

    return gpt_class


def load_json(filename):
    if not filename.endswith('.json'):
        filename += '.json'
    with open(filename, 'r') as fp:
        return json.load(fp)